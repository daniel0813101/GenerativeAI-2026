from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── PDF Parsing ───────────────────────────────────────────────────────────────

class PDFParser:
    """Extract and cache plain text from paper PDFs, with chunking."""

    def __init__(self, cache_dir: str | Path = "paper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, paper_id: str, pdf_path: str | Path) -> str:
        cache_path = self.cache_dir / f"{paper_id}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        text = self._extract(Path(pdf_path))
        cache_path.write_text(text, encoding="utf-8")
        return text

    def _extract(self, pdf_path: Path) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
        except Exception:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            pages = [p.extract_text() or "" for p in reader.pages]

        text = "\n\n".join(pages)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def chunk(text: str, min_len: int = 60) -> List[str]:
        return [c.strip() for c in text.split("\n\n") if len(c.strip()) >= min_len]

    def preprocess_all(self, pdf_dir: str | Path, paper_ids: List[str]) -> None:
        """Warm-up cache: parse every paper once before training begins."""
        pdf_dir = Path(pdf_dir)
        from tqdm import tqdm
        for pid in tqdm(paper_ids, desc="Parsing PDFs"):
            pdf_path = pdf_dir / f"{pid}.pdf"
            if pdf_path.exists():
                self.parse(pid, pdf_path)


# ── Evidence Retrieval ────────────────────────────────────────────────────────

class EvidenceRetriever:
    """BM25-based retrieval of the most relevant paper chunks for a query."""

    def __init__(self, top_k: int = 5, max_tokens: int = 1500):
        self.top_k = top_k
        self.max_tokens = max_tokens
        self._index: Dict[str, tuple] = {}

    def _build(self, chunks: List[str]):
        from rank_bm25 import BM25Okapi
        return BM25Okapi([c.lower().split() for c in chunks])

    def retrieve(self, query: str, paper_id: str, chunks: List[str]) -> str:
        if not chunks:
            return ""
        if paper_id not in self._index:
            self._index[paper_id] = (chunks, self._build(chunks))

        stored_chunks, bm25 = self._index[paper_id]
        scores = bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.top_k]
        top_idx = sorted(top_idx)  # restore document order for coherence

        selected, token_count = [], 0
        for i in top_idx:
            n = len(stored_chunks[i].split())
            if token_count + n > self.max_tokens:
                break
            selected.append(stored_chunks[i])
            token_count += n

        return "\n\n".join(selected) if selected else stored_chunks[0]


# ── Prompt Building ───────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are an expert at detecting hallucinations in AI-generated peer reviews of academic papers.
A hallucination occurs when a review makes a claim that is incorrect, unsupported, or distorted
relative to the actual paper content.

Hallucination types:
{class_block}

Instructions:
1. Identify the specific claim made in the review sentence.
2. Check whether the paper evidence supports or contradicts that claim.
3. Choose the hallucination type that best explains the discrepancy.
4. Output ONLY the exact class name — nothing else.\
"""


class PromptBuilder:
    def __init__(self, classes_path: str | Path):
        with open(classes_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.classes = raw if isinstance(raw, list) else [
            {"id": v["id"], "concept": k, "concept_desc": v.get("concept_desc", "")}
            for k, v in raw.items()
        ]

        self.id_to_name: Dict[int, str] = {c["id"]: c["concept"] for c in self.classes}
        self.name_to_id: Dict[str, int] = {c["concept"]: c["id"] for c in self.classes}

        class_block = "\n".join(
            f"- {c['concept']}: {c.get('concept_desc', '')}" for c in self.classes
        )
        self._system = _SYSTEM_TEMPLATE.format(class_block=class_block)

    def build(
        self,
        text: str,
        evidence: str,
        label_id: int | None = None,
    ) -> List[Dict[str, str]]:
        user = (
            f"Review sentence: {text}\n\n"
            f"Relevant paper evidence:\n{evidence}\n\n"
            "What type of hallucination does this review sentence contain?"
        )
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user},
        ]
        if label_id is not None:
            messages.append({"role": "assistant", "content": self.id_to_name[label_id]})
        return messages


# ── Output Parsing ────────────────────────────────────────────────────────────

class OutputParser:
    def __init__(self, classes_path: str | Path):
        with open(classes_path, encoding="utf-8") as f:
            raw = json.load(f)
        classes = raw if isinstance(raw, list) else [
            {"id": v["id"], "concept": k} for k, v in raw.items()
        ]
        self.name_to_id: Dict[str, int] = {c["concept"]: c["id"] for c in classes}
        self.names = list(self.name_to_id)
        self._default_id: int = classes[0]["id"]

    def parse(self, generated: str) -> int:
        text = generated.strip()
        # Exact match
        if text in self.name_to_id:
            return self.name_to_id[text]
        # Case-insensitive substring match
        lower = text.lower()
        for name, cls_id in self.name_to_id.items():
            if name.lower() in lower:
                return cls_id
        # Fuzzy match as last resort
        matches = difflib.get_close_matches(text, self.names, n=1, cutoff=0.4)
        if matches:
            return self.name_to_id[matches[0]]
        return self._default_id


# ── Class Balancing ───────────────────────────────────────────────────────────

def oversample(
    df: pd.DataFrame,
    label_col: str = "label",
    max_multiplier: int = 5,
) -> pd.DataFrame:
    """Repeat minority-class rows until each class reaches min(median, count * max_multiplier)."""
    counts = df[label_col].value_counts()
    target_count = int(counts.median())

    parts = []
    for label, count in counts.items():
        subset = df[df[label_col] == label]
        desired = min(target_count, count * max_multiplier)
        if desired > count:
            extra = subset.sample(desired - count, replace=True, random_state=42)
            subset = pd.concat([subset, extra], ignore_index=True)
        parts.append(subset)

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


# ── Tokenization helpers ──────────────────────────────────────────────────────

def find_last_sublist(lst: List[int], sublist: List[int]) -> int:
    """Return the start index of the last occurrence of sublist in lst, or -1."""
    n = len(sublist)
    for i in range(len(lst) - n, -1, -1):
        if lst[i : i + n] == sublist:
            return i
    return -1


def mask_prompt_tokens(input_ids: List[int], assistant_header_ids: List[int]) -> List[int]:
    """Copy input_ids into labels and set -100 for all tokens up to and including the assistant header."""
    labels = input_ids.copy()
    pos = find_last_sublist(input_ids, assistant_header_ids)
    if pos >= 0:
        mask_end = pos + len(assistant_header_ids)
        for i in range(mask_end):
            labels[i] = -100
    return labels


# ── Data collator with sample weights ────────────────────────────────────────

@dataclass
class WeightedDataCollator:
    """Pad a batch of pre-tokenized samples and pass through per-sample loss weights."""
    tokenizer: object
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        weights = [float(f.pop("sample_weight", 1.0)) for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # tokenizer.pad fills labels with pad_token_id; replace with -100
        if "labels" in batch:
            batch["labels"] = batch["labels"].masked_fill(
                batch["labels"] == self.tokenizer.pad_token_id, -100
            )
        batch["sample_weights"] = torch.tensor(weights, dtype=torch.float32)
        return batch


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_macro_f1(predictions: List[int], references: List[int]) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(references, predictions, average="macro", zero_division=0))
