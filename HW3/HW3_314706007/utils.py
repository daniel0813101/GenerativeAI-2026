from __future__ import annotations

import difflib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


# ── PDF path resolution ───────────────────────────────────────────────────────

def find_pdf(pdf_dir: Path, paper_id: str) -> Path:
    """Search flat layout and train/dev/test subfolders for a paper PDF."""
    for candidate in [
        pdf_dir / f"{paper_id}.pdf",
        pdf_dir / "train" / f"{paper_id}.pdf",
        pdf_dir / "dev"   / f"{paper_id}.pdf",
        pdf_dir / "test"  / f"{paper_id}.pdf",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"PDF for '{paper_id}' not found under {pdf_dir}")


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
            try:
                self.parse(pid, find_pdf(pdf_dir, pid))
            except FileNotFoundError:
                pass


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
You are an expert hallucination auditor for AI-generated peer reviews of academic papers.

Your task: given a peer review sentence and relevant paper passages, determine which type of hallucination the review sentence contains by directly comparing the claim in the review against what the paper actually states.

Hallucination types:
{class_block}

Reasoning steps:
1. Identify the specific factual claim made in the review sentence.
2. Locate the corresponding content in the paper evidence.
3. Identify the exact discrepancy: wrong entity? wrong number? too broad? wrong tense or modality? unsupported attribution?
4. Map the discrepancy to exactly one of the five hallucination types.

Output format:
First, write a single short sentence describing the discrepancy.
Then end your response with exactly this line:
The type is: <class_name>
where <class_name> must be one of: Attribution Failure, Entity, Number, Overgeneralization, Temporal."""


# Class-grounded reasoning templates used as the supervised CoT signal during
# training. The model has to classify correctly to pick the right template,
# which forces it to attend to the input.
_REASONING_TEMPLATES: Dict[str, str] = {
    "Attribution Failure": "The review's claim cannot be properly grounded in the paper evidence — the source is misattributed or no supporting passage exists.",
    "Entity": "The review references a noun phrase (a name, method, dataset, or technical term) that does not match what the paper actually states.",
    "Number": "The review states a numerical value that differs from the corresponding number in the paper.",
    "Overgeneralization": "The review draws a conclusion that is broader or more absolute than what the paper's evidence actually supports.",
    "Temporal": "The review misrepresents tense, modality, or time reference relative to how the paper frames the same content.",
}


def _assistant_response(class_name: str) -> str:
    reasoning = _REASONING_TEMPLATES.get(
        class_name,
        "The review's claim does not match the paper evidence.",
    )
    return f"{reasoning} The type is: {class_name}"


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
            f"Review sentence:\n{text}\n\n"
            f"Paper evidence:\n{evidence}\n\n"
            "Compare the review sentence with the paper evidence above and identify the hallucination type."
        )
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user},
        ]
        if label_id is not None:
            class_name = self.id_to_name[label_id]
            messages.append({"role": "assistant", "content": _assistant_response(class_name)})
        return messages


# ── Output Parsing ────────────────────────────────────────────────────────────

_TYPE_LINE_RE = re.compile(r"the\s+type\s+is\s*:?\s*([^\n]+)", re.IGNORECASE)


class OutputParser:
    def __init__(self, classes_path: str | Path):
        with open(classes_path, encoding="utf-8") as f:
            raw = json.load(f)
        classes = raw if isinstance(raw, list) else [
            {"id": v["id"], "concept": k} for k, v in raw.items()
        ]
        self.name_to_id: Dict[str, int] = {c["concept"]: c["id"] for c in classes}
        self.names = list(self.name_to_id)
        # Longest names first — avoids "entity"/"number" matching as substrings
        # of longer phrases in free-form output.
        self._names_by_len = sorted(self.names, key=len, reverse=True)
        self._default_id: int = classes[0]["id"]

    def _match_in(self, segment: str) -> int | None:
        if segment in self.name_to_id:
            return self.name_to_id[segment]
        seg_lower = segment.lower()
        for name in self._names_by_len:
            if name.lower() in seg_lower:
                return self.name_to_id[name]
        return None

    def parse(self, generated: str) -> int:
        text = generated.strip()

        # 1) Primary: extract the tail of "The type is: <class_name>"
        match = _TYPE_LINE_RE.search(text)
        if match:
            tail = match.group(1).strip().rstrip(".").strip(" '\"`*")
            hit = self._match_in(tail)
            if hit is not None:
                return hit

        # 2) Fallback: search the whole output, longest name first
        hit = self._match_in(text)
        if hit is not None:
            return hit

        # 3) Fuzzy fallback for typos / casing drift
        matches = difflib.get_close_matches(text, self.names, n=1, cutoff=0.4)
        if matches:
            return self.name_to_id[matches[0]]

        return self._default_id


# ── Class Balancing ───────────────────────────────────────────────────────────

def oversample(
    df: pd.DataFrame,
    label_col: str = "label",
    max_multiplier: int = 10,
) -> pd.DataFrame:
    """Repeat minority-class rows until each class reaches min(majority count, count * max_multiplier).

    Targeting the majority count (instead of the median) gives a near-balanced
    dataset. The 15:1 imbalance described in the assignment is the core
    challenge — for Temporal (130 rows) we expand to ~1300 with multiplier=10.
    """
    counts = df[label_col].value_counts()
    target_count = int(counts.max())

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


# ── Data collator with sample weights ────────────────────────────────────────

@dataclass
class WeightedDataCollator:
    """Pad a batch of pre-tokenized samples and pass through per-sample loss weights."""
    tokenizer: object
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        weights = [float(f.pop("sample_weight", 1.0)) for f in features]

        # Extract labels before passing to tokenizer.pad — it only handles
        # input_ids / attention_mask and will error on variable-length label lists.
        labels_list = [f.pop("labels") for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Right-pad labels with -100 to match the padded input length
        max_len = batch["input_ids"].shape[1]
        padded_labels = [
            lbl + [-100] * (max_len - len(lbl)) for lbl in labels_list
        ]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["sample_weights"] = torch.tensor(weights, dtype=torch.float32)
        return batch


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_macro_f1(predictions: List[int], references: List[int]) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(references, predictions, average="macro", zero_division=0))


def print_prediction_distribution(
    name: str,
    preds: List[int],
    id_to_name: Dict[int, str],
) -> None:
    counts = Counter(preds)
    total = len(preds)
    print(f"[{name}] prediction distribution ({total} samples):")
    for cls_id in sorted(id_to_name):
        c = counts.get(cls_id, 0)
        pct = 100.0 * c / total if total else 0.0
        print(f"  {cls_id} {id_to_name[cls_id]:<22s} {c:>5d}  ({pct:5.1f}%)")
