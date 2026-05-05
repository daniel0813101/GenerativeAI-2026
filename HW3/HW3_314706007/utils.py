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

    _REFERENCES_RE = re.compile(
        r"\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n",
        re.MULTILINE,
    )

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

        # Drop references/bibliography — they pollute BM25 with entity-shaped
        # tokens (author names, citation keys) that aren't the paper's claims.
        m = self._REFERENCES_RE.search(text)
        if m and m.start() > 1000:  # only trim if substantial body precedes it
            text = text[: m.start()]

        return text.strip()

    _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(\[])")

    @staticmethod
    def chunk(
        text: str,
        min_chars: int = 80,
        window: int = 4,
        stride: int = 2,
        min_alpha_ratio: float = 0.55,
    ) -> List[str]:
        """Sliding-window sentence chunks, with a noise filter.

        Sentence-level granularity gives retrieval more focused targets than
        paragraph splits (which can be either huge sections or one-line
        headers). Stride < window keeps overlap so a relevant span isn't
        split across two non-retrieved chunks.

        Drops chunks that look like tables/equations/figure captions:
        the alphabetic-char ratio collapses for those.
        """
        sentences = [
            s.strip()
            for s in PDFParser._SENT_SPLIT_RE.split(text)
            if len(s.strip()) >= 20
        ]
        chunks: List[str] = []
        for i in range(0, max(len(sentences) - 1, 1), stride):
            piece = " ".join(sentences[i : i + window]).strip()
            if len(piece) < min_chars:
                continue
            alpha_ratio = sum(c.isalpha() for c in piece) / len(piece)
            if alpha_ratio < min_alpha_ratio:
                continue
            chunks.append(piece)
        return chunks

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
    """Multi-query hybrid retrieval combined via Reciprocal Rank Fusion.

    Three rankings, all merged via RRF:
      1. BM25 over the full review text — generic lexical match
      2. Embedding similarity (all-MiniLM-L6-v2) — paraphrased semantic match
      3. BM25 over extracted *keys* from the review (capitalized terms,
         numbers, quoted phrases, citation tags) — focused on the exact
         tokens that drive Entity/Number hallucinations

    Stream 3 is the key change: when the review says "BERT baseline" or
    "95% accuracy" or "[Bos02]", we *force* a query that hunts for those
    specific tokens, not just whatever BM25 weights from the full sentence.

    Budget is measured in whitespace-separated words.
    """

    _RRF_K = 60  # standard RRF damping constant

    # Patterns that surface hallucination-relevant tokens in review text.
    _KEY_PATTERNS = [
        re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:[-_][A-Za-z0-9]+)+\b"),                       # ResNet-50, GPT-4
        re.compile(r"\b[A-Z]{2,}[a-zA-Z0-9]*\b"),                                          # BERT, MARL, CTDE
        re.compile(r"\b\d+(?:\.\d+)?[%KMBkmb]?\b"),                                        # 95%, 1.2M, 7.25
        re.compile(r"\b\d+(?:\.\d+)?[eE][-+]?\d+\b"),                                      # 1.2e-3, scientific notation
        re.compile(r"\b\d+/\d+\b"),                                                        # fractions: 3/4, 1/15
        re.compile(r"\[([A-Za-z]+\d{0,4})\]"),                                             # [Bos02], [A]
        re.compile(r'"([^"]{3,40})"'),                                                      # quoted phrases
        # Temporal cues — multi-word phrases (single modal verbs are too common, low IDF)
        re.compile(r"\b(?:future work|plans? to|aims? to|going to|will be|has been|have been|had been)\b", re.IGNORECASE),
        re.compile(r"\b(?:previously|previous work|prior work|existing methods?|already|used to)\b", re.IGNORECASE),
        re.compile(r"\b(?:19|20)\d{2}\b"),                                                  # years 1900-2099
    ]

    def __init__(
        self,
        top_k: int = 5,
        max_tokens: int = 600,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_to: int = 3,
    ):
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.reranker_name = reranker_name
        self.rerank_to = rerank_to  # 0 disables reranking
        self._model = None
        self._reranker = None
        self._index: Dict[str, tuple] = {}  # paper_id -> (chunks, bm25, chunk_embs)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _ensure_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_name)
        return self._reranker

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self._ensure_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # dot product = cosine similarity
            show_progress_bar=False,
        )

    def _build_index(self, chunks: List[str]):
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([c.lower().split() for c in chunks])
        embs = self._encode(chunks)
        return chunks, bm25, embs

    @classmethod
    def _extract_keys(cls, text: str) -> List[str]:
        """Pull entity/number/citation/quote tokens out of the review text."""
        keys: List[str] = []
        for pat in cls._KEY_PATTERNS:
            for m in pat.findall(text):
                tok = m if isinstance(m, str) else next((g for g in m if g), "")
                tok = tok.strip().lower()
                if tok and len(tok) >= 2:
                    keys.append(tok)
        # de-dup preserving order
        seen, out = set(), []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def retrieve(self, query: str, paper_id: str, chunks: List[str]) -> str:
        if not chunks:
            return ""
        if paper_id not in self._index:
            self._index[paper_id] = self._build_index(chunks)

        stored_chunks, bm25, chunk_embs = self._index[paper_id]
        n = len(stored_chunks)

        # Three rankings to fuse, with weighted RRF. The keys-query is a
        # tiebreaker, not a primary driver — equal weight caused Entity to
        # over-predict (any review with a capitalized term retrieved
        # entity-shaped chunks, biasing the model toward the Entity rule).
        rankings = [
            (1.0, np.argsort(-bm25.get_scores(query.lower().split()))),  # BM25 full
            (1.0, np.argsort(-(chunk_embs @ self._encode([query])[0]))),  # embedding
        ]
        keys = self._extract_keys(query)
        if keys:
            rankings.append((0.3, np.argsort(-bm25.get_scores(keys))))    # BM25 keys (further down-weighted)

        rrf = np.zeros(n, dtype=np.float32)
        for w, ranking in rankings:
            for r, idx in enumerate(ranking):
                rrf[idx] += w / (self._RRF_K + r)

        top_idx_full = sorted(range(n), key=lambda i: -rrf[i])[: self.top_k]

        # Cross-encoder rerank picks `rerank_to` chunks by joint relevance.
        # Keep only the reranked chunks; the former extra raw-RRF packing
        # over-expanded evidence and hurt the current dev score.
        if self.rerank_to and len(top_idx_full) > self.rerank_to:
            reranker = self._ensure_reranker()
            pairs = [(query, stored_chunks[i]) for i in top_idx_full]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(top_idx_full, ce_scores), key=lambda x: -x[1])
            top_idx = [idx for idx, _ in ranked[: self.rerank_to]]
        else:
            top_idx = top_idx_full

        top_idx = sorted(top_idx)  # restore document order

        selected, token_count = [], 0
        for i in top_idx:
            words = stored_chunks[i].split()
            n_words = len(words)
            if token_count + n_words > self.max_tokens:
                remaining = self.max_tokens - token_count
                if remaining >= 50:
                    selected.append(" ".join(words[:remaining]))
                break
            selected.append(stored_chunks[i])
            token_count += n_words

        if not selected:
            top_words = stored_chunks[top_idx[0]].split()
            selected.append(" ".join(top_words[: self.max_tokens]))

        return "\n\n".join(selected)


# ── Prompt Building ───────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You audit AI-generated peer reviews for hallucinations. Compare the review sentence against the paper evidence and identify the single hallucination type.

Hallucination types:
{class_block}

Decision rules — check in this order, NOT alphabetical:
- If the paper has a related paragraph but with a different name/term/method → Entity (NOT Attribution Failure).
- If the paper has a related paragraph but with a different number → Number (NOT Attribution Failure).
- If the paper has a related paragraph and the review's claim is broader or more absolute than what is shown → Overgeneralization (NOT Attribution Failure).
- If the paper has a related paragraph but the tense, modality, or time reference differs → Temporal (NOT Attribution Failure).
- Use Attribution Failure ONLY when the paper has no related paragraph at all, or the review credits the wrong source.

Output: one short sentence describing the discrepancy, then end with exactly:
The type is: <class_name>
where <class_name> ∈ {{Attribution Failure, Entity, Number, Overgeneralization, Temporal}}."""


# Training responses are deliberately mixed:
# - AF / Entity / Number use compact class templates from the stronger #8-style
#   lineage. The full contrastive version pushed too many dev examples into
#   Entity.
# - Overgeneralization / Temporal keep more explicit boundary language because
#   the latest dev run under-predicted both classes.
_REASONING_TEMPLATES: Dict[str, str] = {
    "Attribution Failure": (
        "The review's claim cannot be properly grounded in the paper evidence; "
        "the source is misattributed or no supporting passage exists. "
        "The type is: Attribution Failure"
    ),
    "Entity": (
        "The review references a noun phrase, method, dataset, or technical term "
        "that does not match what the paper actually states. "
        "The type is: Entity"
    ),
    "Number": (
        "The review states a numerical value that differs from the corresponding "
        "number in the paper. The type is: Number"
    ),
}


def _build_training_response(class_name: str, review: str, evidence: str) -> str:  # noqa: ARG001
    """Supervised assistant response used as the generated label target."""
    if class_name in _REASONING_TEMPLATES:
        return _REASONING_TEMPLATES[class_name]
    if class_name == "Overgeneralization":
        return (
            "The paper evidence supports a related but narrower claim, while the "
            "review states it too broadly or too absolutely. This is not "
            "Attribution Failure because related evidence exists; the problem is "
            "the unsupported scope of the claim. "
            "The type is: Overgeneralization"
        )
    if class_name == "Temporal":
        return (
            "The paper evidence discusses related content, but the review changes "
            "the tense, modality, or time reference, such as treating future, "
            "past, possible, or planned work as a different temporal claim. "
            "The type is: Temporal"
        )
    return f"The review's claim does not match the paper evidence. The type is: {class_name}"


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

    _MAX_REVIEW_WORDS = 400

    def _cap_review(self, text: str) -> str:
        # Cap long review texts (paragraph-length reviews with bullets) so the
        # prompt fits in max_seq_len. The hallucinated claim is in the first
        # ~few sentences; later bullets rarely add classification signal.
        words = text.split()
        if len(words) > self._MAX_REVIEW_WORDS:
            return " ".join(words[: self._MAX_REVIEW_WORDS]) + " […]"
        return text

    def build(
        self,
        text: str,
        evidence: str,
        label_id: int | None = None,
        verify_for: str | None = None,
    ) -> List[Dict[str, str]]:
        text = self._cap_review(text)
        if verify_for is not None:
            # Self-verification pass: present the first prediction, force the
            # model to re-examine the comparison rather than rubber-stamp it.
            user = (
                f"Review sentence:\n{text}\n\n"
                f"Paper evidence:\n{evidence}\n\n"
                f"An initial pass classified this as: {verify_for}.\n"
                "Re-examine the review against the paper evidence carefully. "
                "Walk through each rule and decide whether the initial classification holds. "
                "If correct, restate it. If not, output the correct classification."
            )
        else:
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
            response = _build_training_response(class_name, text, evidence)
            messages.append({"role": "assistant", "content": response})
        return messages

# ── Output Parsing ────────────────────────────────────────────────────────────

_TYPE_LINE_RE = re.compile(r"the\s+type\s+is\s*:?\s*([^\n]+)", re.IGNORECASE)
_DECISION_LINE_RE = re.compile(
    r"(?:final\s+answer|answer|classification|class)\s*(?:is|:)\s*([^\n]+)",
    re.IGNORECASE,
)


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

    def _match_first_mentioned(self, segment: str) -> int | None:
        """Return the first class name mentioned in a decision tail."""
        exact = self._match_exact(segment)
        if exact is not None:
            return exact

        seg_lower = segment.lower()
        hits = []
        for name in self.names:
            pos = seg_lower.find(name.lower())
            if pos >= 0:
                hits.append((pos, -len(name), name))
        if not hits:
            return None
        _, _, name = min(hits)
        return self.name_to_id[name]

    def _match_exact(self, segment: str) -> int | None:
        cleaned = segment.strip().rstrip(".").strip(" '\"`*")
        if cleaned in self.name_to_id:
            return self.name_to_id[cleaned]
        cleaned_lower = cleaned.lower()
        for name in self.names:
            if cleaned_lower == name.lower():
                return self.name_to_id[name]
        return None

    def parse_with_fallback(self, generated: str) -> tuple[int, bool]:
        """Parse generated class text.

        Returns (label_id, used_fallback). The parser prefers the final valid
        "The type is:" line so self-verification outputs that mention
        an earlier class do not accidentally keep the stale label.
        """
        text = generated.strip()

        # 1) Primary: use the final valid "The type is: <class_name>" line.
        for match in reversed(list(_TYPE_LINE_RE.finditer(text))):
            tail = match.group(1).strip().rstrip(".").strip(" '\"`*")
            hit = self._match_first_mentioned(tail)
            if hit is not None:
                return hit, False

        # 2) Secondary: accept final answer/classification lines.
        for match in reversed(list(_DECISION_LINE_RE.finditer(text))):
            tail = match.group(1).strip().rstrip(".").strip(" '\"`*")
            hit = self._match_first_mentioned(tail)
            if hit is not None:
                return hit, False

        # 3) If the whole output is just a class name, accept it.
        hit = self._match_exact(text)
        if hit is not None:
            return hit, False

        # 4) Fuzzy fallback only for short near-label outputs. Long reasoning
        # often contains negated class names, so whole-output fuzzy matching
        # would be too eager.
        if len(text) <= 80:
            matches = difflib.get_close_matches(text, self.names, n=1, cutoff=0.6)
            if matches:
                return self.name_to_id[matches[0]], False

        # 5) Conservative class-prior fallback for truly unparsable outputs.
        return self._default_id, True

    def parse(self, generated: str) -> int:
        label_id, _ = self.parse_with_fallback(generated)
        return label_id


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
