
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from sentence_transformers import CrossEncoder, SentenceTransformer


console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class Chunk:
    """A retrieval chunk and its tokenized representation."""

    chunk_id: int
    text: str
    tokens: list[str]


@dataclass
class TrainConfig:
    """Configuration for the development RAG pipeline.

    This dataclass is the source of truth for defaults. CLI flags are generated
    from this schema, and parsed CLI values are converted back into this class.
    """

    input: str = str(SCRIPT_DIR / "dataset" / "private_dataset.json")
    output_root: str = str(SCRIPT_DIR / "checkpoints")
    run_name: str | None = None
    submission_filename: str = "submission.json"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    disable_rerank: bool = False
    chunk_size: int = 700
    chunk_overlap: int = 100
    retrieval_k: int = 14
    evidence_k: int = 3
    dynamic_evidence: bool = True
    min_evidence_k: int = 2
    max_evidence_k: int = 6
    evidence_score_threshold: float = 0.78
    rrf_k: int = 60
    use_bge_instructions: bool = True
    evidence_mode: str = "sentence"
    evidence_source_chunks: int = 4
    evidence_max_chars: int = 320
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 220

    @classmethod
    def build_arg_parser(cls, defaults: TrainConfig | None = None) -> argparse.ArgumentParser:
        """Build an argument parser aligned with ``TrainConfig`` fields.

        Args:
            defaults: Optional base config whose values are used as CLI defaults.

        Returns:
            argparse.ArgumentParser: Parser with arguments mapped to config fields.
        """
        defaults = defaults or cls()
        parser = argparse.ArgumentParser(description="HW2 baseline RAG pipeline (development)")
        parser.add_argument(
            "--input",
            type=str,
            default=defaults.input,
            help="Input dataset path",
        )
        parser.add_argument(
            "--output-root",
            type=str,
            default=defaults.output_root,
            help="Root directory for timestamped run folders",
        )
        parser.add_argument(
            "--run-name",
            type=str,
            default=defaults.run_name,
            help="Optional run folder name (default: current timestamp)",
        )
        parser.add_argument(
            "--submission-filename",
            type=str,
            default=defaults.submission_filename,
            help="Filename for submission JSON inside each run folder",
        )
        parser.add_argument(
            "--embedding-model",
            type=str,
            default=defaults.embedding_model,
            help="SentenceTransformer embedding model",
        )
        parser.add_argument(
            "--rerank-model",
            type=str,
            default=defaults.rerank_model,
            help="CrossEncoder reranker model",
        )
        parser.add_argument("--disable-rerank", action="store_true", help="Disable reranking stage")
        parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size, help="Chunk size in characters")
        parser.add_argument(
            "--chunk-overlap",
            type=int,
            default=defaults.chunk_overlap,
            help="Chunk overlap in characters",
        )
        parser.add_argument(
            "--retrieval-k",
            type=int,
            default=defaults.retrieval_k,
            help="Candidates kept after hybrid retrieval",
        )
        parser.add_argument(
            "--evidence-k",
            type=int,
            default=defaults.evidence_k,
            help="Fixed evidence count per question (used when --no-dynamic-evidence)",
        )
        parser.add_argument(
            "--dynamic-evidence",
            action=argparse.BooleanOptionalAction,
            default=defaults.dynamic_evidence,
            help="Use adaptive evidence count per question",
        )
        parser.add_argument(
            "--min-evidence-k",
            type=int,
            default=defaults.min_evidence_k,
            help="Minimum evidence count when dynamic evidence is enabled",
        )
        parser.add_argument(
            "--max-evidence-k",
            type=int,
            default=defaults.max_evidence_k,
            help="Maximum evidence count when dynamic evidence is enabled",
        )
        parser.add_argument(
            "--evidence-score-threshold",
            type=float,
            default=defaults.evidence_score_threshold,
            help="Adaptive threshold in [0,1] for selecting additional evidence",
        )
        parser.add_argument("--rrf-k", type=int, default=defaults.rrf_k, help="RRF constant for rank fusion")
        parser.add_argument(
            "--use-bge-instructions",
            action=argparse.BooleanOptionalAction,
            default=defaults.use_bge_instructions,
            help="Use BGE query/passage instruction prefixes for embeddings",
        )
        parser.add_argument(
            "--evidence-mode",
            type=str,
            choices=("chunk", "sentence"),
            default=defaults.evidence_mode,
            help="Evidence output mode",
        )
        parser.add_argument(
            "--evidence-source-chunks",
            type=int,
            default=defaults.evidence_source_chunks,
            help="How many top chunks to mine snippets from when evidence-mode=sentence",
        )
        parser.add_argument(
            "--evidence-max-chars",
            type=int,
            default=defaults.evidence_max_chars,
            help="Max characters per evidence snippet",
        )
        parser.add_argument(
            "--ollama-base-url",
            type=str,
            default=defaults.ollama_base_url,
            help="Ollama server URL",
        )
        parser.add_argument("--llm-model", type=str, default=defaults.llm_model, help="Ollama model name")
        parser.add_argument(
            "--llm-temperature",
            type=float,
            default=defaults.llm_temperature,
            help="Generation temperature",
        )
        parser.add_argument(
            "--llm-max-tokens",
            type=int,
            default=defaults.llm_max_tokens,
            help="Max output tokens",
        )
        return parser

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> TrainConfig:
        """Create a config object from parsed CLI arguments.

        Args:
            namespace: Parsed namespace from ``argparse``.

        Returns:
            TrainConfig: Validated config instance.

        Raises:
            ValueError: If CLI keys do not exactly align with dataclass fields.
        """
        values = vars(namespace)
        field_names = {f.name for f in fields(cls)}
        missing = field_names - values.keys()
        unknown = values.keys() - field_names
        if missing or unknown:
            raise ValueError(
                "CLI arguments and TrainConfig fields are misaligned. "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**{name: values[name] for name in field_names})


def parse_args() -> TrainConfig:
    """Parse CLI arguments and return a validated ``TrainConfig``.

    Returns:
        TrainConfig: Parsed and validated configuration object.
    """
    parser = TrainConfig.build_arg_parser()
    namespace = parser.parse_args()
    return TrainConfig.from_namespace(namespace)


def resolve_run_name(explicit_name: str | None, now: datetime | None = None) -> str:
    """Resolve run folder name from explicit value or timestamp.

    Args:
        explicit_name: User-provided run name.
        now: Optional datetime override, mainly for deterministic tests.

    Returns:
        str: A filesystem-safe run folder name.
    """
    if explicit_name and explicit_name.strip():
        return explicit_name.strip()
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def create_run_dir(output_root: str, run_name: str) -> Path:
    """Create a unique run directory.

    If the target name already exists, the function appends an incremental
    suffix (e.g. ``_01``) until an unused directory is found.

    Args:
        output_root: Root directory where runs are stored.
        run_name: Preferred run folder name.

    Returns:
        Path: Newly created run directory path.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    candidate = root / run_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    index = 1
    while True:
        next_candidate = root / f"{run_name}_{index:02d}"
        if not next_candidate.exists():
            next_candidate.mkdir(parents=True, exist_ok=False)
            return next_candidate
        index += 1


def write_json(path: Path, payload: object) -> None:
    """Write JSON payload to disk with UTF-8 encoding.

    Args:
        path: Destination path.
        payload: JSON-serializable object.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_run_folder_structure(run_dir: Path, submission_filename: str) -> None:
    """Print the checkpoint folder structure for one run.

    Args:
        run_dir: Generated run directory.
        submission_filename: Submission JSON filename under ``run_dir``.
    """
    console.print("[bold]Run folder structure:[/bold]")
    console.print(f"{run_dir}/")
    console.print("├── config.json")
    console.print("├── run_info.json")
    console.print("├── per_item_stats.json")
    console.print(f"└── {submission_filename}")


def normalize_spaces(text: str) -> str:
    """Normalize redundant spaces and trim surrounding whitespace.

    Args:
        text: Raw input text.

    Returns:
        str: Text with repeated spaces/tabs collapsed.
    """
    return re.sub(r"[ \t]+", " ", text.strip())


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens.

    Args:
        text: Input text to tokenize.

    Returns:
        list[str]: Tokens used for lexical retrieval.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    """Split a long paragraph into chunk-size-bounded units.

    The splitter first tries sentence boundaries and falls back to hard slicing
    when a sentence itself exceeds ``chunk_size``.

    Args:
        paragraph: Paragraph text to split.
        chunk_size: Maximum size for each returned unit in characters.

    Returns:
        list[str]: Paragraph segments no longer than ``chunk_size`` when possible.
    """
    if len(paragraph) <= chunk_size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    result: list[str] = []
    current: list[str] = []
    cur_len = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        add_len = len(sent) + (1 if current else 0)
        if cur_len + add_len <= chunk_size:
            current.append(sent)
            cur_len += add_len
        else:
            if current:
                result.append(" ".join(current))
            if len(sent) <= chunk_size:
                current = [sent]
                cur_len = len(sent)
            else:
                for i in range(0, len(sent), chunk_size):
                    result.append(sent[i : i + chunk_size])
                current = []
                cur_len = 0
    if current:
        result.append(" ".join(current))
    return result


def chunk_text(full_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Convert full paper text into chunked retrieval units.

    The function segments by paragraphs, then merges adjacent units up to
    ``chunk_size``. If overlap is enabled, each chunk (except the first) gets a
    prefix from the previous chunk.

    Args:
        full_text: Full document text.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        list[str]: Chunked strings ready for indexing.
    """
    paragraphs = [normalize_spaces(p) for p in re.split(r"\n{2,}", full_text) if p.strip()]

    units: list[str] = []
    for para in paragraphs:
        units.extend(split_long_paragraph(para, chunk_size))

    base_chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for unit in units:
        sep_len = 2 if buf else 0
        if buf_len + sep_len + len(unit) <= chunk_size:
            buf.append(unit)
            buf_len += sep_len + len(unit)
            continue
        if buf:
            base_chunks.append("\n\n".join(buf))
        buf = [unit]
        buf_len = len(unit)
    if buf:
        base_chunks.append("\n\n".join(buf))

    if not base_chunks:
        raw = normalize_spaces(full_text)
        if not raw:
            return []
        return [raw[:chunk_size]]

    if chunk_overlap <= 0:
        return base_chunks

    overlapped: list[str] = []
    for idx, current in enumerate(base_chunks):
        if idx == 0:
            overlapped.append(current)
            continue
        prefix = base_chunks[idx - 1][-chunk_overlap:]
        merged = (prefix + "\n" + current).strip()
        overlapped.append(merged)
    return overlapped


def build_chunks(full_text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Build ``Chunk`` objects with text and token metadata.

    Args:
        full_text: Full paper text.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Character overlap between neighboring chunks.

    Returns:
        list[Chunk]: Chunk objects with pre-tokenized text.
    """
    chunk_texts = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for idx, text in enumerate(chunk_texts):
        tokens = tokenize(text)
        if not tokens:
            continue
        chunks.append(Chunk(chunk_id=idx, text=text, tokens=tokens))
    return chunks


def build_bm25_stats(chunks: Sequence[Chunk]) -> tuple[dict[str, float], float]:
    """Compute BM25 corpus statistics for a paper-level index.

    Args:
        chunks: Chunks from one document.

    Returns:
        tuple[dict[str, float], float]:
            - Token-level IDF map.
            - Average document length (in tokens).
    """
    n_docs = len(chunks)
    if n_docs == 0:
        return {}, 0.0

    df = Counter()
    total_len = 0
    for chunk in chunks:
        total_len += len(chunk.tokens)
        for tok in set(chunk.tokens):
            df[tok] += 1

    idf = {}
    for tok, freq in df.items():
        idf[tok] = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
    avgdl = total_len / n_docs
    return idf, avgdl


def bm25_scores(
    query_tokens: Sequence[str],
    chunks: Sequence[Chunk],
    idf: dict[str, float],
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    """Score chunks with BM25 for a tokenized query.

    Args:
        query_tokens: Query tokens.
        chunks: Candidate chunks.
        idf: BM25 IDF map computed from the same chunk set.
        avgdl: Average chunk length.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        np.ndarray: BM25 score vector aligned to ``chunks``.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if avgdl <= 0:
        return np.zeros(len(chunks), dtype=np.float32)

    q_counter = Counter(query_tokens)
    scores = np.zeros(len(chunks), dtype=np.float32)

    for i, chunk in enumerate(chunks):
        tf = Counter(chunk.tokens)
        dl = len(chunk.tokens)
        score = 0.0
        for term, qf in q_counter.items():
            if term not in tf:
                continue
            term_idf = idf.get(term, 0.0)
            tf_term = tf[term]
            denom = tf_term + k1 * (1 - b + b * dl / avgdl)
            score += term_idf * ((tf_term * (k1 + 1)) / max(denom, 1e-9)) * qf
        scores[i] = float(score)
    return scores


def argsort_desc(values: np.ndarray) -> list[int]:
    """Return indices sorted by descending score.

    Args:
        values: Score vector.

    Returns:
        list[int]: Indices sorted from largest to smallest value.
    """
    if values.size == 0:
        return []
    return np.argsort(-values).tolist()


def reciprocal_rank_fusion(rank_lists: Sequence[Sequence[int]], rrf_k: int) -> dict[int, float]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion (RRF).

    Args:
        rank_lists: Ranked index lists from different retrievers.
        rrf_k: RRF dampening constant.

    Returns:
        dict[int, float]: Fused score per chunk index.
    """
    scores: dict[int, float] = {}
    for rank_list in rank_lists:
        for rank_idx, doc_idx in enumerate(rank_list, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank_idx)
    return scores


def topn_indices(score_map: dict[int, float], n: int) -> list[int]:
    """Select top-N indices from a score dictionary.

    Args:
        score_map: Mapping from index to score.
        n: Number of indices to keep.

    Returns:
        list[int]: Highest-scoring indices in descending score order.
    """
    return [idx for idx, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:n]]


def make_answer_prompt(title: str, question: str, evidence_texts: Sequence[str]) -> str:
    """Create a grounded QA prompt from selected evidence.

    Args:
        title: Paper title.
        question: User question for the paper.
        evidence_texts: Retrieved evidence passages.

    Returns:
        str: Prompt sent to the generation model.
    """
    evidence_block = "\n\n".join(f"[Evidence {i+1}]\n{txt}" for i, txt in enumerate(evidence_texts))
    return (
        "You are answering a question about one NLP research paper.\n"
        "Rules:\n"
        "1) Use only the provided evidence.\n"
        "2) If evidence is insufficient, say exactly: Insufficient evidence.\n"
        "3) Keep the answer concise (1-2 sentences).\n\n"
        f"Paper title: {title}\n"
        f"Question: {question}\n\n"
        f"{evidence_block}\n\n"
        "Answer:"
    )


def to_text_response(resp: object) -> str:
    """Normalize different LLM response shapes into plain text.

    Args:
        resp: Raw response object from the LLM client.

    Returns:
        str: Extracted textual content.
    """
    if hasattr(resp, "content"):
        content = getattr(resp, "content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return " ".join(parts).strip()
    return str(resp).strip()


def ensure_evidence_count(evidence: list[str], k: int) -> list[str]:
    """Normalize, deduplicate, and clamp evidence list length.

    Args:
        evidence: Candidate evidence strings.
        k: Requested evidence count.

    Returns:
        list[str]: Evidence list with 1..40 entries, deduplicated and cleaned.
    """
    k = max(1, min(40, k))
    filtered = [normalize_spaces(x) for x in evidence if x and x.strip()]
    dedup: list[str] = []
    seen = set()
    for item in filtered:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    if not dedup:
        return ["Insufficient evidence."]
    return dedup[:k]


def apply_bge_query_prefix(question: str, use_bge_instructions: bool) -> str:
    """Apply BGE query instruction prefix when enabled.

    Args:
        question: Original question text.
        use_bge_instructions: Whether to add BGE-specific instruction prefix.

    Returns:
        str: Query text used for embedding.
    """
    q = normalize_spaces(question)
    if not use_bge_instructions:
        return q
    return f"Represent this question for retrieving relevant passages: {q}"


def apply_bge_passage_prefix(texts: Sequence[str], use_bge_instructions: bool) -> list[str]:
    """Apply BGE passage prefix for document-side embeddings when enabled.

    Args:
        texts: Chunk or sentence texts.
        use_bge_instructions: Whether to add passage prefix.

    Returns:
        list[str]: Texts ready for embedding.
    """
    if not use_bge_instructions:
        return [normalize_spaces(t) for t in texts]
    return [f"passage: {normalize_spaces(t)}" for t in texts]


def split_sentences(text: str) -> list[str]:
    """Split text into candidate evidence sentences.

    Args:
        text: Input text.

    Returns:
        list[str]: Sentence-like units with light filtering.
    """
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    result: list[str] = []
    for piece in pieces:
        s = normalize_spaces(piece)
        if len(s) < 25:
            continue
        result.append(s)
    return result


def lexical_overlap_score(sentence: str, question_tokens: Sequence[str]) -> float:
    """Compute a simple lexical overlap score for one sentence.

    Args:
        sentence: Candidate sentence.
        question_tokens: Tokenized question.

    Returns:
        float: Overlap score in [0, 1].
    """
    if not question_tokens:
        return 0.0
    sent_tokens = set(tokenize(sentence))
    if not sent_tokens:
        return 0.0
    q_set = set(question_tokens)
    return len(sent_tokens.intersection(q_set)) / len(q_set)


def trim_evidence_length(text: str, max_chars: int) -> str:
    """Trim evidence text to a maximum character count.

    Args:
        text: Evidence text.
        max_chars: Character limit.

    Returns:
        str: Trimmed text.
    """
    text = normalize_spaces(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def adaptive_k_from_scores(
    scores_desc: Sequence[float],
    min_k: int,
    max_k: int,
    threshold: float,
) -> int:
    """Choose adaptive evidence count from descending confidence scores.

    The score threshold is applied on per-question min-max normalized scores,
    so it is robust to differences in score scale across questions.

    Args:
        scores_desc: Scores sorted in descending order.
        min_k: Minimum number of items to keep.
        max_k: Maximum number of items to keep.
        threshold: Normalized score threshold in [0, 1].

    Returns:
        int: Selected number of items.
    """
    if not scores_desc:
        return 0

    n = min(len(scores_desc), max(1, max_k))
    min_k = max(1, min(min_k, n))
    threshold = max(0.0, min(1.0, threshold))

    window = list(scores_desc[:n])
    top = window[0]
    tail = window[-1]
    if abs(top - tail) <= 1e-8:
        return n

    selected = min_k
    for i in range(min_k, n):
        normalized = (window[i] - tail) / (top - tail)
        if normalized >= threshold:
            selected = i + 1
        else:
            break
    return selected


def select_sentence_evidence(
    question: str,
    chunks: Sequence[Chunk],
    chunk_indices: Sequence[int],
    embedder: SentenceTransformer,
    question_emb: np.ndarray,
    use_bge_instructions: bool,
    evidence_k: int,
    dynamic_evidence: bool,
    min_evidence_k: int,
    max_evidence_k: int,
    evidence_score_threshold: float,
    evidence_source_chunks: int,
    evidence_max_chars: int,
) -> list[str]:
    """Select sentence-level evidence from top-ranked chunks.

    Args:
        question: Original question text.
        chunks: All chunks in the current paper.
        chunk_indices: Ranked chunk indices.
        embedder: Embedding model for sentence scoring.
        question_emb: Precomputed question embedding.
        use_bge_instructions: Whether BGE prefixes are enabled.
        evidence_k: Number of evidence snippets to return.
        dynamic_evidence: Whether to adaptively choose evidence count.
        min_evidence_k: Minimum evidence count in adaptive mode.
        max_evidence_k: Maximum evidence count in adaptive mode.
        evidence_score_threshold: Adaptive normalized score threshold.
        evidence_source_chunks: Number of top chunks to mine sentences from.
        evidence_max_chars: Maximum characters per snippet.

    Returns:
        list[str]: Ranked sentence-level evidence snippets.
    """
    source_count = max(1, min(len(chunk_indices), evidence_source_chunks))
    source_indices = list(chunk_indices[:source_count])

    candidates: list[str] = []
    for idx in source_indices:
        candidates.extend(split_sentences(chunks[idx].text))
    if not candidates:
        return []

    sent_inputs = apply_bge_passage_prefix(candidates, use_bge_instructions)
    sent_emb = embedder.encode(sent_inputs, normalize_embeddings=True, convert_to_numpy=True)
    dense_scores = sent_emb @ question_emb
    q_tokens = tokenize(question)

    blended: list[tuple[float, str]] = []
    for sent, dense_score in zip(candidates, dense_scores):
        lex = lexical_overlap_score(sent, q_tokens)
        score = float(dense_score) + 0.15 * lex
        blended.append((score, sent))

    ranked_pairs = sorted(blended, key=lambda x: x[0], reverse=True)
    ranked_scores = [score for score, _ in ranked_pairs]
    ranked = [s for _, s in ranked_pairs]

    if dynamic_evidence:
        target_k = adaptive_k_from_scores(
            ranked_scores,
            min_k=min_evidence_k,
            max_k=max_evidence_k,
            threshold=evidence_score_threshold,
        )
    else:
        target_k = max(1, min(evidence_k, len(ranked)))

    deduped: list[str] = []
    seen = set()
    for sent in ranked:
        trimmed = trim_evidence_length(sent, evidence_max_chars)
        if trimmed in seen:
            continue
        seen.add(trimmed)
        deduped.append(trimmed)
        if len(deduped) >= target_k:
            break
    return deduped


def run_pipeline(config: TrainConfig) -> tuple[list[dict], list[dict]]:
    """Run end-to-end RAG inference on a dataset file.

    Args:
        config: Runtime configuration for retrieval and generation.

    Returns:
        tuple[list[dict], list[dict]]:
            - Prediction rows in submission format.
            - Per-item retrieval/generation statistics.

    Raises:
        FileNotFoundError: If the input dataset path does not exist.
        ValueError: If the input dataset is not a JSON list.
    """
    input_path = Path(config.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Input dataset must be a list of objects.")

    fixed_evidence_k = max(1, min(40, config.evidence_k))
    min_evidence_k = max(1, min(40, config.min_evidence_k))
    max_evidence_k = max(1, min(40, config.max_evidence_k))
    if max_evidence_k < min_evidence_k:
        max_evidence_k = min_evidence_k

    evidence_threshold = max(0.0, min(1.0, config.evidence_score_threshold))
    target_max_for_retrieval = max_evidence_k if config.dynamic_evidence else fixed_evidence_k
    retrieval_k = max(target_max_for_retrieval, config.retrieval_k)

    console.print(f"[bold]Loading embedding model:[/bold] {config.embedding_model}")
    embedder = SentenceTransformer(config.embedding_model)

    reranker: CrossEncoder | None = None
    if not config.disable_rerank:
        console.print(f"[bold]Loading reranker model:[/bold] {config.rerank_model}")
        reranker = CrossEncoder(config.rerank_model)

    llm = ChatOllama(
        base_url=config.ollama_base_url,
        model=config.llm_model,
        temperature=config.llm_temperature,
        num_predict=config.llm_max_tokens,
    )

    outputs: list[dict] = []
    per_item_stats: list[dict] = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Running RAG", total=len(dataset))
        for item in dataset:
            title = item.get("title", "")
            question = item.get("question", "")
            full_text = item.get("full_text", "")

            chunks = build_chunks(
                full_text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            if not chunks:
                fallback = "Insufficient evidence."
                outputs.append({"title": title, "answer": fallback, "evidence": [fallback]})
                per_item_stats.append(
                    {
                        "title": title,
                        "num_chunks": 0,
                        "num_candidates": 0,
                        "num_evidence": 1,
                        "llm_error": True,
                        "answer_chars": len(fallback),
                    }
                )
                progress.advance(task)
                continue

            chunk_texts = [c.text for c in chunks]
            doc_inputs = apply_bge_passage_prefix(chunk_texts, config.use_bge_instructions)
            query_input = apply_bge_query_prefix(question, config.use_bge_instructions)
            doc_emb = embedder.encode(doc_inputs, normalize_embeddings=True, convert_to_numpy=True)
            query_emb = embedder.encode([query_input], normalize_embeddings=True, convert_to_numpy=True)[0]
            dense_scores = doc_emb @ query_emb
            dense_rank = argsort_desc(dense_scores)

            idf, avgdl = build_bm25_stats(chunks)
            bm25 = bm25_scores(tokenize(question), chunks, idf, avgdl)
            sparse_rank = argsort_desc(bm25)

            fused = reciprocal_rank_fusion([dense_rank, sparse_rank], rrf_k=config.rrf_k)
            candidate_idx = topn_indices(fused, n=min(retrieval_k, len(chunks)))

            ranked_chunk_pairs: list[tuple[int, float]]
            if reranker is not None and candidate_idx:
                pairs = [[question, chunks[i].text] for i in candidate_idx]
                rerank_scores = reranker.predict(pairs)
                reranked = sorted(
                    zip(candidate_idx, rerank_scores),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )
                ranked_chunk_pairs = [(i, float(score)) for i, score in reranked]
            else:
                ranked_chunk_pairs = [(i, float(fused.get(i, 0.0))) for i in candidate_idx]

            ranked_chunk_scores = [score for _, score in ranked_chunk_pairs]
            if config.dynamic_evidence:
                selected_k = adaptive_k_from_scores(
                    ranked_chunk_scores,
                    min_k=min_evidence_k,
                    max_k=max_evidence_k,
                    threshold=evidence_threshold,
                )
            else:
                selected_k = min(fixed_evidence_k, len(ranked_chunk_pairs))

            final_idx = [i for i, _ in ranked_chunk_pairs[:selected_k]]

            if config.evidence_mode == "sentence":
                sentence_evidence = select_sentence_evidence(
                    question=question,
                    chunks=chunks,
                    chunk_indices=final_idx if final_idx else candidate_idx,
                    embedder=embedder,
                    question_emb=query_emb,
                    use_bge_instructions=config.use_bge_instructions,
                    evidence_k=fixed_evidence_k,
                    dynamic_evidence=config.dynamic_evidence,
                    min_evidence_k=min_evidence_k,
                    max_evidence_k=max_evidence_k,
                    evidence_score_threshold=evidence_threshold,
                    evidence_source_chunks=config.evidence_source_chunks,
                    evidence_max_chars=config.evidence_max_chars,
                )
                # Hard cap by adaptive/fixed chunk selection target while preserving rule 1<=k<=40.
                target_k_for_output = selected_k if config.dynamic_evidence else fixed_evidence_k
                evidence = ensure_evidence_count(sentence_evidence, k=target_k_for_output)
            else:
                target_k_for_output = selected_k if config.dynamic_evidence else fixed_evidence_k
                evidence = ensure_evidence_count([chunks[i].text for i in final_idx], k=target_k_for_output)
            prompt = make_answer_prompt(title=title, question=question, evidence_texts=evidence)

            llm_error = False
            try:
                resp = llm.invoke(prompt)
                answer = to_text_response(resp)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]LLM error on '{title}':[/yellow] {exc}")
                answer = "Insufficient evidence."
                llm_error = True

            if not answer:
                answer = "Insufficient evidence."
                llm_error = True

            outputs.append(
                {
                    "title": title,
                    "answer": answer,
                    "evidence": evidence,
                }
            )
            per_item_stats.append(
                {
                    "title": title,
                    "num_chunks": len(chunks),
                    "num_candidates": len(candidate_idx),
                    "selected_k": selected_k,
                    "num_evidence": len(evidence),
                    "dynamic_evidence": config.dynamic_evidence,
                    "evidence_threshold": evidence_threshold,
                    "evidence_mode": config.evidence_mode,
                    "source_chunk_topn": min(
                        len(final_idx if final_idx else candidate_idx),
                        config.evidence_source_chunks,
                    ),
                    "llm_error": llm_error,
                    "answer_chars": len(answer),
                }
            )
            progress.advance(task)
    return outputs, per_item_stats


def main() -> None:
    """Execute the CLI entrypoint and write run artifacts to a checkpoint folder."""
    config = parse_args()

    started_at = datetime.now().astimezone()
    timer_start = time.perf_counter()

    run_name = resolve_run_name(config.run_name, now=started_at)
    run_dir = create_run_dir(config.output_root, run_name)
    submission_path = run_dir / config.submission_filename
    config_path = run_dir / "config.json"
    run_info_path = run_dir / "run_info.json"
    stats_path = run_dir / "per_item_stats.json"

    write_json(config_path, asdict(config))

    predictions, per_item_stats = run_pipeline(config)

    write_json(submission_path, predictions)
    write_json(stats_path, per_item_stats)

    finished_at = datetime.now().astimezone()
    duration_seconds = round(time.perf_counter() - timer_start, 3)

    run_info = {
        "run_name": run_dir.name,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "input_path": str(Path(config.input).resolve()),
        "python_version": sys.version,
        "num_predictions": len(predictions),
        "submission_path": str(submission_path.resolve()),
        "config_path": str(config_path.resolve()),
        "per_item_stats_path": str(stats_path.resolve()),
    }
    write_json(run_info_path, run_info)

    print_run_folder_structure(run_dir, config.submission_filename)
    console.print(f"[green]Saved submission:[/green] {submission_path}")
    console.print(f"Entries: {len(predictions)}")


if __name__ == "__main__":
    main()
