
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
    chunk_size: int = 1000
    chunk_overlap: int = 140
    retrieval_k: int = 18
    evidence_k: int = 6
    rrf_k: int = 60
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
            help="Evidence chunks to submit per question (1-40)",
        )
        parser.add_argument("--rrf-k", type=int, default=defaults.rrf_k, help="RRF constant for rank fusion")
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

    evidence_k = max(1, min(40, config.evidence_k))
    retrieval_k = max(evidence_k, config.retrieval_k)

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
            doc_emb = embedder.encode(chunk_texts, normalize_embeddings=True, convert_to_numpy=True)
            query_emb = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
            dense_scores = doc_emb @ query_emb
            dense_rank = argsort_desc(dense_scores)

            idf, avgdl = build_bm25_stats(chunks)
            bm25 = bm25_scores(tokenize(question), chunks, idf, avgdl)
            sparse_rank = argsort_desc(bm25)

            fused = reciprocal_rank_fusion([dense_rank, sparse_rank], rrf_k=config.rrf_k)
            candidate_idx = topn_indices(fused, n=min(retrieval_k, len(chunks)))

            if reranker is not None and candidate_idx:
                pairs = [[question, chunks[i].text] for i in candidate_idx]
                rerank_scores = reranker.predict(pairs)
                reranked = sorted(
                    zip(candidate_idx, rerank_scores),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )
                final_idx = [i for i, _ in reranked[:evidence_k]]
            else:
                final_idx = candidate_idx[:evidence_k]

            evidence = ensure_evidence_count([chunks[i].text for i in final_idx], k=evidence_k)
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
                    "num_evidence": len(evidence),
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
