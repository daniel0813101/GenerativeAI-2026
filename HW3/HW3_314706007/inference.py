from __future__ import annotations

import argparse
from collections import Counter
import gc
import logging
import os
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

from utils import (
    EvidenceRetriever,
    OutputParser,
    PDFParser,
    PromptBuilder,
    compute_macro_f1,
    find_pdf,
    print_prediction_distribution,
    set_seed,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    data_dir: str = "../dataset"
    adapter_dir: str = "adapter_checkpoint"
    ensemble_dirs: str = ""  # comma-separated adapter dirs; overrides adapter_dir
    cache_dir: str = "paper_cache"
    output_csv: str = "hw3_314706007.csv"
    max_seq_len: int = 3072
    max_new_tokens: int = 220
    seed: int = 42
    dev_only: bool = False   # run dev.csv only, skip test.csv
    test_only: bool = True   # default: run test.csv only, skip dev.csv
    verify: bool = False     # optional second self-verification pass


@dataclass
class InferenceResult:
    predictions: List[int]
    parse_fallbacks: int
    verify_flips: int


# ── Core inference loop ───────────────────────────────────────────────────────

def _generate_class(
    messages, model, tokenizer, output_parser, max_seq_len, max_new_tokens
) -> tuple[int, str, bool]:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    pred, used_fallback = output_parser.parse_with_fallback(generated)
    return pred, generated, used_fallback


def run_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    parser: PDFParser,
    retriever: EvidenceRetriever,
    prompt_builder: PromptBuilder,
    output_parser: OutputParser,
    pdf_dir: Path,
    max_seq_len: int,
    max_new_tokens: int,
    verify: bool = False,
) -> InferenceResult:
    predictions: List[int] = []
    parse_fallbacks = 0
    flips = 0  # how often verification overruled the first pass

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        pdf_path = find_pdf(pdf_dir, str(row["paper_id"]))
        raw_text = parser.parse(str(row["paper_id"]), pdf_path)
        chunks = parser.chunk(raw_text)
        evidence = retriever.retrieve(row["text"], str(row["paper_id"]), chunks)

        # First pass: standard classification
        messages_1 = prompt_builder.build(row["text"], evidence)
        first_pred, first_gen, used_fallback = _generate_class(
            messages_1, model, tokenizer, output_parser, max_seq_len, max_new_tokens,
        )
        parse_fallbacks += int(used_fallback)

        if verify:
            # Second pass: self-verification with the first prediction in context
            first_class_name = prompt_builder.id_to_name[first_pred]
            messages_2 = prompt_builder.build(
                row["text"], evidence, verify_for=first_class_name,
            )
            base_pred, base_gen, used_fallback = _generate_class(
                messages_2, model, tokenizer, output_parser, max_seq_len, max_new_tokens,
            )
            parse_fallbacks += int(used_fallback)
            if base_pred != first_pred:
                flips += 1
        else:
            base_pred, base_gen = first_pred, first_gen

        final_pred = base_pred

        if len(predictions) < 10:
            pred_name = prompt_builder.id_to_name[final_pred]
            tag = "→" if final_pred == first_pred else f"FLIP {first_pred}→{final_pred}"
            print(f"[{len(predictions)}] {tag} pred={final_pred} ({pred_name})")
            print(f"    first:  {first_gen[:250]!r}")
            if verify:
                print(f"    verify: {base_gen[:250]!r}")

        predictions.append(final_pred)

    if verify:
        print(f"[verify] {flips}/{len(predictions)} predictions changed by verification pass")
    print(f"[parser] {parse_fallbacks} generated outputs used conservative fallback")
    return InferenceResult(
        predictions=predictions,
        parse_fallbacks=parse_fallbacks,
        verify_flips=flips,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def _resolve_adapter_dir(adapter_dir: str) -> Path:
    """Return adapter_dir as-is, or its latest timestamped subdirectory."""
    base = Path(adapter_dir)
    if base.is_dir():
        ts_dirs = sorted(
            (d for d in base.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == "-"),
            key=lambda d: d.name,
        )
        if ts_dirs:
            latest = ts_dirs[-1]
            print(f"Using latest checkpoint: {latest}")
            return latest
    return base


def _resolve_adapter_dirs(config: InferenceConfig) -> List[Path]:
    if not config.ensemble_dirs.strip():
        return [_resolve_adapter_dir(config.adapter_dir)]

    raw_dirs = [d.strip() for d in config.ensemble_dirs.split(",") if d.strip()]
    if len(raw_dirs) != 3:
        print(f"[ensemble] warning: expected 3 adapter dirs, got {len(raw_dirs)}")
    return [_resolve_adapter_dir(d) for d in raw_dirs]


def _majority_vote(prediction_sets: List[List[int]]) -> List[int]:
    if not prediction_sets:
        return []
    if len(prediction_sets) == 1:
        return prediction_sets[0]

    voted: List[int] = []
    for row_preds in zip(*prediction_sets):
        counts = Counter(row_preds)
        best_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == best_count]
        # With three seeds, ties mean all three disagree. Use the first adapter
        # as the deterministic tie-breaker, so the strongest/default seed wins.
        voted.append(winners[0] if len(winners) == 1 else row_preds[0])
    return voted


def _release_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(config: InferenceConfig) -> None:
    set_seed(config.seed)

    data_dir = Path(config.data_dir)
    pdf_dir = data_dir / "paper_evidence"
    classes_json = data_dir / "classes.json"

    pdf_parser = PDFParser(config.cache_dir)
    retriever = EvidenceRetriever(top_k=7, max_tokens=550)
    prompt_builder = PromptBuilder(classes_json)
    output_parser = OutputParser(classes_json)
    adapter_dirs = _resolve_adapter_dirs(config)
    ensemble = len(adapter_dirs) > 1

    dev_df = None if config.test_only else pd.read_csv(data_dir / "dev.csv")
    test_df = None if config.dev_only else pd.read_csv(data_dir / "test.csv")
    dev_prediction_sets: List[List[int]] = []
    test_prediction_sets: List[List[int]] = []

    for idx, adapter_dir in enumerate(adapter_dirs, start=1):
        print(f"[adapter {idx}/{len(adapter_dirs)}] {adapter_dir}")
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=config.max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # ── Dev evaluation (skipped when --test_only) ────────────────────────
        if dev_df is not None:
            dev_result = run_inference(
                dev_df, model, tokenizer, pdf_parser, retriever,
                prompt_builder, output_parser, pdf_dir,
                config.max_seq_len, config.max_new_tokens,
                verify=config.verify,
            )
            dev_preds = dev_result.predictions
            dev_prediction_sets.append(dev_preds)
            if ensemble:
                print_prediction_distribution(
                    f"dev adapter {idx}", dev_preds, prompt_builder.id_to_name,
                )
                score = compute_macro_f1(dev_preds, dev_df["label"].tolist())
                print(f"Dev Macro F1 adapter {idx}: {score:.4f}")

        # ── Test inference (skipped when --dev_only) ─────────────────────────
        if test_df is not None:
            test_result = run_inference(
                test_df, model, tokenizer, pdf_parser, retriever,
                prompt_builder, output_parser, pdf_dir,
                config.max_seq_len, config.max_new_tokens,
                verify=config.verify,
            )
            test_prediction_sets.append(test_result.predictions)

        _release_model(model, tokenizer)

    if dev_df is not None:
        dev_preds = _majority_vote(dev_prediction_sets)
        name = "dev ensemble" if ensemble else "dev"
        print_prediction_distribution(name, dev_preds, prompt_builder.id_to_name)
        score = compute_macro_f1(dev_preds, dev_df["label"].tolist())
        label = "Dev Macro F1 ensemble" if ensemble else "Dev Macro F1"
        print(f"{label}: {score:.4f}")

    if test_df is not None:
        preds = _majority_vote(test_prediction_sets)
        name = "test ensemble" if ensemble else "test"
        print_prediction_distribution(name, preds, prompt_builder.id_to_name)

        submission = pd.DataFrame({"id": range(len(preds)), "label": preds})
        submission.to_csv(config.output_csv, index=False)
        print(f"Saved {len(preds)} predictions → {config.output_csv}")


def parse_args() -> InferenceConfig:
    defaults = InferenceConfig()
    parser = argparse.ArgumentParser(description="Inference for hallucination classification")
    for f in fields(InferenceConfig):
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    return InferenceConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(parse_args())
