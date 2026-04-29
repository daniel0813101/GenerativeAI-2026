from __future__ import annotations

import argparse
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
    set_seed,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    data_dir: str = "../data"
    adapter_dir: str = "adapter_checkpoint"
    cache_dir: str = "paper_cache"
    output_csv: str = "hw3_314706007.csv"
    max_seq_len: int = 2048
    max_new_tokens: int = 50
    seed: int = 42


# ── Core inference loop ───────────────────────────────────────────────────────

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
) -> List[int]:
    predictions: List[int] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        pdf_path = find_pdf(pdf_dir, str(row["paper_id"]))
        raw_text = parser.parse(str(row["paper_id"]), pdf_path)
        chunks = parser.chunk(raw_text)
        evidence = retriever.retrieve(row["text"], str(row["paper_id"]), chunks)

        messages = prompt_builder.build(row["text"], evidence)
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
        predictions.append(output_parser.parse(generated))

    return predictions


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config: InferenceConfig) -> None:
    set_seed(config.seed)

    data_dir = Path(config.data_dir)
    pdf_dir = data_dir / "paper_evidence"
    classes_json = data_dir / "classes.json"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.adapter_dir,
        max_seq_length=config.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    pdf_parser = PDFParser(config.cache_dir)
    retriever = EvidenceRetriever(top_k=5, max_tokens=1500)
    prompt_builder = PromptBuilder(classes_json)
    output_parser = OutputParser(classes_json)

    # ── Optional dev evaluation ───────────────────────────────────────────────
    dev_path = data_dir / "dev.csv"
    if dev_path.exists():
        dev_df = pd.read_csv(dev_path)
        dev_preds = run_inference(
            dev_df, model, tokenizer, pdf_parser, retriever,
            prompt_builder, output_parser, pdf_dir,
            config.max_seq_len, config.max_new_tokens,
        )
        score = compute_macro_f1(dev_preds, dev_df["label"].tolist())
        print(f"Dev Macro F1: {score:.4f}")

    # ── Test inference ────────────────────────────────────────────────────────
    test_df = pd.read_csv(data_dir / "test.csv")
    preds = run_inference(
        test_df, model, tokenizer, pdf_parser, retriever,
        prompt_builder, output_parser, pdf_dir,
        config.max_seq_len, config.max_new_tokens,
    )

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
