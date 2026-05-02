from __future__ import annotations

import unsloth

import argparse
from datetime import datetime
import logging
import os
import warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Both `max_new_tokens`.*and `max_length`")
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel

from utils import (
    EvidenceRetriever,
    OutputParser,
    PDFParser,
    PromptBuilder,
    WeightedDataCollator,
    compute_macro_f1,
    find_pdf,
    oversample,
    print_prediction_distribution,
    set_seed,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model_name: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    data_dir: str = "../dataset"
    adapter_dir: str = "adapter_checkpoint"
    cache_dir: str = "paper_cache"
    max_seq_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-4
    epochs: int = 3
    warmup_ratio: float = 0.1
    max_multiplier: int = 10
    seed: int = 42


# ── Weighted Trainer ──────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer with per-sample loss weighting to handle class imbalance."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)

        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)

        mask = (shift_labels != -100).float()
        denom = mask.sum(dim=-1).clamp(min=1)
        per_sample_loss = (per_token_loss * mask).sum(dim=-1) / denom

        if sample_weights is not None:
            w = sample_weights.to(per_sample_loss.device).float()
            w = w / w.mean()  # normalize so loss magnitude stays comparable to CE
            loss = (per_sample_loss * w).mean()
        else:
            loss = per_sample_loss.mean()

        return (loss, outputs) if return_outputs else loss


# ── Dataset construction ──────────────────────────────────────────────────────

def build_tokenized_dataset(
    df: pd.DataFrame,
    parser: PDFParser,
    retriever: EvidenceRetriever,
    prompt_builder: PromptBuilder,
    tokenizer,
    max_seq_len: int,
    pdf_dir: Path,
    class_weight_map: Dict[int, float],
) -> Dataset:
    all_input_ids: List[List[int]] = []
    all_attention_masks: List[List[int]] = []
    all_labels: List[List[int]] = []
    all_weights: List[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        pdf_path = find_pdf(pdf_dir, str(row["paper_id"]))
        raw_text = parser.parse(str(row["paper_id"]), pdf_path)
        chunks = parser.chunk(raw_text)
        evidence = retriever.retrieve(row["text"], str(row["paper_id"]), chunks)

        # Tokenize prompt-only to get exact boundary; add_generation_prompt=True
        # appends the <|im_start|>assistant\n prefix so prompt_len covers it.
        prompt_only = tokenizer.apply_chat_template(
            prompt_builder.build(row["text"], evidence),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(
            tokenizer(prompt_only, truncation=False, add_special_tokens=False)["input_ids"]
        )

        messages = prompt_builder.build(row["text"], evidence, label_id=int(row["label"]))
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )
        input_ids: List[int] = tokens["input_ids"]
        labels = input_ids.copy()
        # Mask everything up to (but not including) the answer tokens.
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # Diagnostic on first sample only — confirms masking actually works.
        if not all_input_ids:
            answer_ids = input_ids[prompt_len:]
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
            unmasked = sum(1 for l in labels if l != -100)
            print(
                f"\n[diag] First training example:\n"
                f"  prompt_len={prompt_len}, total_len={len(input_ids)}\n"
                f"  unmasked label tokens={unmasked}\n"
                f"  answer text={answer_text[:300]!r}\n"
            )
            if unmasked == 0:
                raise RuntimeError(
                    "All labels are -100 — answer was truncated away. "
                    "Increase max_seq_len or shrink the evidence budget."
                )

        all_input_ids.append(input_ids)
        all_attention_masks.append(tokens["attention_mask"])
        all_labels.append(labels)
        all_weights.append(class_weight_map[int(row["label"])])

    return Dataset.from_dict(
        {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
            "sample_weight": all_weights,
        }
    )


# ── Dev evaluation ────────────────────────────────────────────────────────────

def evaluate_dev(
    model,
    tokenizer,
    dev_df: pd.DataFrame,
    parser: PDFParser,
    retriever: EvidenceRetriever,
    prompt_builder: PromptBuilder,
    output_parser: OutputParser,
    pdf_dir: Path,
    max_seq_len: int,
) -> float:
    FastLanguageModel.for_inference(model)
    predictions: List[int] = []

    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="Dev eval"):
        pdf_path = find_pdf(pdf_dir, str(row["paper_id"]))
        raw_text = parser.parse(str(row["paper_id"]), pdf_path)
        chunks = parser.chunk(raw_text)
        evidence = retriever.retrieve(row["text"], str(row["paper_id"]), chunks)

        messages = prompt_builder.build(row["text"], evidence)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred = output_parser.parse(generated)
        if len(predictions) < 5:
            print(f"[dev {len(predictions)}] generated: {generated[:300]!r} → pred={pred}")
        predictions.append(pred)

    model.train()
    print_prediction_distribution("dev", predictions, prompt_builder.id_to_name)
    return compute_macro_f1(predictions, dev_df["label"].tolist())


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config: TrainingConfig) -> None:
    set_seed(config.seed)

    run_dir = Path(config.adapter_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    data_dir = Path(config.data_dir)
    pdf_dir = data_dir / "paper_evidence"
    classes_json = data_dir / "classes.json"

    df = pd.read_csv(data_dir / "train.csv")
    print(f"[diag] Raw class distribution:\n{df['label'].value_counts().sort_index()}")
    df = oversample(df, max_multiplier=config.max_multiplier)
    print(f"[diag] Post-oversample class distribution:\n{df['label'].value_counts().sort_index()}")

    unique_labels = np.unique(df["label"].values)
    raw_weights = compute_class_weight("balanced", classes=unique_labels, y=df["label"].values)
    class_weight_map: Dict[int, float] = {
        int(k): float(v) for k, v in zip(unique_labels, raw_weights)
    }

    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    model.print_trainable_parameters()

    pdf_parser = PDFParser(config.cache_dir)
    pdf_parser.preprocess_all(pdf_dir, df["paper_id"].unique().tolist())

    retriever = EvidenceRetriever(top_k=5, max_tokens=1500)
    prompt_builder = PromptBuilder(classes_json)

    train_dataset = build_tokenized_dataset(
        df, pdf_parser, retriever, prompt_builder, tokenizer,
        config.max_seq_len, pdf_dir, class_weight_map,
    )

    collator = WeightedDataCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()

    dev_path = data_dir / "dev.csv"
    if dev_path.exists():
        output_parser = OutputParser(classes_json)
        dev_df = pd.read_csv(dev_path)
        macro_f1 = evaluate_dev(
            model, tokenizer, dev_df, pdf_parser, retriever,
            prompt_builder, output_parser, pdf_dir, config.max_seq_len,
        )
        print(f"Dev Macro F1: {macro_f1:.4f}")

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    print(f"Adapter saved → {run_dir}/")


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for hallucination classification")
    for f in fields(TrainingConfig):
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    return TrainingConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(parse_args())
