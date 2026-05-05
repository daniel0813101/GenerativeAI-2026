"""Generate paraphrased training rows for the minority hallucination classes.

Target: lift Temporal (130 → ~390) and Number (249 → ~750) class sizes by
asking the base Qwen model to rephrase each row twice while preserving the
exact temporal/numerical signal that defines the label.

Output: train_augmented.csv next to the original train.csv. train.py picks
this up automatically if it exists.

Usage:
    python augment_minorities.py
    # ~13 minutes for 758 generations on a single GPU.
"""

from __future__ import annotations

import unsloth  # noqa: F401  — must precede transformers to apply patches

import argparse
import logging
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

logging.getLogger("transformers.generation").setLevel(logging.ERROR)


# Class-specific paraphrase instructions: preserve the signal that defines
# the label, vary the rest of the sentence.
_INSTRUCTIONS = {
    2: (  # Number
        "You will rephrase a peer-review sentence in different words. "
        "STRICT REQUIREMENT: keep every numerical value EXACTLY as written "
        "(percentages, counts, years, dimensions — do not change any digits). "
        "Vary the surrounding wording. Output ONLY the rephrased sentence, "
        "no preamble, no quotes."
    ),
    4: (  # Temporal
        "You will rephrase a peer-review sentence in different words. "
        "STRICT REQUIREMENT: preserve the exact tense, modality, and time "
        "framing (do not change 'will' to 'has', 'plans to' to 'has done', "
        "etc.). Vary the surrounding wording. Output ONLY the rephrased "
        "sentence, no preamble, no quotes."
    ),
}

_CLASS_NAMES = {2: "Number", 4: "Temporal"}


@dataclass
class AugmentConfig:
    model_name: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    data_dir: str = "../dataset"
    output_csv: str = "../dataset/train_augmented.csv"
    n_paraphrases: int = 2
    max_new_tokens: int = 80
    max_seq_len: int = 1024
    seed: int = 42


def paraphrase_one(model, tokenizer, instruction: str, sentence: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": sentence},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,        # diversity per call
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip surrounding quotes / common preambles if the model emitted them.
    text = text.strip().strip('"').strip("'").strip()
    for prefix in ("Sure, here's", "Here's", "Here is", "Rephrased:", "Rewrite:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip(":,. \n").strip()
    return text


def main(config: AugmentConfig) -> None:
    torch.manual_seed(config.seed)

    train_path = Path(config.data_dir) / "train.csv"
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} train rows from {train_path}")
    print(f"Original class distribution:\n{df['label'].value_counts().sort_index()}")

    targets = df[df["label"].isin([2, 4])].copy()
    print(
        f"\nGenerating {config.n_paraphrases} paraphrases for each of "
        f"{len(targets)} minority rows…"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    new_rows = []
    for _, row in tqdm(targets.iterrows(), total=len(targets), desc="Paraphrasing"):
        label = int(row["label"])
        instruction = _INSTRUCTIONS[label]
        for _ in range(config.n_paraphrases):
            try:
                paraphrased = paraphrase_one(
                    model, tokenizer, instruction,
                    str(row["text"]), config.max_new_tokens,
                )
                if (
                    paraphrased
                    and paraphrased != row["text"]
                    and len(paraphrased) >= 20
                ):
                    new_rows.append({
                        "id": -1,  # filled below
                        "paper_id": row["paper_id"],
                        "text": paraphrased,
                        "label": label,
                    })
            except Exception as e:
                print(f"[skip] paraphrase failed for {row['paper_id']} ({_CLASS_NAMES[label]}): {e}")

    aug_df = pd.DataFrame(new_rows)
    print(f"\nGenerated {len(aug_df)} paraphrased rows.")
    if len(aug_df):
        print(f"Augmented class distribution:\n{aug_df['label'].value_counts().sort_index()}")

    out_df = pd.concat([df, aug_df], ignore_index=True)
    out_df["id"] = range(len(out_df))
    out_path = Path(config.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} total rows → {out_path}")
    print(f"Final class distribution:\n{out_df['label'].value_counts().sort_index()}")


def parse_args() -> AugmentConfig:
    defaults = AugmentConfig()
    parser = argparse.ArgumentParser(description="Paraphrase Number/Temporal training rows")
    for f in fields(AugmentConfig):
        parser.add_argument(
            f"--{f.name}",
            type=type(getattr(defaults, f.name)),
            default=getattr(defaults, f.name),
        )
    return AugmentConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    main(parse_args())
