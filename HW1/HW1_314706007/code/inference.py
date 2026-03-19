from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_eval import PromptOnlyDataset, prompt_collate_fn
from utils import ID_TO_LABEL, get_choice_token_ids, load_dataset


def run_inference(
    model_dir: str | Path,
    benchmark_csv: str | Path,
    output_csv: str | Path | None,
    max_length: int = 512,
    batch_size: int = 4,
) -> None:
    """Run benchmark inference and export predictions to CSV.

    Args:
        model_dir: Directory containing a saved model checkpoint.
        benchmark_csv: Path to the unlabeled benchmark CSV.
        output_csv: Destination path for prediction output. If ``None``, the file
            is saved inside ``model_dir`` as ``submission.csv``.
        max_length: Maximum prompt length.
        batch_size: Evaluation batch size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    benchmark_df = load_dataset(benchmark_csv)
    dataset = PromptOnlyDataset(benchmark_df, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
    )

    choice_token_ids = get_choice_token_ids(tokenizer)
    candidate_ids = torch.tensor(
        [choice_token_ids["A"], choice_token_ids["B"], choice_token_ids["C"], choice_token_ids["D"]],
        device=device,
    )

    predictions = []
    question_ids = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=device)
            next_token_logits = outputs.logits[batch_indices, last_indices]
            choice_logits = next_token_logits.index_select(dim=-1, index=candidate_ids)
            batch_predictions = choice_logits.argmax(dim=-1).detach().cpu().tolist()

            predictions.extend(batch_predictions)
            question_ids.extend(batch["question_id"].tolist())

    submission = pd.DataFrame({"question_id": question_ids, "ans": predictions})
    output_csv = Path(output_csv) if output_csv is not None else Path(model_dir) / "submission.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")
    print("Answer mapping:", ID_TO_LABEL)


def parse_args():
    """Parse CLI arguments for benchmark inference.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run benchmark inference for the PathoQA baseline.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--benchmark_csv", type=str, default="../dataset/benchmark.csv")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_dir=args.model_dir,
        benchmark_csv=args.benchmark_csv,
        output_csv=args.output_csv,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
