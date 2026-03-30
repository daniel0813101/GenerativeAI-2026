from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from transformers import AutoTokenizer

from train_eval import generate_predictions_from_dataframe, load_saved_model
from utils import load_dataset


def run_inference(
    model_dirs: Sequence[str | Path],
    benchmark_csv: str | Path,
    output_csv: str | Path | None,
    max_length: int = 1024,
    batch_size: int = 4,
    option_order_ensemble: bool = True,
    num_option_order_permutations: int = 4,
    max_new_tokens: int = 40,
) -> None:
    """Run benchmark inference and export hard-voted predictions to CSV.

    Args:
        model_dirs: One or more directories containing saved model checkpoints.
        benchmark_csv: Path to the unlabeled benchmark CSV.
        output_csv: Destination path for prediction output. If ``None``, the file
            is saved inside the first model directory as ``submission.csv``.
        max_length: Maximum prompt length.
        batch_size: Evaluation batch size.
        option_order_ensemble: Whether to hard-vote across several answer
            option permutations for each example.
        num_option_order_permutations: Number of permutations used when
            ``option_order_ensemble`` is enabled.
        max_new_tokens: Maximum generation length for each answer.

    Returns:
        None.
    """
    model_dirs = [Path(model_dir) for model_dir in model_dirs]
    if not model_dirs:
        raise ValueError("At least one model directory must be provided for inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    benchmark_df = load_dataset(benchmark_csv)
    all_model_predictions = []
    question_ids = None

    for model_dir in model_dirs:
        model = load_saved_model(model_dir, tokenizer, device)
        model.eval()
        predictions_df = generate_predictions_from_dataframe(
            model,
            benchmark_df,
            tokenizer,
            device,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=device.type == "cuda",
            option_order_ensemble=option_order_ensemble,
            num_option_order_permutations=num_option_order_permutations,
            max_new_tokens=max_new_tokens,
        )
        model_preds = predictions_df["prediction"].tolist()
        batch_q_ids = predictions_df["question_id"].tolist()
        if question_ids is None:
            question_ids = batch_q_ids
        elif batch_q_ids != question_ids:
            raise ValueError("Ensemble models produced a different question ordering.")

        all_model_predictions.append(model_preds)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_ensembled_predictions = []
    for i in range(len(question_ids)):
        votes = [model_preds[i] for model_preds in all_model_predictions]
        winner = Counter(votes).most_common(1)[0][0]
        final_ensembled_predictions.append(winner)

    submission = pd.DataFrame({"question_id": question_ids, "ans": final_ensembled_predictions})
    output_csv = Path(output_csv) if output_csv is not None else model_dirs[0] / "submission.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)
    print(f"Saved Hard-Voted submission to {output_csv}")


def parse_args():
    """Parse CLI arguments for benchmark inference.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run benchmark inference for the PathoQA baseline.")
    parser.add_argument("--model_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--benchmark_csv", type=str, default="../dataset/benchmark.csv")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--option_order_ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_option_order_permutations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_dirs=args.model_dirs,
        benchmark_csv=args.benchmark_csv,
        output_csv=args.output_csv,
        max_length=args.max_length,
        batch_size=args.batch_size,
        option_order_ensemble=args.option_order_ensemble,
        num_option_order_permutations=args.num_option_order_permutations,
        max_new_tokens=args.max_new_tokens,
    )
