from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from transformers import AutoTokenizer

from train_eval import collect_choice_logits_from_dataframe, load_saved_model
from utils import ID_TO_LABEL, load_dataset


def run_inference(
    model_dirs: Sequence[str | Path],
    benchmark_csv: str | Path,
    output_csv: str | Path | None,
    max_length: int = 1024,
    batch_size: int = 4,
    option_order_ensemble: bool = False,
    num_option_order_permutations: int = 4,
) -> None:
    """Run benchmark inference and export predictions to CSV.

    Args:
        model_dirs: One or more directories containing saved model checkpoints.
        benchmark_csv: Path to the unlabeled benchmark CSV.
        output_csv: Destination path for prediction output. If ``None``, the file
            is saved inside the first model directory as ``submission.csv``.
        max_length: Maximum prompt length.
        batch_size: Evaluation batch size.
        option_order_ensemble: Whether to average logits across several answer
            option permutations for each example.
        num_option_order_permutations: Number of permutations used when
            ``option_order_ensemble`` is enabled.
    """
    model_dirs = [Path(model_dir) for model_dir in model_dirs]
    if not model_dirs:
        raise ValueError("At least one model directory must be provided for inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dirs[0])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    benchmark_df = load_dataset(benchmark_csv)
    ensemble_logits = None
    question_ids = None

    for model_dir in model_dirs:
        model = load_saved_model(model_dir, tokenizer, device)
        model.eval()
        stacked_logits, batch_question_ids, _ = collect_choice_logits_from_dataframe(
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
        )
        if ensemble_logits is None:
            ensemble_logits = stacked_logits
            question_ids = batch_question_ids
        else:
            if batch_question_ids != question_ids:
                raise ValueError("Ensemble models produced a different question ordering.")
            ensemble_logits += stacked_logits

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = ensemble_logits.argmax(dim=-1).tolist()

    submission = pd.DataFrame({"question_id": question_ids, "ans": predictions})
    output_csv = Path(output_csv) if output_csv is not None else model_dirs[0] / "submission.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")
    print("Ensembled checkpoints:", [str(model_dir) for model_dir in model_dirs])
    if option_order_ensemble:
        print(f"Option-order permutations per checkpoint: {num_option_order_permutations}")
    print("Answer mapping:", ID_TO_LABEL)


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
    parser.add_argument("--option_order_ensemble", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_option_order_permutations", type=int, default=4)
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
    )
