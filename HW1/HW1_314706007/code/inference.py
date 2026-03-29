from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from transformers import AutoTokenizer

from train_eval import load_saved_model
from utils import extract_prediction, load_dataset


def generate_predictions_from_dataframe(
    model,
    dataframe,
    tokenizer,
    device,
    max_length,
    batch_size,
    option_order_ensemble,
    num_option_order_permutations,
):
    """Generate predictions from a dataframe with optional option-order voting.

    Args:
        model: Fine-tuned causal language model used for generation.
        dataframe: Benchmark dataframe containing unlabeled examples.
        tokenizer: Tokenizer used to encode prompts and decode outputs.
        device: Device used for inference.
        max_length: Maximum prompt length.
        batch_size: Evaluation batch size.
        option_order_ensemble: Whether to run multiple option permutations and
            hard-vote across them.
        num_option_order_permutations: Number of permutations to evaluate when
            option-order ensembling is enabled.

    Returns:
        A tuple containing final predicted labels and their corresponding
        question ids.
    """
    from train_eval import create_prompt_dataloader
    from utils import get_option_permutations

    num_permutations = num_option_order_permutations if option_order_ensemble else 1
    permutations = get_option_permutations(num_permutations)

    all_permutation_preds = []
    question_ids = None

    for permutation in permutations:
        dataloader = create_prompt_dataloader(
            dataframe,
            tokenizer,
            max_length,
            batch_size,
            num_workers=0,
            pin_memory=device.type == "cuda",
            option_permutation=permutation,
        )

        perm_preds = []
        batch_q_ids = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=25,  # Allows short text, prevents long rambling
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
                input_lengths = attention_mask.sum(dim=1).tolist()
                generated_texts = [
                    tokenizer.decode(output_ids[int(input_length):], skip_special_tokens=True)
                    for output_ids, input_length in zip(generated_ids, input_lengths)
                ]

                for text, q_id in zip(generated_texts, batch["question_id"].tolist()):
                    pred_idx = extract_prediction(text)
                    original_idx = permutation[pred_idx]
                    perm_preds.append(original_idx)
                    batch_q_ids.append(q_id)

        all_permutation_preds.append(perm_preds)
        if question_ids is None:
            question_ids = batch_q_ids

    final_preds = []
    for i in range(len(question_ids)):
        votes = [perm_preds[i] for perm_preds in all_permutation_preds]
        winner = Counter(votes).most_common(1)[0][0]
        final_preds.append(winner)

    return final_preds, question_ids


def run_inference(
    model_dirs: Sequence[str | Path],
    benchmark_csv: str | Path,
    output_csv: str | Path | None,
    max_length: int = 1024,
    batch_size: int = 4,
    option_order_ensemble: bool = False,
    num_option_order_permutations: int = 4,
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
        model_preds, batch_q_ids = generate_predictions_from_dataframe(
            model,
            benchmark_df,
            tokenizer,
            device,
            max_length,
            batch_size,
            option_order_ensemble,
            num_option_order_permutations,
        )
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
