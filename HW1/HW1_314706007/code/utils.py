from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn.utils.rnn import pad_sequence


LABEL_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
OPTION_COLUMNS = ["opa", "opb", "opc", "opd"]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Random seed applied to Python and PyTorch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        csv_path: Path to the dataset CSV file.

    Returns:
        A pandas DataFrame. If an ``ans`` column exists, it is cast to ``int``.
    """
    df = pd.read_csv(csv_path)
    if "ans" in df.columns:
        df["ans"] = df["ans"].astype(int)
    return df


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    split_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a labeled dataset into train/validation and optional test DataFrames.

    Args:
        df: Full labeled dataset.
        val_ratio: Fraction of samples assigned to the validation split.
        test_ratio: Fraction of samples assigned to the test split.
        random_state: Random seed used by ``train_test_split``.
        split_path: Optional JSON path used to persist or reuse a fixed split
            based on ``question_id`` values. If the file exists, the saved
            split is loaded. Otherwise a new split is created and saved there.

    Returns:
        A tuple of ``(train_df, val_df, test_df)`` with reset indices. When
        ``test_ratio`` is ``0``, the returned ``test_df`` is empty.

    Raises:
        ValueError: If the split ratios are invalid.
    """
    if split_path is not None and Path(split_path).exists():
        split_spec = load_json(split_path)
        if "question_id" not in df.columns:
            raise ValueError("A persisted split requires a 'question_id' column in the dataset.")

        question_ids = {
            "train": set(split_spec["train_question_ids"]),
            "val": set(split_spec["val_question_ids"]),
            "test": set(split_spec.get("test_question_ids", [])),
        }
        all_saved_ids = question_ids["train"] | question_ids["val"] | question_ids["test"]
        dataset_ids = set(df["question_id"].tolist())
        if all_saved_ids != dataset_ids:
            raise ValueError(
                "Saved split does not match the current dataset question_id values. "
                "Delete the split file or point to a compatible one."
            )

        train_df = df[df["question_id"].isin(question_ids["train"])]
        val_df = df[df["question_id"].isin(question_ids["val"])]
        test_df = df[df["question_id"].isin(question_ids["test"])]
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    if val_ratio <= 0:
        raise ValueError("val_ratio must be greater than 0.")
    if test_ratio < 0:
        raise ValueError("test_ratio must be greater than or equal to 0.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    stratify_labels = df["ans"] if "ans" in df.columns else None

    if test_ratio == 0:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )
        test_df = df.iloc[0:0].copy()
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=val_ratio + test_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )

        temp_stratify = temp_df["ans"] if "ans" in temp_df.columns else None
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)

        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            random_state=random_state,
            shuffle=True,
            stratify=temp_stratify,
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if split_path is not None:
        if "question_id" not in df.columns:
            raise ValueError("A persisted split requires a 'question_id' column in the dataset.")
        split_path = Path(split_path)
        save_json(
            {
                "random_state": random_state,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "train_question_ids": train_df["question_id"].astype(int).tolist(),
                "val_question_ids": val_df["question_id"].astype(int).tolist(),
                "test_question_ids": test_df["question_id"].astype(int).tolist(),
            },
            split_path,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_kfold_splits(
    df: pd.DataFrame,
    num_folds: int = 5,
    random_state: int = 42,
    split_path: str | Path | None = None,
) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    """Build or load fixed stratified k-fold train/validation splits.

    Args:
        df: Full labeled dataset.
        num_folds: Number of folds to create.
        random_state: Random seed used when creating the folds.
        split_path: Optional JSON path used to persist or reuse the fold
            assignments based on ``question_id`` values.

    Returns:
        A list of ``(train_df, val_df)`` tuples, one per fold.

    Raises:
        ValueError: If the dataset cannot support the requested number of folds
            or if a persisted split does not match the current dataset.
    """
    if "ans" not in df.columns:
        raise ValueError("K-fold training requires labeled data with an 'ans' column.")
    if "question_id" not in df.columns:
        raise ValueError("A persisted k-fold split requires a 'question_id' column in the dataset.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    if split_path is not None and Path(split_path).exists():
        split_spec = load_json(split_path)
        folds = split_spec["folds"]
        all_saved_ids = set()
        dataset_ids = set(df["question_id"].tolist())
        split_pairs: List[tuple[pd.DataFrame, pd.DataFrame]] = []

        for fold_spec in folds:
            train_ids = set(fold_spec["train_question_ids"])
            val_ids = set(fold_spec["val_question_ids"])
            all_saved_ids |= train_ids | val_ids
            train_df = df[df["question_id"].isin(train_ids)]
            val_df = df[df["question_id"].isin(val_ids)]
            split_pairs.append((train_df.reset_index(drop=True), val_df.reset_index(drop=True)))

        if all_saved_ids != dataset_ids:
            raise ValueError(
                "Saved k-fold split does not match the current dataset question_id values. "
                "Delete the split file or point to a compatible one."
            )
        return split_pairs

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    split_pairs: List[tuple[pd.DataFrame, pd.DataFrame]] = []
    fold_specs = []

    for train_indices, val_indices in splitter.split(df, df["ans"]):
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        split_pairs.append((train_df, val_df))
        fold_specs.append(
            {
                "train_question_ids": train_df["question_id"].astype(int).tolist(),
                "val_question_ids": val_df["question_id"].astype(int).tolist(),
            }
        )

    if split_path is not None:
        save_json(
            {
                "random_state": random_state,
                "num_folds": num_folds,
                "folds": fold_specs,
            },
            split_path,
        )

    return split_pairs


def build_prompt(row: pd.Series | Dict[str, object]) -> str:
    """Format one MCQ row into an optimized two-shot instruction prompt.

    Args:
        row: A dataset row containing the question and options.

    Returns:
        The prompt string given to the language model.
    """
    system_persona = (
        "You are an expert, dual-board-certified anatomical and clinical pathologist "
        "taking a high-stakes medical licensing examination. Your task is to critically evaluate "
        "the patient's clinical presentation, histopathological findings, and molecular markers "
        "to determine the single best answer. You must synthesize complex medical data with "
        "absolute clinical precision. Output strictly the letter of the correct option (A, B, C, or D).\n\n"
    )

    few_shot_examples = (
        "Question:\n"
        "A 45-year-old man presents with a painless neck mass. Biopsy reveals large cells "
        "with multilobated nuclei and prominent nucleoli resembling 'popcorn' cells. "
        "What is the most likely diagnosis?\n\n"
        "Options:\n"
        "A. Nodular lymphocyte predominant Hodgkin lymphoma\n"
        "B. Nodular sclerosis Hodgkin lymphoma\n"
        "C. Burkitt lymphoma\n"
        "D. Follicular lymphoma\n\n"
        "Answer: A\n\n"
        "---\n\n"
        "Question:\n"
        "A 32-year-old woman undergoes a routine cervical cytology screen. The pathologist notes "
        "scattered squamous epithelial cells characterized by enlarged, hyperchromatic nuclei with "
        "irregular contours, surrounded by a distinct, clear perinuclear halo. Which of the following "
        "is the most likely cause of these specific cellular changes?\n\n"
        "Options:\n"
        "A. Candida albicans infection\n"
        "B. Trichomonas vaginalis infection\n"
        "C. Herpes simplex virus (HSV) infection\n"
        "D. Human papillomavirus (HPV) infection\n\n"
        "Answer: D\n\n"
        "---\n\n"
    )

    return (
        f"{system_persona}"
        f"{few_shot_examples}"
        f"Question:\n{row['question']}\n\n"
        "Options:\n"
        f"A. {row['opa']}\n"
        f"B. {row['opb']}\n"
        f"C. {row['opc']}\n"
        f"D. {row['opd']}\n\n"
        "Answer:"
    )


def permute_answer_options(row: pd.Series | Dict[str, object], permutation: Sequence[int]) -> Dict[str, object]:
    """Permute MCQ options deterministically and remap the label if present.

    Args:
        row: A dataset row containing the question, options, and optional
            integer answer label.
        permutation: Mapping from new option index to original option index.

    Returns:
        A dictionary with permuted option text and a remapped ``ans`` label
        when the input row contains one.
    """
    if len(permutation) != len(OPTION_COLUMNS):
        raise ValueError(f"Expected {len(OPTION_COLUMNS)} permutation entries, got {len(permutation)}.")

    permuted_row = dict(row)
    for new_index, old_index in enumerate(permutation):
        permuted_row[OPTION_COLUMNS[new_index]] = row[OPTION_COLUMNS[old_index]]

    if "ans" in row:
        original_answer = int(row["ans"])
        permuted_row["ans"] = list(permutation).index(original_answer)

    return permuted_row


def shuffle_answer_options(row: pd.Series | Dict[str, object]) -> Dict[str, object]:
    """Shuffle MCQ answer options and remap the label if present.

    Args:
        row: A dataset row containing the question, options, and optional
            integer answer label.

    Returns:
        A dictionary with shuffled option text. If ``ans`` exists in ``row``,
        it is remapped to the new option index after shuffling.
    """
    permutation = list(range(len(OPTION_COLUMNS)))
    random.shuffle(permutation)
    return permute_answer_options(row, permutation)


def get_option_permutations(num_permutations: int = 4) -> List[List[int]]:
    """Return a deterministic list of answer-option permutations.

    Args:
        num_permutations: Number of permutations to return, including the
            identity permutation.

    Returns:
        A list of permutations, each mapping new option index to original
        option index.

    Raises:
        ValueError: If ``num_permutations`` is outside ``[1, 24]`` for four
            answer options.
    """
    max_permutations = 1
    for value in range(2, len(OPTION_COLUMNS) + 1):
        max_permutations *= value
    if num_permutations < 1 or num_permutations > max_permutations:
        raise ValueError(f"num_permutations must be between 1 and {max_permutations}.")
    return [list(permutation) for permutation in itertools.islice(itertools.permutations(range(len(OPTION_COLUMNS))), num_permutations)]


def restore_permuted_choice_logits(choice_logits: torch.Tensor, permutation: Sequence[int]) -> torch.Tensor:
    """Map logits from a permuted option order back to the original order.

    Args:
        choice_logits: Tensor of shape ``(..., 4)`` in permuted option order.
        permutation: Mapping from new option index to original option index.

    Returns:
        A tensor with the same shape as ``choice_logits`` in the original
        ``A/B/C/D`` order.
    """
    restored_logits = torch.empty_like(choice_logits)
    for new_index, old_index in enumerate(permutation):
        restored_logits[..., old_index] = choice_logits[..., new_index]
    return restored_logits


def label_id_to_text(label_id: int) -> str:
    """Convert an integer class id into an option letter.

    Args:
        label_id: Integer label in ``{0, 1, 2, 3}``.

    Returns:
        The corresponding option letter in ``{"A", "B", "C", "D"}``.
    """
    return ID_TO_LABEL[int(label_id)]


def label_text_to_id(label_text: str) -> int:
    """Convert an option letter into an integer class id.

    Args:
        label_text: Answer letter such as ``"A"`` or ``"c"``.

    Returns:
        The corresponding integer label.
    """
    return LABEL_TO_ID[label_text.strip().upper()]


def extract_prediction(text: str) -> int:
    """Parse the first valid answer choice from model text output.

    Args:
        text: Raw generated text from the model.

    Returns:
        The predicted class id.

    Raises:
        ValueError: If no valid answer choice can be parsed.
    """
    normalized = text.strip().upper()
    for choice in ("A", "B", "C", "D"):
        if normalized.startswith(choice):
            return LABEL_TO_ID[choice]
    raise ValueError(f"Unable to parse prediction from text: {text!r}")


def get_choice_token_ids(tokenizer) -> Dict[str, int]:
    """Map answer letters to their tokenizer token ids.

    Args:
        tokenizer: Hugging Face tokenizer used by the baseline model.

    Returns:
        A mapping from answer letters to token ids.

    Raises:
        ValueError: If a choice letter does not map to a single token.
    """
    token_ids: Dict[str, int] = {}
    for choice in ("A", "B", "C", "D"):
        encoded = tokenizer.encode(f" {choice}", add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Choice {choice} is tokenized into {encoded}. "
                "This baseline expects each answer letter to map to a single token."
            )
        token_ids[choice] = encoded[0]
    return token_ids


def compute_accuracy(predictions: Sequence[int], references: Sequence[int]) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted class ids.
        references: Ground-truth class ids.

    Returns:
        The fraction of correct predictions.
    """
    if not references:
        return 0.0
    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    return correct / len(references)


def save_json(data: Dict[str, object], save_path: str | Path) -> None:
    """Save a dictionary as a JSON file.

    Args:
        data: Serializable dictionary content.
        save_path: Destination JSON path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_json(json_path: str | Path) -> Dict[str, object]:
    """Load a JSON file into a dictionary.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def plot_training_history(history: List[Dict[str, float]], save_path: str | Path) -> None:
    """Plot training and validation metrics across epochs.

    Args:
        history: Per-epoch metric dictionaries.
        save_path: Output image path.
    """
    if not history:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    val_accuracies = [item["val_accuracy"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, marker="o", label="train_loss")
    axes[0].plot(epochs, val_losses, marker="o", label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_accuracies, marker="o", color="tab:green", label="val_accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_test_results(
    predictions: Sequence[int],
    references: Sequence[int],
    save_path: str | Path,
) -> None:
    """Plot held-out test correctness and label distributions.

    Args:
        predictions: Predicted class ids on the test split.
        references: Ground-truth class ids on the test split.
        save_path: Output image path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(references)
    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    incorrect = total - correct

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(["correct", "incorrect"], [correct, incorrect], color=["tab:green", "tab:red"])
    axes[0].set_title("Test Prediction Outcome")
    axes[0].set_ylabel("Count")

    labels = ["A", "B", "C", "D"]
    pred_counts = [sum(int(pred == idx) for pred in predictions) for idx in range(4)]
    ref_counts = [sum(int(ref == idx) for ref in references) for idx in range(4)]
    x = range(len(labels))
    width = 0.35
    axes[1].bar([i - width / 2 for i in x], ref_counts, width=width, label="ground_truth")
    axes[1].bar([i + width / 2 for i in x], pred_counts, width=width, label="prediction")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Test Label Distribution")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@dataclass
class MCQBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SupervisedCollator:
    def __init__(self, tokenizer):
        """Initialize the collator with tokenizer padding metadata.

        Args:
            tokenizer: Tokenizer providing the pad token id.
        """
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> MCQBatch:
        """Pad a batch of supervised training features.

        Args:
            features: Tokenized examples with ``input_ids``, ``attention_mask``,
                and ``labels``.

        Returns:
            A padded ``MCQBatch`` ready for model input.
        """
        input_ids = pad_sequence(
            [feature["input_ids"] for feature in features],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [feature["attention_mask"] for feature in features],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=-100,
        )
        return MCQBatch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
