from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


LABEL_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
OPTION_COLUMNS = ["opa", "opb", "opc", "opd"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ans" in df.columns:
        df["ans"] = df["ans"].astype(int)
    return df


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=df["ans"] if "ans" in df.columns else None,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def maybe_create_splits(
    dataset_csv: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = Path(train_csv)
    val_csv = Path(val_csv)

    if train_csv.exists() and val_csv.exists():
        return load_dataset(train_csv), load_dataset(val_csv)

    df = load_dataset(dataset_csv)
    train_df, val_df = split_dataframe(df, val_ratio=val_ratio, random_state=random_state)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    return train_df, val_df


def build_prompt(row: pd.Series | Dict[str, object]) -> str:
    return (
        "You are a medical pathology question answering assistant.\n"
        "Read the multiple-choice question and return only the correct option letter.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Options:\n"
        f"A. {row['opa']}\n"
        f"B. {row['opb']}\n"
        f"C. {row['opc']}\n"
        f"D. {row['opd']}\n\n"
        "Answer:"
    )


def label_id_to_text(label_id: int) -> str:
    return ID_TO_LABEL[int(label_id)]


def label_text_to_id(label_text: str) -> int:
    return LABEL_TO_ID[label_text.strip().upper()]


def extract_prediction(text: str) -> int:
    normalized = text.strip().upper()
    for choice in ("A", "B", "C", "D"):
        if normalized.startswith(choice):
            return LABEL_TO_ID[choice]
    raise ValueError(f"Unable to parse prediction from text: {text!r}")


def get_choice_token_ids(tokenizer) -> Dict[str, int]:
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
    if not references:
        return 0.0
    correct = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    return correct / len(references)


def save_json(data: Dict[str, object], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_json(json_path: str | Path) -> Dict[str, object]:
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def plot_training_history(history: List[Dict[str, float]], save_path: str | Path) -> None:
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


@dataclass
class MCQBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SupervisedCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> MCQBatch:
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
