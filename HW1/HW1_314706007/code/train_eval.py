from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from utils import (
    MCQBatch,
    SupervisedCollator,
    build_prompt,
    compute_accuracy,
    get_choice_token_ids,
    label_id_to_text,
    load_dataset,
    maybe_create_splits,
    plot_training_history,
    save_json,
    set_seed,
)


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_csv: str = "../dataset/dataset.csv"
    train_csv: str = "../dataset/train.csv"
    val_csv: str = "../dataset/val.csv"
    output_dir: str = "../saved_models/checkpoint/baseline"
    batch_size: int = 2
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    grad_accum_steps: int = 8
    val_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 0


class SupervisedMCQDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        prompt = build_prompt(row)
        answer = f" {label_id_to_text(int(row['ans']))}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[: max(self.max_length - len(answer_ids), 1)]

        input_ids = prompt_ids + answer_ids
        prompt_length = len(prompt_ids)
        labels = [-100] * prompt_length + input_ids[prompt_length:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PromptOnlyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        prompt = build_prompt(row)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if "ans" in row.index:
            item["target"] = torch.tensor(int(row["ans"]), dtype=torch.long)
        item["question_id"] = torch.tensor(int(row["question_id"]), dtype=torch.long)
        return item


def prompt_collate_fn(features: List[Dict[str, torch.Tensor]], tokenizer) -> Dict[str, torch.Tensor]:
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [feature["input_ids"] for feature in features],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [feature["attention_mask"] for feature in features],
        batch_first=True,
        padding_value=0,
    )
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "target" in features[0]:
        batch["target"] = torch.stack([feature["target"] for feature in features])
    batch["question_id"] = torch.stack([feature["question_id"] for feature in features])
    return batch


def evaluate_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = MCQBatch(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
            )
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            )
            running_loss += outputs.loss.item()

    return running_loss / max(len(dataloader), 1)


def predict_choice_ids(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
) -> List[int]:
    model.eval()
    choice_token_ids = get_choice_token_ids(tokenizer)
    candidate_ids = torch.tensor([choice_token_ids["A"], choice_token_ids["B"], choice_token_ids["C"], choice_token_ids["D"]], device=device)
    predictions: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=device)
            next_token_logits = outputs.logits[batch_indices, last_indices]
            choice_logits = next_token_logits.index_select(dim=-1, index=candidate_ids)
            batch_predictions = choice_logits.argmax(dim=-1).detach().cpu().tolist()
            predictions.extend(batch_predictions)

    return predictions


def evaluate_accuracy(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
) -> float:
    predictions = predict_choice_ids(model, dataloader, tokenizer, device)
    references: List[int] = []
    for batch in dataloader:
        references.extend(batch["target"].tolist())
    return compute_accuracy(predictions, references)


def save_checkpoint(model, tokenizer, output_dir: str | Path, config: TrainingConfig, history: List[Dict[str, float]]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json(asdict(config), output_dir / "train_config.json")
    save_json({"history": history}, output_dir / "history.json")


def train(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df = maybe_create_splits(
        dataset_csv=config.dataset_csv,
        train_csv=config.train_csv,
        val_csv=config.val_csv,
        val_ratio=config.val_ratio,
        random_state=config.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    train_dataset = SupervisedMCQDataset(train_df, tokenizer, config.max_length)
    val_dataset = SupervisedMCQDataset(val_df, tokenizer, config.max_length)
    val_prompt_dataset = PromptOnlyDataset(val_df, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=SupervisedCollator(tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=SupervisedCollator(tokenizer),
    )
    val_prompt_loader = DataLoader(
        val_prompt_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(len(train_loader) * config.num_epochs // config.grad_accum_steps, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: List[Dict[str, float]] = []
    best_accuracy = -1.0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=False)

        for step, batch in enumerate(progress_bar, start=1):
            batch = MCQBatch(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                labels=batch.labels.to(device),
            )

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )
                loss = outputs.loss / config.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * config.grad_accum_steps

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            progress_bar.set_postfix(loss=f"{running_loss / step:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = evaluate_loss(model, val_loader, device)
        val_accuracy = evaluate_accuracy(model, val_prompt_loader, tokenizer, device)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_log)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(model, tokenizer, config.output_dir, config, history)

    plot_training_history(history, Path(config.output_dir) / "training_curve.png")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Baseline training / evaluation for PathoQA.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_csv", type=str, default="../dataset/dataset.csv")
    parser.add_argument("--train_csv", type=str, default="../dataset/train.csv")
    parser.add_argument("--val_csv", type=str, default="../dataset/val.csv")
    parser.add_argument("--output_dir", type=str, default="../saved_models/checkpoint/baseline")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
