from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
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
    plot_test_results,
    plot_training_history,
    save_json,
    set_seed,
    split_dataframe,
)


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_csv: str = "../dataset/dataset.csv"
    output_dir: str = "../saved_models/checkpoint"
    batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 50
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    grad_accum_steps: int = 4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 4
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"


class SupervisedMCQDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int) -> None:
        """Build supervised examples for causal language model training.

        Args:
            dataframe: Labeled MCQ samples.
            tokenizer: Tokenizer used to encode prompts and answers.
            max_length: Maximum token length per sample.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [self._build_example(index) for index in range(len(self.dataframe))]

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.examples)

    def _build_example(self, index: int) -> Dict[str, torch.Tensor]:
        """Tokenize and cache one supervised training example."""
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Create one supervised training example.

        Args:
            index: Dataset row index.

        Returns:
            A dictionary containing tokenized model inputs and labels.
        """
        return self.examples[index]


class PromptOnlyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int) -> None:
        """Build prompt-only examples for evaluation or inference.

        Args:
            dataframe: MCQ samples with or without labels.
            tokenizer: Tokenizer used to encode prompts.
            max_length: Maximum prompt length.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = [self._build_example(index) for index in range(len(self.dataframe))]

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.examples)

    def _build_example(self, index: int) -> Dict[str, torch.Tensor]:
        """Tokenize and cache one prompt-only example."""
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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Create one prompt-only evaluation example.

        Args:
            index: Dataset row index.

        Returns:
            A dictionary containing tokenized prompt tensors and metadata.
        """
        return self.examples[index]


def prompt_collate_fn(features: List[Dict[str, torch.Tensor]], tokenizer) -> Dict[str, torch.Tensor]:
    """Pad prompt-only examples into a batch.

    Args:
        features: Prompt-only dataset items.
        tokenizer: Tokenizer providing the pad token id.

    Returns:
        A dictionary containing padded tensors and per-item metadata.
    """
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
    """Compute average loss over a supervised dataloader.

    Args:
        model: Causal language model under evaluation.
        dataloader: Dataloader that yields supervised batches.
        device: Device used for evaluation.

    Returns:
        Mean batch loss across the dataloader.
    """
    model.eval()
    running_loss = 0.0
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch in dataloader:
            batch = MCQBatch(
                input_ids=batch.input_ids.to(device, non_blocking=True),
                attention_mask=batch.attention_mask.to(device, non_blocking=True),
                labels=batch.labels.to(device, non_blocking=True),
            )
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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
    """Predict answer class ids from prompt-only batches.

    Args:
        model: Trained causal language model.
        dataloader: Dataloader yielding prompt-only batches.
        tokenizer: Tokenizer used to identify answer token ids.
        device: Device used for inference.

    Returns:
        Predicted class ids for all examples in order.
    """
    model.eval()
    choice_token_ids = get_choice_token_ids(tokenizer)
    candidate_ids = torch.tensor([choice_token_ids["A"], choice_token_ids["B"], choice_token_ids["C"], choice_token_ids["D"]], device=device)
    predictions: List[int] = []
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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
    """Compute classification accuracy from prompt-only evaluation batches.

    Args:
        model: Trained causal language model.
        dataloader: Dataloader yielding prompt-only labeled batches.
        tokenizer: Tokenizer used to identify answer token ids.
        device: Device used for inference.

    Returns:
        Accuracy on the provided dataloader.
    """
    predictions_df = collect_predictions(model, dataloader, tokenizer, device)
    return compute_accuracy(
        predictions_df["prediction"].tolist(),
        predictions_df["target"].tolist(),
    )


def collect_predictions(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
) -> pd.DataFrame:
    """Collect per-example predictions into a DataFrame.

    Args:
        model: Trained causal language model.
        dataloader: Dataloader yielding prompt-only batches.
        tokenizer: Tokenizer used to identify answer token ids.
        device: Device used for inference.

    Returns:
        A DataFrame with question ids, predictions, and optional targets.
    """
    model.eval()
    choice_token_ids = get_choice_token_ids(tokenizer)
    candidate_ids = torch.tensor([choice_token_ids["A"], choice_token_ids["B"], choice_token_ids["C"], choice_token_ids["D"]], device=device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=device)
            next_token_logits = outputs.logits[batch_indices, last_indices]
            choice_logits = next_token_logits.index_select(dim=-1, index=candidate_ids)
            batch_predictions = choice_logits.argmax(dim=-1).detach().cpu().tolist()

            targets = batch["target"].tolist() if "target" in batch else [None] * len(batch_predictions)
            for question_id, prediction, target in zip(
                batch["question_id"].tolist(),
                batch_predictions,
                targets,
            ):
                row = {
                    "question_id": question_id,
                    "prediction": prediction,
                }
                if target is not None:
                    row["target"] = target
                    row["correct"] = int(prediction == target)
                rows.append(row)
    return pd.DataFrame(rows)


def save_checkpoint(model, tokenizer, output_dir: str | Path, config: TrainingConfig, history: List[Dict[str, float]]) -> None:
    """Save the current best checkpoint and related metadata.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        output_dir: Checkpoint directory.
        config: Training configuration.
        history: Training history up to the current checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if not getattr(model, "is_peft_model", False):
        torch.save(model.state_dict(), output_dir / "model.pt")
    save_json(asdict(config), output_dir / "train_config.json")
    save_json({"history": history}, output_dir / "history.json")


def create_model(config: TrainingConfig, tokenizer, device: torch.device):
    """Load a base language model and optionally attach LoRA adapters.

    Args:
        config: Training configuration containing the base model name and
            optional LoRA hyperparameters.
        tokenizer: Tokenizer used to provide the model pad token id.
        device: Target device where the model should be placed.

    Returns:
        A causal language model on ``device``. If ``config.use_lora`` is
        ``True``, the returned model is a PEFT-wrapped model with trainable
        LoRA adapters attached to the requested target modules.
    """
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    if config.use_lora:
        target_modules = [module.strip() for module in config.lora_target_modules.split(",") if module.strip()]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.to(device)
    return model


def load_saved_model(model_dir: str | Path, tokenizer, device: torch.device):
    """Load a checkpoint saved from either LoRA or full fine-tuning.

    Args:
        model_dir: Directory containing a saved checkpoint. This may be a LoRA
            adapter directory with ``adapter_config.json`` or a full model
            directory saved with ``save_pretrained``.
        tokenizer: Tokenizer used to set the model pad token id after loading.
        device: Target device where the loaded model should be placed.

    Returns:
        A causal language model restored from ``model_dir`` and moved to
        ``device``. LoRA checkpoints are reconstructed by loading the base
        model first and then attaching the saved adapters.
    """
    model_dir = Path(model_dir)
    adapter_config_path = model_dir / "adapter_config.json"

    if adapter_config_path.exists():
        peft_config = PeftConfig.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    return model


def create_run_dir(output_root: str | Path) -> Path:
    """Create a timestamped directory for one training run.

    Args:
        output_root: Base checkpoint directory.

    Returns:
        A newly created timestamped run directory.
    """
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def train(config: TrainingConfig) -> None:
    """Run baseline training, validation, and held-out test evaluation.

    Args:
        config: Training configuration values.
    """
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_dir(config.output_dir)

    full_df = load_dataset(config.dataset_csv)
    train_df, val_df, test_df = split_dataframe(
        full_df,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model(config, tokenizer, device)
    pin_memory = device.type == "cuda"

    train_dataset = SupervisedMCQDataset(train_df, tokenizer, config.max_length)
    val_dataset = SupervisedMCQDataset(val_df, tokenizer, config.max_length)
    val_prompt_dataset = PromptOnlyDataset(val_df, tokenizer, config.max_length)
    test_prompt_dataset = PromptOnlyDataset(test_df, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=SupervisedCollator(tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=SupervisedCollator(tokenizer),
    )
    val_prompt_loader = DataLoader(
        val_prompt_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
    )
    test_prompt_loader = DataLoader(
        test_prompt_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
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
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
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

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                )
                loss = outputs.loss / config.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * config.grad_accum_steps

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
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
            save_checkpoint(model, tokenizer, run_dir, config, history)

    plot_training_history(history, run_dir / "training_curve.png")

    best_model = load_saved_model(run_dir, tokenizer, device)

    test_predictions_df = collect_predictions(best_model, test_prompt_loader, tokenizer, device)
    test_accuracy = compute_accuracy(
        test_predictions_df["prediction"].tolist(),
        test_predictions_df["target"].tolist(),
    )
    test_predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)
    plot_test_results(
        predictions=test_predictions_df["prediction"].tolist(),
        references=test_predictions_df["target"].tolist(),
        save_path=run_dir / "test_results.png",
    )

    metrics = {
        "run_dir": str(run_dir),
        "best_val_accuracy": best_accuracy,
        "test_accuracy": test_accuracy,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    save_json(metrics, run_dir / "metrics.json")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": asdict(config),
            "history": history,
            "metrics": metrics,
        },
        run_dir / "training_state.pt",
    )
    print(f"Saved run outputs to {run_dir}")
    print(f"Final held-out test accuracy: {test_accuracy:.4f}")


def parse_args() -> TrainingConfig:
    """Parse CLI arguments into a ``TrainingConfig`` instance.

    Returns:
        Parsed training configuration.
    """
    parser = argparse.ArgumentParser(description="Baseline training / evaluation for PathoQA.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_csv", type=str, default="../dataset/dataset.csv")
    parser.add_argument("--output_dir", type=str, default="../saved_models/checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
