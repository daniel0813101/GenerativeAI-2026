from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from utils import (
    build_kfold_splits,
    build_prompt,
    compute_accuracy,
    get_choice_token_ids,
    load_dataset,
    plot_test_results,
    plot_training_history,
    save_json,
    set_seed,
    shuffle_answer_options,
    split_dataframe,
)


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    dataset_csv: str = "../dataset/dataset.csv"
    output_dir: str = "../saved_models/checkpoint"
    use_kfold: bool = False
    num_folds: int = 5
    kfold_split_path: str = "../saved_models/splits/default_kfold_5.json"
    split_path: str = "../saved_models/splits/default_split.json"
    batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 50
    weight_decay: float = 0.02
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    shuffle_option_augmentation: bool = True
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 0.0
    max_length: int = 1024
    grad_accum_steps: int = 4
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    num_workers: int = 4
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"


class PromptOnlyDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        shuffle_options: bool = False,
    ) -> None:
        """Build prompt-only examples for evaluation or inference.

        Args:
            dataframe: MCQ samples with or without labels.
            tokenizer: Tokenizer used to encode prompts.
            max_length: Maximum prompt length.
            shuffle_options: Whether to randomly permute answer option order
                and remap labels when building each training example.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_options = shuffle_options
        self.examples = None if shuffle_options else [self._build_example(index) for index in range(len(self.dataframe))]

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""
        return len(self.dataframe)

    def _build_example(self, index: int) -> Dict[str, torch.Tensor]:
        """Tokenize and cache one prompt-only example."""
        row = self.dataframe.iloc[index]
        if self.shuffle_options:
            row = shuffle_answer_options(row)
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
        if self.examples is not None:
            return self.examples[index]
        return self._build_example(index)


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


def compute_choice_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute logits over answer choices for the next token position.

    Args:
        model: Causal language model being trained or evaluated.
        input_ids: Prompt token ids with shape ``(batch_size, seq_len)``.
        attention_mask: Attention mask with shape ``(batch_size, seq_len)``.
        candidate_ids: Token ids for the answer choices ``A/B/C/D``.

    Returns:
        A tensor of shape ``(batch_size, 4)`` containing logits for the
        answer choices only.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
    next_token_logits = outputs.logits[batch_indices, last_indices]
    return next_token_logits.index_select(dim=-1, index=candidate_ids)


def evaluate_choice_loss(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
) -> float:
    """Compute choice-only validation loss over A/B/C/D logits.

    Args:
        model: Causal language model under evaluation.
        dataloader: Dataloader yielding prompt-only labeled batches.
        tokenizer: Tokenizer used to identify answer token ids.
        device: Device used for evaluation.

    Returns:
        Mean cross-entropy loss over the four answer choices.
    """
    model.eval()
    running_loss = 0.0
    batch_count = 0
    choice_token_ids = get_choice_token_ids(tokenizer)
    candidate_ids = torch.tensor(
        [choice_token_ids["A"], choice_token_ids["B"], choice_token_ids["C"], choice_token_ids["D"]],
        device=device,
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                choice_logits = compute_choice_logits(model, input_ids, attention_mask, candidate_ids)
            running_loss += F.cross_entropy(choice_logits, targets).item()
            batch_count += 1

    return running_loss / max(batch_count, 1)


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
                choice_logits = compute_choice_logits(model, input_ids, attention_mask, candidate_ids)
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
                choice_logits = compute_choice_logits(model, input_ids, attention_mask, candidate_ids)
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


def train_one_split(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_dir: Path,
    device: torch.device,
    test_df: pd.DataFrame | None = None,
) -> Dict[str, object]:
    """Train and evaluate one train/validation split.

    Args:
        config: Training configuration values.
        train_df: Labeled training split.
        val_df: Labeled validation split.
        run_dir: Output directory for this training run.
        device: Device used for training and evaluation.
        test_df: Optional held-out labeled test split for evaluation.

    Returns:
        A metrics dictionary summarizing the run.
    """
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model(config, tokenizer, device)
    pin_memory = device.type == "cuda"

    train_prompt_dataset = PromptOnlyDataset(
        train_df,
        tokenizer,
        config.max_length,
        shuffle_options=config.shuffle_option_augmentation,
    )
    val_prompt_dataset = PromptOnlyDataset(val_df, tokenizer, config.max_length)
    test_prompt_dataset = (
        PromptOnlyDataset(test_df, tokenizer, config.max_length)
        if test_df is not None
        else None
    )

    train_loader = DataLoader(
        train_prompt_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
    )
    val_prompt_loader = DataLoader(
        val_prompt_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
    )
    test_prompt_loader = (
        DataLoader(
            test_prompt_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
        )
        if test_prompt_dataset is not None
        else None
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
    choice_token_ids = get_choice_token_ids(tokenizer)
    candidate_ids = torch.tensor(
        [choice_token_ids["A"], choice_token_ids["B"], choice_token_ids["C"], choice_token_ids["D"]],
        device=device,
    )
    history: List[Dict[str, float]] = []
    best_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=False)

        for step, batch in enumerate(progress_bar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                choice_logits = compute_choice_logits(model, input_ids, attention_mask, candidate_ids)
                batch_loss = F.cross_entropy(
                    choice_logits,
                    targets,
                    label_smoothing=config.label_smoothing,
                )
                loss = batch_loss / config.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += batch_loss.item()

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
        val_loss = evaluate_choice_loss(model, val_prompt_loader, tokenizer, device)
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

        if val_accuracy > best_accuracy + config.early_stopping_min_delta:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(model, tokenizer, run_dir, config, history)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    "Early stopping triggered: "
                    f"no validation accuracy improvement for {config.early_stopping_patience} epoch(s)."
                )
                break

    plot_training_history(history, run_dir / "training_curve.png")

    best_model = load_saved_model(run_dir, tokenizer, device)

    metrics = {
        "run_dir": str(run_dir),
        "best_val_accuracy": best_accuracy,
        "train_size": len(train_df),
        "val_size": len(val_df),
    }
    if test_prompt_loader is not None and test_df is not None:
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
        metrics["test_accuracy"] = test_accuracy
        metrics["test_size"] = len(test_df)

    save_json({"history": history}, run_dir / "history.json")
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
    if "test_accuracy" in metrics:
        print(f"Final held-out test accuracy: {metrics['test_accuracy']:.4f}")
    return metrics


def train(config: TrainingConfig) -> None:
    """Run either the original split workflow or optional k-fold training.

    Args:
        config: Training configuration values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_dir(config.output_dir)
    save_json(asdict(config), run_dir / "train_config.json")

    full_df = load_dataset(config.dataset_csv)

    if config.use_kfold:
        fold_splits = build_kfold_splits(
            full_df,
            num_folds=config.num_folds,
            random_state=config.seed,
            split_path=config.kfold_split_path,
        )
        fold_metrics = []
        for fold_index, (train_df, val_df) in enumerate(fold_splits, start=1):
            fold_run_dir = run_dir / f"fold_{fold_index}"
            print(f"Starting fold {fold_index}/{config.num_folds}")
            fold_metrics.append(
                train_one_split(
                    config=config,
                    train_df=train_df,
                    val_df=val_df,
                    run_dir=fold_run_dir,
                    device=device,
                    test_df=None,
                )
            )

        summary = {
            "run_dir": str(run_dir),
            "num_folds": config.num_folds,
            "mean_best_val_accuracy": sum(metric["best_val_accuracy"] for metric in fold_metrics) / max(len(fold_metrics), 1),
            "fold_metrics": fold_metrics,
        }
        save_json(summary, run_dir / "cv_metrics.json")
        print(f"Saved k-fold outputs to {run_dir}")
        print(f"Mean best validation accuracy across folds: {summary['mean_best_val_accuracy']:.4f}")
        return

    train_df, val_df, test_df = split_dataframe(
        full_df,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.seed,
        split_path=config.split_path,
    )
    train_one_split(
        config=config,
        train_df=train_df,
        val_df=val_df,
        run_dir=run_dir,
        device=device,
        test_df=test_df,
    )


def parse_args() -> TrainingConfig:
    """Parse CLI arguments into a ``TrainingConfig`` instance.

    Returns:
        Parsed training configuration.
    """
    default_config = TrainingConfig()
    parser = argparse.ArgumentParser(description="Baseline training / evaluation for PathoQA.")
    for config_field in fields(TrainingConfig):
        default_value = getattr(default_config, config_field.name)
        argument_name = f"--{config_field.name}"
        if isinstance(default_value, bool):
            parser.add_argument(
                argument_name,
                action=argparse.BooleanOptionalAction,
                default=default_value,
            )
        else:
            parser.add_argument(
                argument_name,
                type=type(default_value),
                default=default_value,
            )
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
