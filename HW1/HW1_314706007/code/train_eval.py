from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from utils import (
    OPTION_COLUMNS,
    SupervisedCollator,
    build_kfold_splits,
    build_prompt,
    build_training_completion,
    compute_accuracy,
    extract_prediction,
    get_option_permutations,
    load_dataset,
    plot_test_results,
    plot_training_history,
    save_json,
    set_seed,
    permute_answer_options,
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
    use_holdout_test: bool = False
    split_path: str = "../saved_models/splits/default_holdout_split.json"
    tuning_split_path: str = "../saved_models/splits/default_train_val_split.json"
    batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 30
    weight_decay: float = 0.02
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    shuffle_option_augmentation: bool = True
    option_order_ensemble: bool = True
    num_option_order_permutations: int = 4
    generation_max_new_tokens: int = 40
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
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


class PromptOnlyDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        shuffle_options: bool = False,
        option_permutation: Sequence[int] | None = None,
    ) -> None:
        """Build prompt-only examples for evaluation or inference.

        Args:
            dataframe: MCQ samples with or without labels.
            tokenizer: Tokenizer used to encode prompts.
            max_length: Maximum prompt length.
            shuffle_options: Whether to randomly permute answer option order
                and remap labels when building each training example.
            option_permutation: Optional deterministic permutation applied to
                the answer options for each example.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_options = shuffle_options
        self.option_permutation = list(option_permutation) if option_permutation is not None else None
        self.examples = None if shuffle_options else [self._build_example(index) for index in range(len(self.dataframe))]

    def __len__(self) -> int:
        """Return the number of dataset rows.

        Returns:
            The number of examples available in the dataset.
        """
        return len(self.dataframe)

    def _build_example(self, index: int) -> Dict[str, torch.Tensor]:
        """Tokenize one prompt-only example.

        Args:
            index: Dataset row index.

        Returns:
            A dictionary containing encoded prompt tensors and associated
            metadata such as ``question_id`` and optional ``target``.
        """
        row = self.dataframe.iloc[index]
        if self.option_permutation is not None:
            row = permute_answer_options(row, self.option_permutation)
        elif self.shuffle_options:
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
        if "ans" in row:
            item["target"] = torch.tensor(int(row["ans"]), dtype=torch.long)
        item["question_id"] = torch.tensor(int(row["question_id"]), dtype=torch.long)
        item["options"] = tuple(str(row[column]).strip() for column in OPTION_COLUMNS)
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


class SupervisedDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int, shuffle_options: bool = False):
        """Build supervised causal-LM examples for fine-tuning.

        Args:
            dataframe: Labeled MCQ samples used for training or validation.
            tokenizer: Tokenizer used to encode prompts and answers.
            max_length: Maximum combined prompt-answer sequence length.
            shuffle_options: Whether to randomly shuffle answer options for
                augmentation before prompt construction.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_options = shuffle_options

    def __len__(self) -> int:
        """Return the number of rows in the dataset.

        Returns:
            The dataset size.
        """
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Create one supervised example with labels masked over the prompt.

        Args:
            index: Dataset row index.

        Returns:
            A dictionary containing ``input_ids``, ``attention_mask``, and
            ``labels`` for causal language-model training.
        """
        row = self.dataframe.iloc[index]
        if self.shuffle_options:
            row = shuffle_answer_options(row)

        prompt = build_prompt(row)
        answer_text = build_training_completion(row)
        full_text = prompt + answer_text
        encoded = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        prompt_encoded = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def prompt_collate_fn(features: List[Dict[str, torch.Tensor]], tokenizer) -> Dict[str, torch.Tensor]:
    """Pad prompt-only examples into a batch.

    Args:
        features: Prompt-only dataset items.
        tokenizer: Tokenizer providing the pad token id.

    Returns:
        A dictionary containing padded tensors and per-item metadata.
    """
    max_length = max(feature["input_ids"].size(0) for feature in features)
    input_ids = torch.stack(
        [
            torch.nn.functional.pad(
                feature["input_ids"],
                (max_length - feature["input_ids"].size(0), 0),
                value=tokenizer.pad_token_id,
            )
            for feature in features
        ]
    )
    attention_mask = torch.stack(
        [
            torch.nn.functional.pad(
                feature["attention_mask"],
                (max_length - feature["attention_mask"].size(0), 0),
                value=0,
            )
            for feature in features
        ]
    )
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "target" in features[0]:
        batch["target"] = torch.stack([feature["target"] for feature in features])
    batch["question_id"] = torch.stack([feature["question_id"] for feature in features])
    batch["options"] = [feature["options"] for feature in features]
    return batch


def create_prompt_dataloader(
    dataframe: pd.DataFrame,
    tokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    option_permutation: Sequence[int] | None = None,
) -> DataLoader:
    """Create a prompt-only dataloader.

    Args:
        dataframe: Dataframe containing MCQ rows to tokenize.
        tokenizer: Tokenizer used to encode prompts.
        max_length: Maximum prompt length.
        batch_size: Number of examples per batch.
        num_workers: Number of dataloader worker processes.
        pin_memory: Whether to pin CPU memory for faster host-to-device copies.
        option_permutation: Optional deterministic option permutation applied
            to every row before tokenization.

    Returns:
        A dataloader yielding prompt-only batches.
    """
    dataset = PromptOnlyDataset(
        dataframe,
        tokenizer,
        max_length=max_length,
        option_permutation=option_permutation,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: prompt_collate_fn(batch, tokenizer),
    )


def compute_supervised_loss(logits, labels: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    """Compute causal-LM loss with optional label smoothing."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


def evaluate_supervised_loss(
    model,
    dataloader: DataLoader,
    device: torch.device,
    label_smoothing: float,
) -> float:
    """Compute average causal-LM loss over supervised labels.

    Args:
        model: Causal language model under evaluation.
        dataloader: Dataloader yielding supervised training-style batches.
        device: Device used for evaluation.

    Returns:
        Mean supervised loss across all batches.
    """
    model.eval()
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    running_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = compute_supervised_loss(outputs.logits, labels, label_smoothing)
            running_loss += loss.item()
            batch_count += 1

    return running_loss / max(batch_count, 1)


def collect_generated_predictions(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> pd.DataFrame:
    """Generate responses and extract final answer predictions into a DataFrame.

    Args:
        model: Causal language model used for generation.
        dataloader: Dataloader yielding prompt-only evaluation batches.
        tokenizer: Tokenizer used to decode generated tokens.
        device: Device used for inference.

    Returns:
        A DataFrame containing question ids, parsed predictions, generated
        text, and optional labels.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            input_lengths = attention_mask.sum(dim=1).tolist()
            generated_texts = [
                tokenizer.decode(output_ids[int(input_length):], skip_special_tokens=True)
                for output_ids, input_length in zip(generated_ids, input_lengths)
            ]

            targets = batch["target"].tolist() if "target" in batch else [None] * len(generated_texts)
            batch_options = batch.get("options", [None] * len(generated_texts))
            for question_id, text, target, options in zip(
                batch["question_id"].tolist(),
                generated_texts,
                targets,
                batch_options,
            ):
                prediction = extract_prediction(text, options=options)
                row = {
                    "question_id": question_id,
                    "prediction": prediction,
                    "generated_text": text,
                }
                if target is not None:
                    row["target"] = target
                    row["correct"] = int(prediction == target)
                rows.append(row)

    return pd.DataFrame(rows)


def generate_predictions_from_dataframe(
    model,
    dataframe: pd.DataFrame,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    option_order_ensemble: bool,
    num_option_order_permutations: int,
    max_new_tokens: int,
) -> pd.DataFrame:
    """Generate predictions with optional hard voting over option orderings.

    Args:
        model: Causal language model used for generation.
        dataframe: Benchmark or validation dataframe.
        tokenizer: Tokenizer used to decode generated tokens.
        device: Device used for inference.
        max_length: Maximum prompt length.
        batch_size: Evaluation batch size.
        num_workers: Number of dataloader workers.
        pin_memory: Whether to pin host memory for faster device transfers.
        option_order_ensemble: Whether to hard-vote across option permutations.
        num_option_order_permutations: Number of permutations to evaluate.
        max_new_tokens: Maximum completion length during generation.

    Returns:
        A dataframe containing final predictions and optional labels.
    """
    num_permutations = num_option_order_permutations if option_order_ensemble else 1
    permutations = get_option_permutations(num_permutations)

    all_permutation_predictions: List[List[int]] = []
    question_ids = None
    targets = None
    generated_texts = None

    for permutation in permutations:
        dataloader = create_prompt_dataloader(
            dataframe,
            tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            option_permutation=permutation,
        )
        predictions_df = collect_generated_predictions(
            model,
            dataloader,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
        )
        mapped_predictions = [permutation[prediction] for prediction in predictions_df["prediction"].tolist()]
        all_permutation_predictions.append(mapped_predictions)

        if question_ids is None:
            question_ids = predictions_df["question_id"].tolist()
            generated_texts = predictions_df["generated_text"].tolist()
            if "target" in predictions_df:
                targets = predictions_df["target"].tolist()

    final_predictions = []
    for index in range(len(question_ids or [])):
        votes = [predictions[index] for predictions in all_permutation_predictions]
        winner = Counter(votes).most_common(1)[0][0]
        final_predictions.append(winner)

    result = pd.DataFrame(
        {
            "question_id": question_ids,
            "prediction": final_predictions,
            "generated_text": generated_texts,
        }
    )
    if targets is not None:
        result["target"] = targets
        result["correct"] = (result["prediction"] == result["target"]).astype(int)
    return result


def evaluate_generation_accuracy(
    model,
    dataframe: pd.DataFrame,
    tokenizer,
    device: torch.device,
    max_length: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    option_order_ensemble: bool,
    num_option_order_permutations: int,
    max_new_tokens: int,
) -> float:
    """Compute accuracy from generated explanations and final answers."""
    predictions_df = generate_predictions_from_dataframe(
        model,
        dataframe,
        tokenizer,
        device,
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        option_order_ensemble=option_order_ensemble,
        num_option_order_permutations=num_option_order_permutations,
        max_new_tokens=max_new_tokens,
    )
    return compute_accuracy(
        predictions_df["prediction"].tolist(),
        predictions_df["target"].tolist(),
    )


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


def configure_greedy_generation(model) -> None:
    """Normalize generation config for greedy decoding without sampling warnings.

    Args:
        model: Causal language model whose generation config should be updated.

    Returns:
        None.
    """
    if getattr(model, "generation_config", None) is None:
        return
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0


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
    model.config.use_cache = False

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

    model.gradient_checkpointing_enable()

    model.to(device)
    configure_greedy_generation(model)
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
    model.config.use_cache = True
    model.to(device)
    configure_greedy_generation(model)
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
    tokenizer.padding_side = "left"

    model = create_model(config, tokenizer, device)
    pin_memory = device.type == "cuda"

    train_dataset = SupervisedDataset(
        train_df,
        tokenizer,
        config.max_length,
        shuffle_options=config.shuffle_option_augmentation,
    )
    val_supervised_dataset = SupervisedDataset(val_df, tokenizer, config.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=SupervisedCollator(tokenizer),
    )
    val_supervised_loader = DataLoader(
        val_supervised_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        collate_fn=SupervisedCollator(tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max(len(train_loader) * config.num_epochs // config.grad_accum_steps, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    history: List[Dict[str, float]] = []
    best_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=False)

        for step, batch in enumerate(progress_bar, start=1):
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = compute_supervised_loss(
                    outputs.logits,
                    labels,
                    config.label_smoothing,
                ) / config.grad_accum_steps

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
        val_loss = evaluate_supervised_loss(
            model,
            val_supervised_loader,
            device,
            config.label_smoothing,
        )
        val_accuracy = evaluate_generation_accuracy(
            model,
            val_df,
            tokenizer,
            device,
            max_length=config.max_length,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            option_order_ensemble=config.option_order_ensemble,
            num_option_order_permutations=config.num_option_order_permutations,
            max_new_tokens=config.generation_max_new_tokens,
        )

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
        "selection_metric": "best_val_accuracy",
        "selection_score": best_accuracy,
        "best_val_accuracy": best_accuracy,
        "evaluation_mode": "validation_plus_holdout_test" if test_df is not None else "validation_only",
        "train_size": len(train_df),
        "val_size": len(val_df),
    }
    if test_df is not None:
        test_predictions_df = generate_predictions_from_dataframe(
            best_model,
            test_df,
            tokenizer,
            device,
            max_length=config.max_length,
            batch_size=config.eval_batch_size,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            option_order_ensemble=config.option_order_ensemble,
            num_option_order_permutations=config.num_option_order_permutations,
            max_new_tokens=config.generation_max_new_tokens,
        )
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
    print(f"Selection metric (best validation accuracy): {metrics['best_val_accuracy']:.4f}")
    if "test_accuracy" in metrics:
        print(f"Final held-out test accuracy: {metrics['test_accuracy']:.4f}")
    return metrics


def train(config: TrainingConfig) -> None:
    """Run either the original split workflow or optional k-fold training.

    Args:
        config: Training configuration values.

    Returns:
        None.
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

        mean_best_val_accuracy = sum(metric["best_val_accuracy"] for metric in fold_metrics) / max(len(fold_metrics), 1)
        summary = {
            "run_dir": str(run_dir),
            "selection_metric": "mean_best_val_accuracy",
            "selection_score": mean_best_val_accuracy,
            "num_folds": config.num_folds,
            "mean_best_val_accuracy": mean_best_val_accuracy,
            "fold_metrics": fold_metrics,
        }
        save_json(summary, run_dir / "cv_metrics.json")
        print(f"Saved k-fold outputs to {run_dir}")
        print(f"Mean best validation accuracy across folds: {summary['mean_best_val_accuracy']:.4f}")
        return

    if config.use_holdout_test:
        train_df, val_df, test_df = split_dataframe(
            full_df,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            random_state=config.seed,
            split_path=config.split_path,
        )
        print("Running train/validation/test workflow. Use best_val_accuracy for model selection; treat held-out test as a final check.")
    else:
        train_df, val_df, _ = split_dataframe(
            full_df,
            val_ratio=config.val_ratio,
            test_ratio=0.0,
            random_state=config.seed,
            split_path=config.tuning_split_path,
        )
        test_df = None
        print("Running train/validation-only workflow. Use best_val_accuracy as the main model-selection target.")
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
