import argparse
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import BrainrotDataset, collate_batch
from src.diffusion import DiffusionConfig, GaussianDiffusion
from src.model_unet import ConditionalUNet
from src.utils import EMAModel, cosine_lr, ensure_dir, save_json, seed_everything


@dataclass
class TrainConfig:
    image_dir: str = "data/trainset"
    metadata_path: str = "data/train.csv"
    output_dir: str = "checkpoints/ddpm_unet"
    image_size: int = 64
    base_channels: int = 128
    channel_mults: str = "1,2,2,4"
    num_res_blocks: int = 2
    attention_resolutions: str = "16,8"
    dropout: float = 0.1
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    prediction_type: str = "epsilon"
    epochs: int = 400
    max_steps: int = 0
    batch_size: int = 128
    grad_accum_steps: int = 1
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    ema_decay: float = 0.9999
    condition_drop_prob: float = 0.1
    max_grad_norm: float = 1.0
    save_every: int = 5000
    num_workers: int = 8
    seed: int = 42
    mixed_precision: bool = True
    resume: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_model(config: TrainConfig) -> ConditionalUNet:
    return ConditionalUNet(
        image_channels=3,
        base_channels=config.base_channels,
        channel_mults=parse_int_tuple(config.channel_mults),
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=parse_int_tuple(config.attention_resolutions),
        dropout=config.dropout,
        image_size=config.image_size,
    )


def save_checkpoint(path: Path, model: ConditionalUNet, ema: EMAModel, optimizer: AdamW, config: TrainConfig, step: int, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    root = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(root / config.output_dir)
    save_json(output_dir / "config.yaml", asdict(config))

    dataset = BrainrotDataset(root / config.image_dir, root / config.metadata_path, image_size=config.image_size, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
    )
    device = torch.device(config.device)
    model = build_model(config).to(device)
    ema = EMAModel(model, decay=config.ema_decay)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            timesteps=config.timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
        ),
        device=device,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision and device.type == "cuda")
    start_step = 0
    start_epoch = 0
    if config.resume:
        ckpt = torch.load(root / config.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))

    total_steps = config.max_steps or (len(loader) * config.epochs // max(1, config.grad_accum_steps))
    step = start_step
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, config.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for batch_idx, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            animal_id = batch["animal_id"].to(device, non_blocking=True)
            object_id = batch["object_id"].to(device, non_blocking=True)
            drop = torch.rand(images.shape[0], device=device) < config.condition_drop_prob
            animal_id = torch.where(drop, torch.full_like(animal_id, model.null_animal_id), animal_id)
            object_id = torch.where(drop, torch.full_like(object_id, model.null_object_id), object_id)
            t = torch.randint(0, config.timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = diffusion.q_sample(images, t, noise)
            target = diffusion.training_target(images, t, noise)

            with torch.amp.autocast("cuda", enabled=config.mixed_precision and device.type == "cuda"):
                pred = model(noisy, t, animal_id, object_id)
                loss = F.mse_loss(pred, target) / config.grad_accum_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                lr = cosine_lr(step, total_steps, config.lr, config.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                step += 1
                progress.set_postfix(loss=f"{loss.item() * config.grad_accum_steps:.4f}", lr=f"{lr:.2e}", step=step)
                if step % config.save_every == 0:
                    save_checkpoint(output_dir / f"step_{step:07d}.pth", model, ema, optimizer, config, step, epoch)
                    save_checkpoint(output_dir / "latest.pth", model, ema, optimizer, config, step, epoch)
                if config.max_steps and step >= config.max_steps:
                    save_checkpoint(output_dir / "latest.pth", model, ema, optimizer, config, step, epoch)
                    save_checkpoint(output_dir / "model_ema.pth", model, ema, optimizer, config, step, epoch)
                    shutil.copy2(output_dir / "model_ema.pth", root / "checkpoints" / "model_ema.pth")
                    return

    save_checkpoint(output_dir / "latest.pth", model, ema, optimizer, config, step, config.epochs)
    save_checkpoint(output_dir / "model_ema.pth", model, ema, optimizer, config, step, config.epochs)
    shutil.copy2(output_dir / "model_ema.pth", root / "checkpoints" / "model_ema.pth")


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a from-scratch conditional DDPM for HW6.")
    for field, value in asdict(defaults).items():
        if isinstance(value, bool):
            parser.add_argument(f"--{field}", action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(f"--{field}", type=type(value), default=value)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())

