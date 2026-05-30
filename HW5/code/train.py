import argparse
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common import (
    IMAGE_SIZE,
    VAE_MODEL_ID,
    EMAModel,
    build_train_scheduler,
    build_unet,
    save_json,
    seed_everything,
)


@dataclass
class TrainConfig:
    image_dir: str = "HW5/public_data/images"
    latent_cache: str = "HW5/model/cache/latents.pt"
    output_dir: str = "HW5/model/baseline_latent_ddpm"
    rebuild_cache: bool = False
    augment_cache: bool = False
    epochs: int = 350
    max_steps: int = 0
    batch_size: int = 64
    encode_batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    ema_decay: float = 0.9999
    max_grad_norm: float = 1.0
    save_every: int = 5000
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = IMAGE_SIZE, augment: bool = True):
        self.image_paths = sorted(p for p in image_dir.glob("*.png") if p.is_file())
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found under {image_dir}")

        aug = [transforms.RandomHorizontalFlip(p=0.5)] if augment else []
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                *aug,
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


class LatentDataset(Dataset):
    def __init__(self, latent_path: Path):
        payload = torch.load(latent_path, map_location="cpu")
        self.latents = payload["latents"].float()

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.latents[index]


@torch.no_grad()
def cache_latents(config: TrainConfig, device: torch.device) -> None:
    latent_path = Path(config.latent_cache)
    if latent_path.exists() and not config.rebuild_cache:
        print(f"Using existing latent cache: {latent_path}")
        return

    dataset = ImageDataset(Path(config.image_dir), augment=config.augment_cache)
    dataloader = DataLoader(
        dataset,
        batch_size=config.encode_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device)
    vae.requires_grad_(False)
    vae.eval()

    latents = []
    for batch in tqdm(dataloader, desc="Caching VAE latents"):
        pixel_values = batch.to(device, non_blocking=True)
        latent = vae.encode(pixel_values).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
        latents.append(latent.cpu())

    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.cat(latents, dim=0),
            "vae": VAE_MODEL_ID,
            "image_dir": str(config.image_dir),
            "augmented": config.augment_cache,
        },
        latent_path,
    )
    print(f"Saved latent cache to {latent_path}")


def build_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


def save_checkpoint(unet, ema, output_dir: Path, step: int, config: TrainConfig) -> None:
    step_dir = output_dir / f"unet_step_{step:07d}"
    ema_dir = output_dir / f"unet_ema_step_{step:07d}"
    unet.save_pretrained(step_dir)

    ema_unet = build_unet().to(next(unet.parameters()).device)
    ema_unet.load_state_dict(unet.state_dict())
    ema.copy_to(ema_unet)
    ema_unet.save_pretrained(ema_dir)
    torch.save(ema.state_dict(), ema_dir / "ema.pt")

    save_json(
        output_dir / "last_checkpoint.json",
        {
            "step": step,
            "raw_checkpoint": str(step_dir),
            "ema_checkpoint": str(ema_dir),
            "latent_cache": config.latent_cache,
        },
    )
    print(f"Saved checkpoints at step {step}")


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    device = torch.device(config.device)

    cache_latents(config, device)
    dataset = LatentDataset(Path(config.latent_cache))
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    unet = build_unet().to(device)
    unet.train()
    ema = EMAModel(unet, decay=config.ema_decay)
    scheduler = build_train_scheduler()
    optimizer = AdamW(unet.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)

    updates_per_epoch = math.ceil(len(dataloader) / config.grad_accum_steps)
    total_steps = config.max_steps or config.epochs * updates_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(config.warmup_steps, total_steps)
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_config.json", asdict(config))

    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and device.type == "cuda")
    step = 0
    accum_step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps, desc="Training")
    while step < total_steps:
        for latents in dataloader:
            latents = latents.to(device, non_blocking=True)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=config.mixed_precision and device.type == "cuda"):
                pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(pred.float(), noise.float()) / config.grad_accum_steps

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % config.grad_accum_steps == 0:
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema.update(unet)

                step += 1
                accum_step = 0
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item() * config.grad_accum_steps:.4f}", lr=lr_scheduler.get_last_lr()[0])

                if step % config.save_every == 0 or step == total_steps:
                    save_checkpoint(unet, ema, output_dir, step, config)

                if step >= total_steps:
                    break

    final_dir = output_dir / "unet_ema_final"
    final_unet = build_unet().to(device)
    final_unet.load_state_dict(unet.state_dict())
    ema.copy_to(final_unet)
    final_unet.save_pretrained(final_dir)
    torch.save(ema.state_dict(), final_dir / "ema.pt")
    shutil.copyfile(output_dir / "train_config.json", final_dir / "train_config.json")
    print(f"Saved final EMA model to {final_dir}")


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train legal HW5 latent-DDPM baseline from scratch.")
    parser.add_argument("--image_dir", type=str, default=defaults.image_dir)
    parser.add_argument("--latent_cache", type=str, default=defaults.latent_cache)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--rebuild_cache", action="store_true", default=defaults.rebuild_cache)
    parser.add_argument("--augment_cache", action="store_true", default=defaults.augment_cache, help="Cache one horizontally-augmented pass. Usually keep off.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--max_steps", type=int, default=defaults.max_steps)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--encode_batch_size", type=int, default=defaults.encode_batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=defaults.grad_accum_steps)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup_steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--ema_decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--max_grad_norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--save_every", type=int, default=defaults.save_every)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--mixed_precision", action="store_true", default=defaults.mixed_precision)
    parser.add_argument("--device", type=str, default=defaults.device)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
