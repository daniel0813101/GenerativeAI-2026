import argparse
import math
import shutil
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
def cache_latents(args, device: torch.device) -> None:
    latent_path = Path(args.latent_cache)
    if latent_path.exists() and not args.rebuild_cache:
        print(f"Using existing latent cache: {latent_path}")
        return

    dataset = ImageDataset(Path(args.image_dir), augment=args.augment_cache)
    dataloader = DataLoader(
        dataset,
        batch_size=args.encode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
            "image_dir": str(args.image_dir),
            "augmented": args.augment_cache,
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


def save_checkpoint(unet, ema, output_dir: Path, step: int, args) -> None:
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
            "latent_cache": args.latent_cache,
        },
    )
    print(f"Saved checkpoints at step {step}")


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)

    cache_latents(args, device)
    dataset = LatentDataset(Path(args.latent_cache))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    unet = build_unet().to(device)
    unet.train()
    ema = EMAModel(unet, decay=args.ema_decay)
    scheduler = build_train_scheduler()
    optimizer = AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    updates_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    total_steps = args.max_steps or args.epochs * updates_per_epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, total_steps)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_config.json", vars(args))

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")
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

            with torch.cuda.amp.autocast(enabled=args.mixed_precision and device.type == "cuda"):
                pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(pred.float(), noise.float()) / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % args.grad_accum_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema.update(unet)

                step += 1
                accum_step = 0
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item() * args.grad_accum_steps:.4f}", lr=lr_scheduler.get_last_lr()[0])

                if step % args.save_every == 0 or step == total_steps:
                    save_checkpoint(unet, ema, output_dir, step, args)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train legal HW5 latent-DDPM baseline from scratch.")
    parser.add_argument("--image_dir", type=str, default="HW5/public_data/images")
    parser.add_argument("--latent_cache", type=str, default="HW5/model/cache/latents.pt")
    parser.add_argument("--output_dir", type=str, default="HW5/model/baseline_latent_ddpm")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--augment_cache", action="store_true", help="Cache one horizontally-augmented pass. Usually keep off.")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
