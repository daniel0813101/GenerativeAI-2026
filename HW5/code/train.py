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

from utils import (
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
    """Hyperparameters and paths used by the HW5 training script.

    Attributes:
        image_dir: Directory containing 256x256 RGB training PNG images.
        latent_cache: Path where encoded VAE latents are stored.
        output_dir: Directory used for checkpoints and training metadata.
        rebuild_cache: Whether to overwrite an existing latent cache.
        augment_cache: Whether to randomly horizontally augment one cache pass.
        cache_flip_pairs: Whether to cache both original and horizontally flipped latents.
        cache_mild_augments: Whether to cache original, color-jittered, and affine latents.
        unet_channels: Comma-separated U-Net channel widths for four resolution stages.
        init_unet_checkpoint: Optional checkpoint directory used to initialize the U-Net.
        prediction_type: Diffusion target type: "epsilon" or "v_prediction".
        epochs: Number of epochs to train when max_steps is unset.
        max_steps: Explicit optimizer update count. Uses epochs when set to 0.
        batch_size: Latent training batch size.
        encode_batch_size: Image batch size used for VAE latent caching.
        grad_accum_steps: Number of backward passes per optimizer update.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps before cosine LR decay.
        ema_decay: Exponential moving average decay for checkpoint export.
        max_grad_norm: Gradient clipping norm. Disabled when <= 0.
        save_every: Optimizer update interval for checkpoint saves.
        num_workers: DataLoader worker count.
        seed: Random seed for Python, NumPy, and PyTorch.
        mixed_precision: Whether to use CUDA AMP during training.
        device: Torch device string, usually "cuda" or "cpu".
    """

    image_dir: str = "HW5/public_data/images"
    latent_cache: str = "HW5/model/cache/latents.pt"
    output_dir: str = "HW5/model/baseline_latent_ddpm"
    rebuild_cache: bool = False
    augment_cache: bool = False
    cache_flip_pairs: bool = False
    cache_mild_augments: bool = False
    unet_channels: str = "128,256,512,512"
    init_unet_checkpoint: str = ""
    prediction_type: str = "epsilon"
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
    """Loads and normalizes professor face images for VAE encoding."""

    def __init__(self, image_dir: Path, image_size: int = IMAGE_SIZE, augment: bool = True):
        """Initializes the image dataset.

        Args:
            image_dir: Directory containing PNG training images.
            image_size: Output square image size expected by the VAE.
            augment: Whether to apply random horizontal flips.

        Raises:
            FileNotFoundError: If no PNG images are found in image_dir.
        """
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
        """Returns the number of available training images."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Loads one image and returns it normalized to [-1, 1].

        Args:
            index: Image index.

        Returns:
            A transformed RGB image tensor with shape (3, image_size, image_size).
        """
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


class MildAugmentImageDataset(Dataset):
    """Loads original plus mild augmented image variants for VAE caching."""

    def __init__(self, image_dir: Path, image_size: int = IMAGE_SIZE):
        """Initializes the mild augmentation image dataset.

        Args:
            image_dir: Directory containing PNG training images.
            image_size: Output square image size expected by the VAE.

        Raises:
            FileNotFoundError: If no PNG images are found in image_dir.
        """
        self.image_paths = sorted(p for p in image_dir.glob("*.png") if p.is_file())
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found under {image_dir}")

        self.original_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.color_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.affine_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.02, 0.02),
                    scale=(0.97, 1.03),
                    fill=128,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        """Returns the number of available training images."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Loads one image and returns original plus mild augmented variants.

        Args:
            index: Image index.

        Returns:
            A tensor with shape (3, 3, image_size, image_size), where the first
            dimension indexes original, color-jittered, and affine variants.
        """
        image = Image.open(self.image_paths[index]).convert("RGB")
        return torch.stack(
            [
                self.original_transform(image),
                self.color_transform(image),
                self.affine_transform(image),
            ],
            dim=0,
        )


class LatentDataset(Dataset):
    """Loads cached VAE latents for diffusion training."""

    def __init__(self, latent_path: Path):
        """Initializes the latent dataset.

        Args:
            latent_path: Path to a torch file containing a "latents" tensor.
        """
        payload = torch.load(latent_path, map_location="cpu")
        self.latents = payload["latents"].float()

    def __len__(self) -> int:
        """Returns the number of cached latent samples."""
        return self.latents.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Returns one cached latent tensor.

        Args:
            index: Latent index.

        Returns:
            A latent tensor with shape (4, 32, 32).
        """
        return self.latents[index]


def parse_unet_channels(value: str) -> tuple[int, ...]:
    """Parses a comma-separated U-Net channel specification.

    Args:
        value: Comma-separated channel widths, such as "128,256,512,768".

    Returns:
        A tuple of four positive channel widths.

    Raises:
        ValueError: If value does not contain exactly four positive integers.
    """
    try:
        channels = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("--unet_channels must be comma-separated integers") from exc

    if len(channels) != 4 or any(channel <= 0 for channel in channels):
        raise ValueError("--unet_channels must contain exactly four positive integers")
    return channels


@torch.no_grad()
def cache_latents(config: TrainConfig, device: torch.device) -> None:
    """Encodes training images into VAE latent space and saves a cache.

    Args:
        config: Training configuration with image and cache paths.
        device: Device used to run the pretrained VAE encoder.
    """
    latent_path = Path(config.latent_cache)
    if latent_path.exists() and not config.rebuild_cache:
        payload = torch.load(latent_path, map_location="cpu")
        cached_flip_pairs = bool(payload.get("cache_flip_pairs", False))
        cached_augmented = bool(payload.get("augmented", False))
        cached_mild_augments = bool(payload.get("cache_mild_augments", False))
        if (
            cached_flip_pairs != config.cache_flip_pairs
            or cached_augmented != config.augment_cache
            or cached_mild_augments != config.cache_mild_augments
        ):
            raise ValueError(
                "Existing latent cache was built with different cache options. "
                "Use --rebuild_cache or choose a different --latent_cache path."
            )
        print(f"Using existing latent cache: {latent_path}")
        return

    enabled_cache_modes = sum(
        [
            config.augment_cache,
            config.cache_flip_pairs,
            config.cache_mild_augments,
        ]
    )
    if enabled_cache_modes > 1:
        raise ValueError(
            "Use only one of --augment_cache, --cache_flip_pairs, or --cache_mild_augments"
        )

    if config.cache_mild_augments:
        dataset = MildAugmentImageDataset(Path(config.image_dir))
    else:
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
        if pixel_values.ndim == 5:
            batch_size, variants, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.view(batch_size * variants, channels, height, width)

        latent = vae.encode(pixel_values).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
        latents.append(latent.cpu())

        if config.cache_flip_pairs:
            flipped_values = torch.flip(pixel_values, dims=[-1])
            flipped_latent = vae.encode(flipped_values).latent_dist.sample()
            flipped_latent = flipped_latent * vae.config.scaling_factor
            latents.append(flipped_latent.cpu())

    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.cat(latents, dim=0),
            "vae": VAE_MODEL_ID,
            "image_dir": str(config.image_dir),
            "augmented": config.augment_cache,
            "cache_flip_pairs": config.cache_flip_pairs,
            "cache_mild_augments": config.cache_mild_augments,
        },
        latent_path,
    )
    print(f"Saved latent cache to {latent_path}")


def build_lr_lambda(warmup_steps: int, total_steps: int):
    """Builds a warmup plus cosine decay learning-rate schedule.

    Args:
        warmup_steps: Number of linear warmup optimizer updates.
        total_steps: Total number of optimizer updates.

    Returns:
        A LambdaLR-compatible function mapping step index to LR scale.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


def save_checkpoint(unet, ema, output_dir: Path, step: int, config: TrainConfig) -> None:
    """Saves raw and EMA U-Net checkpoints for a training step.

    Args:
        unet: Current denoising U-Net.
        ema: EMA tracker containing smoothed U-Net parameters.
        output_dir: Parent directory where checkpoint folders are written.
        step: Current optimizer update step.
        config: Training configuration to record in checkpoint metadata.
    """
    step_dir = output_dir / f"unet_step_{step:07d}"
    ema_dir = output_dir / f"unet_ema_step_{step:07d}"
    unet.save_pretrained(step_dir)

    ema_unet = build_unet(parse_unet_channels(config.unet_channels)).to(next(unet.parameters()).device)
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
            "unet_channels": config.unet_channels,
            "init_unet_checkpoint": config.init_unet_checkpoint,
            "prediction_type": config.prediction_type,
        },
    )
    print(f"Saved checkpoints at step {step}")


def train(config: TrainConfig) -> None:
    """Trains the from-scratch latent DDPM denoising U-Net.

    Args:
        config: Training hyperparameters, paths, and runtime settings.
    """
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

    unet_channels = parse_unet_channels(config.unet_channels)
    if config.init_unet_checkpoint:
        unet = build_unet(unet_channels)
        loaded_unet = type(unet).from_pretrained(config.init_unet_checkpoint)
        unet.load_state_dict(loaded_unet.state_dict())
        del loaded_unet
        print(f"Initialized U-Net from {config.init_unet_checkpoint}")
    else:
        unet = build_unet(unet_channels)
    unet = unet.to(device)
    unet.train()
    ema = EMAModel(unet, decay=config.ema_decay)
    if config.prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError('--prediction_type must be "epsilon" or "v_prediction"')
    scheduler = build_train_scheduler(config.prediction_type)
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

            if config.prediction_type == "epsilon":
                target = noise
            else:
                target = scheduler.get_velocity(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=config.mixed_precision and device.type == "cuda"):
                pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(pred.float(), target.float()) / config.grad_accum_steps

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
    final_unet = build_unet(parse_unet_channels(config.unet_channels)).to(device)
    final_unet.load_state_dict(unet.state_dict())
    ema.copy_to(final_unet)
    final_unet.save_pretrained(final_dir)
    torch.save(ema.state_dict(), final_dir / "ema.pt")
    shutil.copyfile(output_dir / "train_config.json", final_dir / "train_config.json")
    print(f"Saved final EMA model to {final_dir}")


def parse_args() -> TrainConfig:
    """Parses CLI overrides into a TrainConfig instance.

    Returns:
        TrainConfig populated from dataclass defaults and command-line flags.
    """
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train legal HW5 latent-DDPM baseline from scratch.")
    parser.add_argument("--image_dir", type=str, default=defaults.image_dir)
    parser.add_argument("--latent_cache", type=str, default=defaults.latent_cache)
    parser.add_argument("--output_dir", type=str, default=defaults.output_dir)
    parser.add_argument("--rebuild_cache", action="store_true", default=defaults.rebuild_cache)
    parser.add_argument("--augment_cache", action="store_true", default=defaults.augment_cache, help="Cache one randomly horizontally-augmented pass. Usually keep off.")
    parser.add_argument("--cache_flip_pairs", action="store_true", default=defaults.cache_flip_pairs, help="Cache both original and horizontally flipped latents for each image.")
    parser.add_argument("--cache_mild_augments", action="store_true", default=defaults.cache_mild_augments, help="Cache original, mild color-jittered, and mild affine latents for each image.")
    parser.add_argument("--unet_channels", type=str, default=defaults.unet_channels, help="Comma-separated U-Net channel widths, e.g. 128,256,512,768.")
    parser.add_argument("--init_unet_checkpoint", type=str, default=defaults.init_unet_checkpoint, help="Optional U-Net checkpoint directory to initialize from before training.")
    parser.add_argument("--prediction_type", choices=["epsilon", "v_prediction"], default=defaults.prediction_type, help="Diffusion prediction target used for training and sampling.")
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
