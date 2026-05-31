import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel


VAE_MODEL_ID = "stabilityai/sd-vae-ft-mse"
LATENT_CHANNELS = 4
LATENT_SIZE = 32
IMAGE_SIZE = 256


def seed_everything(seed: int) -> None:
    """Sets random seeds for reproducible training and sampling.

    Args:
        seed: Seed used for Python, NumPy, CPU torch, and CUDA torch RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_unet(block_out_channels: tuple[int, ...] = (128, 256, 512, 512)) -> UNet2DModel:
    """Builds the from-scratch latent denoising U-Net.

    Args:
        block_out_channels: Channel widths for the four U-Net resolution stages.

    Returns:
        A diffusers UNet2DModel configured for 4-channel 32x32 VAE latents.

    Raises:
        ValueError: If block_out_channels does not contain four widths.
    """
    if len(block_out_channels) != 4:
        raise ValueError("block_out_channels must contain exactly four channel widths")

    return UNet2DModel(
        sample_size=LATENT_SIZE,
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def build_train_scheduler() -> DDPMScheduler:
    """Builds the DDPM scheduler shared by training and inference samplers.

    Returns:
        A DDPMScheduler configured to train epsilon prediction on VAE latents.
    """
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type="epsilon",
    )


def save_json(path: Path, data: dict) -> None:
    """Writes a dictionary to a formatted JSON file.

    Args:
        path: Destination JSON path. Parent directories are created if needed.
        data: JSON-serializable dictionary to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


class EMAModel:
    """Tracks an exponential moving average of trainable model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        """Initializes EMA shadow parameters from a model.

        Args:
            model: Model whose trainable parameters should be tracked.
            decay: EMA decay factor. Higher values update the shadow weights slower.
        """
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Updates shadow parameters from the current model weights.

        Args:
            model: Model providing the latest trainable parameters.
        """
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        """Copies EMA shadow parameters into a model.

        Args:
            model: Destination model with matching parameter names and shapes.
        """
        model_state = model.state_dict()
        for name, value in self.shadow.items():
            if name in model_state:
                model_state[name].copy_(value)

    def state_dict(self) -> dict:
        """Returns EMA state for checkpoint serialization.

        Returns:
            A dictionary containing the EMA decay and shadow parameters.
        """
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        """Loads EMA state from a checkpoint dictionary.

        Args:
            state: Dictionary produced by state_dict().
        """
        self.decay = state["decay"]
        self.shadow = state["shadow"]
