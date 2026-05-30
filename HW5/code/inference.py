import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from torchvision import transforms
from tqdm import tqdm

from common import LATENT_CHANNELS, LATENT_SIZE, VAE_MODEL_ID, build_train_scheduler, seed_everything


def build_sampler(name: str, num_steps: int):
    base = build_train_scheduler()
    if name == "ddpm":
        scheduler = DDPMScheduler.from_config(base.config)
    elif name == "ddim":
        scheduler = DDIMScheduler.from_config(base.config)
    elif name == "dpm":
        scheduler = DPMSolverMultistepScheduler.from_config(base.config)
    else:
        raise ValueError(f"Unknown sampler: {name}")
    scheduler.set_timesteps(num_steps)
    return scheduler


@torch.no_grad()
def generate(args) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        for path in output_dir.glob("*.png"):
            path.unlink()

    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID).to(device)
    vae.requires_grad_(False)
    vae.eval()

    unet = UNet2DModel.from_pretrained(args.checkpoint_dir).to(device)
    unet.requires_grad_(False)
    unet.eval()

    scheduler = build_sampler(args.sampler, args.num_inference_steps)
    to_pil = transforms.ToPILImage()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    sample_idx = 0
    total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    for _ in tqdm(range(total_batches), desc="Generating"):
        current_batch = min(args.batch_size, args.num_samples - sample_idx)
        latents = torch.randn(
            current_batch,
            LATENT_CHANNELS,
            LATENT_SIZE,
            LATENT_SIZE,
            generator=generator,
            device=device,
        )

        for timestep in scheduler.timesteps:
            model_output = unet(latents, timestep).sample
            latents = scheduler.step(model_output, timestep, latents).prev_sample

        decoded_latents = latents / vae.config.scaling_factor
        images = vae.decode(decoded_latents, return_dict=False)[0]
        images = (images.clamp(-1, 1) + 1) / 2

        for image in images:
            path = output_dir / f"{sample_idx:04d}.png"
            to_pil(image.cpu()).save(path)
            sample_idx += 1

    print(f"Saved {sample_idx} PNG files to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 256x256 HW5 result PNGs.")
    parser.add_argument("--checkpoint_dir", type=str, default="HW5/model/baseline_latent_ddpm/unet_ema_final")
    parser.add_argument("--output_dir", type=str, default="HW5/scoring_program/input/res")
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sampler", choices=["ddpm", "ddim", "dpm"], default="ddim")
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--clean_output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
