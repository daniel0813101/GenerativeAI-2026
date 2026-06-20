import argparse
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.train import build_model
from src.dataset import read_conditions
from src.diffusion import DiffusionConfig, GaussianDiffusion
from src.sampler import sample_ddim
from src.utils import ensure_dir, seed_everything, to_uint8_image


@dataclass
class GenerateConfig:
    checkpoint: str = "checkpoints/model_ema.pth"
    conditions: str = "data/generate.csv"
    output_dir: str = "scoring_program/input/res"
    batch_size: int = 64
    num_steps: int = 100
    guidance_scale: float = 2.5
    ddim_eta: float = 0.0
    seed: int = 1234
    clean_output: bool = True
    use_ema: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config: GenerateConfig, device: torch.device):
    root = Path(__file__).resolve().parents[1]
    ckpt = torch.load(root / config.checkpoint, map_location=device)
    train_config = ckpt["config"]
    from scripts.train import TrainConfig

    model_config = TrainConfig(**train_config)
    model = build_model(model_config).to(device)
    if config.use_ema and "ema" in ckpt:
        from src.utils import EMAModel

        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            timesteps=model_config.timesteps,
            beta_schedule=model_config.beta_schedule,
            prediction_type=model_config.prediction_type,
        ),
        device=device,
    )
    return model, diffusion, model_config.image_size


@torch.no_grad()
def generate(config: GenerateConfig) -> None:
    seed_everything(config.seed)
    root = Path(__file__).resolve().parents[1]
    output_dir = ensure_dir(root / config.output_dir)
    if config.clean_output:
        for path in output_dir.glob("*.png"):
            path.unlink()
    conditions = read_conditions(root / config.conditions)
    device = torch.device(config.device)
    model, diffusion, image_size = load_model(config, device)
    generator = torch.Generator(device=device).manual_seed(config.seed)

    for start in tqdm(range(0, len(conditions), config.batch_size), desc="Generating"):
        batch = conditions[start : start + config.batch_size]
        animal = torch.tensor([item.animal_id for item in batch], device=device)
        obj = torch.tensor([item.object_id for item in batch], device=device)
        images = sample_ddim(
            model,
            diffusion,
            animal,
            obj,
            image_size=image_size,
            steps=config.num_steps,
            guidance_scale=config.guidance_scale,
            eta=config.ddim_eta,
            generator=generator,
        )
        for image, cond in zip(images, batch, strict=True):
            to_uint8_image(image).save(output_dir / cond.image_name)


def copy_for_submission(student_id: str, source_dir: str = "scoring_program/input/res") -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "submission" / f"HW6_{student_id}" / "generated_images"
    dest.mkdir(parents=True, exist_ok=True)
    for path in dest.glob("*.png"):
        path.unlink()
    for path in sorted((root / source_dir).glob("*.png")):
        shutil.copy2(path, dest / path.name)


def parse_args() -> GenerateConfig:
    defaults = GenerateConfig()
    parser = argparse.ArgumentParser(description="Generate HW6 conditional images.")
    for field, value in asdict(defaults).items():
        if isinstance(value, bool):
            parser.add_argument(f"--{field}", action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(f"--{field}", type=type(value), default=value)
    return GenerateConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    generate(parse_args())

