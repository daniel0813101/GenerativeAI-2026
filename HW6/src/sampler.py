import torch

from .diffusion import GaussianDiffusion
from .model_unet import ConditionalUNet


@torch.no_grad()
def predict_with_guidance(
    model: ConditionalUNet,
    x: torch.Tensor,
    t: torch.Tensor,
    animal_id: torch.Tensor,
    object_id: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    if guidance_scale == 1.0:
        return model(x, t, animal_id, object_id)
    null_animal = torch.full_like(animal_id, model.null_animal_id)
    null_object = torch.full_like(object_id, model.null_object_id)
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    animal_in = torch.cat([null_animal, animal_id], dim=0)
    object_in = torch.cat([null_object, object_id], dim=0)
    uncond, cond = model(x_in, t_in, animal_in, object_in).chunk(2, dim=0)
    return uncond + guidance_scale * (cond - uncond)


@torch.no_grad()
def sample_ddim(
    model: ConditionalUNet,
    diffusion: GaussianDiffusion,
    animal_id: torch.Tensor,
    object_id: torch.Tensor,
    image_size: int = 64,
    steps: int = 100,
    guidance_scale: float = 2.5,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    batch = animal_id.shape[0]
    x = torch.randn(batch, 3, image_size, image_size, device=device, generator=generator)
    times = torch.linspace(diffusion.config.timesteps - 1, 0, steps + 1, device=device).long()
    for i in range(steps):
        t = times[i].repeat(batch)
        t_prev = times[i + 1].repeat(batch) if i + 1 < len(times) else torch.full_like(t, -1)
        eps = predict_with_guidance(model, x, t, animal_id, object_id, guidance_scale)
        x = diffusion.ddim_step(x, t, t_prev, eps, eta=eta)
    return x.clamp(-1.0, 1.0)

