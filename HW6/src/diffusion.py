from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    prediction_type: str = "epsilon"


class GaussianDiffusion:
    def __init__(self, config: DiffusionConfig, device: torch.device | str):
        self.config = config
        self.device = torch.device(device)
        betas = self._build_betas().to(self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def _build_betas(self) -> torch.Tensor:
        if self.config.beta_schedule == "linear":
            return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps)
        if self.config.beta_schedule != "cosine":
            raise ValueError(f"Unsupported beta schedule: {self.config.beta_schedule}")
        steps = self.config.timesteps + 1
        x = torch.linspace(0, self.config.timesteps, steps)
        s = 0.008
        alphas_cumprod = torch.cos(((x / self.config.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def training_target(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.config.prediction_type == "epsilon":
            return noise
        if self.config.prediction_type == "v":
            return _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        raise ValueError(f"Unsupported prediction type: {self.config.prediction_type}")

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        if self.config.prediction_type == "epsilon":
            return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * model_output
        if self.config.prediction_type == "v":
            return _extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * model_output
        raise ValueError(f"Unsupported prediction type: {self.config.prediction_type}")

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor, clip_denoised: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.predict_x0(x_t, t, model_output)
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * x0 + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        log_var = _extract(self.posterior_log_variance, t, x_t.shape)
        return mean, log_var

    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, model_output: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        x0 = self.predict_x0(x_t, t, model_output).clamp(-1.0, 1.0)
        alpha = _extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = _extract(self.alphas_cumprod, t_prev.clamp(min=0), x_t.shape)
        alpha_prev = torch.where((t_prev < 0).view(-1, 1, 1, 1), torch.ones_like(alpha_prev), alpha_prev)
        eps = (x_t - torch.sqrt(alpha) * x0) / torch.sqrt(1.0 - alpha)
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).clamp(min=0)
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        direction = torch.sqrt((1.0 - alpha_prev - sigma ** 2).clamp(min=0)) * eps
        return torch.sqrt(alpha_prev) * x0 + direction + sigma * noise


def _extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = values.gather(0, t.to(values.device))
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

