import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import ANIMALS, OBJECTS


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(1, half - 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_ch, out_ch * 2)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        image_size: int = 64,
    ):
        super().__init__()
        self.image_size = image_size
        self.time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.animal_embed = nn.Embedding(len(ANIMALS) + 1, self.time_dim)
        self.object_embed = nn.Embedding(len(OBJECTS) + 1, self.time_dim)
        self.null_animal_id = len(ANIMALS)
        self.null_object_id = len(OBJECTS)

        self.input = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        skips = [base_channels]
        ch = base_channels
        resolution = image_size
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(ch, out_ch, self.time_dim, dropout))
                ch = out_ch
                if resolution in attention_resolutions:
                    self.downs.append(AttentionBlock(ch))
                skips.append(ch)
            if level != len(channel_mults) - 1:
                self.downs.append(Downsample(ch))
                resolution //= 2
                skips.append(ch)

        self.mid1 = ResBlock(ch, ch, self.time_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResBlock(ch, ch, self.time_dim, dropout)

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(ch + skips.pop(), out_ch, self.time_dim, dropout))
                ch = out_ch
                if resolution in attention_resolutions:
                    self.ups.append(AttentionBlock(ch))
            if level != 0:
                self.ups.append(Upsample(ch))
                resolution *= 2

        self.out = nn.Sequential(
            group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, image_channels, 3, padding=1),
        )

    def condition_embedding(self, t: torch.Tensor, animal_id: torch.Tensor, object_id: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features))
        return emb + self.animal_embed(animal_id) + self.object_embed(object_id)

    def forward(self, x: torch.Tensor, t: torch.Tensor, animal_id: torch.Tensor, object_id: torch.Tensor) -> torch.Tensor:
        emb = self.condition_embedding(t, animal_id, object_id)
        h = self.input(x)
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
                skips.append(h)
            elif isinstance(layer, AttentionBlock):
                h = layer(h)
            else:
                h = layer(h)
                skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, emb)
            elif isinstance(layer, AttentionBlock):
                h = layer(h)
            else:
                h = layer(h)
        return self.out(h)
