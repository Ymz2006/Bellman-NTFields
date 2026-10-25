"""Frozen training-only coefficient; never needed by deployed TravelCost."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fk import PositionFK
from common import RELEASE


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, value):
        return value + self.second(F.silu(self.first(self.norm(value))))/math.sqrt(2.0)


class SpeedFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(80999)
        matrix = torch.randn(128, 6, generator=generator)
        matrix = matrix / torch.linalg.vector_norm(matrix, dim=1, keepdim=True).clamp_min(1e-8)
        scale = torch.randint(1, 9, (128, 1), generator=generator).float()
        self.register_buffer('random_matrix', matrix*scale)
        self.register_buffer('periods', torch.tensor((.5, .5, 1., .5, .5, .5)))
        self.register_buffer('harmonics', torch.arange(1, 5).float())

    def forward(self, q):
        phase = 2.0*math.pi*F.linear(q, self.random_matrix)
        normalized = q/self.periods.to(dtype=q.dtype)
        harmonic = 2.0*math.pi*normalized[:, :, None]*self.harmonics.to(dtype=q.dtype)[None, None, :]
        return torch.cat((q, torch.sin(phase), torch.cos(phase),
                          torch.sin(harmonic).flatten(1), torch.cos(harmonic).flatten(1)), dim=1)


class SpeedTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(310, 512)
        self.blocks = nn.ModuleList([ResidualBlock(512) for _ in range(6)])
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        x = F.silu(self.input(x))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class SpeedField(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = SpeedFeatures()
        self.tower = SpeedTower()
        self.head = nn.Linear(512, 2)
        self.position_fk = PositionFK()
        norm = np.load(RELEASE/'configs/joint_normalization.npz')
        self.register_buffer('fk_center', torch.tensor(norm['q_center']))
        self.register_buffer('fk_range', torch.tensor(norm['q_range']))

    def forward(self, q):
        components = self.head(self.tower(self.features(q))).clamp(-2.302585093, 0).exp()
        xyz = self.position_fk(q*self.fk_range+self.fk_center)
        u = ((xyz[:, 1]+.1)/.2).clamp(0, 1)
        w = u.square()*(3-2*u)
        return w*components[:, 0]+(1-w)*components[:, 1]


def load_speed(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    net = SpeedField().to(device)
    net.load_state_dict(payload['model_state_dict'], strict=True)
    return net.eval().requires_grad_(False)
