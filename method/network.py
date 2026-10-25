"""Fixed 338,177-parameter symmetric Bellman travel-cost network."""
import math
import torch
from torch import nn
from torch.nn import functional as F

CONFIG = dict(dimension=6, harmonics=4, state_width=256, state_blocks=2,
              pair_width=64, pair_blocks=1)


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)
        nn.init.kaiming_uniform_(self.first.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, value):
        return value + self.second(F.silu(self.first(self.norm(value)))) / math.sqrt(2.0)


class JointFeatures(nn.Module):
    def forward(self, q):
        harmonic = torch.arange(1, 5, device=q.device, dtype=q.dtype)
        phase = (2.0 * math.pi * q[:, :, None] * harmonic).flatten(1)
        return torch.cat((q, torch.sin(phase), torch.cos(phase)), dim=1)


class TravelCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = JointFeatures()
        self.state_input = nn.Linear(54, 256)
        self.state_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(2)])
        self.state_norm = nn.LayerNorm(256)
        self.pair_input = nn.Linear(792, 64)
        self.pair_blocks = nn.ModuleList([ResidualBlock(64)])
        self.pair_norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, math.log(math.expm1(0.05)))

    def encode(self, q):
        h = F.silu(self.state_input(self.features(q)))
        for block in self.state_blocks:
            h = block(h)
        return self.state_norm(h)

    def forward(self, q, goal):
        encoded = self.encode(torch.cat((q, goal), dim=0))
        a, b = encoded[:len(q)], encoded[len(q):]
        harmonic = torch.arange(1, 5, device=q.device, dtype=q.dtype)
        phase = (2.0 * math.pi * (q-goal)[:, :, None] * harmonic).flatten(1)
        pair = torch.cat((a+b, (a-b).square(), a*b, torch.cos(phase)), dim=1)
        h = F.silu(self.pair_input(pair))
        for block in self.pair_blocks:
            h = block(h)
        residual = self.output(self.pair_norm(h))
        distance = torch.linalg.vector_norm(q-goal, dim=1, keepdim=True)
        return distance * (1.0 + F.softplus(residual))


def load_checkpoint(path, device='cuda:0'):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload['model_config'] != CONFIG:
        raise ValueError('Checkpoint does not match the clean baseline architecture')
    net = TravelCost().to(device)
    net.load_state_dict(payload['model_state_dict'], strict=True)
    return net.eval().requires_grad_(False)
