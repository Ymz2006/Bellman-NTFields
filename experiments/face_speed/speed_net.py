"""Per-piece speed network: the release SpeedField trunk with one output per piece, combined by a minimum.

The release coefficient (method/speed.py SpeedField) maps q to two outputs in [0.1, 1] that the cabinet task
trains on the same clearance label. PieceSpeedField keeps the trunk (Fourier features, width-512 tower of six
residual blocks) and gives the head one output per piece (box faces or spheres plus the self-collision
piece); each output is trained on its own piece speed and the body speed is their minimum, which places
the kinks of the clearance speed exactly where the limiting piece changes instead of asking one smooth
output to fit them. `forward(q)` is the scalar speed used by the Bellman Eikonal term and the bank costs.
"""
import sys
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/'method', Path('/clean/method')):
    if p.is_dir() and str(p) not in sys.path: sys.path.insert(0, str(p))
from speed import SpeedFeatures, SpeedTower, SpeedField

LOG_FLOOR = -2.302585093   # log 0.1: the speed floor of the release label


class PieceSpeedField(nn.Module):
    def __init__(self, n_pieces, names=None):
        super().__init__()
        self.features = SpeedFeatures(); self.tower = SpeedTower(); self.head = nn.Linear(512, n_pieces)
        self.n_pieces = n_pieces; self.names = list(names) if names else [f'piece_{i}' for i in range(n_pieces)]

    def pieces(self, q):
        return self.head(self.tower(self.features(q))).clamp(LOG_FLOOR, 0).exp()

    def forward(self, q):
        return self.pieces(q).amin(1)


class SeparateSpeedField(nn.Module):
    """One complete speed network per piece (its own Fourier features, tower and one-output head), combined by a
    minimum. With cover='split' this is one model for environment collision and one for self-collision."""
    def __init__(self, n_pieces, names=None):
        super().__init__()
        self.nets = nn.ModuleList([PieceSpeedField(1, [n]) for n in (names or [f'piece_{i}' for i in range(n_pieces)])])
        self.n_pieces = n_pieces; self.names = list(names) if names else [f'piece_{i}' for i in range(n_pieces)]

    def pieces(self, q):
        return torch.cat([net.pieces(q) for net in self.nets], 1)

    def forward(self, q):
        return self.pieces(q).amin(1)


def components(net, q):
    """The trained outputs of any of the networks, (B, outputs)."""
    if hasattr(net, 'pieces'): return net.pieces(q)
    return net.head(net.tower(net.features(q))).clamp(LOG_FLOOR, 0).exp()


def build(contract):
    """The release SpeedField of a cabinet run predates the cover fields: no cover means release."""
    if contract.get('cover', 'release') == 'release': return SpeedField()
    if contract.get('speed_model', 'shared') == 'separate': return SeparateSpeedField(contract['n_pieces'], contract.get('pieces'))
    return PieceSpeedField(contract['n_pieces'], contract.get('pieces'))


def load_piece_speed(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    net = build(payload['training_contract']).to(device)
    net.load_state_dict(payload['model_state_dict'], strict=True)
    return net.eval().requires_grad_(False)
