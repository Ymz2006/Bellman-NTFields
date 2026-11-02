"""A zoomed ntrl-demo field on the original coordinates (zoom.py control).

A network trained on coordinates x' = s x computes T_net(x'); the NN with B' = s B and the same weights computes
T_net(s x) on the original coordinates exactly (its first operation is x @ 2 pi B), and Scaled divides the output by
s, so the field is in the original time units and |grad_x| matches the speed label.
"""
import torch
import plan as P


class Scaled(torch.nn.Module):
    """net.out with the output multiplied by `factor` (same leaf, so P.field_grad gives the scaled gradient)."""

    def __init__(self, net, factor):
        super().__init__()
        self.net = net
        self.factor = float(factor)

    def out(self, coords):
        t, w, xp = self.net.out(coords)
        return t * self.factor, w, xp

    @property
    def B(self):
        return self.net.B


def zoomed_network(B, state_dict, zoom, device='cuda:0'):
    net = P.network_class('metric')(device, 3, B * zoom)
    net.load_state_dict(state_dict, strict=True)
    net.to(torch.device(device)).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return Scaled(net, 1.0 / zoom)


def load_zoomed(checkpoint, zoom, device='cuda:0'):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    return zoomed_network(payload['B_state_dict'], payload['model_state_dict'], zoom, device)
