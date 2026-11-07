"""Batched evaluation of an ntrl-demo models/metric NN with the encoder split from the metric.

NN.out(pairs) encodes x0 and x1 with the same encoder (Fourier features, two gated sine blocks, last linear layer,
per-row InstanceNorm) and returns T = 0.2 sum_groups smoothmax_16 sqrt((f0 - f1)^2 + 1e-6). encode() and metric()
are that computation split in two, so a goal's features can be computed once and reused for every rollout state of an
MPPI step (half the work of NN.out on (state, goal) pairs). Frozen() precomputes the Lipschitz-normalized weights for
no-grad use (MPPI, targets); encode(net, x) differentiates through the network's own parameters (training).
check() compares both against NN.out.
"""
import math
import torch

LOG16 = math.log(16.0)


def _layers(net):
    """(weight, bias) of every linear layer NN.out uses, weights Lipschitz-normalized as NN.lip_norm does."""
    lin = [net.pe_gate[0], net.pe_gate[1]] + [net.encoder[k] for k in range(1, 3 * net.nl1 + 1)] + [net.encoder[-1]]
    return [(net.lip_norm(m.weight), m.bias) for m in lin]


def _forward(B2pi, layers, gates, nl1, x):
    proj = x @ B2pi
    x = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    (w0, b0), (w1, b1) = layers[0], layers[1]
    u = torch.sin(torch.addmm(b0, x, w0.T))
    v = torch.sin(torch.addmm(b1, x, w1.T))
    k = 2
    for ii in range(nl1):
        x_tmp = x
        w, b = layers[k]
        s = torch.sin(torch.addmm(b, x, w.T))
        x = u * s + v * (1 - s)
        w, b = layers[k + 1]
        s = torch.sin(torch.addmm(b, x, w.T))
        x = u * s + v * (1 - s)
        w, b = layers[k + 2]
        y = torch.addmm(b, x, w.T)
        g = gates[ii]
        x = (1 - g) * x_tmp + g * torch.sin(y)
        k += 3
    w, b = layers[k]
    y = torch.addmm(b, x, w.T)
    mean = y.mean(1, keepdim=True)
    var = y.var(1, unbiased=False, keepdim=True)
    return (y - mean) / torch.sqrt(var + 1e-5)


def encode(net, x):
    """Features (N, 256) of configurations x (N, dim), differentiable in the network's parameters (and x)."""
    gates = [torch.sigmoid(0.1 * net.gate[ii].weight) for ii in range(net.nl1)]
    return _forward(2.0 * math.pi * net.B, _layers(net), gates, net.nl1, x)


def metric(f0, f1):
    """T for feature rows f0, f1 (N, 256) -> (N,)."""
    d = torch.sqrt((f0 - f1) ** 2 + 1e-6).view(f0.shape[0], -1, 16)
    return 0.2 * ((torch.logsumexp(10 * d, 2) - LOG16) / 10).sum(1)


def travel(net, pairs):
    """T(x0, x1) for pairs (N, 2 dim), differentiable (training)."""
    dim = pairs.shape[1] // 2
    f = encode(net, torch.cat([pairs[:, :dim], pairs[:, dim:]]))
    return metric(f[:len(pairs)], f[len(pairs):])


class Frozen:
    """No-grad evaluator with the normalized weights computed once (valid until the network's parameters change)."""

    def __init__(self, net, chunk=1_000_000):
        with torch.no_grad():
            self.B2pi = (2.0 * math.pi * net.B).detach().clone()
            self.layers = [(w.detach().clone(), b.detach().clone()) for w, b in _layers(net)]
            self.gates = [torch.sigmoid(0.1 * net.gate[ii].weight).detach().clone() for ii in range(net.nl1)]
        self.nl1, self.chunk = net.nl1, int(chunk)

    @torch.no_grad()
    def encode(self, x):
        if len(x) <= self.chunk:
            return _forward(self.B2pi, self.layers, self.gates, self.nl1, x)
        return torch.cat([_forward(self.B2pi, self.layers, self.gates, self.nl1, x[i:i + self.chunk]) for i in range(0, len(x), self.chunk)])

    @torch.no_grad()
    def travel(self, pairs):
        dim = pairs.shape[1] // 2
        out = []
        step = max(1, self.chunk // 2)
        for i in range(0, len(pairs), step):
            p = pairs[i:i + step]
            f = _forward(self.B2pi, self.layers, self.gates, self.nl1, torch.cat([p[:, :dim], p[:, dim:]]))
            out.append(metric(f[:len(p)], f[len(p):]))
        return torch.cat(out) if out else pairs.new_zeros(0)

    @torch.no_grad()
    def to_goal(self, states, goal_feat, index=None):
        """T(states, goal) with goal features precomputed; index maps each state row to its goal_feat row."""
        out = []
        for i in range(0, len(states), self.chunk):
            f = _forward(self.B2pi, self.layers, self.gates, self.nl1, states[i:i + self.chunk])
            g = goal_feat if index is None else goal_feat[index[i:i + self.chunk]]
            out.append(metric(f, g))
        return torch.cat(out) if out else states.new_zeros(0)


def check(net, n=4096, device='cuda:0', seed=0):
    """Max |T - NN.out| for random pairs (Frozen and differentiable paths)."""
    g = torch.Generator(device).manual_seed(seed)
    pairs = torch.rand(n, 6, generator=g, device=device) - 0.5
    with torch.no_grad():
        ref = net.out(pairs)[0][:, 0]
        a = Frozen(net).travel(pairs)
        b = travel(net, pairs)
    return dict(frozen=float((a - ref).abs().max()), grad_path=float((b - ref).abs().max()), ref_mean=float(ref.mean()))
