"""P-NTFields training loop (Model.train in model_res_sigmoid_multi.py) for one environment.

Kept: AdamW(lr 1e-3, weight decay 0.1), 6 inner batches of 10000 pairs per epoch, speed annealing
s_alpha = alpha s + 1 - alpha with alpha rising 0.5 -> 1.07 over the schedule, lr 1e-3 -> 5e-4 after 80 % of the
schedule, loss scaled by beta = 1 / previous residual, viscosity gamma = 1e-3, and the rollback rule (if the epoch
residual rises by more than 20 % the network is reset to one of the five previous epoch states and the epoch repeats).
`epochs` may be shorter than the original 10000; the schedule is compressed proportionally.
"""
import copy
import random
import time
import numpy as np
import torch
from model import eikonal_loss


def train(net, points, speed, epochs, log, batch=10000, inner=6, gamma=1e-3, lr=1e-3, weight_decay=0.1, seed=0,
          schedule_epochs=10000, laplacian=True, log_every=100, label='', mode='jet', fd_h=2.0 / 128):
    """mode: 'jet' (forward-mode derivatives, one pass), 'fd' (jet gradient, finite-difference Laplacian at fd_h) or
    'autograd' (double backward)."""
    dev = points.device
    trainable = lambda: {k: v.detach().clone() for k, v in net.state_dict().items() if not k.startswith('features.')}  # noqa: E731
    g = torch.Generator(device=dev).manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = points.shape[0]
    scale = schedule_epochs / epochs
    step = -0.5
    beta, prev_diff = 1.0, 1.0
    queue = []
    history, repeats = [], 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        e = epoch * scale
        alpha = min(max(0.5, 0.5 + 0.5 * step), 1.07)
        step += scale / 4000.0 / (int(e / 4000) + 1.0)
        for grp in opt.param_groups:
            grp['lr'] = float(np.clip(lr * (1 - (e - 0.8 * schedule_epochs) / (0.1 * schedule_epochs)), lr * 0.5, lr))
        queue.append((trainable(), copy.deepcopy(opt.state_dict())))  # frozen feature grids are not copied
        if len(queue) > 5:
            queue.pop(0)
        tries = 0
        while True:
            perm = torch.randperm(n, generator=g, device=dev)
            total = 0.0
            for ii in range(inner):
                idx = perm[ii * batch:(ii + 1) * batch]
                s = alpha * speed[idx] + 1 - alpha
                diff = eikonal_loss(net, points[idx], s, gamma, laplacian, mode, fd_h)
                loss = beta * diff
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(diff)
            total /= inner
            ratio = total / prev_diff
            if 0 < ratio < 1.2 or tries >= 10:
                break
            tries += 1
            repeats += 1
            sd, od = queue[rng.randint(0, len(queue) - 1)]
            net.load_state_dict(sd, strict=False)
            opt.load_state_dict(od)
        prev_diff = total
        beta = 1.0 / total
        if epoch % log_every == 0 or epoch == epochs or epoch == 1:
            history.append(dict(epoch=epoch, residual=total, alpha=alpha, repeats=repeats))
            log(f'  {label} epoch {epoch}/{epochs} residual {total:.4e} alpha {alpha:.3f} lr {opt.param_groups[0]["lr"]:.1e} '
                f'repeats {repeats} ({time.time() - t0:.0f}s)')
    return history
