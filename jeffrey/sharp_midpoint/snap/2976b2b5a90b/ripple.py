"""Travel time along the true paths (copy of jeffrey/sharp_features/ripple.py analyse(), for the frozen snapshot): local minima,
their depth and spacing, ripple spectrum, share of the true path running uphill toward the goal, eikonal residual |S |grad T| - 1|."""
import numpy as np
import torch
import plan as P

BANDS = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]


def analyse(maze, prepared, cost, grad=None, res=0.01):
    q = np.load(prepared / 'queries.npz')
    tp = np.load(prepared / 'truth_paths.npy', allow_pickle=True)
    depths, spacings, counts, power = [], [], [], np.zeros(len(BANDS))
    uphill_len, total_len, eik_res, eik_up = 0.0, 0.0, [], []
    for k in np.nonzero(q['bin'] >= 3)[0]:
        p = P.densify(torch.tensor(np.asarray(tp[k], np.float32), device=maze.device), res * maze.pitch)
        g = torch.tensor(q['goal'][k], device=maze.device, dtype=torch.float32).expand(len(p), 3)
        t = cost(torch.cat([p, g], 1)).cpu().numpy() / maze.pitch          # travel time in cells
        seg_c = np.linalg.norm(np.diff(p[:, :2].cpu().numpy(), axis=0), axis=1) * maze.E
        up = np.diff(t) > 0                                                  # T rises toward the goal on this step
        uphill_len += float(seg_c[up].sum())
        total_len += float(seg_c.sum())
        if grad is not None:
            _, gr = grad(torch.cat([p, g], 1))
            S = maze.train_speed(maze.clearance(p))
            r = (S * torch.linalg.vector_norm(gr, dim=1)).cpu().numpy()      # eikonal: S |grad T| = 1
            eik_res += np.abs(r - 1).tolist()
            mid = np.concatenate([up, [False]])
            eik_up += np.abs(r[mid] - 1).tolist()
        seg = np.linalg.norm(np.diff(p[:, :2].cpu().numpy(), axis=0), axis=1) * maze.E
        s = np.concatenate([[0], np.cumsum(seg)])                            # path length in cells
        dt = np.diff(t)
        sgn = np.sign(dt)
        mins = np.nonzero((sgn[:-1] < 0) & (sgn[1:] > 0))[0] + 1
        counts.append(len(mins))
        for m in mins:
            j = m
            while j + 1 < len(t) and t[j + 1] >= t[j]:
                j += 1
            depths.append(float(t[j] - t[m]))
        if len(mins) > 1:
            spacings += np.diff(s[mins]).tolist()
        # ripple spectrum: resample on a uniform 0.01-cell grid, subtract a 1-cell moving average
        if s[-1] > 4:
            u = np.arange(0, s[-1], res)
            tu = np.interp(u, s, t)
            w = int(round(1.0 / res))
            trend = np.convolve(tu, np.ones(w) / w, mode='same')
            r = (tu - trend)[w:-w]
            if len(r) > 16:
                F = np.abs(np.fft.rfft(r * np.hanning(len(r)))) ** 2
                lam = 1.0 / np.maximum(np.fft.rfftfreq(len(r), d=res), 1e-12)
                for b, (lo, hi) in enumerate(BANDS):
                    power[b] += F[(lam >= lo) & (lam < hi)].sum()
    power = power / power.sum() if power.sum() else power
    extra = dict(uphill_share=uphill_len / total_len if total_len else None)
    if eik_res:
        extra.update(eikonal_residual_median=float(np.median(eik_res)), eikonal_residual_uphill_median=float(np.median(eik_up)) if eik_up else None)
    return dict(**extra, paths=len(counts), minima_per_path=float(np.mean(counts)), depth_median=float(np.median(depths)) if depths else None,
                depth_q90=float(np.quantile(depths, 0.9)) if depths else None,
                spacing_median=float(np.median(spacings)) if spacings else None,
                spacing_quartiles=[float(x) for x in np.quantile(spacings, [0.25, 0.5, 0.75])] if spacings else None,
                ripple_power_by_wavelength={f'{lo:g}-{hi:g}': float(v) for (lo, hi), v in zip(BANDS, power)})
