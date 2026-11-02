"""Plan the prepared queries of a maze with a learned travel-time field and diagnose the field against the truth.

evaluate(maze, prepared, cost, grad, out, log): ntrl-demo's MPPI (plan.mppi with plan.settings(maze), per-query budget
from the truth path) for planning seeds 0/1/2, exact scoring, success by distance bin; field diagnostics per query:
T(start, goal) / T*, Spearman rank correlation of T with T* (all queries and per bin), cosine between the field's
descent direction at the start (-dT/dx0) and the truth's, xy-length and travel time of successful paths over the
truth's. Paths of planning seed 0 are saved (paths_seed0.npy); per-query rows go to rows.npz.
cost(pairs (P, 6), ids) -> T; grad(pairs) -> (T, dT/dx0).
"""
import time
import numpy as np
import torch
from scipy.stats import spearmanr
import plan as P
from common import BINS, PLAN_SEEDS, bin_labels, by_bin, save_paths


def load_queries(prepared, device):
    z = np.load(prepared / 'queries.npz')
    Q = {k: z[k] for k in z.files}
    Q['start_t'] = torch.tensor(Q['start'], device=device)
    Q['goal_t'] = torch.tensor(Q['goal'], device=device)
    return Q


def evaluate(maze, prepared, cost, grad, out, log, label='field', plan_seeds=PLAN_SEEDS):
    dev = maze.device
    Q = load_queries(prepared, dev)
    S, G, bins = Q['start_t'], Q['goal_t'], Q['bin']
    st = P.settings(maze)
    budget = torch.tensor(np.maximum(st['min_steps'], np.ceil(st['budget_factor'] * Q['path_clen'] / st['step'])), device=dev)
    res = dict(planner=dict(st, budget_median=float(budget.median()), budget_max=float(budget.max())), planning={})
    success = np.zeros((len(plan_seeds), len(S)), bool)
    collision = np.zeros_like(success)
    converged = np.zeros_like(success)
    length_ratio = np.full(success.shape, np.nan)
    time_ratio = np.full(success.shape, np.nan)
    for k, ps in enumerate(plan_seeds):
        tp = time.time()
        paths, stop, conv = P.mppi(cost, S, G, ps, budget, maze.lo, maze.hi, **st)
        rows = P.score(maze, [paths[i] for i in range(len(S))], conv)
        success[k] = [r['success'] for r in rows]
        collision[k] = [r['collision'] for r in rows]
        converged[k] = [r['converged'] for r in rows]
        length_ratio[k] = [r['length_cells'] / max(Q['path_cells'][i], 1e-6) if r['success'] else np.nan for i, r in enumerate(rows)]
        time_ratio[k] = [r['time'] / Q['t_star'][i] if r['success'] else np.nan for i, r in enumerate(rows)]
        res['planning'][str(ps)] = dict(success=int(success[k].sum()), converged=int(converged[k].sum()), collision=int(collision[k].sum()),
                                        by_bin=by_bin(rows, bins), by_bin_collision=by_bin(rows, bins, 'collision'),
                                        steps_median=float(stop.float().median()), seconds=time.time() - tp)
        if k == 0:
            save_paths(out / 'paths_seed0.npy', [paths[i] for i in range(len(S))])
        log(f'{label} MPPI seed {ps}: {int(success[k].sum())}/{len(S)} (converged {int(converged[k].sum())}, collisions {int(collision[k].sum())}) '
            f'by bin {res["planning"][str(ps)]["by_bin"]} ({time.time() - tp:.0f}s)')
    labels = bin_labels()
    res['success_by_bin'] = {labels[b]: [int(success[:, bins == b].sum()), int(len(plan_seeds) * (bins == b).sum())] for b in sorted(set(bins.tolist()))}
    res['success_total'] = [int(success.sum()), int(success.size)]
    # field diagnostics
    pairs = torch.cat([S, G], 1)
    t_pred, g = grad(pairs)
    pred_dir = torch.nn.functional.normalize(-g, dim=1)
    true_dir = torch.tensor(Q['true_dir'], device=dev)
    cos = (pred_dir * true_dir).sum(1).cpu().numpy()
    t_pred = t_pred.cpu().numpy()
    ratio = t_pred / Q['t_star']
    field = dict(t_over_tstar_median=float(np.median(ratio)), spearman=float(spearmanr(t_pred, Q['t_star'])[0]), descent_cos_median=float(np.median(cos)),
                 by_bin={})
    for b in sorted(set(bins.tolist())):
        m = bins == b
        field['by_bin'][labels[b]] = dict(n=int(m.sum()), t_over_tstar_median=float(np.median(ratio[m])),
                                          t_over_tstar_q10_q90=[float(np.quantile(ratio[m], 0.1)), float(np.quantile(ratio[m], 0.9))],
                                          spearman=float(spearmanr(t_pred[m], Q['t_star'][m])[0]) if m.sum() > 2 else None,
                                          descent_cos_median=float(np.median(cos[m])), descent_cos_negative=int((cos[m] < 0).sum()),
                                          success=[int(success[:, m].sum()), int(success[:, m].size)],
                                          length_ratio_median=float(np.nanmedian(length_ratio[:, m])) if np.isfinite(length_ratio[:, m]).any() else None,
                                          time_ratio_median=float(np.nanmedian(time_ratio[:, m])) if np.isfinite(time_ratio[:, m]).any() else None)
    res['field'] = field
    np.savez(out / 'rows.npz', success=success, collision=collision, converged=converged, length_ratio=length_ratio, time_ratio=time_ratio,
             t_pred=t_pred, cos=cos, bin=bins)
    log(f'{label} field: T/T* {field["t_over_tstar_median"]:.3f}, Spearman {field["spearman"]:.3f}, descent cos {field["descent_cos_median"]:.3f}; by bin ' +
        ', '.join(f'{k} {v["t_over_tstar_median"]:.2f}/{v["descent_cos_median"]:.2f}' for k, v in field['by_bin'].items()))
    return res


def slice_values(maze, cost, goal, res=None, chunk=100_000):
    """Field T((x, y, phi_goal), goal) on a res x res grid over the maze (res = 8 n + 1)."""
    res = 8 * maze.n + 1 if res is None else res
    xs = torch.linspace(-0.5, 0.5, res, device=maze.device)
    X, Y = torch.meshgrid(xs, xs, indexing='ij')
    q = torch.stack([X.reshape(-1), Y.reshape(-1), torch.full((res * res,), float(goal[2]), device=maze.device)], 1)
    pairs = torch.cat([q, goal[None].expand(len(q), 3)], 1)
    vals = torch.cat([cost(pairs[i:i + chunk], None) for i in range(0, len(pairs), chunk)])
    return vals.reshape(res, res).cpu().numpy()


def speed_fit(maze, prepared, grad):
    """Field speed 1 / |dT/dx0| vs the trained label at x0 on the held-out pairs."""
    hx = torch.tensor(np.load(prepared / 'heldout/sampled_points.npy'), device=maze.device)
    hd = torch.tensor(np.load(prepared / 'heldout/clearance_cells.npy'), device=maze.device)
    _, g = grad(hx)
    s_pred = 1 / torch.linalg.vector_norm(g, dim=1).clamp(min=1e-6)
    s_true = maze.train_speed(hd[:, 0])
    lp, lt = torch.log(s_pred), torch.log(s_true)
    return dict(log_speed_corr=float(torch.corrcoef(torch.stack([lp, lt]))[0, 1]), log_speed_mae=float((lp - lt).abs().mean()),
                speed_pred_median_at_floor=float(s_pred[s_true < 0.05].median()) if bool((s_true < 0.05).any()) else None,
                speed_pred_median_free=float(s_pred[s_true > 0.95].median()) if bool((s_true > 0.95).any()) else None)
