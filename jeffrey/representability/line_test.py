"""Can the travel-time networks represent the slope of a wall? One block, one line (host venv, read-only analysis).

Setup. The middle-compartment pose q* of the Test 4 viewer grid; u = the least-norm IK joint direction that moves the
gripper in +y there (release units, unit length). The joint-space line x(s) = x* + s u, s in [-S, S]. The scene is ONE
cube of 0.10 m centred on the gripper position at s = s_hit (plus the arm's self distance), so moving along +y hits it.
Truth: the release label d(s) on the line, target speed s'(d) = 1.025 S^2 (2 - S)^2 - 0.025 with S = clip(d, 5 mm,
5 cm) / 5 cm, travel time T(a, b) = |int_a^b ds / s'| (exact in 1-D). Training pairs: s0, s1 uniform on the line, labels
at both ends, unit 6-D clearance gradients as normals. The networks see only this line.

Variants. ntrl-demo metric NTFields (models/metric, its own Loss: sqrt-eikonal 1e-2 + normal 1e-3 + TD 1e-3, AdamW
5e-4 wd 0.5, batches of 2,000) with knobs: freq (B = freq x N(0, 1)), scale (the metric factor, default 0.2; the
embedding stretch), lip (lip_norm on/off), loss sqrt | log (the bounded (sqrt(s'|dT|) - 1)^2 or the original
method's (log s'|dT|)^2). The original method's TravelCost with the eikonal specialist loss (eikonal.py: (log s'|dT|)^2
at both ends, AdamW 2e-4, clip 2, batch 256) with knobs: harmonics (4 or more) and inscale (input coordinates scaled
before the features, i.e. all frequencies multiplied).

Metrics per variant, from the free reference point s_ref: T(s_ref, s) along the line vs truth (far-end ratio), the
steepest slope |dT/ds| in the wall band vs truth (1/s' = 83 at the floor), predicted speed 1/|dT/dx| at contact and
in free space, RMS log error over random pairs. Writes results.json, curves.npz and curves.png.

  CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/representability/line_test.py --output jeffrey/representability/line_v1
"""
import argparse, importlib.util, json, math, sys, time, types
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'jeffrey/joint_range'), str(ROOT/'method'), str(ROOT)]
import timefield as TF
from experiments.face_speed import robot as R
from network import TravelCost, ResidualBlock
from torch import nn
from torch.nn import functional as F
NTRL = ROOT/'baselines/ntrl-demo'
GRID = ROOT/'jeffrey/joint_range/test4_viewer/cabinet'; PER_POSE = ROOT/'jeffrey/joint_range/axis_sensitivity/per_pose.npz'
NODE, S_MAX, S_HIT, S_REF = 16261, 0.05, 0.03, -0.04
PANEL = (0.40, 0.02, 0.70)                                  # a side-panel-like slab (x, y, z extents): thin along the +y motion
ALPHA, MARGIN, OFFSET = 1.025, .05, .005

NTRL_VARIANTS = {
    'ntrl-base': dict(freq=1., scale=None, lip=True, loss='sqrt'),
    'ntrl-f4': dict(freq=4., scale=None, lip=True, loss='sqrt'),
    'ntrl-f16': dict(freq=16., scale=None, lip=True, loss='sqrt'),
    'ntrl-s1': dict(freq=1., scale=1., lip=True, loss='sqrt'),
    'ntrl-s5': dict(freq=1., scale=5., lip=True, loss='sqrt'),
    'ntrl-nolip': dict(freq=1., scale=None, lip=False, loss='sqrt'),
    'ntrl-log': dict(freq=1., scale=None, lip=True, loss='log'),
    'ntrl-f4-s1-log': dict(freq=4., scale=1., lip=True, loss='log'),
    'ntrl-f16-s5-log-nolip': dict(freq=16., scale=5., lip=False, loss='log'),
}
TC_VARIANTS = {'tc-h4': dict(harmonics=4, inscale=1.), 'tc-h16': dict(harmonics=16, inscale=1.), 'tc-h4-in4': dict(harmonics=4, inscale=4.), 'tc-h16-in4': dict(harmonics=16, inscale=4.),
               'tc-h32': dict(harmonics=32, inscale=1.),
               # + the normal term of ntrl-demo's loss: (1.001 - s') |s' grad T + n|^2 at both ends (n = unit clearance gradient), weight w
               'tc-h4-n0.1': dict(harmonics=4, inscale=1., normal=0.1), 'tc-h4-n1': dict(harmonics=4, inscale=1., normal=1.), 'tc-h4-n10': dict(harmonics=4, inscale=1., normal=10.),
               'tc-h16-n1': dict(harmonics=16, inscale=1., normal=1.)}


def target_speed(d):
    s = (d.clamp(OFFSET, MARGIN)/MARGIN).double(); return ALPHA*s**2*(2-s)**2+1-ALPHA


class Wide(TravelCost):
    """TravelCost with H harmonics of the joint features and an input scale (same trunk widths)."""
    def __init__(self, harmonics=4, inscale=1.):
        super().__init__(); self.H, self.inscale = harmonics, inscale; self.state_input = nn.Linear(6+12*harmonics, 256)
        self.pair_input = nn.Linear(768+6*harmonics, 64)

    def feats(self, q):
        h = torch.arange(1, self.H+1, device=q.device, dtype=q.dtype); phase = (2*math.pi*(self.inscale*q)[:, :, None]*h).flatten(1)
        return torch.cat((q, torch.sin(phase), torch.cos(phase)), 1)

    def encode(self, q):
        h = F.silu(self.state_input(self.feats(q)))
        for block in self.state_blocks: h = block(h)
        return self.state_norm(h)

    def forward(self, q, goal):
        encoded = self.encode(torch.cat((q, goal), 0)); a, b = encoded[:len(q)], encoded[len(q):]
        h = torch.arange(1, self.H+1, device=q.device, dtype=q.dtype); phase = (2*math.pi*(self.inscale*(q-goal))[:, :, None]*h).flatten(1)
        pair = torch.cat((a+b, (a-b).square(), a*b, torch.cos(phase)), 1); x = F.silu(self.pair_input(pair))
        for block in self.pair_blocks: x = block(x)
        residual = self.output(self.pair_norm(x)); return torch.linalg.vector_norm(q-goal, dim=1, keepdim=True)*(1+F.softplus(residual))


def load_source(name, path, subs):
    src = Path(path).read_text()
    for old, new in subs: assert src.count(old) == 1, old; src = src.replace(old, new)
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(path)); m = importlib.util.module_from_spec(spec); m.__file__ = str(path)
    exec(compile(src, str(path), 'exec'), m.__dict__); return m


def ntrl_modules(v):
    if 'igl' not in sys.modules: sys.modules['igl'] = types.ModuleType('igl')
    net_subs = []
    if v['scale'] is not None: net_subs.append(('x = 0.2*(torch.sum(x,dim=1,keepdim=True))', f'x = {v["scale"]!r}*(torch.sum(x,dim=1,keepdim=True))'))
    if not v['lip']: net_subs.append(('scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)', 'scale = torch.ones_like(absrowsum)'))
    fn_subs = []
    if v['loss'] == 'log':
        fn_subs += [('loss0 = loss_weight*(l0_2-1)**2', 'loss0 = loss_weight*torch.log(l0+1e-8)**2'), ('loss1 = loss_weight*(l1_2-1)**2', 'loss1 = loss_weight*torch.log(l1+1e-8)**2')]
    N = load_source('rep_ntrl_network', NTRL/'models/metric/model_network_metric.py', net_subs); Fm = load_source('rep_ntrl_function', NTRL/'models/metric/model_function_metric.py', fn_subs)
    return N.NN, Fm.Function


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--variants', nargs='*', default=list(NTRL_VARIANTS)+list(TC_VARIANTS)); ap.add_argument('--updates-ntrl', type=int, default=6000); ap.add_argument('--updates-tc', type=int, default=15620)
    ap.add_argument('--pairs', type=int, default=200000); ap.add_argument('--seed', type=int, default=87358)
    ap.add_argument('--free-only', action='store_true', help='training pairs with both ends collision-free (clearance > 5 mm), as the original method\'s pool; the wall is then seen only from its slow band')
    ap.add_argument('--save-models', type=Path, help='directory: save every trained variant (state dict, variant settings, ntrl B matrix) as <name>.pt')
    ap.add_argument('--tube', type=float, default=0., help='training points x = line(s) + r v, v a random unit direction orthogonal to u, r ~ U(0, tube) (0: the line only)'); a = ap.parse_args()
    dev = a.device; out = a.output; out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed); t0 = time.monotonic()
    m = json.loads((GRID/'meta.json').read_text()); N = m['nodes']; A = np.fromfile(GRID/'nodes.bin', np.float32).reshape(N, 11); qr = np.array(m['q_range'])
    q_star = A[NODE, 3:9].astype(np.float64); x_star = torch.tensor(q_star/qr, device=dev); dq = np.load(PER_POSE)['dq_ik_leastnorm'][NODE, :, 1]
    u = torch.tensor(dq/qr, device=dev); u = u/u.norm(); qr_t = torch.tensor(qr, device=dev)
    line = lambda s: x_star[None]+s[:, None]*u[None]
    # the block: a cube on the gripper position at s_hit
    kin = R.Kinematics().to(dev); fk = lambda x: kin.transforms((x*qr_t), [R.EE_LINK])[R.EE_LINK][:, :3, 3]
    with torch.no_grad(): p_hit = fk(line(torch.tensor([S_HIT], dtype=torch.float64, device=dev)))[0]; p0 = fk(line(torch.tensor([0.], dtype=torch.float64, device=dev)))[0]
    label = TF.Label(dev); label.cen = p_hit.float()[None].clone(); label.half = torch.tensor([PANEL], device=dev)/2
    d_of = lambda x: label(x.float()*qr_t.float())
    # truth on a fine grid
    sg = torch.linspace(-S_MAX, S_MAX, 4001, dtype=torch.float64, device=dev); dg = d_of(line(sg)).double(); spg = target_speed(dg)
    inv = 1/spg; cum = torch.cat([torch.zeros(1, device=dev, dtype=torch.float64), torch.cumulative_trapezoid(inv, sg)])
    truth_T = lambda sa, sb: (torch.from_numpy(np.interp(sb.cpu().numpy(), sg.cpu().numpy(), cum.cpu().numpy())).to(dev)-torch.from_numpy(np.interp(sa.cpu().numpy(), sg.cpu().numpy(), cum.cpu().numpy())).to(dev)).abs()
    contact = dg <= 0; band = (dg > 0) & (dg < MARGIN); thickness = float((sg[1]-sg[0])*(dg < MARGIN).sum())
    with torch.no_grad(): pg = fk(line(sg))
    setup = dict(free_only=a.free_only, node=NODE, q_star=q_star.round(5).tolist(), u=u.cpu().numpy().round(5).tolist(), s_max=S_MAX, s_hit=S_HIT, panel_m=PANEL, tube=a.tube, block_centre=p_hit.cpu().numpy().round(4).tolist(),
                 gripper_at_0=p0.cpu().numpy().round(4).tolist(), gripper_travel_y_per_unit_s=float((pg[-1, 1]-pg[0, 1])/(2*S_MAX)), slow_band_units=thickness, contact_units=float((sg[1]-sg[0])*contact.sum()),
                 contact_range=[float(sg[contact].min()), float(sg[contact].max())] if contact.any() else None, truth_T_ref_to_end=float(truth_T(torch.tensor([S_REF], device=dev), sg[-1:])[0]),
                 truth_max_slope=float(inv.max()), truth_T_ref_to_hit=float(truth_T(torch.tensor([S_REF], device=dev), torch.tensor([S_HIT], device=dev))[0]))
    print('setup', json.dumps(setup), flush=True); assert contact.any(), 'the block is not hit on the line'
    # training pairs on the line
    g = torch.Generator(device=dev).manual_seed(a.seed); s0 = (torch.rand(a.pairs, generator=g, device=dev, dtype=torch.float64)*2-1)*S_MAX; s1 = (torch.rand(a.pairs, generator=g, device=dev, dtype=torch.float64)*2-1)*S_MAX
    x0, x1 = line(s0), line(s1)
    if a.tube > 0:   # off-line training points: the slab is thin in the directions orthogonal to the line
        def off(x):
            v = torch.randn(len(x), 6, generator=g, device=dev, dtype=torch.float64); v = v-(v@u)[:, None]*u[None]; v = v/v.norm(dim=1, keepdim=True)
            return x+(torch.rand(len(x), 1, generator=g, device=dev, dtype=torch.float64)*a.tube)*v
        x0, x1 = off(x0), off(x1)
    d0, d1 = d_of(x0).double(), d_of(x1).double(); S0 = (d0.clamp(OFFSET, MARGIN)/MARGIN); S1 = (d1.clamp(OFFSET, MARGIN)/MARGIN)
    def normal(x):
        eps = 1e-4; g6 = torch.stack([(d_of(x+eps*torch.eye(6, device=dev, dtype=x.dtype)[i])-d_of(x-eps*torch.eye(6, device=dev, dtype=x.dtype)[i]))/(2*eps) for i in range(6)], 1).double()
        return g6/g6.norm(dim=1, keepdim=True)
    n0, n1 = normal(x0), normal(x1); good = torch.isfinite(n0).all(1) & torch.isfinite(n1).all(1) & ((s0-s1).abs() > 1e-4)
    if a.free_only: good &= (d0 > OFFSET) & (d1 > OFFSET)
    x0, x1, S0, S1, n0, n1, s0, s1 = x0[good], x1[good], S0[good], S1[good], n0[good], n1[good], s0[good], s1[good]; P = len(x0)
    yobs = torch.stack([ALPHA*S0**2*(2-S0)**2+1-ALPHA, ALPHA*S1**2*(2-S1)**2+1-ALPHA], 1).float()
    pts = torch.cat([x0, x1], 1).float(); nrm = torch.cat([n0, n1], 1).float(); sp0, sp1 = yobs[:, 0], yobs[:, 1]
    print('pairs', P, 'contact share', float((d1 <= 0).float().mean()), flush=True)
    results = dict(setup=setup, pairs=P, variants={}); curves = dict(s=sg.cpu().numpy(), d=dg.cpu().numpy(), truth=truth_T(torch.full_like(sg, S_REF), sg).cpu().numpy(), truth_speed=spg.cpu().numpy())

    def evaluate(T_fn, grad_fn, name, train_seconds, history):
        """T_fn(xa, xb) -> T; grad_fn(xa, xb) -> dT/dxb (release units)."""
        with torch.no_grad():
            xr = line(torch.full_like(sg, S_REF)).float(); xs = line(sg).float()
            T = torch.cat([T_fn(xr[i:i+1024], xs[i:i+1024]) for i in range(0, len(sg), 1024)]).double()
        G = torch.cat([grad_fn(xr[i:i+1024], xs[i:i+1024]) for i in range(0, len(sg), 1024)]).double()
        slope = (G@u.double()).abs(); gnorm = G.norm(dim=1); speed = 1/gnorm.clamp_min(1e-9); tr = curves['truth']; trt = torch.tensor(tr, device=dev)
        rnd = torch.randint(P, (20000,), generator=g, device=dev); Tt = truth_T(s0[rnd], s1[rnd])
        with torch.no_grad(): Tl = T_fn(x0[rnd].float(), x1[rnd].float()).double()
        r = dict(train_seconds=train_seconds, T_end_learned=float(T[-1]), T_end_truth=float(trt[-1]), end_ratio=float(T[-1]/trt[-1]),
                 T_hit_learned=float(T[(sg-S_HIT).abs().argmin()]), T_hit_truth=float(trt[(sg-S_HIT).abs().argmin()]),
                 T_contact_learned=float(T[contact.nonzero()[0, 0]]), T_contact_truth=float(trt[contact.nonzero()[0, 0]]),
                 T_band_learned=float(T[(dg < MARGIN).nonzero()[0, 0]]), T_band_truth=float(trt[(dg < MARGIN).nonzero()[0, 0]]),
                 max_slope_learned=float(slope.max()), max_slope_truth=float(inv.max()), slope_in_band_median=float(slope[band | contact].median()), truth_slope_in_band_median=float(inv[band | contact].median()),
                 speed_contact_median=float(speed[contact].median()), speed_contact_target=float(spg[contact].median()), speed_free_median=float(speed[dg > MARGIN].median()),
                 speed_band_median=float(speed[band].median()), speed_band_target=float(spg[band].median()), pairs_rms_log=float((Tl.clamp_min(1e-6)/Tt.clamp_min(1e-6)).log().square().mean().sqrt()),
                 pairs_rms_log_contact_end=float((Tl/Tt).clamp_min(1e-6).log()[(d1[rnd] <= 0) | (d0[rnd] <= 0)].square().mean().sqrt()), history=history)
        results['variants'][name] = r; curves[name] = T.cpu().numpy(); curves[name+'_speed'] = speed.cpu().numpy()
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != 'history'}), flush=True)
        (out/'results.json').write_text(json.dumps(results, indent=1)); np.savez(out/'curves.npz', **curves)

    for name in a.variants:
        t1 = time.monotonic(); torch.manual_seed(a.seed)
        if name in NTRL_VARIANTS:
            v = NTRL_VARIANTS[name]; NN, Fn = ntrl_modules(v); B = v['freq']*torch.randn(128, 6, generator=torch.Generator().manual_seed(a.seed))
            net = NN(dev, 6, B).to(dev); net.apply(net.init_weights); fn = Fn('.', dev, net, 6)
            opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.5); hist = []
            for step in range(1, a.updates_ntrl+1):
                rows = torch.randint(P, (2000,), generator=g, device=dev)
                loss, loss_n, diff = fn.Loss(pts[rows], yobs[rows], nrm[rows], 1.0, 0.001, step//5+1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if step % 500 == 0: hist.append([step, float(loss)]); print(name, step, round(float(loss), 6), flush=True)
            net.eval()
            def T_fn(xa, xb):
                with torch.enable_grad(): tau, _, _ = net.out(torch.cat([xa, xb], 1))
                return tau[:, 0].detach()
            def grad_fn(xa, xb):
                with torch.enable_grad():
                    tau, _, xp = net.out(torch.cat([xa, xb], 1)); gg = torch.autograd.grad(tau.sum(), xp)[0]
                return gg[:, 6:].detach()
        else:
            v = dict(TC_VARIANTS[name]); w_n = v.pop('normal', 0.); net = Wide(**v).to(dev).train(); opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-6); hist = []
            from optimizer import stable_clip_grad_norm_
            for step in range(1, a.updates_tc+1):
                rows = torch.randint(P, (256,), generator=g, device=dev); q = x0[rows].float().requires_grad_(True); gl = x1[rows].float().requires_grad_(True)
                value = net(q, gl)[:, 0]; dq_, dg_ = torch.autograd.grad(value.sum(), (q, gl), create_graph=True)
                nq = torch.sqrt(dq_.square().sum(1)+1e-12); ng = torch.sqrt(dg_.square().sum(1)+1e-12)
                res = torch.cat([torch.log((sp0[rows]*nq).clamp_min(1e-8)), torch.log((sp1[rows]*ng).clamp_min(1e-8))]); loss = res.square().mean()
                if w_n:
                    n_loss = ((1.001-sp0[rows])[:, None]*(sp0[rows][:, None]*dq_+n0[rows].float())**2).sum(1).mean()+((1.001-sp1[rows])[:, None]*(sp1[rows][:, None]*dg_+n1[rows].float())**2).sum(1).mean()
                    loss = loss+w_n*n_loss
                opt.zero_grad(set_to_none=True); loss.backward(); stable_clip_grad_norm_(net.parameters(), 2.0); opt.step()
                if step % 1000 == 0: hist.append([step, float(loss)]); print(name, step, round(float(loss), 6), flush=True)
            net.eval().requires_grad_(False)
            T_fn = lambda xa, xb: net(xa, xb)[:, 0]
            def grad_fn(xa, xb):
                with torch.enable_grad():
                    xb = xb.clone().requires_grad_(True); gg = torch.autograd.grad(net(xa, xb)[:, 0].sum(), xb)[0]
                return gg.detach()
        evaluate(T_fn, grad_fn, name, time.monotonic()-t1, hist)
        if a.save_models:
            a.save_models.mkdir(parents=True, exist_ok=True)
            torch.save(dict(name=name, family='ntrl' if name in NTRL_VARIANTS else 'tc', variant=NTRL_VARIANTS.get(name) or TC_VARIANTS[name], state_dict=net.state_dict(),
                            B=B if name in NTRL_VARIANTS else None, tube=a.tube, free_only=a.free_only, seed=a.seed), a.save_models/f'{name}.pt')
        del net; torch.cuda.empty_cache()
    # chart
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = [n for n in a.variants if n in curves]; fig, ax = plt.subplots(2, 2, figsize=(14, 9)); s = curves['s']
    for col, fam in enumerate(('ntrl', 'tc')):
        for row, key in enumerate(('', '_speed')):
            axx = ax[row, col]; axx.plot(s, curves['truth' if key == '' else 'truth_speed'], 'k-', lw=2, label='truth')
            for n in names:
                if n.startswith(fam): axx.plot(s, curves[n+key], lw=1, label=n)
            axx.set_xlabel('s (release units along the +y joint direction)'); axx.set_ylabel('T from s_ref' if key == '' else 'speed 1/|dT/dx|'); axx.set_title(f'{fam}: {"travel time" if key == "" else "field speed"}')
            if key: axx.set_yscale('log')
            axx.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'curves.png', dpi=110); print('done', round(time.monotonic()-t0), 's', flush=True)


if __name__ == '__main__': main()
