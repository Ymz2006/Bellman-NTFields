"""Data for the line-test explorer page (host venv): every saved line-test model (line_test.py --save-models) evaluated
along eight axes through picked points on the trained line, with the 1-D truth along each axis.

Picked points: x_b = x* + s_b u for 21 values s_b in [-0.05, 0.05] (the trained line). Axes, all unit length in release
units: u (the trained +y IK direction) and the six joint axes e_1..e_6. Points: x = x_b + t a, t in [-0.05, 0.05], 161
values. At every point: the clearance label d (m), the target speed s'(d), the 1-D travel time from x_b along the axis
(trapezoid of 1/s' on a 20x finer grid), and per model the learned T(x_b, x) and field speed 1/|grad_x T(x_b, x)|.

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/.venvs/bellman-ntfield/bin/python jeffrey/representability/explorer_export.py
"""
import base64, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import line_test as LT

SETS = {'line': HERE/'line_v3/models', 'tube': HERE/'tube_v3/models'}
ORDER = ['ntrl-base', 'ntrl-log', 'ntrl-s1', 'tc-h4', 'tc-h4-n1']
NB, NT, FINE = 21, 161, 20


def b64(a): return base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()


def load(path, dev):
    c = torch.load(path, map_location=dev, weights_only=False)
    if c['family'] == 'ntrl':
        NN, _ = LT.ntrl_modules(c['variant']); net = NN(dev, 6, c['B'].cpu()).to(dev); net.load_state_dict(c['state_dict']); net.eval()
        def tg(xa, xb):
            with torch.enable_grad():
                tau, _, xp = net.out(torch.cat([xa, xb], 1)); g = torch.autograd.grad(tau.sum(), xp)[0]
            return tau[:, 0].detach(), g[:, 6:].detach()
    else:
        v = dict(c['variant']); v.pop('normal', None); net = LT.Wide(**v).to(dev); net.load_state_dict(c['state_dict']); net.eval().requires_grad_(False)
        def tg(xa, xb):
            with torch.enable_grad():
                xb = xb.clone().requires_grad_(True); t = net(xa, xb)[:, 0]; g = torch.autograd.grad(t.sum(), xb)[0]
            return t.detach(), g.detach()
    return tg


def main():
    dev = 'cuda:0'
    m = json.loads((LT.GRID/'meta.json').read_text()); A = np.fromfile(LT.GRID/'nodes.bin', np.float32).reshape(m['nodes'], 11); qr = np.array(m['q_range'])
    q_star = A[LT.NODE, 3:9].astype(np.float64); x_star = torch.tensor(q_star/qr, device=dev)
    u = torch.tensor(np.load(LT.PER_POSE)['dq_ik_leastnorm'][LT.NODE, :, 1]/qr, device=dev); u = u/u.norm(); qr_t = torch.tensor(qr, device=dev)
    kin = LT.R.Kinematics().to(dev); fk = lambda x: kin.transforms((x*qr_t), [LT.R.EE_LINK])[LT.R.EE_LINK][:, :3, 3]
    with torch.no_grad(): p_hit = fk((x_star+LT.S_HIT*u)[None])[0]
    label = LT.TF.Label(dev); label.cen = p_hit.float()[None].clone(); label.half = torch.tensor([LT.PANEL], device=dev)/2
    d_of = lambda x: label(x.float()*qr_t.float()).double()
    axes = torch.cat([u[None], torch.eye(6, dtype=torch.float64, device=dev)])
    sb = torch.linspace(-LT.S_MAX, LT.S_MAX, NB, dtype=torch.float64, device=dev); tt = torch.linspace(-LT.S_MAX, LT.S_MAX, NT, dtype=torch.float64, device=dev)
    tf = torch.linspace(-LT.S_MAX, LT.S_MAX, (NT-1)*FINE+1, dtype=torch.float64, device=dev); mid = (len(tf)-1)//2
    base = x_star[None]+sb[:, None]*u[None]                                                        # [NB, 6]
    X = base[:, None, None, :]+tt[None, None, :, None]*axes[None, :, None, :]                       # [NB, 8, NT, 6]
    XF = base[:, None, None, :]+tf[None, None, :, None]*axes[None, :, None, :]
    d = d_of(X.reshape(-1, 6)).reshape(NB, 7, NT); S = LT.target_speed(d)
    inv = 1/LT.target_speed(d_of(XF.reshape(-1, 6)).reshape(NB, 7, -1))
    cum = torch.cat([torch.zeros(NB, 7, 1, dtype=torch.float64, device=dev), torch.cumulative_trapezoid(inv, tf, dim=2)], 2)
    T = (cum-cum[:, :, mid:mid+1]).abs()[:, :, ::FINE]
    with torch.no_grad(): grip = fk(X.reshape(-1, 6)).reshape(NB, 7, NT, 3)
    out = dict(note='explorer_export.py', q_star=q_star.tolist(), q_range=qr.tolist(), u=u.cpu().tolist(), s_hit=LT.S_HIT, panel=list(LT.PANEL), block_centre=p_hit.cpu().tolist(),
               bases=sb.cpu().tolist(), t=tt.cpu().tolist(), axes=['+y line', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6'], shape=[NB, 7, NT],
               d=b64(d.cpu().numpy()), S=b64(S.cpu().numpy()), T=b64(T.cpu().numpy()), grip=b64(grip.cpu().numpy()), models={})
    for set_name, folder in SETS.items():
        out['models'][set_name] = {}
        for name in ORDER:
            tg = load(folder/f'{name}.pt', dev); Tl, Sl = [], []
            xa = base[:, None, None, :].expand(NB, 7, NT, 6).reshape(-1, 6).float(); xb = X.reshape(-1, 6).float()
            for i in range(0, len(xa), 4096):
                t_, g_ = tg(xa[i:i+4096], xb[i:i+4096]); Tl.append(t_.double()); Sl.append(1/g_.double().norm(dim=1).clamp_min(1e-9))
            Tl = torch.cat(Tl).reshape(NB, 7, NT); Sl = torch.cat(Sl).reshape(NB, 7, NT)
            out['models'][set_name][name] = dict(T=b64(Tl.cpu().numpy()), S=b64(Sl.cpu().numpy()))
            print(set_name, name, 'T from s=-0.04 to s=+0.01 along u:', round(float(Tl[2, 0, -1]), 3), 'truth', round(float(T[2, 0, -1]), 3), flush=True)
            del tg; torch.cuda.empty_cache()
    (HERE/'explorer/data.json').write_text(json.dumps(out)); print('wrote', (HERE/'explorer/data.json').stat().st_size, 'bytes')


if __name__ == '__main__': main()
