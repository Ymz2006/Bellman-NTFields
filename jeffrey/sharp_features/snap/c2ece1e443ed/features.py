"""Sharp features on top of the grid-synced octave Fourier embedding (jeffrey/grid_sync `octave`, "the octaves").

Base embedding: ntrl-demo's seeded draw B (128 x 3) turned into the octave matrix (octave(): the same operations as
jeffrey/grid_sync/features.py mode `octave` and jeffrey/octave_detune/features.py, bit for bit): every row keeps its
xy direction, the rows are sorted by ntrl-demo's |f_xy| into K equal groups and group k is put on the square
max(|f_x|, |f_y|) = f_grid / L_k, L = 2^(K-1) .. 2, 1 cells (f_grid = E / 2 cycles per normalized unit: half period one
cell), heading column b_phi x s_eff. With z_i = 2 pi B_i . x (x = (x, y, phi) normalized) the network's input is
[sin z_1 .. sin z_128, cos z_1 .. cos z_128] (256 features).

A spec is 'octave' or one or more parts joined by '+':

  R<fn>@<sel>        replace: the pair (sin z_i, cos z_i) of every selected row becomes fn's pair (width stays 256;
                     the network is ntrl-demo's, unchanged)
  A<fn>@<sel>[x<k>]  add: fn's features of the selected rows are appended (width 256 + m); the three layers that read
                     the input (pe_gate 0 / 1 and encoder 1) take 256 + m inputs, and the residual of the first block
                     carries the 256 base features (sharpnet.py). x<k>: k copies per row (gates: k offsets h per row).

Functions fn of the phase z (pairs unless noted):
  pow<p>     (sin^p z, cos^p z), p even: period pi, non-negative, a peak of width ~1/sqrt(p) rad at every crest
  pow<p>z    pow<p> standardized over a uniform phase: (f - mu_p) / sd_p / sqrt(2), the mean and SD of sin (0, 0.707)
  spow<p>    (sgn(sin) |sin|^p, sgn(cos) |cos|^p): the same sharpness with the sign kept (period 2 pi, no fold)
  powmix     pow<p> with p = 2, 4, 8, 16 in turn over the selected rows of each octave level
  powcat     (sin^2, cos^2, sin^4, cos^4, sin^8, cos^8, sin^16, cos^16): every power for every row (8 per row; add only)
  tri<A>     A (T_s(z), T_c(z)), T_c = (2 / pi) arcsin(cos z), T_s = (2 / pi) arcsin(sin z) (unit triangle waves);
             'tri' = tri4 = the user's f(x) = (8 / pi) arcsin(cos x) (amplitude 4); tri1 unit amplitude
  sc<g>      g (sin z, cos z): sc1 = a copy of the base pair, sc3.27 = sin / cos with the RMS of tri4 (gain control)
  harm<k>    (sin k z, cos k z): the same row at k times its frequency
  gtri<A>    max(0, tanh(a (z - h))) A T_c(z)  (1 per offset h; default A = 4: the user's idea 3 with f of idea 2)
  gtrip<A>   the gate times A (T_s, T_c)  (2 per h)
  gsc        the gate times (sin z, cos z)  (2 per h)
  gate       max(0, tanh(a (z - h))) alone  (1 per h)
Options (after the fn, each '.<name>[value]'):
  .partner   (R only) a replaced row takes fn of its partner row (the neighbouring kept row of the same level), so
             every kept row carries (sin, cos) and fn of itself: the same width with half the directions
  .grid      gate offsets h at the phase of a random grid vertex (default: of a uniformly random point of the box)
  .a<v>      gate slope a (default 1, the user's tanh(x - h))
Selections <sel>:
  all, half (every other row of each octave level, by row index), quarter (every fourth), top (the level L = 1, half
  period one cell), low (every level but the top), new<M> (M new rows: a second draw, rng [seed, 11], put on the
  octave levels by the same construction; add only)

Randomness: numpy default_rng([seed, tag]) only (tag 11 new rows, 12 gate offsets), never the torch generator, so the
network initialization and the data order are those of plain ntrl-demo with the same seed whenever the width is 256.
"""
import math
import re
import numpy as np
import torch

RHO = 0.25                      # jeffrey/maze_scale/maze.py: phi = RHO theta / E
POWERS = (2, 4, 8, 16)
FN_PAIR = ('pow', 'spow', 'powmix', 'tri', 'sc', 'harm')
FN_ADD_ONLY = ('powcat', 'gtri', 'gtrip', 'gsc', 'gate')
SELS = ('all', 'half', 'quarter', 'top', 'low')


# ---------------------------------------------------------------- base (octave) matrix

def draw(seed):
    """ntrl-demo's B for a training seed without touching the global generator (= jeffrey/grid_sync/features.draw)."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        B = torch.normal(0, 1, size=(128, 3))
        torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
    finally:
        torch.random.set_rng_state(state)
    return B


def octaves(n):
    """Half periods L (cells) of the octave levels, largest first: 2^k <= n."""
    L = [1]
    while L[-1] * 2 <= n:
        L.append(L[-1] * 2)
    return L[::-1]


def octave_rows(b, E, s_eff=None):
    """The grid-sync octave construction on a float64 (M, 3) array b (same operations and order as grid_sync)."""
    xy, ph = b[:, :2].copy(), b[:, 2].copy()
    grid = E / 2.0
    dom = np.abs(xy).max(1)
    norm = np.linalg.norm(xy, axis=1)
    cos = dom / norm
    unit = xy / norm[:, None]
    if s_eff is None:
        s_eff = grid / float(np.median(dom))
    Ls = octaves(int(round(E - 0.1)))
    order = np.argsort(norm, kind='stable')
    L = np.empty(len(b))
    for k, idx in enumerate(np.array_split(order, len(Ls))):
        L[idx] = Ls[k]
    mag = grid / cos
    xy = unit * (mag / L)[:, None]
    return np.concatenate([xy, (ph * s_eff)[:, None]], 1), L, s_eff


def octave(B, E):
    """jeffrey/grid_sync's octave matrix (128 x 3, B's dtype and device) from ntrl-demo's draw B for extent E cells."""
    b = B.detach().cpu().double().numpy().copy()
    out, _, _ = octave_rows(b, E)
    return torch.tensor(out, dtype=B.dtype, device=B.device)


def levels(Bo, E):
    """Octave level L (half period in cells) of every row of an octave matrix: f_grid / dominant component."""
    b = Bo.detach().cpu().double().numpy() if torch.is_tensor(Bo) else np.asarray(Bo, float)
    L = (E / 2.0) / np.abs(b[:, :2]).max(1)
    Lr = 2.0 ** np.round(np.log2(L))
    assert np.allclose(L, Lr, rtol=1e-5), 'not an octave matrix'
    return Lr


# ---------------------------------------------------------------- spec parsing

PART = re.compile(r'(R|A)([a-z]+?)((?:[0-9]+(?:\.[0-9]+)?)?)(z?)((?:\.[a-z]+[0-9.]*)*)@(all|half|quarter|top|low|new[0-9]+)(?:x([0-9]+))?')


def parse(spec):
    """List of part dicts (empty for 'octave')."""
    if spec == 'octave':
        return []
    parts = []
    for tok in spec.split('+'):
        m = PART.fullmatch(tok)
        assert m, f'bad part {tok!r} in {spec!r}'
        op, fn, num, zflag, opts, sel, copies = m.groups()
        fn_all = FN_PAIR + FN_ADD_ONLY + ('dup',)
        assert fn in fn_all, f'unknown fn {fn!r}'
        part = dict(token=tok, op=op, fn=fn, num=float(num) if num else None, std=bool(zflag), sel=sel,
                    copies=int(copies) if copies else 1, partner=False, grid=False, a=1.0)
        for o in re.findall(r'\.([a-z]+)([0-9.]*)', opts):
            name, val = o
            if name == 'partner':
                part['partner'] = True
            elif name == 'grid':
                part['grid'] = True
            elif name == 'a':
                part['a'] = float(val)
            else:
                raise AssertionError(f'unknown option {name!r}')
        if fn == 'dup':
            part['fn'], part['num'] = 'sc', 1.0
        if part['fn'] in ('pow', 'spow'):
            assert part['num'] is not None and int(part['num']) == part['num'] and part['num'] >= 2, tok
        if part['std']:
            assert part['fn'] == 'pow', 'standardization only for pow'
        if part['fn'] == 'harm':
            assert part['num'] is not None, tok
        if op == 'R':
            assert part['fn'] in FN_PAIR, f'{tok}: replace needs a pair function'
            assert not sel.startswith('new') and part['copies'] == 1, tok
        if part['partner']:
            assert op == 'R' and sel in ('half',), 'partner only with R...@half'
        parts.append(part)
    return parts


def tri_amp(part):
    return 4.0 if part['num'] is None else part['num']


def sc_gain(part):
    return 1.0 if part['num'] is None else part['num']


def pow_stats(p):
    """Mean and SD of sin^p over a uniform phase (p even): mu = C(p, p/2) / 2^p, E[sin^2p] = C(2p, p) / 4^p."""
    mu = math.comb(p, p // 2) / 2.0 ** p
    m2 = math.comb(2 * p, p) / 4.0 ** p
    return mu, math.sqrt(m2 - mu * mu)


def width_of(part, rows):
    fn, k = part['fn'], part['copies']
    if fn == 'powcat':
        return 8 * rows * k
    if fn in ('gtri', 'gate'):
        return rows * k
    return 2 * rows * k


# ---------------------------------------------------------------- the plan

def select(sel, L):
    """Row indices of a selection (base rows, row-index order within each level)."""
    idx = np.arange(len(L))
    if sel == 'all':
        return idx
    if sel == 'top':
        return idx[L == 1]
    if sel == 'low':
        return idx[L > 1]
    step = dict(half=2, quarter=4)[sel]
    out = []
    for lv in sorted(set(L.tolist()), reverse=True):
        out.extend(idx[L == lv][::step].tolist())
    return np.array(sorted(out), dtype=np.int64)


def partners(L):
    """For R...@half.partner: replaced row -> the next row of the same level (the kept one)."""
    idx = np.arange(len(L))
    rep, src = [], []
    for lv in sorted(set(L.tolist()), reverse=True):
        rows = idx[L == lv]
        for j in range(0, len(rows) - 1, 2):
            rep.append(rows[j])
            src.append(rows[j + 1])
    return np.array(rep, dtype=np.int64), np.array(src, dtype=np.int64)


def mix_powers(rows, L):
    """powmix: p = 2, 4, 8, 16 in turn over the selected rows of each level."""
    p = np.empty(len(rows))
    for lv in sorted(set(L[rows].tolist()), reverse=True):
        at = np.nonzero(L[rows] == lv)[0]
        p[at] = np.array(POWERS)[np.arange(len(at)) % len(POWERS)]
    return p


def truncated_normal(rng, shape):
    out = rng.standard_normal(shape)
    bad = np.abs(out) > 2.0
    while bad.any():
        out[bad] = rng.standard_normal(int(bad.sum()))
        bad = np.abs(out) > 2.0
    return out


def box_points(rng, count, E, n, grid):
    """Points where the gate boundaries pass: uniform in the C-space box, or random grid vertices (phi = 0)."""
    if grid:
        k = rng.integers(0, n + 1, size=(count, 2))
        xy = (k - n / 2.0) / E
        return np.concatenate([xy, np.zeros((count, 1))], 1)
    phi_max = RHO * (math.pi / 2) / E
    xy = rng.uniform(-0.5, 0.5, size=(count, 2))
    ph = rng.uniform(-phi_max, phi_max, size=(count, 1))
    return np.concatenate([xy, ph], 1)


def plan(spec, Bo, E, n, seed):
    """Everything the network needs for a spec, as numpy arrays (sharpnet.py turns them into buffers).

    Bo: the octave matrix (128 x 3). Returns dict(spec, replace=[...], add=[...], extra_width, rows_replaced, summary)."""
    parts = parse(spec)
    b = Bo.detach().cpu().double().numpy() if torch.is_tensor(Bo) else np.asarray(Bo, float)
    L = levels(b, E)
    M = len(b)
    out = dict(spec=spec, E=float(E), n=int(n), seed=int(seed), replace=[], add=[], extra_width=0)
    taken = np.zeros(M, bool)
    for part in parts:
        fn = part['fn']
        if part['op'] == 'R':
            if part['partner']:
                rows, src = partners(L)
            else:
                rows = select(part['sel'], L)
                src = rows.copy()
            assert not taken[rows].any(), f'{spec}: two replace parts overlap'
            taken[rows] = True
            mask = np.zeros(M, bool)
            mask[rows] = True
            source = np.arange(M)
            source[rows] = src
            blk = dict(part=part, mask=mask, source=source, p=None)
            if fn == 'powmix':
                p = np.ones(M)
                p[rows] = mix_powers(rows, L)
                blk['p'] = p
            out['replace'].append(blk)
            continue
        # add
        blk = dict(part=part)
        if part['sel'].startswith('new'):
            count = int(part['sel'][3:])
            rng = np.random.default_rng([seed, 11])
            raw = truncated_normal(rng, (count, 3))
            s_eff = (E / 2.0) / float(np.median(np.abs(draw(seed)[:, :2].double().numpy()).max(1)))
            newB, Lnew, _ = octave_rows(raw, E, s_eff)
            blk['B_new'] = newB
            blk['rows'] = None
            rows_count = count
            blk['levels'] = Lnew
        else:
            rows = select(part['sel'], L)
            blk['rows'] = rows
            rows_count = len(rows)
            blk['levels'] = L[rows]
        if fn == 'powmix':
            blk['p'] = mix_powers(np.arange(rows_count), blk['levels'])
        if fn in ('gtri', 'gtrip', 'gsc', 'gate'):
            assert blk['rows'] is not None, 'gates on base rows only'
            k = part['copies']
            rr = np.repeat(blk['rows'], k)                    # row of every gate (k consecutive gates per row)
            rng = np.random.default_rng([seed, 12])
            pts = box_points(rng, len(rr), E, n, part['grid'])
            h = 2.0 * math.pi * np.einsum('ij,ij->i', pts, b[rr])
            blk['gate_rows'] = rr
            blk['h'] = h
        elif part['copies'] != 1:
            raise AssertionError(f'{part["token"]}: copies only for gates')
        blk['width'] = width_of(part, rows_count)
        out['extra_width'] += blk['width']
        out['add'].append(blk)
    out['rows_replaced'] = int(taken.sum())
    out['summary'] = summary(out, L)
    return out


def summary(pl, L):
    rep = []
    for blk in pl['replace']:
        part = blk['part']
        lv = {str(int(x)): int(((L == x) & blk['mask']).sum()) for x in sorted(set(L.tolist()), reverse=True)}
        rep.append(dict(token=part['token'], rows=int(blk['mask'].sum()), by_level=lv,
                        powers=None if blk['p'] is None else {str(int(q)): int((blk['p'][blk['mask']] == q).sum()) for q in POWERS}))
    add = []
    for blk in pl['add']:
        part = blk['part']
        rec = dict(token=part['token'], width=int(blk['width']), rows=None if blk['rows'] is None else int(len(blk['rows'])))
        if 'h' in blk:
            rec['gates'] = int(len(blk['h']))
            rec['h_abs_median'] = float(np.median(np.abs(blk['h'])))
        if 'B_new' in blk:
            rec['new_rows'] = int(len(blk['B_new']))
        add.append(rec)
    return dict(replace=rep, add=add, extra_width=int(pl['extra_width']), input_width=256 + int(pl['extra_width']))


# ---------------------------------------------------------------- elementwise functions (torch, twice differentiable)

def tri_c(z):
    """(2 / pi) arcsin(cos z) = 1 - 2 |w| / pi, w = z wrapped to [-pi, pi): no infinite derivative at the crests."""
    w = torch.remainder(z + math.pi, 2.0 * math.pi) - math.pi
    return 1.0 - (2.0 / math.pi) * torch.abs(w)


def tri_s(z):
    """(2 / pi) arcsin(sin z) = tri_c(z - pi / 2)."""
    return tri_c(z - 0.5 * math.pi)


def powers(s):
    """s^2, s^4, s^8, s^16 by repeated squaring (exact even powers)."""
    s2 = s * s
    s4 = s2 * s2
    s8 = s4 * s4
    return {2: s2, 4: s4, 8: s8, 16: s8 * s8}


def even_power(s, p):
    if p in POWERS:
        return powers(s)[p]
    return s ** int(p)


def pair(part, z, p_rows=None):
    """fn's pair (f1, f2) of a phase tensor z (..., R); p_rows: per-column powers for powmix."""
    fn = part['fn']
    if fn == 'pow':
        p = int(part['num'])
        f1, f2 = even_power(torch.sin(z), p), even_power(torch.cos(z), p)
        if part['std']:
            mu, sd = pow_stats(p)
            sd = sd * math.sqrt(2.0)
            f1, f2 = (f1 - mu) / sd, (f2 - mu) / sd
        return f1, f2
    if fn == 'spow':
        p = int(part['num'])
        s, c = torch.sin(z), torch.cos(z)
        return s * torch.abs(s) ** (p - 1), c * torch.abs(c) ** (p - 1)
    if fn == 'powmix':
        s, c = powers(torch.sin(z)), powers(torch.cos(z))
        f1, f2 = s[16], c[16]
        for q in (8, 4, 2):
            sel = p_rows == q
            f1 = torch.where(sel, s[q], f1)
            f2 = torch.where(sel, c[q], f2)
        return f1, f2
    if fn == 'tri':
        A = tri_amp(part)
        return A * tri_s(z), A * tri_c(z)
    if fn == 'sc':
        g = sc_gain(part)
        if g == 1.0:
            return torch.sin(z), torch.cos(z)
        return g * torch.sin(z), g * torch.cos(z)
    if fn == 'harm':
        k = part['num']
        return torch.sin(k * z), torch.cos(k * z)
    raise AssertionError(fn)


def add_features(part, z, p_rows=None, h=None):
    """Appended features of phases z (N, R) (for gates: z of the gate rows, h (R,) offsets)."""
    fn = part['fn']
    if fn == 'powcat':
        s, c = powers(torch.sin(z)), powers(torch.cos(z))
        return torch.cat([t for q in POWERS for t in (s[q], c[q])], -1)
    if fn in ('gtri', 'gtrip', 'gsc', 'gate'):
        g = torch.relu(torch.tanh(part['a'] * (z - h)))
        if fn == 'gate':
            return g
        if fn == 'gtri':
            return g * (tri_amp(part) * tri_c(z))
        if fn == 'gtrip':
            A = tri_amp(part)
            return torch.cat([g * (A * tri_s(z)), g * (A * tri_c(z))], -1)
        return torch.cat([g * torch.sin(z), g * torch.cos(z)], -1)
    f1, f2 = pair(part, z, p_rows)
    return torch.cat([f1, f2], -1)


def describe(pl):
    return pl['summary']
