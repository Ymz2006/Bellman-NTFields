"""The numbers of page_text.html, all from report/summary.json."""
import json
from pathlib import Path


def n(v, d=2):
    return '–' if v is None else f'{v:.{d}f}'.replace('-', '−')


def rk(x):
    return f"{x:+.2f}".replace('-', '−')


def compare(S, T12, SC):
    """The three mazes of last study: its recipe, the line loss with limits, the density form."""
    R = S['regions']
    rec, line, c, flow = S['recipe']['both'], T12['Long runs pushed, stiff near walls, stretch bound 8'], T12['Cartogram, h 2, ε 0.3'], T12['Learned flow, transport 1 (small network, 3 seeds)']
    cal = {b: {r['key']: r for r in S['calibration'][b]['rows']} for b in ('c22', 'plus', 'm10')}
    head = ['×12 maze (10×10 quadrants)', 'R² near walls', 'quadrant walls', 'band walls', 'R² over the box', 'Wall zone q05 / median', 'Zone spread', 'Anisotropy', 'Long-run share']
    r = lambda name, w, q, b, r2, x: [name, w if isinstance(w, str) else n(w, 3), q if isinstance(q, str) else n(q, 3), b if isinstance(b, str) else n(b, 3), n(r2, 3), f"{n(x.get('zone_q05'))} / {n(x.get('zone_q50'))}", n(x.get('zone_log_std')), n(x.get('aniso_q50')), n(x.get('long_share'))]
    span = lambda v: '–' if not v else f"{n(v[0], 3)}–{n(v[1], 3)}"
    rows = [r('Last study\'s recipe (separable)', R['recipe']['wall_r2'][0], R['recipe'].get('r2_quadrants'), R['recipe'].get('r2_band'), R['recipe']['r2'][0], rec),
            r('<b>Line loss with limits</b> (3 fit seeds)', span(line.get('wall_r2_range')), span(line.get('wall_r2_quadrants_range')), span(line.get('wall_r2_band_range')), line.get('r2'), line),
            r('<b>Density form</b> (no training)', c.get('wall_r2'), c.get('wall_r2_quadrants'), c.get('wall_r2_band'), c.get('r2'), c),
            r('Learned density form (3 fit seeds)', span(flow.get('wall_r2_range')), span(flow.get('wall_r2_quadrants_range')), span(flow.get('wall_r2_band_range')), flow.get('r2'), flow)]
    import page
    t1 = page.table(head, rows)
    w = lambda q: n((q or {}).get('wall_r2'), 3)
    rows2 = [['22×22 quadrants, ×12', n(cal['c22']['identity'].get('wall0'), 3), n(cal['c22']['both'].get('wall0'), 3), w(SC['c22'].get('lines')), w(SC['c22'].get('carto'))],
             ['Plus maze', n(cal['plus']['identity'].get('wall0'), 3), n(cal['plus']['dil'].get('wall0'), 3), w(SC['plus'].get('lines')), w(SC['plus'].get('carto'))]]
    t2 = page.table(['R² near walls', 'No warp', 'Last study\'s recipe', 'Line loss with limits', 'Density form'], rows2)
    return t1 + t2


def values(S, T10, T12, SC):
    runs = json.load(open(Path(__file__).resolve().parent / 'report' / 'runs.json'))
    W = lambda T, k, q='wall_r2', d=3: n((T.get(k) or {}).get(q), d)
    ID, LIT, ANY = T10['No warp'], T10['Every run of a line equal, as asked (coupling)'], T10['Fixed target 0.10, every angle (coupling)']
    C10, C12 = T10['Cartogram, h 2, ε 0.3'], T12['Cartogram, h 2, ε 0.3']
    NF = T12['Cartogram with the frame left in']
    FL = T12['Learned flow, transport 1 (small network, 3 seeds)']
    LINE = T12['Long runs pushed, stiff near walls, stretch bound 8']
    OLD = T12['Old loss with shear']
    EQ = T12['Equal runs, stiff near walls']
    R = S['regions']
    rec = S['recipe']['both']
    a = S['arm']
    rot = SC['r30g12']
    ident_w = ID.get('identity', {}).get('wall_r2')
    v = dict(N_RUNS=len(runs))
    span = lambda x, d=2: '–' if not x else f"{n(x[0], d)}–{n(x[1], d)}"
    v.update(T_LINE=span(LINE.get('wall_r2_range')), T_REC=n(R['recipe']['wall_mean'], 2), T_CARTO=n(C12.get('wall_r2'), 2), T_ID='below 0', T_ROT=n(rot['carto'].get('wall_r2'), 2), T_ROT_LINE=n((rot.get('lines') or {}).get('wall_r2'), 2), T_ROT_SEP=n(rot['sep'].get('wall_r2'), 2),
             T_P10=n(C10.get('wall_r2'), 3), T_P10_ID=n(ident_w, 3), T_P10_LIT=n(LIT.get('wall_r2'), 3), T_ARM=n(a[0]['r2'], 2), T_ARM_ID=n(a[0]['r2_identity'], 2))
    v.update(ID_CVL=n(ID['cv_line']), LIT_CVL=n(LIT['cv_line']), ID_LONG=n(ID['long_share']), LIT_LONG=n(LIT['long_share']), ID_W=n(ident_w, 3), LIT_W=n(LIT.get('wall_r2'), 3), ID_ANYT=n(ID['any_stripe']), ANY_ANYT=n(ANY['any_stripe']), ANY_W=n(ANY.get('wall_r2'), 3))
    turned = T10.get('Long runs pushed, stiff near walls, stretch bound 8')
    v['TURN_NOTE'] = f"The line loss with all its limits simply turns this maze by about 20°, the free win from reading along the axes: long runs almost vanish ({n(turned.get('long_share'))}) and the walls stay straight, but it scores {n(turned.get('wall_r2'), 3)} near walls. With lines at every angle the same limits leave the maze nearly alone." if turned else ''
    v.update(OLD_W=n(OLD.get('wall_r2'), 3), OLD_AN=n(OLD['aniso_q50'], 1), TRUST_W=W(T12, 'Old loss, update size capped'), ANCHOR_W=W(T12, 'Old loss, others held still within an update'), POW2_W=W(T12, 'Equal runs, worst run of a line weighted (power 2)'), TOP_W=W(T12, 'Equal runs, only the worst 5 % of runs pushed'),
             EQ_Q=n(EQ.get('wall_r2_quadrants'), 3), REC_Q=n(R['recipe'].get('r2_quadrants'), 3), EQ_B=n(EQ.get('wall_r2_band'), 3))
    v.update(LINE_W=span(LINE.get('wall_r2_range'), 3), LINE_Q=span(LINE.get('wall_r2_quadrants_range'), 3), LINE_B=span(LINE.get('wall_r2_band_range'), 3), LINE_ZQ=n(LINE['zone_q05']), LINE_ZM=n(LINE['zone_q50']), LINE_AN=n(LINE['aniso_q50']), REC_W=n(R['recipe']['wall_mean'], 3), REC_B=n(R['recipe'].get('r2_band'), 3), REC_ZQ=n(rec['zone_q05']), REC_ZM=n(rec['zone_q50']), LINE_SP=n(LINE['spacing_cv']))
    k6, k12 = T12.get('Long runs pushed, stiff near walls, stretch bound 6'), T12.get('Long runs pushed, stiff near walls, stretch bound 12')
    v['LINE_BOUND'] = f"A bound of 6 gives {n(k6.get('wall_r2'), 3)} on its one seed, with the most even wall zones of any map here (spread {n(k6['zone_log_std'])})." if k6 else ''
    v['KNONE_W'] = W(T12, 'Only long runs pushed, target from run count, stiff near walls')
    v.update(K6_W=n((k6 or {}).get('wall_r2'), 3), K8_W=n(LINE.get('wall_r2'), 3) + ' on average', K12_W=n((k12 or {}).get('wall_r2'), 3))
    v.update(CG_ZM=n(C12['zone_q50']), NF_ZM=n(NF['zone_q50']), CG_Q=n(C12.get('wall_r2_quadrants'), 3), NF_Q=n(NF.get('wall_r2_quadrants'), 3), COMPARE=compare(S, T12, SC))
    v['CG_MIN'] = 'seconds, plus about a minute for the measurements'
    fr = [abs(x['r2'] - x['r2_identity']) for x in S['fresh']]
    v['FRESH'] = n(max(fr), 3) if fr else '–'
    v.update(G12_SP_ID=n(C12['identity']['spacing_cv']), G12_SP=n(C12['spacing_cv']), G12_CR_ID=n(C12['identity'].get('crossing_cv')), G12_CR=n(C12.get('crossing_cv')), P10_SP_ID=n(C10['identity']['spacing_cv']), P10_SP=n(C10['spacing_cv']))
    l = rot.get('lines') or {}
    v.update(ROT_SEP_ZM=n(rot['sep']['zone_q50']), ROT_SEP_W=n(rot['sep'].get('wall_r2'), 3), ROT_ZM=n(rot['carto']['zone_q50']), ROT_W=n(rot['carto'].get('wall_r2'), 3), ROT_LINE_ZM=n(l.get('zone_q50')), ROT_LINE_W=n(l.get('wall_r2'), 3))
    la = rot.get('lines_any')
    v['ROT_ANY'] = f"Reading lines at every angle does not rescue it ({n(la.get('zone_q50'))} and {n(la.get('wall_r2'), 3)})." if la and (la.get('wall_r2') or 0) < (rot['carto'].get('wall_r2') or 0) - 0.05 else (f"Reading lines at every angle helps ({n(la.get('zone_q50'))} and {n(la.get('wall_r2'), 3)})." if la else '')
    iw = lambda k: n(SC[k]['identity'].get('wall_r2'), 3)
    cw = lambda k: n(SC[k]['carto'].get('wall_r2'), 3)
    v.update(ROOM_ID=iw('rooms'), ROOM_W=cw('rooms'), GIB13_ID=iw('gib13'), GIB13_W=cw('gib13'), P10_ID=iw('p10'), P10_W=cw('p10'))
    v.update(SE4_ID=iw('se2g4'), SE4_W=cw('se2g4'), SE12_ID=iw('se2g12'), SE12_W=cw('se2g12'), GIB1_ID=iw('gib1'), GIB1_W=cw('gib1'), GIB0_ID=iw('gib0'), GIB0_W=cw('gib0'), GIB0_ZQ=n(SC['gib0']['carto']['zone_q05']))
    rng = FL['r2_range']
    v.update(FLOW_R2_LO=n(rng[0], 3), FLOW_R2_HI=n(rng[1], 3), FLOW_ZQ=n(FL['zone_q05']), FLOW_ZM=n(FL['zone_q50']), FLOW_MIN=f"{FL['minutes']:.0f} minutes", FLOW_W=n(FL.get('wall_r2'), 3), CG_W=n(C12.get('wall_r2'), 3), FLOW_B=n(FL.get('wall_r2_band'), 3), CG_B=n(C12.get('wall_r2_band'), 3), FLOW_DEEP=W(T12, 'Learned flow, 10 layers, 256 wide'))
    lo = min(a, key=lambda r: r['r2'])
    v.update(ARM_ID=n(a[0]['r2_identity']), ARM_LO=n(lo['r2']), ARM_AN=n(lo['aniso_q50'], 0), ARM_CR=n(a[0]['identity']['crossing_cv']))
    c = S['calibration']
    v.update(CAL_PLUS=rk(c['plus']['wall']['rank']), CAL_M10=rk(c['m10']['wall']['rank']), CAL_C22=rk(c['c22']['wall']['rank']), CALR_PLUS=rk(c['plus']['r2']['rank']), CALR_M10=rk(c['m10']['r2']['rank']), CALR_C22=rk(c['c22']['r2']['rank']))
    return v
