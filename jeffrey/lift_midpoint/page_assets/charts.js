// Charts of the lift + midpoint findings page. Plain SVG, data from the embedded D object, colors from CSS tokens.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const D = window.LM_DATA;
  const ARM_VAR = { ntrl: '--s1', lift: '--s2', liftmid: '--s3', mid: '--s4', drive: '--s5', liftpush: '--ref' };
  const SHAPE = { ntrl: 'circle', lift: 'square', liftmid: 'diamond', mid: 'triangle', drive: 'tridown', liftpush: 'ring' };
  const tip = document.createElement('div');
  tip.className = 'tip';
  tip.hidden = true;
  document.body.appendChild(tip);

  function mk(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, anchor) {
    const t = mk('text', { x: x, y: y, class: cls || 'ax', 'text-anchor': anchor || 'start' }, parent);
    t.textContent = s;
    return t;
  }
  function fmt(v, d) { return v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(d === undefined ? 1 : d); }
  function showTip(ev, html) {
    tip.innerHTML = html;
    tip.hidden = false;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
    tip.style.left = Math.max(8, x) + 'px';
    tip.style.top = Math.max(8, y) + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function hover(node, html) {
    node.addEventListener('mousemove', ev => showTip(ev, html));
    node.addEventListener('mouseleave', hideTip);
    node.addEventListener('focus', ev => { const b = node.getBoundingClientRect(); showTip({ clientX: b.right, clientY: b.top }, html); });
    node.addEventListener('blur', hideTip);
    node.setAttribute('tabindex', '0');
  }
  function marker(parent, arm, x, y, r) {
    const st = `fill: var(${ARM_VAR[arm]}); stroke: var(--surface); stroke-width: 2`;
    let e;
    if (SHAPE[arm] === 'circle') e = mk('circle', { cx: x, cy: y, r: r, style: st }, parent);
    else if (SHAPE[arm] === 'square') e = mk('rect', { x: x - r * 0.9, y: y - r * 0.9, width: 1.8 * r, height: 1.8 * r, rx: 1.5, style: st }, parent);
    else if (SHAPE[arm] === 'diamond') e = mk('path', { d: `M${x} ${y - r * 1.25} L${x + r * 1.25} ${y} L${x} ${y + r * 1.25} L${x - r * 1.25} ${y} Z`, style: st }, parent);
    else if (SHAPE[arm] === 'triangle') e = mk('path', { d: `M${x} ${y - r * 1.2} L${x + r * 1.15} ${y + r * 0.85} L${x - r * 1.15} ${y + r * 0.85} Z`, style: st }, parent);
    else if (SHAPE[arm] === 'tridown') e = mk('path', { d: `M${x} ${y + r * 1.2} L${x + r * 1.15} ${y - r * 0.85} L${x - r * 1.15} ${y - r * 0.85} Z`, style: st }, parent);
    else e = mk('circle', { cx: x, cy: y, r: r * 0.85, style: `fill: var(--surface); stroke: var(--ref); stroke-width: 2` }, parent);
    return e;
  }
  function legend(host, arms, extra) {
    const lg = document.createElement('div');
    lg.className = 'legend';
    for (const a of arms) {
      const s = document.createElement('span');
      const svg = mk('svg', { width: 16, height: 16, viewBox: '0 0 16 16', 'aria-hidden': 'true' });
      marker(svg, a, 8, 8, 5);
      s.appendChild(svg);
      s.appendChild(document.createTextNode(D.armName[a]));
      lg.appendChild(s);
    }
    for (const [cls, label] of extra || []) {
      const s = document.createElement('span');
      s.innerHTML = `<i class="${cls}"></i>${label}`;
      lg.appendChild(s);
    }
    host.appendChild(lg);
  }
  function width(host) { return Math.max(320, Math.min(1080, host.clientWidth || 800)); }
  function present(scene, arms) { return arms.filter(a => D.rows[scene] && D.rows[scene][a]); }

  // 1. queries solved: grouped horizontal bars, lift + push as a reference ring
  function solved(host) {
    const arms = D.arms, W = width(host), narrow = W < 560;
    const left = narrow ? 104 : 196, right = 46, top = 24, barH = 9, gap = 2, groupGap = 16;
    const groups = D.order.map(s => ({ s, arms: present(s, arms) }));
    let H = top;
    groups.forEach(g => { g.y = H; H += g.arms.length * (barH + gap) - gap + groupGap; });
    H += 8;
    legend(host, arms.concat(D.rows[D.order[0]] && D.order.some(s => D.rows[s].liftpush) ? ['liftpush'] : []));
    const svg = mk('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Queries solved per scene and arm' }, host);
    const x = v => left + (W - left - right) * v / 100;
    for (const t of [0, 25, 50, 75, 100]) {
      mk('line', { x1: x(t), x2: x(t), y1: top - 6, y2: H - 10, class: t ? 'grid' : 'base' }, svg);
      txt(svg, x(t), top - 10, t + (t === 100 ? ' %' : ''), 'ax', 'middle');
    }
    for (const g of groups) {
      const r = D.rows[g.s];
      const h = g.arms.length * (barH + gap) - gap;
      txt(svg, left - 10, g.y + h / 2 + 4, narrow ? D.short[g.s] : D.title[g.s], 'lab', 'end');
      g.arms.forEach((a, i) => {
        const v = r[a].solved, y = g.y + i * (barH + gap), w = Math.max(0, x(v) - x(0));
        const p = mk('path', { d: `M${x(0)} ${y} h${Math.max(0, w - 4)} a4 4 0 0 1 4 4 v${barH - 8} a4 4 0 0 1 -4 4 h${-Math.max(0, w - 4)} Z`, style: `fill: var(${ARM_VAR[a]})` }, svg);
        const hit = mk('rect', { x: left, y: y - 1, width: W - left - right, height: barH + 2, class: 'hit' }, svg);
        const html = `<div class="h">${D.title[g.s]}</div><div class="r"><b>${fmt(v)} %</b> solved · ${D.armName[a]}</div>` +
          `<div class="n">stalled ${fmt(r[a].stall)} % · through a wall ${fmt(r[a].wall)} % · ${r[a].success[0]} / ${r[a].success[1]} runs</div>`;
        hover(hit, html);
        if (a === 'liftmid') txt(svg, x(v) + 6, y + barH - 1, fmt(v), 'val');
      });
      if (r.liftpush) {
        const cy = g.y + h / 2;
        mk('line', { x1: x(r.liftpush.solved), x2: x(r.liftpush.solved), y1: g.y - 3, y2: g.y + h + 3, class: 'refline' }, svg);
        const m = marker(svg, 'liftpush', x(r.liftpush.solved), g.y - 6, 4);
        hover(m, `<div class="h">${D.title[g.s]}</div><div class="r"><b>${fmt(r.liftpush.solved)} %</b> solved · lift + push (reference)</div>`);
      }
    }
  }

  // 2. dot plots by scene (line of sight T/T*, detour index)
  function dots(host, key, opts) {
    const arms = D.arms, W = width(host), narrow = W < 560;
    const left = narrow ? 104 : 196, right = 24, top = 28, rowH = 26;
    const vals = [];
    D.order.forEach(s => present(s, arms).forEach(a => { const v = D.rows[s][a][key]; if (v !== null && v !== undefined) vals.push(v); }));
    let lo = Math.min(opts.lo, ...vals), hi = Math.max(opts.hi, ...vals);
    const pad = (hi - lo) * 0.04; lo -= pad; hi += pad;
    const H = top + D.order.length * rowH + 10;
    legend(host, arms);
    const svg = mk('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opts.label }, host);
    const x = v => left + (W - left - right) * (v - lo) / (hi - lo);
    const ticks = opts.ticks.filter(t => t >= lo && t <= hi);
    for (const t of ticks) {
      mk('line', { x1: x(t), x2: x(t), y1: top - 8, y2: H - 8, class: opts.refs.includes(t) ? 'base' : 'grid' }, svg);
      txt(svg, x(t), top - 12, opts.tickFmt ? opts.tickFmt(t) : String(t), 'ax', 'middle');
    }
    for (const [v, s] of opts.refNotes || []) if (v >= lo && v <= hi) txt(svg, x(v) + 4, H - 1, s, 'note');
    D.order.forEach((s, i) => {
      const y = top + i * rowH + rowH / 2;
      mk('line', { x1: left, x2: W - right, y1: y, y2: y, class: 'row' }, svg);
      txt(svg, left - 10, y + 4, narrow ? D.short[s] : D.title[s], 'lab', 'end');
      const pts = present(s, arms).map(a => ({ a, v: D.rows[s][a][key] })).filter(p => p.v !== null && p.v !== undefined);
      const lm = pts.find(p => p.a === 'liftmid'), la = pts.find(p => p.a === 'lift');
      if (lm && la) mk('line', { x1: x(la.v), x2: x(lm.v), y1: y, y2: y, class: 'link' }, svg);
      for (const p of pts) {
        const m = marker(svg, p.a, x(p.v), y, 5);
        hover(m, `<div class="h">${D.title[s]}</div><div class="r"><b>${fmt(p.v, 2)}</b> ${opts.unit} · ${D.armName[p.a]}</div>`);
      }
    });
  }

  // 3. how runs end: stacked bars (solved / stalled / through a wall) per arm
  function fails(host) {
    const arms = D.arms, W = width(host), narrow = W < 560;
    const left = narrow ? 150 : 350, right = 16, top = 24, barH = 10, gap = 3, groupGap = 14;
    const lg = document.createElement('div');
    lg.className = 'legend';
    lg.innerHTML = '<span><i class="sw ok"></i>solved</span><span><i class="sw stall"></i>did not arrive (stalled)</span><span><i class="sw wall"></i>arrived through a wall</span>';
    host.appendChild(lg);
    const groups = D.order.map(s => ({ s, arms: present(s, arms) }));
    let H = top;
    groups.forEach(g => { g.y = H; H += g.arms.length * (barH + gap) - gap + groupGap; });
    const svg = mk('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'How the runs end, per scene and arm' }, host);
    const x = v => left + (W - left - right) * v / 100;
    for (const t of [0, 50, 100]) { mk('line', { x1: x(t), x2: x(t), y1: top - 6, y2: H - 10, class: t ? 'grid' : 'base' }, svg); txt(svg, x(t), top - 10, t + ' %', 'ax', 'middle'); }
    for (const g of groups) {
      const h = g.arms.length * (barH + gap) - gap;
      txt(svg, 6, g.y + h / 2 + 4, narrow ? D.short[g.s] : D.title[g.s], 'lab', 'start');
      g.arms.forEach((a, i) => {
        const r = D.rows[g.s][a], y = g.y + i * (barH + gap);
        const legW = narrow ? 64 : 168;
        const ky = y + barH / 2;
        marker(svg, a, left - legW - 2, ky, 3.6);
        if (!narrow) txt(svg, left - legW + 6, ky + 4, D.armName[a], 'small', 'start');
        const segs = [['ok', r.solved], ['stall', r.stall], ['wall', r.wall]];
        let acc = 0;
        segs.forEach(([cls, v], k) => {
          if (v === null || v === undefined || v <= 0) return;
          const x0 = x(acc) + (acc > 0 ? 1 : 0), x1 = x(acc + v) - 1;
          if (x1 > x0) mk('rect', { x: x0, y: y, width: x1 - x0, height: barH, class: 'seg ' + cls }, svg);
          acc += v;
        });
        const hit = mk('rect', { x: left, y: y - 1, width: W - left - right, height: barH + 2, class: 'hit' }, svg);
        hover(hit, `<div class="h">${D.title[g.s]} · ${D.armName[a]}</div><div class="r">solved <b>${fmt(r.solved)} %</b></div><div class="r">stalled <b>${fmt(r.stall)} %</b></div><div class="r">through a wall <b>${fmt(r.wall)} %</b></div>`);
      });
    }
  }

  // 4. small multiples: one panel per scene, lines over categories
  function multiples(host, opts) {
    const grid = document.createElement('div');
    grid.className = 'multi';
    legend(host, opts.arms, opts.extraLegend);
    host.appendChild(grid);
    for (const s of D.order) {
      const series = opts.series(s).filter(sr => sr.pts.length);
      if (!series.length || (opts.only && !opts.only(s))) continue;
      let yhi = opts.yhi, yticks = opts.yticks;
      if (yhi === 'auto') {
        const m = Math.max(1, ...series.flatMap(sr => sr.pts.map(p => p[1])));
        const step = m <= 2 ? 0.5 : m <= 5 ? 1 : m <= 10 ? 2 : m <= 25 ? 5 : m <= 50 ? 10 : 25;
        yhi = Math.ceil(m / step) * step;
        yticks = [0, yhi / 2, yhi];
      }
      const panel = document.createElement('div');
      panel.className = 'panel';
      panel.innerHTML = `<h4>${D.title[s]}</h4>`;
      grid.appendChild(panel);
      const W = 320, H = 190, l = 34, r = 10, t = 10, b = 30;
      const svg = mk('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${opts.label}, ${D.title[s]}` }, panel);
      const cats = opts.cats(s);
      const xs = i => l + (W - l - r) * (cats.length === 1 ? 0.5 : i / (cats.length - 1));
      const y = v => t + (H - t - b) * (1 - (v - opts.ylo) / (yhi - opts.ylo));
      for (const v of yticks) { mk('line', { x1: l, x2: W - r, y1: y(v), y2: y(v), class: v === opts.ylo ? 'base' : 'grid' }, svg); txt(svg, l - 6, y(v) + 4, String(v), 'ax', 'end'); }
      cats.forEach((c, i) => { if (opts.showCat(i, cats)) txt(svg, xs(i), H - b + 16, opts.catLabel(c), 'ax', 'middle'); });
      for (const sr of series) {
        if (sr.flat !== undefined) {
          mk('line', { x1: l, x2: W - r, y1: y(sr.flat), y2: y(sr.flat), class: 'flat', style: `stroke: var(${ARM_VAR[sr.arm]})` }, svg);
          continue;
        }
        const P = sr.pts.map(p => [xs(cats.indexOf(p[0])), y(p[1]), p]).filter(p => p[0] >= 0 && !Number.isNaN(p[1]));
        if (P.length > 1) mk('path', { d: 'M' + P.map(p => p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' L'), class: 'ln', style: `stroke: var(${ARM_VAR[sr.arm]})` }, svg);
        for (const p of P) {
          const m = marker(svg, sr.arm, p[0], p[1], 4);
          hover(m, `<div class="h">${D.title[s]} · ${opts.catLabel(p[2][0])}${opts.catUnit || ''}</div><div class="r"><b>${fmt(p[2][1])}</b> ${opts.unit} · ${D.armName[sr.arm]}</div>`);
        }
      }
    }
  }

  const builders = {
    solved: h => solved(h),
    los: h => dots(h, 'los_ratio', { lo: 0.5, hi: 1.2, ticks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], refs: [1.0], unit: 'learned / true time', label: 'Line-of-sight travel times, learned over true' }),
    rho: h => dots(h, 'rho', { lo: -1, hi: 1.5, ticks: [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], refs: [0, 1], unit: 'detour index', label: 'Detour index (0 = obstacle seen through, 1 = true detour)', refNotes: [[0, 'see-through'], [1, 'truth']] }),
    fails: h => fails(h),
    bins: h => multiples(h, {
      arms: D.arms, label: 'Queries solved by true distance', unit: '% solved', ylo: 0, yhi: 100, yticks: [0, 50, 100], catUnit: ' cells',
      cats: s => D.bins[s], catLabel: c => c, showCat: (i, cats) => cats.length <= 6 || i % 2 === 0,
      series: s => present(s, D.arms).map(a => ({ arm: a, pts: D.bins[s].map(b => [b, D.rows[s][a].by_bin[b]]).filter(p => p[1] !== undefined && p[1] !== null) })),
    }),
    course: h => multiples(h, {
      arms: ['lift', 'liftmid', 'mid', 'drive'], label: 'Queries solved during training, planning seed 0', unit: '% solved', ylo: 0, yhi: 100, yticks: [0, 50, 100], catUnit: '',
      extraLegend: [['flatkey', 'ntrl-demo at epoch 5,000']],
      cats: () => [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000], catLabel: c => String(c), showCat: (i) => i % 2 === 0,
      series: s => {
        const r = D.rows[s], out = [];
        if (r.ntrl) out.push({ arm: 'ntrl', flat: r.ntrl.seed0, pts: [[0, 0]] });
        if (r.lift && r.lift.curve) out.push({ arm: 'lift', pts: r.lift.curve });
        if (r.liftmid && r.liftmid.curve) out.push({ arm: 'liftmid', pts: r.liftmid.curve });
        if (r.mid && r.mid.curve) out.push({ arm: 'mid', pts: r.mid.curve.filter(p => p[0] >= 2000) });
        if (r.drive && r.drive.curve) out.push({ arm: 'drive', pts: r.drive.curve });
        return out;
      },
    }),
    detours: h => multiples(h, {
      arms: ['liftmid', 'mid', 'drive'], label: 'Training drives that reach their goal along a detour', unit: '% of 8,192 drives', ylo: 0, yhi: 'auto', catUnit: '',
      cats: s => D.refreshEpochs, catLabel: c => String(c), showCat: (i, cats) => i % 10 === 0 || i === cats.length - 1, only: s => ['liftmid', 'mid', 'drive'].some(a => D.rows[s][a] && D.rows[s][a].detour_drives),
      series: s => ['liftmid', 'mid', 'drive'].filter(a => D.rows[s][a] && D.rows[s][a].detour_drives).map(a => ({ arm: a, pts: D.rows[s][a].detour_drives })),
    }),
  };
  document.querySelectorAll('[data-chart]').forEach(h => {
    const b = builders[h.dataset.chart];
    if (b) b(h);
  });
})();
