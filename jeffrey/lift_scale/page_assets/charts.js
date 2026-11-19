(function () {
  const DATA = JSON.parse(document.getElementById('data').textContent);
  const tip = document.getElementById('tip');
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, attrs) {
    const t = el('text', Object.assign({ x: x, y: y, fill: css('--ink-2'), 'font-size': 12 }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }
  function fmt(v, d) { if (v == null || !isFinite(v)) return '–'; return (+v).toFixed(d == null ? 3 : d); }
  function showTip(evt, rows) {
    tip.replaceChildren();
    rows.forEach(function (r) {
      const row = document.createElement('div'); row.className = 'row';
      if (r.color) { const k = document.createElement('span'); k.className = 'k'; k.style.background = r.color; row.appendChild(k); }
      const v = document.createElement('span'); v.className = 'v'; v.textContent = r.value; row.appendChild(v);
      const n = document.createElement('span'); n.className = 'n'; n.textContent = r.name; row.appendChild(n);
      tip.appendChild(row);
    });
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = evt.clientX + 14, y = evt.clientY + 14;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - 14;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  function hideTip() { tip.hidden = true; }

  // Horizontal bars: rows grouped by scene; bar = value (R); colour = series; reference ticks per group.
  function hbar(box, spec) {
    const W = Math.max(320, box.clientWidth - 28);
    const narrow = W < 560;
    const labW = narrow ? Math.min(150, W * 0.42) : 250, padR = 58, rowH = 22, grpH = 26;
    let n = 0; spec.groups.forEach(function (g) { n += g.rows.length; });
    const H = 30 + n * rowH + spec.groups.length * grpH + 8;
    const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': spec.title || 'bar chart' });
    const x0 = labW, x1 = W - padR;
    const xmax = spec.xmax, xmin = spec.xmin || 0;
    const X = function (v) { return x0 + (Math.min(v, xmax) - xmin) / (xmax - xmin) * (x1 - x0); };
    const g0 = el('g', {}, svg);
    const ticks = spec.ticks || [];
    ticks.forEach(function (t) {
      el('line', { x1: X(t), x2: X(t), y1: 18, y2: H - 6, stroke: css('--grid'), 'stroke-width': 1 }, g0);
      txt(g0, X(t), 12, String(t), { 'text-anchor': 'middle', 'font-size': 11, fill: css('--ink-3') });
    });
    if (spec.unit) {
      el('line', { x1: X(1), x2: X(1), y1: 18, y2: H - 6, stroke: css('--ink-3'), 'stroke-width': 1 }, g0);
    }
    let y = 24;
    spec.groups.forEach(function (g) {
      txt(svg, 0, y + 16, g.name, { 'font-weight': 600, fill: css('--ink'), 'font-size': 12.5 });
      const gy0 = y + grpH - 4;
      y += grpH;
      g.rows.forEach(function (r) {
        const c = css('--s' + (r.slot || 1));
        const lab = txt(svg, labW - 8, y + rowH / 2 + 4, r.label, { 'text-anchor': 'end', 'font-size': narrow ? 11 : 12, fill: css('--ink-2') });
        if (narrow && r.label.length > 22) { lab.textContent = r.label.slice(0, 21) + '…'; }
        if (r.value != null && isFinite(r.value)) {
          const xb = X(Math.max(r.value, xmin)), w = Math.max(1, xb - x0);
          const bh = 14, by = y + (rowH - bh) / 2;
          const path = 'M' + x0 + ',' + by + 'h' + (w - 4) + 'a4,4 0 0 1 4,4v' + (bh - 8) + 'a4,4 0 0 1 -4,4h-' + (w - 4) + 'z';
          const bar = el('path', { d: w > 5 ? path : ('M' + x0 + ',' + by + 'h' + w + 'v' + bh + 'h-' + w + 'z'), fill: c, opacity: r.invalid ? 0.35 : 1 }, svg);
          const dg = spec.digits == null ? 3 : spec.digits;
          const val = txt(svg, xb + 6, y + rowH / 2 + 4, (r.value > xmax ? '>' : '') + fmt(r.value, dg) + (spec.unit_suffix || '') + (r.invalid ? ' invalid' : ''), { 'font-size': 11.5, fill: css('--ink'), 'font-family': css('--font-mono') });
          const hit = el('rect', { x: 0, y: y, width: W, height: rowH, fill: 'transparent' }, svg);
          const rows = [{ color: c, value: (spec.value_name || 'R') + ' = ' + fmt(r.value, dg + 1) + (spec.unit_suffix || ''), name: r.label }];
          if (r.note) rows.push({ value: '', name: r.note });
          hit.addEventListener('pointermove', function (e) { bar.setAttribute('opacity', 0.75); showTip(e, rows); });
          hit.addEventListener('pointerleave', function () { bar.setAttribute('opacity', r.invalid ? 0.35 : 1); hideTip(); });
        } else {
          txt(svg, x0 + 4, y + rowH / 2 + 4, r.missing || 'not available', { 'font-size': 11.5, fill: css('--ink-3') });
        }
        y += rowH;
      });
      (g.refs || []).forEach(function (rf) {
        if (rf.value == null || !isFinite(rf.value) || rf.value > xmax) return;
        const xr = X(rf.value);
        el('line', { x1: xr, x2: xr, y1: gy0, y2: y, stroke: css('--ref'), 'stroke-width': 1.6, 'stroke-dasharray': rf.dash || '4 3' }, svg);
        const t = txt(svg, xr + 3, gy0 + 9, rf.label, { 'font-size': 10.5, fill: css('--ink-2') });
        if (xr + 3 + rf.label.length * 5.6 > W) { t.setAttribute('x', xr - 3); t.setAttribute('text-anchor', 'end'); }
      });
    });
    return svg;
  }

  // Lines: series of [x, y]; optional log x; reference horizontals; crosshair readout snapping to the nearest x.
  function line(box, spec) {
    const W = Math.max(320, box.clientWidth - 28), H = spec.height || 300;
    const m = { l: 52, r: 18, t: spec.ylabel ? 30 : 12, b: 38 };
    const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': spec.title || 'line chart' });
    const xs = [], ys = [];
    spec.series.forEach(function (s) { s.points.forEach(function (p) { xs.push(p[0]); ys.push(p[1]); }); });
    (spec.refs || []).forEach(function (r) { ys.push(r.y); });
    const lx = function (v) { return spec.logx ? Math.log10(v) : v; };
    const xa = spec.xmin != null ? spec.xmin : Math.min.apply(null, xs), xb = spec.xmax != null ? spec.xmax : Math.max.apply(null, xs);
    const ya = spec.ymin != null ? spec.ymin : Math.min.apply(null, ys), yb = spec.ymax != null ? spec.ymax : Math.max.apply(null, ys);
    const ly = function (v) { return spec.logy ? Math.log10(Math.max(v, 1e-9)) : v; };
    const X = function (v) { return m.l + (lx(v) - lx(xa)) / (lx(xb) - lx(xa)) * (W - m.l - m.r); };
    const Y = function (v) { return H - m.b - (ly(Math.min(Math.max(v, ya), yb)) - ly(ya)) / (ly(yb) - ly(ya)) * (H - m.t - m.b); };
    (spec.yticks || []).forEach(function (t) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: css('--grid'), 'stroke-width': 1 }, svg);
      txt(svg, m.l - 6, Y(t) + 4, String(t), { 'text-anchor': 'end', 'font-size': 11, fill: css('--ink-3') });
    });
    (spec.xticks || []).forEach(function (t) {
      const lab = (spec.xticklabels && spec.xticklabels[String(t)] != null) ? spec.xticklabels[String(t)] : String(t);
      txt(svg, X(t), H - m.b + 16, lab, { 'text-anchor': 'middle', 'font-size': 11, fill: css('--ink-3') });
    });
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css('--ink-3'), 'stroke-width': 1 }, svg);
    if (spec.xlabel) txt(svg, (m.l + W - m.r) / 2, H - 4, spec.xlabel, { 'text-anchor': 'middle', 'font-size': 12 });
    if (spec.ylabel) txt(svg, 4, 13, spec.ylabel, { 'font-size': 12 });
    (spec.refs || []).forEach(function (r) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), stroke: css('--ref'), 'stroke-width': 1.4, 'stroke-dasharray': '5 4' }, svg);
      if (r.label) txt(svg, W - m.r - 2, Y(r.y) - 5, r.label, { 'text-anchor': 'end', 'font-size': 11, fill: css('--ink-2') });
    });
    spec.series.forEach(function (s) {
      const c = css('--s' + s.slot);
      const d = s.points.map(function (p, i) { return (i ? 'L' : 'M') + X(p[0]) + ',' + Y(p[1]); }).join('');
      if (!spec.scatter) el('path', { d: d, fill: 'none', stroke: c, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dash || '' }, svg);
      if (spec.dots !== false && s.dots !== false) s.points.forEach(function (p) { el('circle', { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: c, stroke: css('--surface'), 'stroke-width': 2 }, svg); });
      if (s.endlabel && s.points.length) { const p = s.points[s.points.length - 1]; txt(svg, X(p[0]) - 4, Y(p[1]) - 7, s.endlabel, { 'text-anchor': 'end', 'font-size': 11, fill: css('--ink-2') }); }
    });
    if (spec.scatter) {
      const hitS = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
      hitS.addEventListener('pointermove', function (e) {
        const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width, py = (e.clientY - r.top) * H / r.height;
        let best = null, bd = Infinity;
        spec.series.forEach(function (s) { s.points.forEach(function (p) { const d = Math.hypot(X(p[0]) - px, Y(p[1]) - py); if (d < bd) { bd = d; best = [s, p]; } }); });
        if (best && bd < 30) showTip(e, [{ color: css('--s' + best[0].slot), value: fmt(best[1][1], spec.digits || 3), name: (best[1][2] || best[0].name) + ' (' + (spec.xname || 'x') + ' ' + fmt(best[1][0], 3) + ')' }]);
        else hideTip();
      });
      hitS.addEventListener('pointerleave', hideTip);
      return svg;
    }
    const allx = Array.from(new Set(xs)).sort(function (a, b) { return a - b; });
    const cross = el('line', { y1: m.t, y2: H - m.b, stroke: css('--ink-3'), 'stroke-width': 1, visibility: 'hidden' }, svg);
    const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', function (e) {
      const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
      let best = allx[0], bd = Infinity;
      allx.forEach(function (v) { const d = Math.abs(X(v) - px); if (d < bd) { bd = d; best = v; } });
      cross.setAttribute('x1', X(best)); cross.setAttribute('x2', X(best)); cross.setAttribute('visibility', 'visible');
      const bl = (spec.xticklabels && spec.xticklabels[String(best)] != null) ? spec.xticklabels[String(best)] : best;
      const rows = [{ value: (spec.xname || 'x') + ' = ' + bl, name: '' }];
      spec.series.forEach(function (s) {
        const p = s.points.find(function (q) { return q[0] === best; });
        if (p) rows.push({ color: css('--s' + s.slot), value: fmt(p[1], spec.digits || 4), name: s.name });
      });
      showTip(e, rows);
    });
    hit.addEventListener('pointerleave', function () { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    return svg;
  }

  function legend(box, spec) {
    const items = spec.legend || [];
    if (!items.length) return;
    const lg = document.createElement('div'); lg.className = 'legend';
    items.forEach(function (it) {
      const s = document.createElement('span');
      const i = document.createElement('i');
      if (it.ref) { i.className = 'ref'; }
      else { i.style.background = css('--s' + it.slot); if (spec.type === 'line') i.className = 'line'; }
      s.appendChild(i); s.appendChild(document.createTextNode(it.name)); lg.appendChild(s);
    });
    box.appendChild(lg);
  }

  function render() {
    document.querySelectorAll('.chart[data-chart]').forEach(function (box) {
      const spec = DATA.charts[box.dataset.chart];
      if (!spec) return;
      box.replaceChildren();
      const t = document.createElement('div'); t.className = 'ctitle'; t.textContent = spec.title; box.appendChild(t);
      if (spec.sub) { const s = document.createElement('div'); s.className = 'csub'; s.textContent = spec.sub; box.appendChild(s); }
      legend(box, spec);
      box.appendChild(spec.type === 'line' ? line(box, spec) : hbar(box, spec));
    });
  }
  render();
  let rt = null;
  window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(render, 120); });
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  if (mq.addEventListener) mq.addEventListener('change', render);
  new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
