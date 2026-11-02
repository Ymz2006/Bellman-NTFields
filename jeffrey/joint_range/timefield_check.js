// Offline check of the time-field page's JavaScript (the <script id="core"> block of timefield_template.html) against
// the PyTorch / SciPy values stored by timefield.py (meta.json) and timefield_lines.py (lines.json): network forward
// pass, travel time and field speed (input gradient) at both pair ends, Dijkstra on the grid graph, forward kinematics,
// and the page's lines through a picked pose: ground-truth time and speed at every line point, learned T and speed.
//   gjs jeffrey/joint_range/timefield_check.js jeffrey/joint_range/timefield_viewer jeffrey/joint_range/timefield_template.html
//   gjs jeffrey/joint_range/timefield_check.js jeffrey/joint_range/test4_viewer/<scene> jeffrey/joint_range/test4_template.html   (adds the arm spheres)
const GLib = imports.gi.GLib;
const [dir, tpl] = ARGV;
const read = p => { const [ok, b] = GLib.file_get_contents(p); if (!ok) throw new Error('cannot read ' + p); return b; };
const text = p => new TextDecoder().decode(read(p));
const f32 = p => new Float32Array(new Uint8Array(read(p)).slice().buffer);
(0, eval)(text(tpl).match(/<script id="core">([\s\S]*?)<\/script>/)[1]);

const meta = JSON.parse(text(dir + '/meta.json')), LJ = JSON.parse(text(dir + '/lines.json')), N = meta.nodes, K = meta.dirs.length, [NX, NY, NZ] = meta.grid.shape;
const A = f32(dir + '/nodes.bin'), Qr = new Float32Array(6*N), D = new Float32Array(N), X = new Float32Array(6*N), GI = new Int32Array(3*N), G2N = new Int32Array(NX*NY*NZ).fill(-1);
for (let i = 0; i < N; i++) { for (let j = 0; j < 6; j++) { Qr[6*i+j] = A[11*i+3+j]; X[6*i+j] = A[11*i+3+j]/meta.q_range[j]; } D[i] = A[11*i+9];
  for (let a = 0; a < 3; a++) GI[3*i+a] = Math.round(A[11*i+a]); G2N[(GI[3*i]*NY+GI[3*i+1])*NZ+GI[3*i+2]] = i; }
const xs = i => X.subarray(6*i, 6*i+6); let fail = false;
const check = (name, value, limit) => { const ok = value <= limit; if (!ok) fail = true; print(`${ok ? 'OK  ' : 'FAIL'} ${name}: ${value.toExponential(2)} (limit ${limit})`); };
const nets = {};
for (const [key, c] of Object.entries(meta.checkpoints)) {
  const net = nets[key] = unpackNet(f32(`${dir}/net_${key}.bin`)), v = c.validation;
  let eT = 0, eS1 = 0, eS0 = 0, eHead = 0;
  meta.validation.pairs.forEach(([a, b], k) => {
    const ea = forward(net, xs(a)).e, eb = forward(net, xs(b)).e, r1 = fieldAt(net, ea, xs(b));
    eT = Math.max(eT, Math.abs(metric(ea, 0, eb, 0)-v.T[k])/v.T[k], Math.abs(r1[0]-v.T[k])/v.T[k]);
    eS1 = Math.max(eS1, Math.abs(r1[1]-v.speed1[k])/v.speed1[k]);
    eS0 = Math.max(eS0, Math.abs(fieldAt(net, eb, xs(a))[1]-v.speed0[k])/v.speed0[k]);
    if (k === 0) for (let j = 0; j < 8; j++) eHead = Math.max(eHead, Math.abs(eb[j]-v.e_head[j]));
  });
  check(`${key} embedding head, abs`, eHead, 1e-4); check(`${key} travel time, rel`, eT, 1e-4); check(`${key} speed at x1, rel`, eS1, 1e-3); check(`${key} speed at x0, rel`, eS0, 1e-3);
}
const E = f32(dir + '/edges.bin'), NBR = new Int32Array(N*K).fill(-1);
for (let i = 0; i < N; i++) for (let k = 0; k < K; k++) { const [dx, dy, dz] = meta.dirs[k], a = GI[3*i]+dx, b = GI[3*i+1]+dy, c = GI[3*i+2]+dz;
  if (a >= 0 && a < NX && b >= 0 && b < NY && c >= 0 && c < NZ) NBR[i*K+k] = G2N[(a*NY+b)*NZ+c]; }
let t0 = Date.now(); const G = buildGraph(N, NBR, E, K), r = dijkstra(G, meta.validation.dijkstra_start); const ms = Date.now()-t0;
let eD = 0, unreach = 0;
for (const [i, d] of meta.validation.dijkstra_sample) { if (d < 0) { if (isFinite(r.dist[i])) eD = Infinity; unreach++; continue; } eD = Math.max(eD, Math.abs(r.dist[i]-d)/Math.max(d, 1e-9)); }
check('dijkstra vs scipy, rel', eD, 1e-5); print(`     graph build + dijkstra ${ms} ms, ${G.to.length} directed edges, ${unreach} unreachable samples`);
let eF = 0;
for (let t = 0; t < 60; t++) { const i = Math.floor(t*N/60), T = forwardKinematics(meta.chain, A.subarray(11*i+3, 11*i+9))[meta.arm_links[meta.arm_links.length-1]];
  eF = Math.max(eF, Math.hypot(T[3]-(meta.grid.lo[0]+GI[3*i]*meta.grid.h), T[7]-(meta.grid.lo[1]+GI[3*i+1]*meta.grid.h), T[11]-(meta.grid.lo[2]+GI[3*i+2]*meta.grid.h))); }
check('forward kinematics grasp point vs grid, m', eF, 1e-5);

// the lines through the validation pose
const KDIR = [0, 1, 2].map(a => meta.dirs.findIndex(d => d.every((v, j) => v === (j === a ? 1 : 0))));
const C = { shape: meta.grid.shape, lo: meta.grid.lo, h: meta.grid.h, sub: LJ.sub, GI, G2N, Q: Qr, D, LD: f32(dir + '/lines.bin'), E, K, kdir: KDIR, qr: meta.q_range,
            P: { alpha: meta.alpha, margin: meta.margin, offset: meta.offset } };
const V = LJ.validation, rs = dijkstra(G, V.source); let eLT = 0, eLS = 0, count = true; const lines = [];
for (const l of V.lines) { const L = lineTruth(C, buildLine(C, V.source, l.axis), rs.dist); lines[l.axis] = L; if (L.n !== l.count) count = false;
  for (let p = 0; p < Math.min(L.n, l.count); p++) { const ref = l.T[p]; if (ref < 0) { if (isFinite(L.T[p])) eLT = Infinity; continue; }
    eLT = Math.max(eLT, Math.abs(L.T[p]-ref)/Math.max(ref, 1e-9)); eLS = Math.max(eLS, Math.abs(L.S[p]-l.S[p])/l.S[p]); } }
check('line point count matches python', count ? 0 : 1, 0); check('line true time vs scipy, rel', eLT, 1e-5); check('line true speed, rel', eLS, 1e-5);
print(`     lines through pose ${V.source}: ${V.lines.map(l => l.count).join(' + ')} points`);
for (const [key, v] of Object.entries(V.learned)) { const net = nets[key], es = forward(net, xs(V.source)).e; let eT = 0, eS = 0; t0 = Date.now();
  v.points.forEach(([axis, p], i) => { const r = fieldAt(net, es, lines[axis].x.subarray(6*p, 6*p+6)); eT = Math.max(eT, Math.abs(r[0]-v.T[i])/v.T[i]); eS = Math.max(eS, Math.abs(r[1]-v.S[i])/v.S[i]); });
  check(`${key} learned T at line points, rel`, eT, 1e-4); check(`${key} learned speed at line points, rel`, eS, 1e-3); print(`     fieldAt ${((Date.now()-t0)/v.points.length).toFixed(2)} ms per point`); }
// the arm's collision spheres placed by the page's forward kinematics (pages that draw the full arm)
if (typeof sphereCenters === 'function' && meta.arm_spheres && meta.validation.spheres) {
  let eSp = 0; const V2 = meta.validation.spheres;
  V2.nodes.forEach((i, k) => { const w = sphereCenters(meta.chain, meta.arm_spheres, A.subarray(11*i+3, 11*i+9)); V2.world[k].forEach((c, s) => { for (let r = 0; r < 3; r++) eSp = Math.max(eSp, Math.abs(w[3*s+r]-c[r])); }); });
  check(`arm sphere centres vs python (${meta.arm_spheres.length} spheres), m`, eSp, 1e-5);
}
print(fail ? 'SOME CHECKS FAILED' : 'ALL CHECKS PASSED');
