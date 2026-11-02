// Checks the viewer's JavaScript geometry against the experiment's Python clearances stored by export.py.
//   node experiments/face_speed/visualizer/check.cjs
const fs = require('fs'), path = require('path');
const A = require('./geometry.js');
const G = JSON.parse(fs.readFileSync(path.join(__dirname, 'geometry.json'), 'utf8'));
const M = A.build(G), ref = G.reference; let worst = {}, ok = true;
ref.q.forEach((q, n) => {
  const e = A.evaluate(M, q);
  for (const cover of ['spheres', 'links', 'faces']) {
    const r = ref[cover]; if (JSON.stringify(r.names) !== JSON.stringify(e.pieces[cover].names)) { console.log('NAME MISMATCH', cover); ok = false; }
    r.pieces[n].forEach((v, i) => { const d = Math.abs(v-e.pieces[cover].d[i]); if (!(d <= (worst[cover] ?? 0))) worst[cover] = d; });
  }
  const d = Math.hypot(...e.ee.map((x, i) => x-ref.ee[n][i])); if (!(d <= (worst.ee ?? 0))) worst.ee = d;
});
console.log('configurations', ref.q.length, 'worst |js - python| (m):', worst);
for (const v of Object.values(worst)) if (!(v < 2e-5)) ok = false;
console.log(ok ? 'CHECK PASSED' : 'CHECK FAILED'); process.exit(ok ? 0 : 1);
