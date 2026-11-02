// Collision geometry of the UR5e + Robotiq arm, a line-by-line port of experiments/face_speed/{robot,faces,clearance}.py.
// Pure functions on plain arrays (4x4 matrices row-major, length 16) so the same file runs in the page and in check.mjs.
const ArmGeometry = (() => {
  const EPS = 1e-9;
  const mul = (A, B) => { const C = new Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += A[4*i+k]*B[4*k+j]; C[4*i+j] = s; } return C; };
  const rotZ = t => { const c = Math.cos(t), s = Math.sin(t); return [c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1]; };
  const I4 = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
  const apply = (T, p) => [T[0]*p[0]+T[1]*p[1]+T[2]*p[2]+T[3], T[4]*p[0]+T[5]*p[1]+T[6]*p[2]+T[7], T[8]*p[0]+T[9]*p[1]+T[10]*p[2]+T[11]];
  const rotate = (T, v) => [T[0]*v[0]+T[1]*v[1]+T[2]*v[2], T[4]*v[0]+T[5]*v[1]+T[6]*v[2], T[8]*v[0]+T[9]*v[1]+T[10]*v[2]];
  const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
  const scl = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
  const dot = (a, b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const norm = a => Math.hypot(a[0], a[1], a[2]);
  const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

  function build(G) {
    const jointOf = {}; G.joints.forEach(j => { jointOf[j.child] = j; });
    const linkIndex = Object.fromEntries(G.collision_links.map((l, i) => [l, i]));
    const keep = G.spheres.map(s => !s.ignored);
    const faces = [];
    G.bodies.forEach((b, kb) => {
      for (let axis = 0; axis < 3; axis++) for (const sign of [1, -1]) {
        const i = (axis+1) % 3, j = (axis+2) % 3, e = k => [0,1,2].map(x => x === k ? 1 : 0);
        const center = b.center.slice(); center[axis] += sign*b.half[axis];
        faces.push({ body: kb, reference: b.reference, center, U: e(i), V: e(j), n: scl(e(axis), sign), a: b.half[i], b: b.half[j],
                     name: `${b.name}:${sign > 0 ? '+' : '-'}${'xyz'[axis]}` });
      }
    });
    const linksWithSpheres = [...new Set(G.spheres.filter(s => !s.ignored).map(s => s.link))].sort((a, b) => a - b);
    return { G, jointOf, linkIndex, keep, faces, linksWithSpheres, C: G.constants };
  }

  function transforms(M, q) {
    const memo = { base_link: I4 };
    const get = link => {
      if (memo[link]) return memo[link];
      const j = M.jointOf[link]; let T = mul(get(j.parent), j.origin);
      if (j.index >= 0) T = mul(T, rotZ(q[j.index]));
      return (memo[link] = T);
    };
    M.G.joints.forEach(j => get(j.child));
    return memo;
  }

  const pointAabb = (p, lo, hi) => Math.hypot(Math.max(lo[0]-p[0], 0)+Math.max(p[0]-hi[0], 0), Math.max(lo[1]-p[1], 0)+Math.max(p[1]-hi[1], 0), Math.max(lo[2]-p[2], 0)+Math.max(p[2]-hi[2], 0));
  function pointRect(w, f) {
    const d = sub(w, f.c), u = clamp(dot(d, f.U), -f.a, f.a), v = clamp(dot(d, f.V), -f.b, f.b);
    return norm(sub(d, add(scl(f.U, u), scl(f.V, v))));
  }
  function segSeg(p1, q1, p2, q2) {          // Ericson 5.1.9, the same branch structure as faces.segment_segment
    const d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
    const a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r), c = dot(d1, r), b = dot(d1, d2), denom = a*e-b*b;
    let s = denom > EPS ? clamp((b*f-c*e)/Math.max(denom, EPS), 0, 1) : 0;
    let t = (b*s+f)/Math.max(e, EPS);
    if (t < 0) s = clamp(-c/Math.max(a, EPS), 0, 1); else if (t > 1) s = clamp((b-c)/Math.max(a, EPS), 0, 1);
    t = clamp(t, 0, 1);
    return norm(sub(add(p1, scl(d1, s)), add(p2, scl(d2, t))));
  }
  const EX = [[1,0,0],[0,1,0],[0,0,1]];
  function satSeparation(f, lo, hi) {
    const m = scl(add(lo, hi), .5), h = scl(sub(hi, lo), .5), n = cross(f.U, f.V);
    const axes = [...EX, f.U, f.V, n]; for (const w of [f.U, f.V, n]) for (const e of EX) axes.push(cross(w, e));
    const T = sub(f.c, m); let best = -Infinity;
    for (const L0 of axes) {
      const ln = norm(L0); if (ln <= 1e-6) continue; const L = scl(L0, 1/Math.max(ln, EPS));
      const gap = Math.abs(dot(T, L)), rA = h[0]*Math.abs(L[0])+h[1]*Math.abs(L[1])+h[2]*Math.abs(L[2]);
      const rB = f.a*Math.abs(dot(f.U, L))+f.b*Math.abs(dot(f.V, L));
      best = Math.max(best, gap-rA-rB);
    }
    return best;
  }
  function rectCorners(f) {
    const A = scl(f.U, f.a), B = scl(f.V, f.b);
    return [add(add(f.c, A), B), add(sub(f.c, A), B), sub(sub(f.c, A), B), sub(add(f.c, A), B)];
  }
  function exactDistance(f, lo, hi) {
    const P = rectCorners(f); let d = Infinity;
    for (const p of P) d = Math.min(d, pointAabb(p, lo, hi));
    const W = []; for (let k = 0; k < 8; k++) W.push([k >> 2 & 1 ? hi[0] : lo[0], k >> 1 & 1 ? hi[1] : lo[1], k & 1 ? hi[2] : lo[2]]);
    for (const w of W) d = Math.min(d, pointRect(w, f));
    for (let axis = 0; axis < 3; axis++) for (let k = 0; k < 8; k++) {
      if ((k >> (2-axis)) & 1) continue; const E1 = W[k], E2 = W[k | (1 << (2-axis))];
      for (let r = 0; r < 4; r++) d = Math.min(d, segSeg(P[r], P[(r+1) % 4], E1, E2));
    }
    return d;
  }
  function rectAabbSigned(f, lo, hi, far) {
    const sep = satSeparation(f, lo, hi);
    return sep > 0 && sep < far ? exactDistance(f, lo, hi) : sep;
  }

  // One configuration (radians) -> everything the viewer draws and reports.
  function evaluate(M, q) {
    const G = M.G, C = M.C, T = transforms(M, q);
    const spheres = G.spheres.map(s => ({ c: apply(T[G.collision_links[s.link]], s.center), r: s.radius }));
    const sphereClear = spheres.map((s, i) => { let d = Infinity, k = -1; G.cuboids.forEach((cb, kc) => { const v = pointAabb(s.c, cb.lo, cb.hi)-s.r; if (v < d) { d = v; k = kc; } }); return { d, cuboid: k }; });
    let dSelf = Infinity, selfPair = null;
    for (const [i, j] of G.self_pairs) { const v = norm(sub(spheres[i].c, spheres[j].c))-spheres[i].r-spheres[j].r; if (v < dSelf) { dSelf = v; selfPair = [i, j]; } }
    const selfPiece = C.self_scale*dSelf;
    const faces = M.faces.map(f => {
      const R = T[f.reference], placed = { c: apply(R, f.center), U: rotate(R, f.U), V: rotate(R, f.V), n: rotate(R, f.n), a: f.a, b: f.b };
      let d = Infinity, k = -1; G.cuboids.forEach((cb, kc) => { const v = rectAabbSigned(placed, cb.lo, cb.hi, C.far); if (v < d) { d = v; k = kc; } });
      return { ...placed, d, cuboid: k, name: f.name, body: f.body };
    });
    const sphereNames = G.spheres.map(s => s.name);
    const pieces = {
      spheres: { names: sphereNames.filter((_, i) => M.keep[i]).concat('self'), d: sphereClear.filter((_, i) => M.keep[i]).map(x => x.d).concat(selfPiece) },
      links: { names: M.linksWithSpheres.map(k => G.collision_links[k]).concat('self'),
               d: M.linksWithSpheres.map(k => Math.min(...sphereClear.filter((_, i) => M.keep[i] && G.spheres[i].link === k).map(x => x.d))).concat(selfPiece) },
      faces: { names: faces.map(f => f.name).concat('self'), d: faces.map(f => f.d).concat(selfPiece) },
    };
    for (const p of Object.values(pieces)) {
      p.speed = p.d.map(speedOf.bind(null, C)); let k = 0; p.d.forEach((v, i) => { if (v < p.d[k]) k = i; });
      p.limiting = k; p.clearance = p.d[k]; p.bodySpeed = speedOf(C, p.d[k]);
    }
    const bodyBoxes = G.bodies.map(b => ({ T: T[b.reference], center: b.center, half: b.half }));
    return { T, spheres, sphereClear, dSelf, selfPair, faces, pieces, bodyBoxes, ee: apply(T[G.ee_link], [0, 0, 0]) };
  }
  const speedOf = (C, d) => clamp(d, C.offset, C.margin)/C.margin;

  return { build, evaluate, transforms, speedOf, apply, rotate, mul };
})();
if (typeof module !== 'undefined') module.exports = ArmGeometry;
