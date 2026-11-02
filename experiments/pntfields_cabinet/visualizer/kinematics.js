// Forward kinematics of a URDF joint tree (revolute joints about any axis), row-major 4x4 matrices as length-16 arrays.
// Runs in the page and in check.cjs, which compares it with the Python kinematics the experiments use.
const Kinematics = (() => {
  const I4 = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
  const mul = (A, B) => { const C = new Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += A[4*i+k]*B[4*k+j]; C[4*i+j] = s; } return C; };
  function rot(axis, t) {   // Rodrigues
    const n = Math.hypot(...axis), [x, y, z] = axis.map(v => v/n), c = Math.cos(t), s = Math.sin(t), C = 1-c;
    return [c+x*x*C, x*y*C-z*s, x*z*C+y*s, 0,  y*x*C+z*s, c+y*y*C, y*z*C-x*s, 0,  z*x*C-y*s, z*y*C+x*s, c+z*z*C, 0,  0, 0, 0, 1];
  }
  function build(robot) { const jointOf = {}; robot.joints.forEach(j => { jointOf[j.child] = j; }); return { robot, jointOf }; }
  function transforms(K, q, links) {
    const memo = { [K.robot.base]: I4 };
    const get = link => {
      if (memo[link]) return memo[link];
      const j = K.jointOf[link]; let T = mul(get(j.parent), j.origin);
      if (j.index >= 0) T = mul(T, rot(j.axis, q[j.index]));
      return (memo[link] = T);
    };
    return Object.fromEntries(links.map(l => [l, get(l)]));
  }
  return { build, transforms, mul };
})();
if (typeof module !== 'undefined') module.exports = Kinematics;
