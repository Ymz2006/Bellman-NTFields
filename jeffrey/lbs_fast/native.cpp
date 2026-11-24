// Native CPU kernel of the fast LBS planner (jeffrey/lbs_fast/native.py loads it through ctypes).
//
// The whole MPPI loop of fast.Planner for one query in one call: sampling, the exact clearance of the rollout ends (maze: rectangle
// against the wall boxes of the 3 x 3 cells around it, as maze.Maze.signed_cells; 3-D house: the trilinear voxel table, as
// scenes3.House3.signed_cells), the lift profile G, the encoder (9 linear layers 256 x 256, as fastnet._forward) and the metric.
// The rows of one step are split over OpenMP threads, each thread running the whole network on its rows: one parallel region per
// step instead of one per matrix product.
//
// Build: g++ -O3 -march=native -mprefer-vector-width=512 -ffast-math -fopenmp -shared -fPIC native.cpp -o native.so
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace {

const int HD = 256;       // hidden / feature width
const int NF = 128;       // Fourier frequencies
const int NL = 9;         // linear layers

struct Scene {
    int kind = 0;                     // 0 none (no lift), 1 maze, 2 house3
    // maze
    int n = 0;
    float E = 1, rho = 0.25f, hl = 0.25f, hw = 0.1f, twall = 0.1f;
    std::vector<uint8_t> V, H;        // V (n+1, n), H (n, n+1)
    // house3
    int nx = 0, ny = 0, nz = 0;
    float q0[3] = {0, 0, 0}, per_q = 1, umax[3] = {0, 0, 0}, vpc = 10;
    std::vector<float> table;
};

struct Model {
    int in_dim = 3, nthreads = 1;
    float* W = nullptr;               // (in_dim, NF) Fourier matrix times 2 pi
    float* Wt[NL];                    // transposed weights [k][j]
    float* b[NL];
    float gate[2] = {0, 0};
    // lift profile
    int lift = 0, nodes = 0;
    std::vector<float> G;
    float offset = 0, margin = 0, G_offset = 0, floor_slope = 0, depth = 0, hscale = 0;
    Scene sc;
};

inline float* alloc(size_t n) { return static_cast<float*>(aligned_alloc(64, ((n * sizeof(float) + 63) / 64) * 64)); }

// ---------------------------------------------------------------------------------------------------------------- clearance
inline float rect_box(float px, float py, float c, float s, float cx, float cy, float hx, float hy, float hl, float hw) {
    const float ux = c, uy = s, vx = -s, vy = c;
    static const float sg[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
    float d1 = 1e30f, d2 = 1e30f;
    for (int i = 0; i < 4; i++) {
        float qx = px + sg[i][0] * hl * ux + sg[i][1] * hw * vx, qy = py + sg[i][0] * hl * uy + sg[i][1] * hw * vy;
        float ex = std::max(std::fabs(qx - cx) - hx, 0.0f), ey = std::max(std::fabs(qy - cy) - hy, 0.0f);
        d1 = std::min(d1, std::sqrt(ex * ex + ey * ey + 1e-12f));
        float rx = cx + hx * sg[i][0] - px, ry = cy + hy * sg[i][1] - py;
        float lu = std::max(std::fabs(rx * ux + ry * uy) - hl, 0.0f), lv = std::max(std::fabs(rx * vx + ry * vy) - hw, 0.0f);
        d2 = std::min(d2, std::sqrt(lu * lu + lv * lv + 1e-12f));
    }
    const float kx = px - cx, ky = py - cy;
    float g0 = std::fabs(kx) - (hl * std::fabs(ux) + hw * std::fabs(vx)) - hx;
    float g1 = std::fabs(ky) - (hl * std::fabs(uy) + hw * std::fabs(vy)) - hy;
    float g2 = std::fabs(kx * ux + ky * uy) - hl - (hx * std::fabs(ux) + hy * std::fabs(uy));
    float g3 = std::fabs(kx * vx + ky * vy) - hw - (hx * std::fabs(vx) + hy * std::fabs(vy));
    float gap = std::max(std::max(g0, g1), std::max(g2, g3));
    return gap > 0 ? std::min(d1, d2) : gap;
}

inline float clear_maze(const Scene& sc, const float* q) {
    const int n = sc.n;
    const float x = q[0] * sc.E + 0.5f * n, y = q[1] * sc.E + 0.5f * n, th = q[2] * sc.E / sc.rho;
    const int ci = std::min(std::max(static_cast<int>(std::floor(x)), 0), n - 1), cj = std::min(std::max(static_cast<int>(std::floor(y)), 0), n - 1);
    const float c = std::cos(th), s = std::sin(th), h = 0.5f * sc.twall;
    float d = 1e4f;
    for (int di = -1; di <= 2; di++)
        for (int dj = -1; dj <= 1; dj++) {
            int vi = ci + di, vj = cj + dj;                       // vertical wall at x = vi over y in [vj, vj + 1]
            if (vi >= 0 && vi <= n && vj >= 0 && vj <= n - 1 && sc.V[vi * n + vj])
                d = std::min(d, rect_box(x, y, c, s, static_cast<float>(vi), vj + 0.5f, h, 0.5f + h, sc.hl, sc.hw));
            int hi = ci + dj, hj = cj + di;                       // horizontal wall at y = hj over x in [hi, hi + 1]
            if (hi >= 0 && hi <= n - 1 && hj >= 0 && hj <= n && sc.H[hi * (n + 1) + hj])
                d = std::min(d, rect_box(x, y, c, s, hi + 0.5f, static_cast<float>(hj), 0.5f + h, h, sc.hl, sc.hw));
        }
    return d;
}

inline float clear_house(const Scene& sc, const float* q) {
    float fr[3], over = 0;
    int i[3];
    for (int k = 0; k < 3; k++) {
        float u = (q[k] - sc.q0[k]) * sc.per_q - 0.5f;
        float uc = std::min(std::max(u, 0.0f), sc.umax[k]);
        i[k] = static_cast<int>(std::floor(uc));
        fr[k] = uc - i[k];
        over += std::fabs(u - uc);
    }
    const float* t = sc.table.data();
    const size_t base = (static_cast<size_t>(i[0]) * sc.ny + i[1]) * sc.nz + i[2], sy = sc.nz, sx = static_cast<size_t>(sc.ny) * sc.nz;
    const float fx = fr[0], fy = fr[1], fz = fr[2];
    float v = (1 - fx) * (1 - fy) * (1 - fz) * t[base] + (1 - fx) * (1 - fy) * fz * t[base + 1]
            + (1 - fx) * fy * (1 - fz) * t[base + sy] + (1 - fx) * fy * fz * t[base + sy + 1]
            + fx * (1 - fy) * (1 - fz) * t[base + sx] + fx * (1 - fy) * fz * t[base + sx + 1]
            + fx * fy * (1 - fz) * t[base + sx + sy] + fx * fy * fz * t[base + sx + sy + 1];
    return v - over / sc.vpc;
}

inline float clearance(const Model& m, const float* q) { return m.sc.kind == 1 ? clear_maze(m.sc, q) : clear_house(m.sc, q); }

inline float lift_G(const Model& m, float d) {
    if (d < m.offset) return m.G_offset + (m.offset - std::max(d, -m.depth)) * m.floor_slope;
    float dc = std::min(d, m.margin);
    float u = (dc - m.offset) / (m.margin - m.offset) * (m.nodes - 1);
    int i = std::min(std::max(static_cast<int>(std::floor(u)), 0), m.nodes - 2);
    float w = u - i;
    return (1 - w) * m.G[i] + w * m.G[i + 1];
}

// ---------------------------------------------------------------------------------------------------------------- network
// Y (R, HD) = X (R, HD) Wt + b, Wt[k][j] = weight[j][k]
inline void linear(const float* __restrict X, const float* __restrict Wt, const float* __restrict b, float* __restrict Y, int R) {
    for (int r = 0; r < R; r++) std::memcpy(Y + r * HD, b, HD * sizeof(float));
    for (int k = 0; k < HD; k += 8) {
        const float* w0 = Wt + k * HD;
        const float *w1 = w0 + HD, *w2 = w0 + 2 * HD, *w3 = w0 + 3 * HD, *w4 = w0 + 4 * HD, *w5 = w0 + 5 * HD, *w6 = w0 + 6 * HD, *w7 = w0 + 7 * HD;
        for (int r = 0; r < R; r++) {
            const float* x = X + r * HD + k;
            const float a0 = x[0], a1 = x[1], a2 = x[2], a3 = x[3], a4 = x[4], a5 = x[5], a6 = x[6], a7 = x[7];
            float* y = Y + r * HD;
#pragma omp simd aligned(y, w0, w1, w2, w3, w4, w5, w6, w7 : 64)
            for (int j = 0; j < HD; j++)
                y[j] += (a0 * w0[j] + a1 * w1[j]) + (a2 * w2[j] + a3 * w3[j]) + (a4 * w4[j] + a5 * w5[j]) + (a6 * w6[j] + a7 * w7[j]);
        }
    }
}

struct Work {                         // per-thread buffers for R rows
    float *x, *u, *v, *s, *t, *y;
    explicit Work(int R) { x = alloc(R * HD); u = alloc(R * HD); v = alloc(R * HD); s = alloc(R * HD); t = alloc(R * HD); y = alloc(R * HD); }
    ~Work() { free(x); free(u); free(v); free(s); free(t); free(y); }
};

// rows (R, 3) configurations -> features (R, HD). Computes the lift column from the clearance.
void encode_rows(const Model& m, const float* q, int R, float* out, Work& w) {
    const int N = R * HD;
    for (int r = 0; r < R; r++) {
        float in[4] = {q[3 * r], q[3 * r + 1], q[3 * r + 2], 0};
        if (m.lift) in[3] = m.hscale * lift_G(m, clearance(m, q + 3 * r));
        float* p = w.s + r * HD;                                   // projection in the first NF entries
        for (int f = 0; f < NF; f++) {
            float a = 0;
            for (int d = 0; d < m.in_dim; d++) a += in[d] * m.W[d * NF + f];
            p[f] = a;
        }
    }
    for (int r = 0; r < R; r++) {
        const float* p = w.s + r * HD;
        float* x = w.x + r * HD;
#pragma omp simd
        for (int f = 0; f < NF; f++) { x[f] = std::sin(p[f]); x[NF + f] = std::cos(p[f]); }
    }
    linear(w.x, m.Wt[0], m.b[0], w.u, R);
    linear(w.x, m.Wt[1], m.b[1], w.v, R);
#pragma omp simd
    for (int i = 0; i < N; i++) { w.u[i] = std::sin(w.u[i]); w.v[i] = std::sin(w.v[i]); }
    float* x = w.x;
    float* tmp = w.t;
    int k = 2;
    for (int ii = 0; ii < 2; ii++) {
        const float g = m.gate[ii];
        linear(x, m.Wt[k], m.b[k], w.s, R);
#pragma omp simd
        for (int i = 0; i < N; i++) { float sv = std::sin(w.s[i]); w.y[i] = w.u[i] * sv + w.v[i] * (1 - sv); }
        linear(w.y, m.Wt[k + 1], m.b[k + 1], w.s, R);
#pragma omp simd
        for (int i = 0; i < N; i++) { float sv = std::sin(w.s[i]); w.y[i] = w.u[i] * sv + w.v[i] * (1 - sv); }
        linear(w.y, m.Wt[k + 2], m.b[k + 2], w.s, R);
#pragma omp simd
        for (int i = 0; i < N; i++) tmp[i] = (1 - g) * x[i] + g * std::sin(w.s[i]);
        std::swap(x, tmp);
        k += 3;
    }
    linear(x, m.Wt[k], m.b[k], w.s, R);
    for (int r = 0; r < R; r++) {
        const float* y = w.s + r * HD;
        float mean = 0;
#pragma omp simd reduction(+ : mean)
        for (int j = 0; j < HD; j++) mean += y[j];
        mean /= HD;
        float var = 0;
#pragma omp simd reduction(+ : var)
        for (int j = 0; j < HD; j++) var += (y[j] - mean) * (y[j] - mean);
        const float inv = 1.0f / std::sqrt(var / HD + 1e-5f);
        float* o = out + r * HD;
#pragma omp simd
        for (int j = 0; j < HD; j++) o[j] = (y[j] - mean) * inv;
    }
}

inline float metric(const float* f0, const float* f1) {
    static const float LOG16 = std::log(16.0f);
    float total = 0;
    for (int g = 0; g < HD / 16; g++) {
        float d[16], mx = 0;
#pragma omp simd
        for (int j = 0; j < 16; j++) { float e = f0[g * 16 + j] - f1[g * 16 + j]; d[j] = 10.0f * std::sqrt(e * e + 1e-6f); }
        for (int j = 0; j < 16; j++) mx = std::max(mx, d[j]);
        float sum = 0;
#pragma omp simd reduction(+ : sum)
        for (int j = 0; j < 16; j++) sum += std::exp(d[j] - mx);
        total += (mx + std::log(sum) - LOG16) / 10.0f;
    }
    return 0.2f * total;
}

const int CH = 16;                    // rows per network call inside a thread

// rows [a, b) in pieces of at most CH
inline void encode_range(const Model& m, const float* q, int a, int b, float* feat, Work& w) {
    for (int i = a; i < b; i += CH) encode_rows(m, q + 3 * i, std::min(CH, b - i), feat + static_cast<size_t>(i) * HD, w);
}

void encode_many(const Model& m, const float* q, int n, float* feat) {
    const int P = std::max(1, std::min(m.nthreads, n));
#pragma omp parallel num_threads(P)
    {
        Work w(CH);
#pragma omp for schedule(static, 1)
        for (int t = 0; t < P; t++) encode_range(m, q, static_cast<int>(static_cast<long>(n) * t / P), static_cast<int>(static_cast<long>(n) * (t + 1) / P), feat, w);
    }
}

// xoshiro256++ and Box-Muller: the planner's own random stream (not torch's)
struct Rng {
    uint64_t s[4];
    explicit Rng(uint64_t seed) {
        uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4; i++) { z += 0x9E3779B97F4A7C15ULL; uint64_t x = z; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; s[i] = x ^ (x >> 31); }
    }
    inline uint64_t next() {
        auto rotl = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
        uint64_t r = rotl(s[0] + s[3], 23) + s[0], t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    inline float uni() { return (static_cast<float>(next() >> 40) + 0.5f) * (1.0f / 16777216.0f); }
    void normals(float* out, int n) {
        for (int i = 0; i < n; i += 2) {
            float r = std::sqrt(-2.0f * std::log(uni())), a = 6.28318530718f * uni();
            out[i] = r * std::cos(a);
            if (i + 1 < n) out[i + 1] = r * std::sin(a);
        }
    }
};

}  // namespace

extern "C" {

void* lbs_create(int in_dim, const float* W, const float* weights, const float* biases, const float* gates, int nthreads) {
    Model* m = new Model();
    m->in_dim = in_dim;
    m->nthreads = nthreads;
    m->W = alloc(in_dim * NF);
    std::memcpy(m->W, W, sizeof(float) * in_dim * NF);
    for (int l = 0; l < NL; l++) {
        m->Wt[l] = alloc(HD * HD);
        m->b[l] = alloc(HD);
        const float* w = weights + static_cast<size_t>(l) * HD * HD;           // torch layout [out j][in k]
        for (int j = 0; j < HD; j++)
            for (int k = 0; k < HD; k++) m->Wt[l][k * HD + j] = w[j * HD + k];
        std::memcpy(m->b[l], biases + l * HD, sizeof(float) * HD);
    }
    m->gate[0] = gates[0];
    m->gate[1] = gates[1];
    return m;
}

void lbs_set_threads(void* h, int nthreads) { static_cast<Model*>(h)->nthreads = nthreads; }

void lbs_set_lift(void* h, const float* G, int nodes, float offset, float margin, float G_offset, float floor_slope, float depth, float hscale) {
    Model* m = static_cast<Model*>(h);
    m->lift = 1;
    m->nodes = nodes;
    m->G.assign(G, G + nodes);
    m->offset = offset; m->margin = margin; m->G_offset = G_offset; m->floor_slope = floor_slope; m->depth = depth; m->hscale = hscale;
}

void lbs_set_maze(void* h, int n, float E, float rho, float hl, float hw, float twall, const uint8_t* V, const uint8_t* H) {
    Scene& sc = static_cast<Model*>(h)->sc;
    sc.kind = 1; sc.n = n; sc.E = E; sc.rho = rho; sc.hl = hl; sc.hw = hw; sc.twall = twall;
    sc.V.assign(V, V + (n + 1) * n);
    sc.H.assign(H, H + n * (n + 1));
}

void lbs_set_house(void* h, int nx, int ny, int nz, const float* table, const float* q0, float per_q, const float* umax, float vpc) {
    Scene& sc = static_cast<Model*>(h)->sc;
    sc.kind = 2; sc.nx = nx; sc.ny = ny; sc.nz = nz; sc.per_q = per_q; sc.vpc = vpc;
    for (int k = 0; k < 3; k++) { sc.q0[k] = q0[k]; sc.umax[k] = umax[k]; }
    sc.table.assign(table, table + static_cast<size_t>(nx) * ny * nz);
}

void lbs_clearance(void* h, const float* q, int n, float* out) {
    const Model& m = *static_cast<Model*>(h);
    for (int i = 0; i < n; i++) out[i] = clearance(m, q + 3 * i);
}

void lbs_encode(void* h, const float* q, int n, float* feat) { encode_many(*static_cast<Model*>(h), q, n, feat); }

void lbs_travel(void* h, const float* pairs, int n, float* out) {
    const Model& m = *static_cast<Model*>(h);
    std::vector<float> q(6 * static_cast<size_t>(n));
    for (int i = 0; i < n; i++) { std::memcpy(&q[3 * i], pairs + 6 * i, 12); std::memcpy(&q[3 * (n + i)], pairs + 6 * i + 3, 12); }
    float* f = alloc(2 * static_cast<size_t>(n) * HD);
    encode_many(m, q.data(), 2 * n, f);
    for (int i = 0; i < n; i++) out[i] = metric(f + static_cast<size_t>(i) * HD, f + static_cast<size_t>(n + i) * HD);
    free(f);
}

// One query. meet = 0: one agent start -> goal. meet = 1: two agents chasing each other (agent 1 from the goal; within `freeze` of each
// other agent 1 stops). trace: (budget, A, 3) positions after every step. Returns the number of steps; *converged is set.
int lbs_plan(void* h, const float* start, const float* goal, int meet, int S, int H, float step, float temperature, float tol, float freeze,
             float lo2, float hi2, int budget, uint64_t seed, float first_weight, float* trace, int* converged) {
    const Model& m = *static_cast<Model*>(h);
    const int A = meet ? 2 : 1, NR = A * S * 2, NT = NR + (meet ? A : 0);
    Rng rng(seed);
    std::vector<float> z(static_cast<size_t>(A) * S * (H + 1) * 3), first(static_cast<size_t>(A) * S * 3), rows(static_cast<size_t>(NT) * 3), T(NR), cost(S);
    float* feat = alloc(static_cast<size_t>(NT) * HD);
    float* tfeat = alloc(static_cast<size_t>(A) * HD);
    float cur[2][3], prior[2][3] = {{0, 0, 0}, {0, 0, 0}};
    for (int d = 0; d < 3; d++) { cur[0][d] = start[d]; cur[1][d] = goal[d]; }
    if (!meet) encode_many(m, goal, 1, tfeat);
    const int P = std::max(1, std::min(m.nthreads, NT / 4));
    int steps = budget;
    *converged = 0;
#pragma omp parallel num_threads(P)
    {
        Work w(CH);
        for (int it = 0; it < budget; it++) {
#pragma omp single
            {
                rng.normals(z.data(), static_cast<int>(z.size()));
                for (int a = 0; a < A; a++)
                    for (int s = 0; s < S; s++) {
                        const float* z0 = &z[(static_cast<size_t>(a) * S + s) * (H + 1) * 3];
                        float acc[3] = {0, 0, 0};
                        float* r0 = &rows[(static_cast<size_t>(a) * S + s) * 6];
                        for (int hh = 0; hh < H; hh++) {
                            float dp[3], nrm = 0;
                            for (int d = 0; d < 3; d++) { dp[d] = step * z0[d] + step * z0[3 * (hh + 1) + d] + 2 * prior[a][d]; nrm += dp[d] * dp[d]; }
                            nrm = std::max(std::sqrt(nrm), step) / step;
                            for (int d = 0; d < 3; d++) { dp[d] /= nrm; acc[d] += dp[d]; }
                            if (hh == 0)
                                for (int d = 0; d < 3; d++) { first[(static_cast<size_t>(a) * S + s) * 3 + d] = dp[d]; r0[d] = cur[a][d] + acc[d]; }
                        }
                        for (int d = 0; d < 3; d++) r0[3 + d] = cur[a][d] + acc[d];
                        r0[2] = std::min(std::max(r0[2], lo2), hi2);
                        r0[5] = std::min(std::max(r0[5], lo2), hi2);
                    }
                if (meet)
                    for (int a = 0; a < A; a++)
                        for (int d = 0; d < 3; d++) rows[static_cast<size_t>(NR + a) * 3 + d] = cur[a][d];
            }
#pragma omp for schedule(static, 1)
            for (int t = 0; t < P; t++)
                encode_range(m, rows.data(), static_cast<int>(static_cast<long>(NT) * t / P), static_cast<int>(static_cast<long>(NT) * (t + 1) / P), feat, w);
#pragma omp for schedule(static)
            for (int i = 0; i < NR; i++) {
                int a = i / (S * 2);
                const float* tf = meet ? feat + static_cast<size_t>(NR + (1 - a)) * HD : tfeat;
                T[i] = metric(feat + static_cast<size_t>(i) * HD, tf);
            }
#pragma omp single
            {
                bool near = false;
                if (meet) {
                    float dd = 0;
                    for (int d = 0; d < 3; d++) dd += (cur[0][d] - cur[1][d]) * (cur[0][d] - cur[1][d]);
                    near = std::sqrt(dd) < freeze;
                }
                for (int a = 0; a < A; a++) {
                    float mn = 1e30f;
                    for (int s = 0; s < S; s++) { cost[s] = first_weight * T[(a * S + s) * 2] + T[(a * S + s) * 2 + 1]; mn = std::min(mn, cost[s]); }
                    float sum = 0, upd[3] = {0, 0, 0};
                    for (int s = 0; s < S; s++) {
                        float wgt = std::exp(-temperature * (cost[s] - mn));
                        sum += wgt;
                        for (int d = 0; d < 3; d++) upd[d] += wgt * first[(static_cast<size_t>(a) * S + s) * 3 + d];
                    }
                    for (int d = 0; d < 3; d++) upd[d] /= sum;
                    if (meet && a == 1 && near) upd[0] = upd[1] = upd[2] = 0;
                    for (int d = 0; d < 3; d++) { prior[a][d] = upd[d]; cur[a][d] += upd[d]; }
                    cur[a][2] = std::min(std::max(cur[a][2], lo2), hi2);
                    for (int d = 0; d < 3; d++) trace[(static_cast<size_t>(it) * A + a) * 3 + d] = cur[a][d];
                }
                float dd = 0;
                for (int d = 0; d < 3; d++) { float e = (meet ? cur[1][d] : goal[d]) - cur[0][d]; dd += e * e; }
                if (std::sqrt(dd) < tol && !*converged) { *converged = 1; steps = it + 1; }
            }
            if (*converged) break;
        }
    }
    free(feat);
    free(tfeat);
    return steps;
}

}  // extern "C"
