/* Single-source shortest paths on the truth.py C-space grid (same 98-neighbourhood, same edge costs:
   C-space length x mean of 1/S at both ends), seeded like Grid.seed_block; Dijkstra instead of Jacobi sweeps, so
   the result is the fixed point Grid.solve converges to. OpenMP over sources.

   usage: dijkstra inv.f32 nx nt hx ht seeds.bin nsrc lines.i32 nline out.f32 threads
     inv.f32   : 1/S at every node, (nx, nx, nt) C order
     seeds.bin : per source 64 x (int32 flat index, float32 cost); cost inf = unused
     lines.i32 : per source nline flat node indices to report
     out.f32   : per source nline travel times                                                    */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

typedef struct { float d; int v; } item;
typedef struct { item *a; long n, cap; } heap;

static void push(heap *h, float d, int v) {
    if (h->n == h->cap) { h->cap *= 2; h->a = realloc(h->a, h->cap * sizeof(item)); }
    long i = h->n++;
    while (i > 0) { long p = (i - 1) / 2; if (h->a[p].d <= d) break; h->a[i] = h->a[p]; i = p; }
    h->a[i].d = d; h->a[i].v = v;
}
static item pop(heap *h) {
    item top = h->a[0], last = h->a[--h->n];
    long i = 0;
    for (;;) {
        long c = 2 * i + 1; if (c >= h->n) break;
        if (c + 1 < h->n && h->a[c + 1].d < h->a[c].d) c++;
        if (h->a[c].d >= last.d) break;
        h->a[i] = h->a[c]; i = c;
    }
    h->a[i] = last; return top;
}
static void *slurp(const char *p, long *bytes) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *bytes = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*bytes); if (fread(b, 1, *bytes, f) != (size_t)*bytes) { perror(p); exit(1); }
    fclose(f); return b;
}
int main(int argc, char **argv) {
    if (argc != 12) { fprintf(stderr, "bad args\n"); return 1; }
    long b; float *inv = slurp(argv[1], &b);
    int nx = atoi(argv[2]), nt = atoi(argv[3]); double hx = atof(argv[4]), ht = atof(argv[5]);
    char *seeds = slurp(argv[6], &b); int nsrc = atoi(argv[7]);
    int *lines = slurp(argv[8], &b); int nline = atoi(argv[9]);
    int threads = atoi(argv[11]);
    long N = (long)nx * nx * nt;
    int off[98][3]; float len[98]; int no = 0;
    for (int a = -2; a <= 2; a++) for (int c = -2; c <= 2; c++) for (int e = -2; e <= 2; e++) {
        if (!a && !c && !e) continue;
        int g = abs(a), x = abs(c), y = abs(e), t;
        while (x) { t = g % x; g = x; x = t; } x = g; while (y) { t = x % y; x = y; y = t; }
        if (x != 1) continue;
        off[no][0] = a; off[no][1] = c; off[no][2] = e;
        len[no] = (float)sqrt((a * hx) * (a * hx) + (c * hx) * (c * hx) + (e * ht) * (e * ht)); no++;
    }
    if (no != 98) { fprintf(stderr, "offsets %d\n", no); return 1; }
    float *out = malloc((long)nsrc * nline * sizeof(float));
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        float *dist = malloc(N * sizeof(float)); char *done = malloc(N);
        heap h = { malloc((1L << 20) * sizeof(item)), 0, 1 << 20 };
        #pragma omp for schedule(dynamic, 1)
        for (int s = 0; s < nsrc; s++) {
            for (long i = 0; i < N; i++) dist[i] = INFINITY;
            memset(done, 0, N); h.n = 0;
            char *sb = seeds + (long)s * 64 * 8;
            for (int k = 0; k < 64; k++) {
                int v; float c; memcpy(&v, sb + 8 * k, 4); memcpy(&c, sb + 8 * k + 4, 4);
                if (isfinite(c) && c < dist[v]) { dist[v] = c; push(&h, c, v); }
            }
            while (h.n) {
                item it = pop(&h); int v = it.v;
                if (done[v]) continue; done[v] = 1;
                int i = v / (nx * nt), j = (v / nt) % nx, k = v % nt;
                float iv = inv[v], dv = dist[v];
                for (int o = 0; o < 98; o++) {
                    int a = i + off[o][0], c = j + off[o][1], e = k + off[o][2];
                    if (a < 0 || a >= nx || c < 0 || c >= nx || e < 0 || e >= nt) continue;
                    long u = ((long)a * nx + c) * nt + e;
                    if (done[u]) continue;
                    float nd = dv + len[o] * 0.5f * (iv + inv[u]);
                    if (nd < dist[u]) { dist[u] = nd; push(&h, nd, (int)u); }
                }
            }
            int *ln = lines + (long)s * nline;
            for (int q = 0; q < nline; q++) out[(long)s * nline + q] = ln[q] >= 0 ? dist[ln[q]] : NAN;
        }
        free(dist); free(done); free(h.a);
    }
    FILE *f = fopen(argv[10], "wb"); fwrite(out, sizeof(float), (long)nsrc * nline, f); fclose(f);
    return 0;
}
