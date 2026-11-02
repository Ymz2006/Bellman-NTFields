"""Guarded field planner: a learned value field proposes motion, the hierarchical distance model guards it.

Every query runs K particles. Each particle grows a forward branch from the start and (bidirectional mode) a
backward branch from the goal; the heads descend the value field towards each other (forward head towards the
backward head and vice versa, P-NTFields style). Guards, all driven by geometry.ArmClearance:

  1. departure guard  an endpoint closer than the guard threshold is first moved straight out (ascent of the
                      learned clearance of its tightest pieces), every scorer-resolution point checked: a piece
                      below the threshold must not get closer than it was at the (certified) endpoint
  2. barrier          candidate steps are ranked by V(candidate, target) + lambda relu(soft - surplus) / soft
  3. certified step   a step is taken only if the whole segment is certified (conservative advancement); else
                      the next candidate, then a half step; if none certifies the head waits (stall)
  4. connection       heads are joined to the other side's heads / branch nodes of every particle of the same
                      query (and to the start / goal) whenever the joining segment certifies; the first certified
                      join ends the query, so the returned path consists of certified segments only
  5. escape           a particle whose V(head, target) has not improved for `stall` iterations jumps to a random
                      node of its own branch and follows a persistent random direction for `escape` iterations
Particles of a query differ by the noise of their candidate directions (particle 0 is the least noisy).
Unidirectional mode keeps only the forward branch and connects it to the goal. Nothing here reads the exact
scene unless ArmClearance was built without a learned distance (the exact-distance control).
"""
import math, time
import numpy as np
import torch
from .geometry import OBSTACLE_MARGIN, SELF_MARGIN


class GuardedPlanner:
    def __init__(self, field, clear, particles=4, iters=300, step=.01, candidates=8, sigmas=(.3, .6, 1., 1.5), soft=.01, lam=2.,
                 bidirectional=True, connect_every=2, connect_radius=.25, connect_nodes=6, stall=12, escape=10, depart_kappa=.1, depart_buffer=.002,
                 depart_steps=60, depart_step=.002, device='cuda:0'):
        self.field, self.clear = field, clear
        self.K, self.iters, self.h, self.C = particles, iters, step, candidates
        self.sigmas = torch.tensor([sigmas[k % len(sigmas)] for k in range(particles)], device=device)
        self.soft, self.lam, self.bi = soft, lam, bidirectional
        self.every, self.radius, self.nodes = connect_every, connect_radius, connect_nodes
        self.stall, self.escape = stall, escape
        self.kappa, self.buffer, self.dsteps, self.dstep = depart_kappa, depart_buffer, depart_steps, depart_step
        self.device = device

    # ------------------------------------------------------------------ guard 1: departure from tight endpoints
    def _pieces(self, q):
        env, d = self.clear.components(q)
        return env-OBSTACLE_MARGIN-self.clear.eps, d-SELF_MARGIN, env

    def _tight_rows(self, q):
        """Joint gradients (n, S, 6) of every sphere's obstacle clearance and the mask of spheres below the buffer."""
        with torch.enable_grad():
            x = q.detach().requires_grad_(True)
            f = lambda y: self._pieces(y)[0]
            G = torch.autograd.functional.jacobian(lambda y: f(y).sum(0), x, vectorize=True).permute(1, 0, 2)
        with torch.no_grad(): tight = self._pieces(q)[0] < self.buffer
        return G.detach(), tight

    @torch.no_grad()
    def _project(self, d0, G, mask, rate, sweeps=40):
        """Closest direction to d0 with G_i . d >= rate |G_i| on the masked rows (Hildreth), normalized."""
        gg = (G*G).sum(-1).clamp_min(1e-12); h = rate*gg.sqrt(); w = d0.clone(); lam = torch.zeros_like(h); m = mask.to(d0.dtype)
        for _ in range(sweeps):
            for i in range(G.shape[1]):
                if not bool(mask[:, i].any()): continue
                gi = G[:, i]; new = ((lam[:, i]+(h[:, i]-(gi*w).sum(-1))/gg[:, i]).clamp_min(0))*m[:, i]
                w = w+(new-lam[:, i])[:, None]*gi; lam[:, i] = new
        return w/torch.linalg.vector_norm(w, dim=1, keepdim=True).clamp_min(1e-12)

    def depart(self, q0, gen=None, candidates=16):
        """Departure chains (list of (n_k, 6) tensors) from endpoints q0 (B, 6). Each step tries the soft-min ascent
        direction of the guard surplus, noisy versions of it and random directions, at two step sizes; a candidate
        is valid when every scorer-resolution point of its segment keeps each piece above the threshold or no
        closer than at the endpoint (+ kappa x the sphere's displacement); the valid candidate with the largest
        minimum surplus is taken. A step without a valid candidate counts as a failed departure."""
        q_range = self.clear.robot.q_range; dev = self.device
        with torch.no_grad():
            se0, ss0, env0 = self._pieces(q0); c0 = self.clear.spheres(q0)[..., :3]
        chains = [[q0[i:i+1]] for i in range(len(q0))]
        cur = q0.clone(); active = torch.minimum(se0.amin(1), ss0.amin(1)) < self.buffer
        self.depart_report = dict(needed=int(active.sum()), reached=0, failed=0)
        status = torch.where(active, 1, 0)                                                         # 0 not needed, 1 running, 2 reached, 3 failed
        for _ in range(self.dsteps):
            ids = torch.nonzero(active).squeeze(1)
            if not len(ids): break
            n = len(ids)
            with torch.enable_grad():
                x = cur[ids].detach().requires_grad_(True); se, ss, _ = self._pieces(x)
                allp = torch.cat([se, ss], 1); w = torch.softmax(-allp.detach()/.003, 1)
                g = torch.autograd.grad((w*allp).sum(), x)[0]
            G, tight = self._tight_rows(cur[ids])                                                 # (n, S, 6), (n, S)
            with torch.no_grad():
                g = g/torch.linalg.vector_norm(g, dim=1, keepdim=True).clamp_min(1e-12)
                cone = [self._project(g, G, tight, rate) for rate in (.3, .6)]                      # every tight sphere moves outward
                noise = torch.randn(n, candidates, 6, device=dev, generator=gen)
                scale = torch.linspace(0, 2., candidates, device=dev)[None, :, None]; scale[:, -2:] = 1e3   # last two: random directions
                d = g[:, None]+scale*noise; d = torch.cat([d, *[c[:, None] for c in cone]], 1)
                d = d/torch.linalg.vector_norm(d, dim=2, keepdim=True).clamp_min(1e-12)
                d = torch.cat([d*self.dstep, d*self.dstep/4], 1); M = d.shape[1]                    # (n, M, 6)
                nxt = (cur[ids][:, None]+d).clamp(-.5, .5)
                m = max(1, int(math.ceil(float(((nxt-cur[ids][:, None])*q_range).abs().amax())/.001)))
                w8 = torch.arange(1, m+1, device=dev, dtype=cur.dtype)/m
                pts = cur[ids][:, None, None]+(nxt-cur[ids][:, None])[:, :, None]*w8[None, None, :, None]   # (n, M, m, 6)
                flat = pts.reshape(-1, 6); se, ss, env = self._pieces(flat)
                c = self.clear.spheres(flat)[..., :3]; rep = M*m
                rel = env >= env0[ids].repeat_interleave(rep, 0)+self.kappa*torch.linalg.vector_norm(c-c0[ids].repeat_interleave(rep, 0), dim=-1)
                ok = (((se >= 0) | rel).all(1) & (ss >= 0).all(1)).reshape(n, M, m).all(2)
                surplus = torch.minimum(se.amin(1), ss.amin(1)).reshape(n, M, m)[:, :, -1]
                score = torch.where(ok, surplus, -torch.inf); best = score.argmax(1); valid = torch.isfinite(score.max(1).values)
                bad = ids[~valid]; good = ids[valid]
                active[bad] = False; status[bad] = 3; self.depart_report['failed'] += len(bad)
                step = nxt[valid, best[valid]]; cur[good] = step
                for i, x in zip(good.tolist(), step): chains[i].append(x[None])
                fin = surplus[valid, best[valid]] >= self.buffer
                active[good[fin]] = False; status[good[fin]] = 2; self.depart_report['reached'] += int(fin.sum())
        self.depart_report['unfinished'] = int(active.sum()); self.depart_status = status.cpu().numpy()
        return [torch.cat(c) for c in chains]

    def shortcut(self, paths, rounds=64, tries=8):
        """Greedy certified shortcutting of connected paths (lists of (T, 6) arrays, normalized): from the current
        waypoint, jump to the farthest of `tries` later waypoints (last one included) whose straight segment
        certifies; the next waypoint always qualifies (original certified segment or departure step)."""
        dev = self.device; out = []
        P = [torch.as_tensor(p, device=dev) for p in paths]; cur = [0]*len(P); keep = [[0] for _ in P]
        for _ in range(rounds*4):
            act = [k for k in range(len(P)) if cur[k] < len(P[k])-1]
            if not act: break
            a, b, owner, idx = [], [], [], []
            for k in act:
                n = len(P[k]); i = cur[k]
                cand = sorted(set([n-1]+np.linspace(i+2, n-1, tries).astype(int).tolist()))
                cand = [j for j in cand if i+1 < j <= n-1]
                for j in cand: a.append(P[k][i]); b.append(P[k][j]); owner.append(k); idx.append(j)
            ok = self.clear.certify(torch.stack(a), torch.stack(b))[0].tolist() if a else []
            best = {k: cur[k]+1 for k in act}
            for k, j, o in zip(owner, idx, ok):
                if o and j > best[k]: best[k] = j
            for k in act: cur[k] = best[k]; keep[k].append(best[k])
        for k, p in enumerate(P):
            if cur[k] < len(p)-1: keep[k] += list(range(cur[k]+1, len(p)))
            out.append(p[keep[k]].cpu().numpy())
        return out

    # ------------------------------------------------------------------ main loop
    def plan(self, starts, goals, seed=0):
        t0 = time.monotonic(); dev = self.device; Q, K = len(starts), self.K; P = Q*K
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        starts = torch.as_tensor(starts, device=dev, dtype=torch.float32); goals = torch.as_tensor(goals, device=dev, dtype=torch.float32)
        s_chain = self.depart(starts, gen); depart = dict(start=self.depart_report); s_status = self.depart_status
        g_chain = self.depart(goals, gen); depart['goal'] = self.depart_report; g_status = self.depart_status
        g_end = torch.stack([c[-1] for c in g_chain])                    # unidirectional target: end of the goal's departure
        qid = torch.arange(Q, device=dev).repeat_interleave(K); sig = self.sigmas.repeat(Q)
        N = self.iters+max(len(c) for c in s_chain+g_chain)+2
        sides = 2 if self.bi else 1
        nodes = torch.zeros(sides, P, N, 6, device=dev); parent = torch.full((sides, P, N), -1, dtype=torch.long, device=dev)
        count = torch.zeros(sides, P, dtype=torch.long, device=dev); head = torch.zeros(sides, P, dtype=torch.long, device=dev)
        for side, chains in enumerate([s_chain, g_chain][:sides]):
            for q in range(Q):
                c = chains[q]; n = len(c)
                for k in range(K):
                    p = q*K+k; nodes[side, p, :n] = c; parent[side, p, 1:n] = torch.arange(n-1, device=dev); count[side, p] = n; head[side, p] = n-1
        solved = torch.zeros(Q, dtype=torch.bool, device=dev); join = {}
        best = torch.full((P,), torch.inf, device=dev); last = torch.zeros(P, dtype=torch.long, device=dev)
        esc_left = torch.zeros(P, dtype=torch.long, device=dev); esc_dir = torch.zeros(sides, P, 6, device=dev)
        stats = dict(steps=0, stalled=0, escapes=0, certify_segments=0, certify_evals=0)
        rows = torch.arange(P, device=dev)

        def heads(side): return nodes[side, rows, head[side]]

        def certify(a, b):
            ok, ev = self.clear.certify(a, b); stats['certify_segments'] += len(a); stats['certify_evals'] += int(ev.sum()); return ok

        def try_connect(it):
            """Joins between side-0 points and side-1 points (or the goal) of the same query; marks solved queries."""
            act = torch.nonzero(~solved[qid]).squeeze(1)
            if not len(act): return
            if not self.bi:
                a = heads(0)[act]; b = g_end[qid[act]]
                cand = [(act, head[0, act], None, None, a, b)]
            else:
                cand = []
                f, b = heads(0), heads(1)
                for k in range(K):                                             # head x head across particles
                    other = qid[act]*K+k
                    cand.append((act, head[0, act], other, head[1, other], f[act], b[other]))
                for side in (0, 1):                                            # head x random branch nodes of other particles
                    for _ in range(self.nodes):
                        other = qid[act]*K+torch.randint(K, (len(act),), device=dev, generator=gen)
                        j = (torch.rand(len(act), device=dev, generator=gen)*count[1-side, other]).long()
                        hp = heads(side)[act]; npnt = nodes[1-side, other, j]
                        if side == 0: cand.append((act, head[0, act], other, j, hp, npnt))
                        else: cand.append((other, j, act, head[1, act], npnt, hp))
            keep = []
            for p0, i0, p1, i1, a, b in cand:
                near = (torch.linalg.vector_norm(a-b, dim=1) <= self.radius) & ~solved[qid[p0]]
                sel = torch.nonzero(near).squeeze(1)
                if len(sel): keep.append((p0[sel], i0[sel], None if p1 is None else p1[sel], None if p1 is None else i1[sel], a[sel], b[sel]))
            if not keep: return
            P0, I0 = torch.cat([k[0] for k in keep]), torch.cat([k[1] for k in keep])
            P1 = None if keep[0][2] is None else torch.cat([k[2] for k in keep]); I1 = None if P1 is None else torch.cat([k[3] for k in keep])
            ok = certify(torch.cat([k[4] for k in keep]), torch.cat([k[5] for k in keep]))
            for s in torch.nonzero(ok).squeeze(1).tolist():
                q = int(qid[P0[s]])
                if q in join: continue
                join[q] = (int(P0[s]), int(I0[s]), None if P1 is None else int(P1[s]), None if P1 is None else int(I1[s]), it)
                solved[q] = True

        try_connect(0)
        for it in range(1, self.iters+1):
            act = torch.nonzero(~solved[qid]).squeeze(1)
            if not len(act): break
            A = len(act)
            x = torch.cat([heads(s)[act] for s in range(sides)])
            y = torch.cat([heads(1)[act], heads(0)[act]]) if self.bi else g_end[qid[act]]
            grad, v = self.field.gradient(x, y)
            d0 = -grad/torch.linalg.vector_norm(grad, dim=1, keepdim=True).clamp_min(1e-12)
            escaping = (esc_left[act] > 0).repeat(sides)
            ed = torch.cat([esc_dir[s, act] for s in range(sides)])
            d0 = torch.where(escaping[:, None], ed, d0)
            s_rep = sig[act].repeat(sides)
            noise = torch.randn(len(x), self.C, 6, device=dev, generator=gen); noise[:, 0] = 0
            dirs = d0[:, None]+s_rep[:, None, None]*noise
            dirs = dirs/torch.linalg.vector_norm(dirs, dim=2, keepdim=True).clamp_min(1e-12)
            gap = torch.linalg.vector_norm(x-y, dim=1)
            h = torch.minimum(torch.full_like(gap, self.h), gap.clamp_min(1e-4))[:, None, None]
            cand = (x[:, None]+h*dirs).clamp(-.5, .5)                                           # (2A, C, 6)
            flat = cand.reshape(-1, 6)
            with torch.no_grad():
                surplus = self.clear.margins(flat).reshape(len(x), self.C)
                val = self.field.value(flat, y.repeat_interleave(self.C, 0)).reshape(len(x), self.C)
            score = val+self.lam*torch.relu(self.soft-surplus)/self.soft+torch.where(surplus < 0, torch.inf, 0.)
            score = torch.where(escaping[:, None], self.lam*torch.relu(self.soft-surplus)/self.soft+torch.where(surplus < 0, torch.inf, 0.)+.01*torch.rand_like(score), score)
            order = score.argsort(1)
            tries = [cand[torch.arange(len(x)), order[:, 0]], cand[torch.arange(len(x)), order[:, 1]], x+.5*(cand[torch.arange(len(x)), order[:, 0]]-x)]
            finite = [torch.isfinite(score[torch.arange(len(x)), order[:, 0]]), torch.isfinite(score[torch.arange(len(x)), order[:, 1]]), torch.ones(len(x), dtype=torch.bool, device=dev)]
            T = torch.stack(tries); F = torch.stack(finite)                                       # (3, 2A, 6), (3, 2A)
            ok = certify(x.repeat(3, 1), T.reshape(-1, 6)).reshape(3, -1) & F
            chosen = torch.where(ok.any(0), ok.float().argmax(0), torch.full_like(ok[0], -1, dtype=torch.long))
            new = torch.where((chosen >= 0)[:, None], T[chosen.clamp_min(0), torch.arange(len(x), device=dev)], x)
            moved = chosen >= 0
            stats['steps'] += len(x); stats['stalled'] += int((~moved).sum())
            for s in range(sides):
                part = slice(s*A, (s+1)*A); m = moved[part]; p = act[m]
                if not len(p): continue
                n = count[s, p]; nodes[s, p, n] = new[part][m]; parent[s, p, n] = head[s, p]; head[s, p] = n; count[s, p] = n+1
            # progress and escape bookkeeping
            f_now, t_now = heads(0)[act], (heads(1)[act] if self.bi else g_end[qid[act]])
            v_now = self.field.value(f_now, t_now)
            improved = v_now < best[act]-1e-3
            best[act] = torch.where(improved, v_now, best[act]); last[act] = torch.where(improved, torch.full_like(last[act], it), last[act])
            esc_left[act] = (esc_left[act]-1).clamp_min(0)
            start_esc = (esc_left[act] == 0) & (it-last[act] > self.stall)
            if bool(start_esc.any()):
                p = act[start_esc]; stats['escapes'] += len(p)
                esc_left[p] = self.escape; last[p] = it; best[p] = torch.inf
                for s in range(sides):
                    r = torch.randn(len(p), 6, device=dev, generator=gen); esc_dir[s, p] = r/torch.linalg.vector_norm(r, dim=1, keepdim=True)
                    jump = torch.rand(len(p), device=dev, generator=gen) < .5
                    j = (torch.rand(len(p), device=dev, generator=gen)*count[s, p]).long()
                    head[s, p] = torch.where(jump, j, head[s, p])
            if it % self.every == 0 or it == self.iters: try_connect(it)

        # Paths (normalized coordinates); unsolved queries fall back to particle 0's forward branch + the goal.
        def chain(side, p, i):
            out = []
            while i >= 0: out.append(nodes[side, p, i]); i = int(parent[side, p, i])
            return torch.stack(out[::-1])
        paths, info = [], []
        for q in range(Q):
            if q in join:
                p0, i0, p1, i1, it = join[q]; fw = chain(0, p0, i0)
                bw = chain(1, p1, i1).flip(0) if p1 is not None else g_chain[q].flip(0)
                paths.append(torch.cat([fw, bw]).cpu().numpy()); info.append(dict(connected=True, iteration=it, depart=(int(s_status[q]), int(g_status[q]))))
            else:
                fw = chain(0, q*K, int(head[0, q*K])); paths.append(torch.cat([fw, goals[q:q+1]]).cpu().numpy()); info.append(dict(connected=False, iteration=None, depart=(int(s_status[q]), int(g_status[q]))))
        stats['seconds'] = time.monotonic()-t0; stats['connected'] = int(solved.sum()); stats['departure'] = depart
        return paths, info, stats
