"""Path-level self-supervision: self-rollout, path-integrated eikonal, ranking loss.

Gradients flow only through tau. Oracle outputs and safe neighbors are detached upstream.
"""
import torch


def self_rollout(tau_fn, q_start, q_goal, steps, step_size, dim, q_min=None, q_max=None):
    """Roll out the field's own -grad(tau) path from q_start toward q_goal.

    Args:
        tau_fn: callable (points (M,2*dim)) -> tau (M,) or (M,1), differentiable.
        q_start, q_goal: (M, dim) normalized configs.
        steps: K rollout steps. step_size: eta (unit-length step).
        q_min, q_max: optional (dim,) clamp bounds in normalized space.
    Returns:
        path: (M, K+1, dim). Index 0 is q_start (a leaf; losses on it have zero
        grad); indices 1..K are differentiable w.r.t. tau_fn's parameters.
    """
    pts = [q_start]
    q = q_start
    for _ in range(steps):
        if not q.requires_grad:
            q = q.detach().requires_grad_(True)
        inp = torch.cat([q, q_goal], dim=1)                 # (M, 2*dim)
        tau = tau_fn(inp).reshape(-1)                       # (M,)
        g = torch.autograd.grad(tau.sum(), q, create_graph=True)[0]   # (M, dim)
        g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
        q = q - step_size * g
        if q_min is not None:
            q = q.clamp(min=q_min, max=q_max)   # keyword form: robust for tensor bounds
        pts.append(q)
    return torch.stack(pts, dim=1)                          # (M, K+1, dim)


def _speed_from_dobs(d_obs, margin, offset):
    """Map signed obstacle distance to speed in [offset/margin, 1] (matches data-gen)."""
    d = torch.clamp(d_obs, min=offset, max=margin)
    return d / margin


def rollout_and_path_loss(tau_fn, q_start, q_goal, oracle_query, safe_neighbor,
                          steps, step_size, dim, margin, cup_tilt_max, pour_tilt_max,
                          rank_margin, path_eik_weight, rank_weight,
                          use_rank=True, use_path_eik=True,
                          q_min=None, q_max=None, offset=None,
                          handover_accept=False, corner_overlap_deg=45.0,
                          handover_mono_tol=0.10, handover_weight=0.0,
                          self_collision_margin=0.002,
                          collision_only=False):
    """Path-integrated eikonal + collision/tilt repulsion ranking.

    tau_fn: (points (M,2*dim)) -> tau (M,), differentiable.
    oracle_query: (q_norm (M,dim)) -> dict d_obs/d_self/tilt_cup/tilt_pour/ee_y (detached).
    safe_neighbor: (q_norm (M,dim)) -> (q_safe (M,dim) detached, found_mask (M,) bool).
    Returns dict {path_eik, rank}.
    """
    if offset is None:
        offset = margin / 10.0 if margin > 0 else 0.0
    path = self_rollout(tau_fn, q_start, q_goal, steps, step_size, dim, q_min, q_max)
    B, Kp1, _ = path.shape
    flat = path.reshape(B * Kp1, dim)                       # (B*(K+1), dim)
    goal_rep = q_goal.unsqueeze(1).expand(B, Kp1, dim).reshape(B * Kp1, dim)

    info = oracle_query(flat.detach())                      # all detached
    if margin > 0:
        s = _speed_from_dobs(info["d_obs"], margin, offset)
    else:
        s = torch.ones(B * Kp1, device=flat.device)

    # --- path-integrated eikonal ---
    path_eik = flat.new_zeros(())
    if use_path_eik:
        q_var = flat.detach().requires_grad_(True)
        inp = torch.cat([q_var, goal_rep], dim=1)
        tau = tau_fn(inp).reshape(-1)
        grad = torch.autograd.grad(tau.sum(), q_var, create_graph=True)[0]
        gnorm = torch.sqrt((grad ** 2).sum(1) + 1e-8)
        path_eik = path_eik_weight * ((torch.sqrt(s * gnorm) - 1.0) ** 2).mean()

    # --- collision/tilt repulsion ranking ---
    rank = flat.new_zeros(())
    handover_mono = flat.new_zeros(())
    if use_rank:
        if collision_only:
            tilt_violate = torch.zeros_like(info["d_obs"], dtype=torch.bool)
        elif handover_accept and "ee_x" in info:
            from morph_orient import corner_blend_param
            ee_x = info["ee_x"].reshape(B, Kp1)
            ee_y = info["ee_y"].reshape(B, Kp1)
            cup = info["tilt_cup"].reshape(B, Kp1)
            pour = info["tilt_pour"].reshape(B, Kp1)
            s = corner_blend_param(ee_x, ee_y, overlap_deg=corner_overlap_deg)
            cup_wing = (s <= 1e-6)
            pour_wing = (s >= 1.0 - 1e-6)
            tilt_violate = (
                (cup_wing & (cup > cup_tilt_max))
                | (pour_wing & (pour > pour_tilt_max))
            ).reshape(B * Kp1)
        else:
            tilt_violate = (
                (info["tilt_cup"] > cup_tilt_max)
                | (info["tilt_pour"] > pour_tilt_max)
            )
        d_self = info.get("d_self")
        if d_self is not None:
            collision_violate = (
                (info["d_obs"] < margin)
                | (d_self < self_collision_margin)
            )
        else:
            collision_violate = info["d_obs"] < margin
        violate = collision_violate | tilt_violate       # (B*(K+1),)
        if violate.any():
            qv = flat[violate]                              # (V, dim) differentiable
            gv = goal_rep[violate]
            q_safe, found = safe_neighbor(flat[violate].detach())
            tau_v = tau_fn(torch.cat([qv, gv], dim=1)).reshape(-1)            # (V,)
            loss_terms = []
            if found.any():
                tau_safe = tau_fn(torch.cat([q_safe[found].detach(),
                                             gv[found]], dim=1)).reshape(-1)
                # Violating rollout states should be more expensive than a nearby
                # detached safe state for the same goal.
                hinge = torch.clamp(tau_safe - tau_v[found] + rank_margin, min=0.0)
                loss_terms.append(hinge)
            no_safe = ~found
            if no_safe.any():
                # fallback: push violating tau up toward at least rank_margin
                loss_terms.append(torch.clamp(rank_margin - tau_v[no_safe], min=0.0))
            rank = rank_weight * torch.cat(loss_terms).mean()

        if (handover_accept and handover_weight > 0.0
                and "ee_x" in info and "tilt_cup" in info and "tilt_pour" in info):
            from morph_orient import corner_blend_param
            ee_x = info["ee_x"].reshape(B, Kp1)
            ee_y = info["ee_y"].reshape(B, Kp1)
            cup = info["tilt_cup"].reshape(B, Kp1)
            pour = info["tilt_pour"].reshape(B, Kp1)
            s = corner_blend_param(ee_x, ee_y, overlap_deg=corner_overlap_deg)
            corner = (s > 1e-6) & (s < 1.0 - 1e-6)
            denom = (cup + pour).clamp(min=1e-3)
            h = cup / denom
            bad_step = corner[:, :-1] & corner[:, 1:] & (
                (h[:, 1:] - h[:, :-1]) < -handover_mono_tol)
            if bad_step.any():
                q_prev = path[:, :-1, :][bad_step]
                q_next = path[:, 1:, :][bad_step]
                g_bad = q_goal.unsqueeze(1).expand(B, steps, dim)[bad_step]
                tau_prev = tau_fn(torch.cat([q_prev, g_bad], dim=1)).reshape(-1)
                tau_next = tau_fn(torch.cat([q_next, g_bad], dim=1)).reshape(-1)
                hinge = torch.clamp(tau_prev - tau_next + rank_margin, min=0.0)
                handover_mono = handover_weight * hinge.mean()
    return {"path_eik": path_eik, "rank": rank, "handover_mono": handover_mono}


def _corner_tilt_violate(info, cup_tilt_max, pour_tilt_max,
                         handover_accept=False, corner_overlap_deg=45.0):
    """Return per-config tilt-invalid mask using the eval wing rule."""
    if handover_accept and "ee_x" in info and "ee_y" in info:
        from morph_orient import corner_blend_param
        s = corner_blend_param(
            info["ee_x"], info["ee_y"], overlap_deg=corner_overlap_deg)
        cup_wing = s <= 1e-6
        pour_wing = s >= 1.0 - 1e-6
        return (
            (cup_wing & (info["tilt_cup"] > cup_tilt_max))
            | (pour_wing & (info["tilt_pour"] > pour_tilt_max))
        )
    return (
        (info["tilt_cup"] > cup_tilt_max)
        | (info["tilt_pour"] > pour_tilt_max)
    )


def local_segment_barrier_loss(tau_fn, q_anchor, oracle_query, *,
                               dim, step_size, n_dirs, margin,
                               cup_tilt_max, pour_tilt_max,
                               rank_margin, weight,
                               self_collision_margin=0.002,
                               q_min=None, q_max=None,
                               handover_accept=False,
                               corner_overlap_deg=45.0,
                               check_midpoint=True,
                               collision_only=False):
    """Rank random local unsafe directions above random local safe directions.

    This is no-teacher local supervision: directions are random, and labels come
    only from detached collision/tilt feasibility. It does not use, store, or
    imitate a teacher trajectory or teacher direction.
    """
    if q_anchor.numel() == 0 or n_dirs <= 1 or weight <= 0.0:
        zero = q_anchor.new_zeros(())
        return {"segment_barrier": zero, "invalid_rate": zero}

    B = q_anchor.shape[0]
    dirs = torch.randn(B, n_dirs, dim, device=q_anchor.device,
                       dtype=q_anchor.dtype)
    dirs = dirs / dirs.norm(dim=2, keepdim=True).clamp(min=1e-8)
    q_next = q_anchor[:, None, :] + step_size * dirs
    if q_min is not None:
        q_next = q_next.clamp(min=q_min.view(1, 1, dim),
                              max=q_max.view(1, 1, dim))

    flat_next = q_next.reshape(B * n_dirs, dim)
    check_points = [flat_next.detach()]
    if check_midpoint:
        flat_anchor = q_anchor[:, None, :].expand(B, n_dirs, dim).reshape(
            B * n_dirs, dim)
        check_points.append((0.5 * (flat_anchor + flat_next)).detach())
    all_check = torch.cat(check_points, dim=0)
    info = oracle_query(all_check)
    d_obs = info["d_obs"].reshape(len(check_points), B * n_dirs)
    d_self_raw = info.get("d_self")
    if d_self_raw is not None:
        d_self = d_self_raw.reshape(len(check_points), B * n_dirs)
        collision_bad = (d_obs < margin) | (d_self < self_collision_margin)
    else:
        collision_bad = d_obs < margin
    if collision_only:
        tilt_bad = torch.zeros_like(d_obs, dtype=torch.bool)
    else:
        tilt_bad = _corner_tilt_violate(
            info, cup_tilt_max, pour_tilt_max,
            handover_accept=handover_accept,
            corner_overlap_deg=corner_overlap_deg,
        ).reshape(len(check_points), B * n_dirs)
    invalid = (collision_bad | tilt_bad).any(dim=0).reshape(B, n_dirs)
    valid = ~invalid

    anchor_rep = q_anchor[:, None, :].expand(B, n_dirs, dim).reshape(
        B * n_dirs, dim)
    local_pairs = torch.cat([anchor_rep, flat_next], dim=1)
    tau = tau_fn(local_pairs).reshape(B, n_dirs)

    inf = torch.full_like(tau, float("inf"))
    valid_ref = torch.where(valid, tau, inf).min(dim=1).values
    has_valid = torch.isfinite(valid_ref)
    has_invalid = invalid.any(dim=1)
    usable = has_valid & has_invalid
    if not usable.any():
        zero = tau.new_zeros(())
        return {
            "segment_barrier": zero,
            "invalid_rate": invalid.float().mean().detach(),
        }

    hinge = torch.relu(
        valid_ref[:, None].detach() - tau + rank_margin)
    loss = weight * hinge[usable[:, None] & invalid].mean()
    return {
        "segment_barrier": loss,
        "invalid_rate": invalid.float().mean().detach(),
    }
