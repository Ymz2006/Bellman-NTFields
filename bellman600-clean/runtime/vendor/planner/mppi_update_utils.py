"""Small, testable update rules for NTField MPPI integration."""

from __future__ import annotations

import torch


def constant_arc_update(update, current, goal, step_size):
    """Preserve update direction and integrate at a fixed local arc length."""
    update_norm = torch.norm(update, dim=1, keepdim=True)
    remaining_dist = torch.norm(goal - current, dim=1, keepdim=True)
    target_arc = torch.minimum(
        remaining_dist, torch.full_like(remaining_dist, step_size))
    scaled = update / update_norm.clamp(min=1e-8) * target_arc
    return torch.where(update_norm > 1e-8, scaled, update)


def budget_feasible_update(update, current, goal, step_size, remaining_steps):
    """Raise only updates too short to reach within the integration budget.

    The returned direction is always the learned MPPI direction. No goal
    direction is injected; the goal contributes only the scalar lower bound on
    arc length required per remaining step.
    """
    if remaining_steps <= 0:
        raise ValueError("remaining_steps must be positive")
    update_norm = torch.norm(update, dim=1, keepdim=True)
    remaining_dist = torch.norm(goal - current, dim=1, keepdim=True)
    min_required_arc = remaining_dist / float(remaining_steps)
    target_arc = torch.maximum(update_norm, min_required_arc)
    target_arc = torch.minimum(
        target_arc, torch.full_like(target_arc, step_size))
    target_arc = torch.minimum(target_arc, remaining_dist)
    scaled = update / update_norm.clamp(min=1e-8) * target_arc
    return torch.where(update_norm > 1e-8, scaled, update)
