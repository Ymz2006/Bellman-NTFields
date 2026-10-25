"""Numerically stable gradient clipping used by the baseline optimizer."""
import torch

def stable_clip_grad_norm_(
    parameters,
    max_norm: float,
) -> torch.Tensor:
    """Clip a global L2 norm without overflowing finite float32 gradients."""
    gradients = [
        parameter.grad for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros((), dtype=torch.float64)
    maxima = torch.stack([
        gradient.detach().abs().max().to(torch.float64)
        for gradient in gradients
    ])
    maximum = maxima.max()
    if not bool(torch.isfinite(maximum)):
        return maximum
    if float(maximum) == 0.0:
        return maximum
    scaled_square_sum = sum(
        (gradient.detach() / maximum.to(gradient.dtype)).square().sum(
            dtype=torch.float64)
        for gradient in gradients
    )
    total_norm = maximum * torch.sqrt(scaled_square_sum)
    coefficient = (max_norm / (total_norm + 1.0e-12)).clamp(max=1.0)
    torch._foreach_mul_(gradients, coefficient.to(gradients[0].dtype))
    return total_norm
