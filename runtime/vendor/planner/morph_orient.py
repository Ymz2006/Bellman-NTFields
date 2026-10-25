"""Shared morphing-orientation tilt for UR5e cross. Single source of truth used by
BOTH pkl_to_training_data_multi_ur5e.py (labels) and ur5e_plan_eval_multi.py (acceptance).

In the overlap band [y_divide-overlap, y_divide+overlap] the effective tilt morphs from
pure cupping (cup edge, +Y) to pure pouring (pour edge, -Y). Outside the band it is the
pure zone tilt. Works on numpy arrays or torch tensors (duck-typed).
"""
def _blend_param(ee_y, y_divide, overlap):
    # s = 0 at cup edge (ee_y = y_divide+overlap), s = 1 at pour edge (ee_y = y_divide-overlap)
    s = (y_divide + overlap - ee_y) / (2.0 * overlap)
    # clamp to [0,1]
    try:
        s = s.clamp(0.0, 1.0)            # torch
    except AttributeError:
        import numpy as np
        s = np.clip(s, 0.0, 1.0)         # numpy
    return s

def _morph_combine(cup_tilt, pour_tilt, s, mode):
    """Blend cup_tilt->pour_tilt by progress s in [0,1]. Shared by the y-divide
    and corner morphs so both use identical linear/max semantics."""
    if mode == "linear":
        return (1.0 - s) * cup_tilt + s * pour_tilt
    elif mode == "max":
        a = (1.0 - s) * cup_tilt
        b = s * pour_tilt
        try:
            import torch
            if hasattr(a, "clamp"):
                return torch.maximum(a, b)
        except Exception:
            pass
        import numpy as np
        return np.maximum(a, b)
    else:
        raise ValueError(f"unknown morph mode: {mode}")


def morph_tilt_deg(cup_tilt, pour_tilt, ee_y, y_divide, overlap, mode="linear"):
    """cup_tilt, pour_tilt, ee_y: same-shape arrays/tensors (deg, deg, meters).
    Returns morphed tilt (deg). mode: 'linear' or 'max'. 1D y-divide geometry."""
    s = _blend_param(ee_y, y_divide, overlap)
    return _morph_combine(cup_tilt, pour_tilt, s, mode)


# --- 2D corner-table (L-shape) angular morph --------------------------------
# The y-divide morph above blends along ee_y, which is geometrically wrong for the
# corner-table: there the transition is a 90-degree SWEEP through the inner corner,
# from the cup wing (extending toward -X) to the pour wing (extending toward -Y).
# We parameterize progress by the EE's polar bearing atan2(ee_y, ee_x) around the
# robot base. Cup-arm bearing ~= +pi (-X axis); pour-arm bearing ~= -pi/2 (-Y axis).
# The corner sweep goes clockwise from -X to -Y across the third quadrant.
#   s = 0 on the cup wing (pure cup-tilt target)
#   s = 1 on the pour wing (pure pour-tilt target)
#   s sweeps smoothly through the inner corner -> a single morphed target there.
# This gives the field one continuous, physically-correct orientation target through
# the corner, learned from the SAME zone-sampled configs (no hand-picked corner pairs).
import math as _math

_CUP_BEARING = _math.pi          # -X axis  (cup wing direction)
_POUR_BEARING = -_math.pi / 2.0  # -Y axis  (pour wing direction)
_CORNER_MID = -3.0 * _math.pi / 4.0  # inner diagonal bearing (-135deg)


def corner_blend_param(ee_x, ee_y, overlap_deg=45.0):
    """Progress s in [0,1] from cup wing (-X) to pour wing (-Y) by EE polar bearing.

    The cup wing points along -X (bearing +pi), the pour wing along -Y (-pi/2);
    the inner diagonal (corner midpoint) is at -3pi/4 (-135deg). We center the
    frame on that midpoint so the corner arc maps to phi in [-pi/2, +pi/2], far
    from the +-pi atan2 cut (no wrap headaches):
        cup wing  -> phi = -pi/4 ; diagonal -> phi = 0 ; pour wing -> phi = +pi/4.

    overlap_deg is the HALF-WIDTH (in degrees, around the diagonal) of the morph
    band -- the angular analogue of the y-divide `overlap`. Outside +-overlap_deg
    the target is PURE cup (s=0) or PURE pour (s=1); inside, s ramps linearly so
    EITHER constraint can carry the waypoint (OR-logic in the eval). The two
    wings keep their pure single-task orientation; only the corner morphs.
      overlap_deg = 45 -> ramp spans the whole 90deg arc (no pure-wing margin).
      overlap_deg = 25 -> wings pure beyond 25deg; central 50deg morphs.
    """
    band = _math.radians(overlap_deg)
    is_torch = hasattr(ee_x, "clamp")
    if is_torch:
        import torch
        theta = torch.atan2(ee_y, ee_x)
        phi = torch.atan2(torch.sin(theta - _CORNER_MID), torch.cos(theta - _CORNER_MID))
        s = (phi + band) / (2.0 * band)
        return s.clamp(0.0, 1.0)
    else:
        import numpy as np
        theta = np.arctan2(ee_y, ee_x)
        phi = np.arctan2(np.sin(theta - _CORNER_MID), np.cos(theta - _CORNER_MID))
        s = (phi + band) / (2.0 * band)
        return np.clip(s, 0.0, 1.0)


def morph_tilt_deg_corner(cup_tilt, pour_tilt, ee_x, ee_y, mode="linear",
                          overlap_deg=45.0):
    """Corner-table (L-shape) morphed tilt by EE polar bearing. Same linear/max
    semantics as morph_tilt_deg; geometry is the 2D 90-degree corner sweep.
    overlap_deg is the half-width of the morph band around the inner diagonal;
    outside it the wings keep pure cup/pour targets."""
    s = corner_blend_param(ee_x, ee_y, overlap_deg=overlap_deg)
    return _morph_combine(cup_tilt, pour_tilt, s, mode)
