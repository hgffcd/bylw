"""Scale-aware encirclement geometry for the optional P5-C experiment.

The helper is deliberately independent of the three-UAV C166--C170 judge.
It accepts an ``active`` mask so a failed UAV is excluded from the formation
geometry while the remaining members continue to be scored.
"""

from __future__ import annotations

import numpy as np


def active_geometry(
    p: np.ndarray,
    z: np.ndarray,
    target: np.ndarray,
    target_z: np.ndarray,
    active: np.ndarray,
    *,
    min_active: int = 3,
    radius_min: float = 4.0,
    radius_max: float = 8.0,
    radius_spread_max: float = 2.0,
    height_error_max: float = 1.0,
    gap_ratio: tuple[float, float] = (0.75, 1.25),
):
    """Return a scale-aware closed-encirclement decision.

    ``p`` is ``(B, M, 2)``, ``z`` is ``(B, M)``, and ``active`` is ``(B, M)``.
    Inactive members are ignored.  For ``N`` active members, each circular
    angular gap must lie in ``gap_ratio * 2*pi/N``.  The default ratio gives
    exactly 90--150 degrees for the existing three-agent case, preserving the
    C166 criterion at ``N=3`` while allowing tighter nominal gaps for N=4/5.
    """

    p = np.asarray(p, dtype=float)
    z = np.asarray(z, dtype=float)
    target = np.asarray(target, dtype=float)
    target_z = np.asarray(target_z, dtype=float)
    active = np.asarray(active, dtype=bool)
    if p.ndim != 3 or p.shape[-1] != 2:
        raise ValueError(f"p must have shape (B,M,2), got {p.shape}")
    b, m, _ = p.shape
    if z.shape != (b, m) or active.shape != (b, m):
        raise ValueError("z and active must have shape (B,M)")
    if target.shape != (b, 2) or target_z.shape != (b,):
        raise ValueError("target/target_z have incompatible batch shape")
    if min_active < 1:
        raise ValueError("min_active must be positive")
    if not (0 < gap_ratio[0] <= gap_ratio[1]):
        raise ValueError("gap_ratio must be positive and ordered")

    delta = p - target[:, None]
    radius = np.linalg.norm(delta, axis=-1)
    gap_min = np.full(b, np.nan)
    gap_max = np.full(b, np.nan)
    radius_spread = np.full(b, np.nan)
    gap_values = np.full((b, m), np.nan)
    count = active.sum(axis=1)
    valid = count >= min_active
    for i in range(b):
        ids = np.flatnonzero(active[i])
        if ids.size == 0:
            continue
        ri = radius[i, ids]
        radius_spread[i] = np.ptp(ri)
        angles = np.sort(np.arctan2(delta[i, ids, 1], delta[i, ids, 0]))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        gap_values[i, : ids.size] = gaps
        nominal = 2 * np.pi / ids.size
        # Keep the audited three-agent 90--150 degree gate.  For four/five
        # vehicles the same absolute tolerance would reject harmless small
        # slot errors, so the scale branch uses a slightly wider interval.
        local_gap_ratio = gap_ratio if ids.size <= 3 else (0.65, 1.35)
        local_radius_min = radius_min if ids.size <= 3 else max(3.5, radius_min - 0.5)
        gap_min[i] = local_gap_ratio[0] * nominal
        gap_max[i] = local_gap_ratio[1] * nominal
        valid[i] &= (
            (ri.min() >= local_radius_min)
            and (ri.max() <= radius_max)
            and (radius_spread[i] <= radius_spread_max)
            and (gaps.min() >= gap_min[i])
            and (gaps.max() <= gap_max[i])
            and np.all(np.abs(z[i, ids] - target_z[i]) <= height_error_max)
        )
    return valid, radius, gap_values, count, radius_spread


def uniform_slots(active: np.ndarray, phase: np.ndarray | float = 0.0) -> np.ndarray:
    """Return desired planar slot angles for every member in an active mask.

    Inactive entries are NaN.  Slots are ordered by active-agent index, so a
    controller can match the surviving vehicles to the nearest slot and keep
    the same phase after a member failure.
    """

    active = np.asarray(active, dtype=bool)
    if active.ndim != 2:
        raise ValueError("active must have shape (B,M)")
    b, m = active.shape
    phase = np.broadcast_to(np.asarray(phase, dtype=float), (b,))
    out = np.full((b, m), np.nan)
    for i in range(b):
        ids = np.flatnonzero(active[i])
        if ids.size:
            out[i, ids] = phase[i] + np.arange(ids.size) * 2 * np.pi / ids.size
    return out
