"""CPU-only reset geometry aligned with the frozen 2-D tracked-start contract.

The sampler intentionally has no Isaac or Torch dependency.  It reproduces
the *semantics* of ``PursuitLostEnv._sample_initially_visible_pursuers`` for a
fixed centered Isaac map: a free target near the historical right-side start,
unobstructed candidates inside the sensing range, and farthest-point pursuer
selection.  It does not claim bitwise identity with the 2-D RNG stream because
the 2-D environment samples a random obstacle map before the reset positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackedRingReset:
    target_xy_m: np.ndarray
    pursuer_xy_m: np.ndarray
    pursuer_yaw_rad: np.ndarray
    maximum_angular_gap_rad: float
    near_ring_curriculum_applied: bool = False


def sample_irregular_obstacle_layout(
    seed: int,
    obstacle_radius_m: np.ndarray,
    *,
    arena_size_m: float = 100.0,
    sampling_half_extent_m: float = 32.0,
    minimum_surface_gap_m: float = 2.0,
    central_keepout_m: float = 5.0,
    maximum_attempts_per_obstacle: int = 2_000,
) -> np.ndarray:
    """Sample one deterministic, non-grid static-cylinder layout.

    Positions change between episodes but remain fixed during an episode.  The
    radii are intentionally not randomized because each Isaac cylinder keeps
    its authored collision shape; only its kinematic root pose is changed.
    """
    radii = np.asarray(obstacle_radius_m, dtype=float)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("obstacle_radius_m must be a nonempty vector")
    if np.any(radii <= 0.0):
        raise ValueError("obstacle radii must be positive")
    if not 0.0 < sampling_half_extent_m < 0.5 * arena_size_m:
        raise ValueError("sampling extent must lie inside the arena")
    if minimum_surface_gap_m < 0.0 or central_keepout_m < 0.0:
        raise ValueError("layout clearances must be non-negative")
    if maximum_attempts_per_obstacle <= 0:
        raise ValueError("maximum attempts must be positive")

    generator = np.random.default_rng(int(seed) % (2**32))
    positions: list[np.ndarray] = []
    # Place large cylinders first to make rejection sampling robust, then
    # restore the authored obstacle order so geometry and radius indices agree.
    placed_by_index: dict[int, np.ndarray] = {}
    for index in np.argsort(-radii):
        radius = float(radii[index])
        limit = min(
            sampling_half_extent_m,
            0.5 * arena_size_m - radius - 2.0,
        )
        accepted = None
        for _ in range(maximum_attempts_per_obstacle):
            candidate = generator.uniform(-limit, limit, size=2)
            if np.linalg.norm(candidate) <= radius + central_keepout_m:
                continue
            if any(
                np.linalg.norm(candidate - other)
                <= radius + float(radii[other_index]) + minimum_surface_gap_m
                for other_index, other in placed_by_index.items()
            ):
                continue
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError(
                f"could not sample obstacle {int(index)} for seed {seed}"
            )
        placed_by_index[int(index)] = accepted
        positions.append(accepted)
    del positions
    return np.stack([placed_by_index[index] for index in range(radii.size)])


def segment_circle_intersects(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> bool:
    """Use the same closest-point LOS test as the frozen 2-D environment."""
    direction = end - start
    squared_length = float(direction @ direction)
    if squared_length < 1.0e-12:
        return float(np.linalg.norm(start - center)) <= radius
    projection = -float((start - center) @ direction) / squared_length
    projection = min(max(projection, 0.0), 1.0)
    closest = start + projection * direction
    return float(np.linalg.norm(closest - center)) <= radius


def maximum_angular_gap(points: np.ndarray, center: np.ndarray) -> float:
    angles = np.sort(
        np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2.0 * math.pi)
    )
    gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * math.pi)))
    return float(gaps.max())


def _target_is_free(
    point: np.ndarray,
    arena_size_m: float,
    obstacle_xy_m: np.ndarray,
    obstacle_radius_m: np.ndarray,
) -> bool:
    half = 0.5 * arena_size_m
    if np.any(point < -half + 1.0) or np.any(point > half - 1.0):
        return False
    return all(
        np.linalg.norm(point - center) > radius + 2.0
        for center, radius in zip(obstacle_xy_m, obstacle_radius_m)
    )


def sample_tracked_ring_reset(
    seed: int,
    obstacle_xy_m: np.ndarray,
    obstacle_radius_m: np.ndarray,
    *,
    arena_size_m: float = 100.0,
    sense_range_m: float = 25.0,
    capture_radius_m: float = 6.0,
    pursuer_radius_m: float = 0.5,
    safe_agent_distance_m: float = 2.5,
    target_center_xy_m: tuple[float, float] = (35.0, 0.0),
    target_spread_m: float = 6.0,
    visual_occlusion: bool = True,
    n_pursuers: int = 3,
) -> TrackedRingReset:
    """Sample one deterministic, visible, non-capturing tracked-ring reset."""
    obstacle_xy = np.asarray(obstacle_xy_m, dtype=float)
    obstacle_radius = np.asarray(obstacle_radius_m, dtype=float)
    if obstacle_xy.ndim != 2 or obstacle_xy.shape[-1] != 2:
        raise ValueError("obstacle_xy_m must have shape [obstacle, 2]")
    if obstacle_radius.shape != obstacle_xy.shape[:1]:
        raise ValueError("obstacle_radius_m shape mismatch")
    if not sense_range_m > capture_radius_m > 0.0:
        raise ValueError("sense_range_m must exceed positive capture_radius_m")
    if n_pursuers <= 0:
        raise ValueError("n_pursuers must be positive")

    generator = np.random.default_rng(int(seed) % (2**32))
    target_center = np.asarray(target_center_xy_m, dtype=float)
    target = target_center.copy()
    for _ in range(100):
        candidate = target_center + generator.uniform(
            -target_spread_m, target_spread_m, size=2
        )
        if _target_is_free(candidate, arena_size_m, obstacle_xy, obstacle_radius):
            target = candidate
            break
    else:
        if not _target_is_free(target, arena_size_m, obstacle_xy, obstacle_radius):
            raise RuntimeError("could not place a free tracked-start target")

    outer = min(0.8 * sense_range_m, 0.25 * arena_size_m)
    lower = capture_radius_m + max(2.0, 2.0 * pursuer_radius_m)
    if n_pursuers == 3:
        # Preserve the frozen P5 reset radii bit-for-bit for the default path.
        radii = np.asarray(
            (
                max(lower, outer),
                max(lower, min(0.7 * sense_range_m, 0.22 * arena_size_m)),
                max(lower, min(0.6 * sense_range_m, 0.18 * arena_size_m)),
            ),
            dtype=float,
        )
    else:
        # P5-C uses a monotone set of visible rings.  The outer radius remains
        # below the sensing range and all additional agents get a candidate
        # ring rather than silently truncating the reset to three vehicles.
        fractions = np.linspace(0.80, 0.55, n_pursuers)
        radii = np.clip(fractions * sense_range_m, lower, outer)
    phase = float(generator.uniform(-math.pi / 36.0, math.pi / 36.0))
    angles = math.pi + phase + np.linspace(-math.pi, math.pi, 144, endpoint=False)
    candidates: list[np.ndarray] = []
    half = 0.5 * arena_size_m
    for radius in radii:
        if not (capture_radius_m < radius <= sense_range_m):
            continue
        for angle in angles:
            point = target + radius * np.asarray(
                [math.cos(angle), math.sin(angle)], dtype=float
            )
            if np.any(point < -half + 1.0) or np.any(point > half - 1.0):
                continue
            if any(
                np.linalg.norm(point - center)
                <= obstacle_r + pursuer_radius_m + 0.5
                for center, obstacle_r in zip(obstacle_xy, obstacle_radius)
            ):
                continue
            if visual_occlusion and any(
                segment_circle_intersects(point, target, center, obstacle_r)
                for center, obstacle_r in zip(obstacle_xy, obstacle_radius)
            ):
                continue
            candidates.append(point)

    selected: list[np.ndarray] = []
    minimum_separation = max(
        safe_agent_distance_m, 2.0 * pursuer_radius_m + 0.5
    )
    while candidates and len(selected) < n_pursuers:
        if not selected:
            scores = [point[0] for point in candidates]
            index = int(np.argmin(scores))
        else:
            scores = [
                min(np.linalg.norm(point - other) for other in selected)
                for point in candidates
            ]
            index = int(np.argmax(scores))
        point = candidates.pop(index)
        if selected and min(
            np.linalg.norm(point - other) for other in selected
        ) < minimum_separation:
            continue
        selected.append(point)
    if len(selected) != n_pursuers:
        raise RuntimeError(
            f"could not place {n_pursuers} tracked-start pursuers for seed {seed}"
        )

    pursuer_xy = np.asarray(selected, dtype=float)
    delta = target[None, :] - pursuer_xy
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    return TrackedRingReset(
        target_xy_m=target,
        pursuer_xy_m=pursuer_xy,
        pursuer_yaw_rad=yaw,
        maximum_angular_gap_rad=maximum_angular_gap(pursuer_xy, target),
    )


def sample_near_ring_curriculum_reset(
    seed: int,
    original: TrackedRingReset,
    obstacle_xy_m: np.ndarray,
    obstacle_radius_m: np.ndarray,
    *,
    arena_size_m: float = 100.0,
    pursuer_radius_m: float = 0.5,
    safe_agent_distance_m: float = 2.5,
    maximum_attempts: int = 128,
) -> TrackedRingReset:
    """C96 training-only perturbations, never changing the map or target.

    All radii exceed the frozen 5.5 m strong-ring outer edge. Initial yaw is
    target-facing, as in the parent reset; velocities and phase gates are not
    altered. An infeasible map keeps its original reset rather than being
    resampled or omitted. Circle envelopes conservatively include box shapes.
    """
    centers = np.asarray(obstacle_xy_m, dtype=float)
    obstacle_radii = np.asarray(obstacle_radius_m, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("obstacle_xy_m must have shape [obstacle, 2]")
    if obstacle_radii.shape != centers.shape[:1]:
        raise ValueError("obstacle_radius_m shape mismatch")
    if original.pursuer_xy_m.shape != (3, 2):
        raise ValueError("C96 is preregistered for three pursuers only")
    if maximum_attempts <= 0:
        raise ValueError("maximum_attempts must be positive")
    generator = np.random.default_rng((int(seed) ^ 0xC960904) % (2**32))
    target = original.target_xy_m
    minimum_separation = max(safe_agent_distance_m, 2 * pursuer_radius_m + 0.5)
    for _ in range(maximum_attempts):
        radii = generator.uniform(5.6, 6.2, 3)
        angles = (
            generator.uniform(-math.pi, math.pi)
            + np.arange(3) * (2 * math.pi / 3)
            + generator.uniform(-math.pi / 18, math.pi / 18, 3)
        )
        points = target + radii[:, None] * np.stack(
            (np.cos(angles), np.sin(angles)), axis=-1
        )
        if np.any(np.abs(points) > arena_size_m / 2 - 1.0):
            continue
        if any(np.linalg.norm(points[i] - points[j]) < minimum_separation
               for i in range(3) for j in range(i + 1, 3)):
            continue
        if any(
            np.linalg.norm(point - center) <= radius + pursuer_radius_m + 0.5
            or segment_circle_intersects(point, target, center, radius)
            for point in points for center, radius in zip(centers, obstacle_radii)
        ):
            continue
        delta = target - points
        return TrackedRingReset(
            target_xy_m=target.copy(),
            pursuer_xy_m=points,
            pursuer_yaw_rad=np.arctan2(delta[:, 1], delta[:, 0]),
            maximum_angular_gap_rad=maximum_angular_gap(points, target),
            near_ring_curriculum_applied=True,
        )
    return original
