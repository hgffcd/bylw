"""Validity repair only: all-pair obstacle separation and legal route anchors.

No future observation, pursuer performance or capture-ring feasibility test.
"""
import numpy as np
from experiments.c179_random_map_generalization.world import RandomMapWorld

ANCHORS = np.array([[0., 4.], [8., 12.], [25., 12.], [8., 2.], [25., 0.]])


def validity(obstacles, radii, scale=1.6):
    distance = np.linalg.norm(obstacles[:, None]-obstacles[None], axis=-1)-radii[:, None]-radii[None]
    pair = distance[np.triu_indices(len(radii), 1)]
    anchor = (np.linalg.norm(ANCHORS[:, None]-obstacles, axis=-1)-radii).min(1)*scale
    return bool((pair>=1.5-1e-8).all() and (anchor>=1.4).all()), float(pair.min()*scale), float(anchor.min())


def sample_valid(rng, n_obstacles, shape_profile, layout_profile, scale):
    for attempt in range(5000):
        local, radii = RandomMapWorld._sample_layout(rng, n_obstacles, shape_profile, layout_profile)
        valid, pair, anchor = validity(local, radii, scale)
        if valid:
            return local, radii, attempt, pair, anchor
    raise RuntimeError('C191 valid-map sampler exhausted; do not omit the seed')
