"""Parallelism-independent episode seed mapping for matched evaluation."""

from __future__ import annotations


UINT32_MAX_PLUS_ONE = 2**32


def episode_seed(base_seed: int, episode_id: int) -> int:
    if base_seed < 0 or episode_id < 0:
        raise ValueError("base_seed and episode_id must be non-negative")
    return (int(base_seed) + int(episode_id)) % UINT32_MAX_PLUS_ONE


def parallel_episode_seeds(base_seed: int, first_episode_id: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    return tuple(
        episode_seed(base_seed, first_episode_id + offset) for offset in range(count)
    )
