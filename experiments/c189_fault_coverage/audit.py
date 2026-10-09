"""Temporal loss/recapture audit: no credit from before the final withdrawal."""
import numpy as np

from experiments.c180_random_map_variable.metrics import post_fault_metrics


def intervals(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def temporal_coverage(measured, geometry, unsafe, failure_ticks, dt=.1):
    measured, geometry = np.asarray(measured, bool), np.asarray(geometry, bool)
    if measured.ndim != 1 or geometry.shape != measured.shape:
        raise ValueError('one episode of aligned observation/judge samples required')
    if not failure_ticks or not 0 <= min(failure_ticks) <= max(failure_ticks) < len(measured):
        raise ValueError('withdrawal ticks must be inside the episode')
    losses = intervals(~measured)
    final_tick = max(failure_ticks)
    result = dict(lost_at_withdrawals=[not bool(measured[t]) for t in failure_ticks],
                  observed_at_end=bool(measured[-1]),
                  loss_intervals_s=[[float(s*dt), float(e*dt)] for s, e in losses])
    for phase, tick in [('first', min(failure_ticks)), ('last', final_tick)]:
        overlap = [(max(s, tick), e) for s, e in losses if e > tick]
        longest = max((e-s for s, e in overlap), default=0)
        reacquired = bool(overlap and overlap[-1][1] < len(measured))
        capture_after_reacquisition = False
        hold = 0.
        if reacquired:
            start = max(final_tick, overlap[-1][1])
            score = post_fault_metrics(geometry[:, None], [unsafe], start, dt=dt)
            capture_after_reacquisition = bool(score['safe_capture5'][0])
            hold = float(score['max_post_hold_s'][0])
        result.update({
            f'loss_after_{phase}_withdrawal': bool(overlap),
            f'loss_after_{phase}_total_s': float(sum(e-s for s, e in overlap)*dt),
            f'loss_after_{phase}_longest_s': float(longest*dt),
            f'reacquired_after_{phase}_last_loss': reacquired,
            f'safe_capture_after_{phase}_last_reacquisition': capture_after_reacquisition,
            f'hold_after_{phase}_last_reacquisition_s': hold,
        })
    return result
