"""Strict post-event 5 s hold; no pre-fault credit or sample deletion."""
import numpy as np


def post_fault_metrics(geometry, unsafe, failure_tick, dt=.1, hold_s=5.):
    g = np.asarray(geometry, bool)
    if not 0 <= failure_tick < len(g):
        raise ValueError('fault must occur inside the recorded episode')
    post = g[failure_tick:]
    run = np.zeros(g.shape[1], int)
    maximum = run.copy()
    hit = np.full(g.shape[1], np.nan)
    for i, valid in enumerate(post):
        run = np.where(valid, run+1, 0)
        maximum = np.maximum(maximum, run)
        done = (run >= round(hold_s/dt)) & np.isnan(hit)
        hit[done] = (i+1)*dt
    return dict(safe_capture5=np.isfinite(hit)&~np.asarray(unsafe, bool),
                post_geometry_fraction=post.mean(0), max_post_hold_s=maximum*dt,
                post_capture_delay_s=hit)
