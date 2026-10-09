"""Visibility history advances once per environment and physical policy tick."""
import torch


def advance_visibility_history(lost_steps, recapture_count, previous_visible,
                               visible, last_tick, current_tick):
    fresh = last_tick != current_tick
    lost = torch.where(fresh, torch.where(visible, 0, lost_steps + 1), lost_steps)
    recaptured = recapture_count + (fresh & visible & ~previous_visible).long()
    # Refreshes after a partial reset may initialize visibility at the same
    # tick, but must not add elapsed time or events to untouched environments.
    return lost, recaptured, visible.clone(), torch.full_like(last_tick, current_tick)
