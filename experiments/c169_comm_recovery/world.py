"""C169 communication-recovery worlds.

The physical LOS sensor remains available as ``natural_mask``.  During the
registered broadcast-fault window the high-level team receives no new shared
target measurement and uses its last belief.  This is a team-level shared
measurement outage, not a claim of fully decentralized per-agent perception.
"""
import numpy as np
from experiments.c166_unified_matrix.world import HiddenBranchWorld as _CornerWorld
from experiments.c166_unified_matrix.world import CorridorWorld as _CorridorWorld


class CommunicationMixin:
    def _init_comm(self, mode, start, duration):
        if mode not in {"ideal_share", "stale_share", "recovery_share", "adaptive_share"}:
            raise ValueError(mode)
        self.comm_mode = mode
        self.comm_fault_start = float(start)
        self.comm_fault_end = float(start + duration)
        self.natural_mask = np.zeros((len(self.seeds), 3), dtype=bool)
        self.comm_fault_active = False
        self.comm_fault_ticks = np.zeros(len(self.seeds), dtype=int)
        self.comm_fault_seen = np.zeros(len(self.seeds), dtype=bool)
        self._base_role_modes = self.role_modes.copy()

    def sensor(self):
        natural_mask, p, v = super().sensor()
        self.natural_mask = natural_mask.copy()
        active = (self.comm_mode != "ideal_share") and (self.comm_fault_start <= self.time < self.comm_fault_end)
        self.comm_fault_active = bool(active)
        self.role_modes[:] = self._base_role_modes
        if active:
            self.comm_fault_ticks += 1
            self.comm_fault_seen[:] = True
            if self.comm_mode == "stale_share":
                self.role_modes[:] = 0
            elif self.comm_mode == "recovery_share":
                self.role_modes[:] = 2
            elif self.comm_mode == "adaptive_share":
                # If a local camera still has a legal target ray, keep the
                # stale formation; if no local ray exists or the estimate is
                # already old, deploy all agents to search.
                local_visible = natural_mask.any(1)
                old = getattr(self, "age", np.zeros(len(self.seeds))) >= 2.5
                self.role_modes[:] = np.where(local_visible & ~old, 0, 2)
            # The natural local masks remain recorded, while the broadcast
            # interface exposes no new shared measurement to the coordinator.
            return np.zeros_like(natural_mask), np.full_like(p, np.nan), np.full_like(v, np.nan)
        return natural_mask, p, v

    def observe(self):
        # Restore the normal deployment before the next sensor/control cycle.
        self.role_modes[:] = self._base_role_modes
        return super().observe()

    def commands(self, o, actions=None):
        if self.comm_mode == "adaptive_share" and self.comm_fault_active:
            local_visible = self.natural_mask.any(1)
            old = np.asarray(o["age"]) >= 2.5
            self.role_modes[:] = np.where(local_visible & ~old, 0, 2)
        return super().commands(o, actions)


class CommCornerWorld(CommunicationMixin, _CornerWorld):
    def __init__(self, seeds, task, scale, mode, start, duration):
        super().__init__(seeds, task, scale)
        # Keep the already audited low-level response/staging layer identical
        # to C166 while this class changes only the shared communication mode.
        self.configure(slice(None), "staged_brake")
        self._init_comm(mode, start, duration)


class CommCorridorWorld(CommunicationMixin, _CorridorWorld):
    def __init__(self, seeds, task, scale, mode, start, duration):
        super().__init__(seeds, task, scale)
        self.configure(slice(None), "staged_brake")
        self._init_comm(mode, start, duration)


def make_world(scene, seeds, task, scale, mode, start, duration):
    cls = CommCornerWorld if scene == "corner" else CommCorridorWorld if scene == "corridor" else None
    if cls is None:
        raise ValueError(scene)
    return cls(seeds, task, scale, mode, start, duration)
