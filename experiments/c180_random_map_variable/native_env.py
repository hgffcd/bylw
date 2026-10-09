"""Safety accounts for surviving/withdrawn body separation at physics ticks."""
import numpy as np
from experiments.c169_comm_recovery.native_env import ResponseEnv


class AuditEnv(ResponseEnv):
    def record_safety(self):
        super().record_safety()
        if self.pilot is None:
            return
        w = self.pilot
        xyz = np.concatenate([w.p, w.z[..., None]], axis=-1)
        for i in range(xyz.shape[1]):
            for j in range(i+1, xyz.shape[1]):
                cross = w.active[:, i] ^ w.active[:, j]
                d = np.linalg.norm(xyz[:, i]-xyz[:, j], axis=-1)
                self.failures[:, 1] |= cross & (d < 1.)
                self.minima[:, 1] = np.minimum(self.minima[:, 1], np.where(cross, d, np.inf))
