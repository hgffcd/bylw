"""Read-only snapshots at the actual command application phase."""
import numpy as np
from experiments.c180_random_map_variable.native_env import AuditEnv as ParentEnv
class AuditEnv(ParentEnv):
    def _pre_physics_step(self,actions):
        super()._pre_physics_step(actions)
        w=self.pilot;planner=w.recovery_planner
        self.last_progress_applied=getattr(planner,'bypass_applied',np.zeros(w.p.shape[:2],bool)).copy()
        self.last_progress_on=getattr(planner,'bypass_on',np.zeros(w.p.shape[:2],bool)).copy()
        self.last_progress_stall=getattr(planner,'bypass_stall',np.zeros(w.p.shape[:2])).copy()
        self.last_progress_base=planner.audit_last['base'].copy()
        self.last_progress_proposal=planner.audit_last['proposal'].copy()
