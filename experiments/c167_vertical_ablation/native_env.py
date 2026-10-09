from experiments.c150_native_search.native_env import SearchEnv


class ResponseEnv(SearchEnv):
    vertical_mode = "free"

    def _pre_physics_step(self,actions):
        super()._pre_physics_step(actions)
        if self.vertical_mode == "fixed":
            # Keep the same horizontal command and sensing model; only replace
            # the pursuer altitude reference with the fixed 3 m reference.
            self.last_altitude[:] = 3.0
            self._high_level_command.desired_altitude_m[:] = 3.0
        # Capture at command time; subsequent feature queries call commands again.
        self.last_raw_command=self.pilot.raw_command.copy()
        self.last_guard_delta=self.pilot.guard_delta.copy()
        self.last_phase_mix=self.last_obs['phase_mix'].copy()
        self.last_deployed_searcher=self.last_obs['deployed_searcher'].copy()
        self.last_stage_entries=self.pilot.stage_entries.copy()
