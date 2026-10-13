"""Read-only command-time/post-step diagnostics; no sensor or command calls."""
import numpy as np


def snapshot(env):
    w=env.pilot;o=env.last_obs;post=w.cached[1]
    active=(o['deployed_searcher']>=0)&(o['phase_mix']>1e-8)
    selector=np.arange(w.n_pursuers)[None]==o['deployed_searcher'][:,None]
    selector=np.where(w.role_modes[:,None]==2,True,selector)&active[:,None]
    return dict(cmd_phase_mix=o['phase_mix'].copy(),cmd_search_selector=selector.copy(),
        cmd_measured=o['measured'].copy(),cmd_age=o['age'].copy(),cmd_mask=o['mask'].copy(),
        cmd_altitude=env.last_altitude.copy(),post_mask=post['mask'].copy(),post_yaw=w.yaw.copy(),
        post_stage_on=w.stage_on.copy(),post_stage_entries=w.stage_entries.copy())


def reset_snapshot(w):
    fields={k:np.asarray(getattr(w,k)).copy() for k in ['p','z','yaw','target','tz','pv','routes','sigmas','noise']}
    # Noise is included to verify exactly paired sensor draws, not accessed by controllers here.
    return {'reset_'+k:v for k,v in fields.items()}
