"""Frozen C195 scoring with legacy missing-bypass telemetry compatibility."""
import json
import numpy as np
from experiments.c180_random_map_variable.metrics import post_fault_metrics
from experiments.c189_fault_coverage.audit import temporal_coverage,intervals


def audit(group,paths):
    rows=[];reference=None;reference_summary=None
    for arm,path in paths.items():
        summary=json.loads((path/'summary.json').read_text())
        assert summary['recovery']==arm
        assert summary['map_protocol']=='valid' and summary['opponent']=='routed' and not summary['initial_ring_screen']
        assert summary['failure_indices']==group['indices'] and summary['failure_times']==group['times']
        with np.load(path/'trace.npz') as z:
            ticks=[round(t*10) for t in group['times']];last=ticks[-1]
            assert len(z['geometry_history'])==round(group['duration']*10)
            if reference is None:
                reference={k:z[k].copy() for k in ['seeds','obstacles','obstacle_radii','p_history','target_history','active_history']}
                reference_summary=summary
            for key in ['seeds','obstacles','obstacle_radii','active_history']:np.testing.assert_array_equal(z[key],reference[key])
            for key in ['p_history','target_history']:np.testing.assert_array_equal(z[key][:ticks[0]],reference[key][:ticks[0]])
            for key in ['duration','agents','shape_profile','layout_profile','scene','reconfig']:
                assert summary[key]==reference_summary[key],key
            if arm=='sector_bypass':assert not (z['bypass_history']&~z['active_history']).any()
            for window in group['windows']:
                end=last+window*10;unsafe=z['failure_history'][end-1].any(-1)
                score=post_fault_metrics(z['geometry_history'][:end],unsafe,last)
                if window==max(group['windows']):np.testing.assert_array_equal(score['safe_capture5'],z['safe_capture5'])
                for i,seed in enumerate(z['seeds']):
                    speed=np.r_[np.nan,np.linalg.norm(np.diff(z['target_history'][:end,i],axis=0),axis=-1)*10]
                    valid=z['geometry_history'][:end,i].copy();valid[:last]=False
                    holds=[(a,b) for a,b in intervals(valid) if b-a>=50]
                    row=dict(cohort=group['cohort'],group=group['label'],stratum=group['stratum'],arm=arm,
                        seed=int(seed),window_s=window,success=bool(score['safe_capture5'][i]),safe=not bool(unsafe[i]),
                        max_hold_s=float(score['max_post_hold_s'][i]),
                        first_hold5_after_withdrawal_s=float(score['post_capture_delay_s'][i]) if np.isfinite(score['post_capture_delay_s'][i]) else None,
                        observed_postfinal_fraction=float(z['measured_history'][last:end,i].mean()),
                        target_stationary_fraction=float((speed[last:]<.2).mean()),
                        moving_success=bool(holds and not unsafe[i] and np.median(speed[holds[0][0]:holds[0][1]])>1),
                        bypass_applied_s_per_agent=(z['bypass_history'][:end,i].sum(0)*.1).tolist() if 'bypass_history' in z else [0.]*group['agents'],
                        route_done=bool(z['route_done_history'][end-1,i]))
                    row.update(temporal_coverage(z['measured_history'][:end,i],z['geometry_history'][:end,i],unsafe[i],ticks))
                    rows.append(row)
    return rows
