"""Fixed-three strict visibility/geometry/whole-window safety, paired analysis."""
import json
import numpy as np
from experiments.c189_fault_coverage.audit import intervals
from experiments.c150_native_search.task import geometry3,visibility3,Task
from experiments.c145_feasibility_reset.planar import wrap
from experiments.c225_search_removal.design import ARMS,CONFIG


def hold_score(geometry,measured,speed,safe):
    holds=intervals(np.asarray(geometry)&np.asarray(measured))
    long=[(a,b) for a,b in holds if b-a>=50]
    moving=bool(long and np.median(speed[long[0][0]:long[0][0]+50])>1.)
    return dict(success=bool(long and safe),moving_success=bool(moving and safe),
        max_hold_s=max([b-a for a,b in holds],default=0)*.1,
        first_completion_s=(long[0][0]+50)*.1 if long else None,
        first_hold_speed=float(np.median(speed[long[0][0]:long[0][0]+50])) if long else None)


def sensing_audit(z):
    p=z['p_history'];t=z['target_history'];yaw=z['post_yaw'];zz=z['pursuer_z_history'];tz=z['target_z_history']
    d=t[:,:,None]-p;h=np.linalg.norm(d,axis=-1);dz=tz[:,:,None]-zz
    in_range=np.sqrt(h*h+dz*dz)<=20.
    fov=(np.abs(wrap(np.arctan2(d[...,1],d[...,0])-yaw))<=np.pi/4)&(np.abs(np.arctan2(dz,h))<=np.pi/6)
    obs=z['obstacles'];r=z['obstacle_radii'];rel=obs[None,:,None]-p[:,:,:,None]
    ratio=np.clip((rel*d[:,:,:,None]).sum(-1)/np.maximum((d*d).sum(-1),1e-9)[:,:,:,None],0,1)
    closest=p[:,:,:,None]+ratio[...,None]*d[:,:,:,None]
    ray_z=zz[:,:,:,None]+ratio*dz[:,:,:,None]
    blocked=((np.linalg.norm(closest-obs[None,:,None],axis=-1)<r[None,:,None])&(ray_z>=0)&(ray_z<=8)).any(-1)
    mask=in_range&fov&~blocked
    np.testing.assert_array_equal(mask,z['post_mask'])
    np.testing.assert_array_equal(mask.any(-1),z['measured_history'])
    # Ordered categories partition lost frames; mixed underlying causes remain possible.
    cause=np.full(mask.shape[:2],0,int)
    lost=~mask.any(-1);cause[lost]=3
    cause[lost&~(in_range&~blocked).any(-1)]=2
    cause[lost&~in_range.any(-1)]=1
    return cause


def audit_group(g,paths):
    ref=None;traces={};rows=[];audit={};seenmaps=None
    for arm,path in paths.items():
        s=json.loads((path/'summary.json').read_text())
        assert s['agents']==3 and s['n']==g['n'] and s['duration']==g['duration']
        assert s['failure_times']==[] and s['failure_indices']==[] and s['reconfig']=='active_polar'
        assert s['map_protocol']=='valid' and s['opponent']=='routed' and not s['initial_ring_screen']
        assert s['recovery']==CONFIG[arm]['recovery'] and s['search_mode']==CONFIG[arm]['search_mode']
        with np.load(path/'trace.npz') as zz:z={k:zz[k] for k in zz.files}
        traces[arm]=z
        assert z['active_history'].all() and len(z['p_history'])==round(g['duration']*10)
        np.testing.assert_array_equal(z['seeds'],np.arange(g['seed'],g['seed']+g['n']))
        equal_keys=['seeds','obstacles','obstacle_radii']+[k for k in z if k.startswith('reset_')]
        if ref is None:ref={k:z[k] for k in equal_keys}
        for k in equal_keys:np.testing.assert_array_equal(z[k],ref[k],err_msg=k)
        np.testing.assert_array_equal(z['input_p'],z['p_history'])
        np.testing.assert_array_equal(z['input_target'],z['belief_history'])
        np.testing.assert_array_equal(z['input_measured'],z['measured_history'])
        np.testing.assert_allclose(z['input_time'],np.arange(1,len(z['input_time'])+1)*.1,atol=1e-8,rtol=0)
        T,N=z['measured_history'].shape
        geo,_,_=geometry3(z['p_history'].reshape(T*N,3,2),z['pursuer_z_history'].reshape(T*N,3),
            z['target_history'].reshape(T*N,2),z['target_z_history'].reshape(T*N))
        np.testing.assert_array_equal(geo.reshape(T,N),z['geometry_history'])
        cause=sensing_audit(z)
        np.testing.assert_array_equal(z['cmd_measured'][1:],z['measured_history'][:-1])
        np.testing.assert_array_equal(z['cmd_mask'].any(-1),z['cmd_measured'])
        actual=z['cmd_search_selector'].sum(-1)
        assert np.isin(actual,[0,{'predict':0,'single':1,'all':3}[CONFIG[arm]['search_mode']]]).all()
        assert not (z['cmd_search_selector']&(z['cmd_phase_mix']<=1e-8)[...,None]).any()
        for end_s in g['windows']:
            end=round(end_s*10)
            for i,seed in enumerate(z['seeds']):
                safe=not bool(z['failure_history'][end-1,i].any())
                geom=z['geometry_history'][:end,i];vis=z['measured_history'][:end,i]
                speed=np.linalg.norm(np.diff(np.vstack([z['reset_target'][i],z['target_history'][:end,i]]),axis=0),axis=-1)*10
                loss=intervals(~vis);last=loss[-1] if loss else None
                reacquired=bool(last and last[1]<end)
                recover=hold_score(geom[last[1]:],vis[last[1]:],speed[last[1]:],safe) if reacquired else None
                strict=hold_score(geom,vis,speed,safe);legacy=hold_score(geom,np.ones_like(vis),speed,safe)
                rows.append(dict(group=g['label'],arm=arm,seed=int(seed),window_s=end_s,safe=safe,**strict,
                    legacy_success=legacy['success'],had_loss=bool(loss),reacquired=reacquired,
                    capture_after_last_reacquisition=bool(recover and recover['moving_success']),
                    loss_runs_s=[round((b-a)*.1,6) for a,b in loss],max_loss_s=max([b-a for a,b in loss],default=0)*.1,
                    total_loss_s=float((~vis).sum()*.1),loss_at_least_1_5=any(b-a>=15 for a,b in loss),
                    search_deployed=bool(actual[:end,i].any()),search_team_time_s=float((actual[:end,i]>0).sum()*.1),
                    search_agent_time_s=float(actual[:end,i].sum()*.1),
                    loss_cause_s={key:float((cause[:end,i]==j).sum()*.1) for j,key in [(1,'range'),(2,'occlusion'),(3,'field_of_view')]},
                    geometry_without_visibility_samples=int((geom&~vis).sum()),
                    z_min=float(z['pursuer_z_history'][:end,i].min()),z_max=float(z['pursuer_z_history'][:end,i].max()),
                    longest_legacy_hold_s=legacy['max_hold_s']))
        audit[arm]=dict(passed=True,frames=T*N,initial_states_and_noise_identical=True,sensing_geometry_reconstructed=True)
    for prefix in ['ordinary','budget']:
        for left,right in [('predict','single'),('predict','all'),('single','all')]:
            if prefix+'_'+left not in traces or prefix+'_'+right not in traces:continue
            a,b=[traces[prefix+'_'+mode] for mode in [left,right]];records=[]
            for i,seed in enumerate(a['seeds']):
                when=np.flatnonzero(a['cmd_search_selector'][:,i].any(-1)|b['cmd_search_selector'][:,i].any(-1))
                k=int(when[0]) if len(when) else len(a['p_history'])
                for key in ['p_history','target_history','command_history','measured_history','belief_history','cmd_altitude','pursuer_z_history','post_yaw']:
                    np.testing.assert_array_equal(a[key][:k,i],b[key][:k,i],err_msg=f'{prefix} {left}/{right} {seed} {key} before search')
                records.append(dict(seed=int(seed),first_search_command_time_s=k*.1 if len(when) else None,
                    full_trajectory_identical=bool(np.array_equal(a['p_history'][:,i],b['p_history'][:,i]))))
            audit[prefix+'_'+left+'_vs_'+right+'_before_search']=records
    return rows,audit


def summarize(rows):
    # Canonical seed order makes floating summaries independent of worker order.
    out={}
    for window in [30,45,60]:
        data={a:{r['seed']:r for r in rows if r['arm']==a and r['window_s']==window} for a in ARMS}
        if not all(data.values()):continue
        seeds=sorted(data[ARMS[0]])
        assert all(sorted(v)==seeds for v in data.values())
        result={'n':len(seeds),'arms':{},'contrasts':{}}
        for arm,records in data.items():
            v=[records[s] for s in seeds]
            result['arms'][arm]={k:sum(r[k] for r in v) for k in ['success','moving_success','legacy_success','safe','had_loss','loss_at_least_1_5','search_deployed','reacquired','capture_after_last_reacquisition']}
            result['arms'][arm].update(mean_total_loss_s=float(np.mean([r['total_loss_s'] for r in v])),mean_max_hold_s=float(np.mean([r['max_hold_s'] for r in v])))
        draws=np.random.default_rng(225).integers(0,len(seeds),(10000,len(seeds)))
        for core in ['ordinary','budget']:
            for mode in ['single','all']:
                base=data[core+'_predict'];active=data[core+'_'+mode]
                diff=np.array([float(active[s]['moving_success'])-float(base[s]['moving_success']) for s in seeds])
                result['contrasts'][core+'_'+mode+'_minus_predict']=dict(delta_pp=float(diff.mean()*100),bootstrap95_pp=(np.quantile(diff[draws].mean(-1),[.025,.975])*100).tolist(),gains=[s for s in seeds if diff[seeds.index(s)]>0],losses=[s for s in seeds if diff[seeds.index(s)]<0])
        out[str(window)]=result
    return out
