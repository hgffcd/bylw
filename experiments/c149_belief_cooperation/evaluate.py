"""All seeds, same physical initial conditions. Raw 10-Hz geometry is retained.

Safety is the inherited 50-Hz substep accumulated flags. It is not inferred
from sparse trajectory samples. 'post_hold5' starts at/after observation return.
"""
import argparse,json,hashlib,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .world import LocalWorld,Conditions,action_residual
from .policy import Policy,features
from .common import snapshot,write_json

CASES={
    'natural':Conditions(loss=0,asymmetric=False),
    'asymmetric':Conditions(loss=0,asymmetric=True),
    **{f'loss{str(d).replace(".","p")}':Conditions(loss=d) for d in [.8,1.2,2.,3.,5.]}}


def score(trace,conditions,seeds):
    t=trace['time'];g=trace['geometry'];safe=~trace['unsafe'];final=safe[-1]
    start=conditions.start;end=start+conditions.loss
    window=(t>start+1e-7)&(t<=end+5+1e-7)
    # Sensors are sampled before each dynamics step. Physical visibility can
    # delay reacquisition beyond the scheduled end of artificial interruption.
    observation_time=t-.1
    reacquired=(observation_time[:,None]>=end-1e-7)&trace['mask'].any(-1)
    first_index=np.argmax(reacquired,axis=0)
    reacquisition=np.where(reacquired.any(0),observation_time[first_index],np.inf)
    post=t[:,None]>reacquisition[None]+1e-7
    return5=post&(t[:,None]<=reacquisition[None]+5+1e-7)
    run=np.zeros(len(seeds),int);maxrun=run.copy()
    for idx in range(len(t)):
        run=np.where(post[idx]&g[idx]&safe[idx],run+1,0);maxrun=np.maximum(run,maxrun)
    first=np.full(len(seeds),np.nan)
    for idx in range(len(t)):
        mask=return5[idx]&g[idx]&safe[idx]&np.isnan(first);first[mask]=t[idx]-reacquisition[mask]
    duration=.1*window.sum()
    records=[]
    for i,seed in enumerate(seeds):
        records.append(dict(seed=int(seed),safe_full=bool(final[i]),
            window_geometry_fraction=float((g[window,i]&safe[window,i]).mean()),
            window_all_geometry=bool((g[window,i]&safe[window,i]).all()&final[i]),
            recovery_within5=bool(np.isfinite(first[i])&final[i]),
            recovery_delay=float(first[i]) if np.isfinite(first[i]) else None,
            reacquisition_delay=float(reacquisition[i]-end) if np.isfinite(reacquisition[i]) else None,
            post_hold5=bool((maxrun[i]>=50)&final[i]),
            natural_individual_loss_seconds=float((~trace['natural'][:,i]).sum()*.1/3),
            natural_team_loss_seconds=float((~trace['natural'][:,i].any(-1)).sum()*.1),
            effective_team_loss_seconds=float((~trace['mask'][:,i].any(-1)).sum()*.1),
            mean_agent_age_spread=float(np.ptp(np.minimum(trace['age'][:,i],30),axis=-1).mean()),
            action_nonzero_fraction=float((trace['action'][:,i]!=0).mean())))
    summary={k:float(np.mean([r[k] for r in records])) for k in [
        'safe_full','window_geometry_fraction','window_all_geometry','recovery_within5','post_hold5',
        'natural_individual_loss_seconds','natural_team_loss_seconds','effective_team_loss_seconds','mean_agent_age_spread','action_nonzero_fraction']}
    summary['unsafe']=1-summary['safe_full'];summary['n']=len(seeds);summary['window_duration']=duration
    return records,summary


def run(seeds,conditions,fusion='precision',predictor='ct',net=None):
    w=LocalWorld(seeds,conditions,fusion=fusion,predictor=predictor)
    d={k:[] for k in ['time','p','target','geometry','unsafe','natural','artificial','mask','age','action','belief_error','attention','failures']}
    for tick in range(300):
        o=w.observe()
        if tick%2==0:
            if net is None:actions=np.zeros((len(seeds),3),int);attention=np.zeros((len(seeds),3,3))
            else:
                c,n=features(w,o)
                with torch.no_grad():dist,_,att=net(torch.from_numpy(c),torch.from_numpy(n))
                actions=dist.logits.argmax(-1).numpy();attention=att.numpy()
        error=np.linalg.norm(o['target']-w.target,axis=-1)
        natural=w.last_natural.copy();artificial=w.last_artificial.copy()
        info=w.step(o,action_residual(o,actions) if net else None)
        values=[w.time,w.p.copy(),w.target.copy(),info['geometry'],w.unsafe.copy(),natural,artificial,
                o['local_mask'],o['local_age'],actions.copy(),error,attention.copy(),
                np.stack([w.collision_target,w.collision_agent,w.collision_obstacle,w.out_of_bounds,w.target_obstacle_invalid],-1)]
        for key,value in zip(d,values):d[key].append(value)
    return {k:np.array(v) for k,v in d.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--runs',type=Path)
    p.add_argument('--base',type=int,default=15550000);p.add_argument('--n',type=int,default=128)
    p.add_argument('--cases',nargs='+',default=list(CASES));a=p.parse_args();torch.set_num_threads(1)
    methods=[('ct_precision','precision','ct',None),('ct_average','average','ct',None),
             ('ct_freshest','freshest','ct',None),('cv_precision','precision','cv',None),
             ('damped_ct_precision','precision','damped_ct',None)]
    hashes={}
    if a.runs:
        for arm in ['plain','confidence']:
            for seed in range(3):
                path=a.runs/f'{arm}{seed}'/'latest.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
                if ck['update']!=96:raise ValueError('incomplete policy')
                net=Policy(arm=='confidence');net.load_state_dict(ck['state_dict']);net.eval()
                methods.append((f'{arm}{seed}','precision','ct',net));hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot(a.out,dict(schema='c149_confirm_v1',seed_base=a.base,n=a.n,cases={k:asdict(CASES[k]) for k in a.cases},
        methods=[m[0] for m in methods],checkpoint_sha256=hashes,geometry_hz=10,safety_hz=50))
    seeds=np.arange(a.base,a.base+a.n);summary={};start=time.monotonic()
    for case in a.cases:
        summary[case]={};folder=a.out/case;folder.mkdir()
        for name,fusion,predictor,net in methods:
            trace=run(seeds,CASES[case],fusion,predictor,net);records,stats=score(trace,CASES[case],seeds)
            np.savez_compressed(folder/f'{name}.npz',seeds=seeds,**trace)
            write_json(folder/f'{name}.json',dict(summary=stats,episodes=records));summary[case][name]=stats
            print(json.dumps(dict(case=case,method=name,**stats)),flush=True)
        write_json(a.out/'summary.json',summary)
    write_json(a.out/'completion.json',dict(episodes=len(seeds)*len(a.cases)*len(methods),unique_initial_conditions=len(seeds),
        wall_seconds=time.monotonic()-start))

if __name__=='__main__':main()
