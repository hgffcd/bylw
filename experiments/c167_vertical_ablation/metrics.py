import numpy as np
from experiments.c150_native_search.task import geometry3

def longest(x):
    run=np.zeros(x.shape[1],int);maximum=run.copy()
    for row in x:run=np.where(row,run+1,0);maximum=np.maximum(maximum,run)
    return maximum

def score(data,seeds):
    g=data['geometry'];visible=data['mask'].any(-1);safe=~data['unsafe'];final=safe[-1]
    good=g&visible&safe;hold=(longest(good)>=50)&final;episodes=[]
    _,r,gaps=geometry3(data['p'].reshape(-1,3,2),data['z'].reshape(-1,3),data['target'].reshape(-1,2),data['target_z'].reshape(-1))
    legacy=((r.max(1)<=5.5)&(gaps.max(1)<=np.deg2rad(150))).reshape(g.shape)
    legacy_hold=(longest(legacy&visible&safe)>=50)&final
    for i,seed in enumerate(seeds):
        lost=~visible[:,i];edges=np.diff(np.r_[False,lost,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
        events=[]
        for s,e in zip(starts,ends):
            events.append(dict(start_s=round(float(data['time'][s]),3),start_tick=int(s),end_tick=int(e),
                duration_s=round((e-s)*.1,3),reacquired=bool(e<len(lost))))
        meaningful=[e for e in events if e['duration_s']>=.2]
        after=None
        if meaningful:
            last=meaningful[-1];idx=last['end_tick']
            after=bool(last['reacquired'] and (longest(good[idx:,i:i+1])[0]>=50) and final[i])
        episodes.append(dict(seed=int(seed),safe_capture5=bool(hold[i]),legacy_geometry_safe_capture5=bool(legacy_hold[i]),safe_full=bool(final[i]),
            geometry_fraction=float(good[:,i].mean()),max_hold_s=float(longest(good[:,i:i+1])[0]*.1),
            team_lost_s=float(lost.sum()*.1),loss_events=events,meaningful_loss_events=len(meaningful),
            post_last_loss_capture5=after,failures=data['failures'][-1,i].tolist(),
            altitude_min=float(data['z'][:,i].min()),altitude_max=float(data['z'][:,i].max()),
            altitude_excursion=float(np.ptp(data['z'][:,i],axis=0).max())))
    lost_indices=[i for i,e in enumerate(episodes) if e['post_last_loss_capture5'] is not None]
    events=[e for ep in episodes for e in ep['loss_events'] if e['duration_s']>=.2]
    summary=dict(n=len(seeds),safe_capture5=float(hold.mean()),legacy_geometry_safe_capture5=float(legacy_hold.mean()),unsafe=float((~final).mean()),geometry_fraction=float(good.mean()),
        episodes_with_team_loss=len(lost_indices),mean_team_loss_s=float(np.mean([e['team_lost_s'] for e in episodes])),
        max_team_loss_s=max([e['duration_s'] for e in events],default=0),natural_loss_events=len(events),
        reacquired_events=sum(e['reacquired'] for e in events),
        post_loss_capture5=float(np.mean([episodes[i]['post_last_loss_capture5'] for i in lost_indices])) if lost_indices else None,
        failure_counts=data['failures'][-1].sum(0).tolist(),altitude_min=float(data['z'].min()),altitude_max=float(data['z'].max()),
        mean_altitude_excursion=float(np.mean([e['altitude_excursion'] for e in episodes])))
    return episodes,summary
