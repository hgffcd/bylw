"""Preregistered descriptive uncertainty; no parameter search or significance gate."""
import math
import numpy as np


def wilson(success,n):
    if not 0<=success<=n or n<=0:raise ValueError('Invalid count')
    z=1.959963984540054;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def paired_contrast(rows,base,new,metric='moving_success',repetitions=10000):
    b={x['seed']:x for x in rows if x['arm']==base};a={x['seed']:x for x in rows if x['arm']==new}
    if not a or a.keys()!=b.keys():raise ValueError('Complete paired seeds required')
    seeds=sorted(a)
    if any(a[s]['fault_pattern']!=b[s]['fault_pattern'] for s in seeds):raise ValueError('Fault mismatch')
    delta=np.array([int(a[s][metric])-int(b[s][metric]) for s in seeds],dtype=float)
    patterns=np.array([a[s]['fault_pattern'] for s in seeds]);rng=np.random.default_rng(20261012)
    means=np.zeros(repetitions)
    for pattern in sorted(set(patterns)):
        indices=np.flatnonzero(patterns==pattern)
        samples=rng.choice(indices,size=(repetitions,len(indices)),replace=True)
        means+=delta[samples].sum(1)/len(seeds)
    both=[s for s in seeds if a[s]['success'] and b[s]['success']]
    times=[a[s]['first_hold5_after_withdrawal_s']-b[s]['first_hold5_after_withdrawal_s'] for s in both]
    return dict(base=base,new=new,metric=metric,n=len(seeds),difference_percentage_points=float(100*delta.mean()),
        stratified_paired_bootstrap95_percentage_points=(100*np.quantile(means,[.025,.975])).tolist(),
        bootstrap_seed=20261012,repetitions=repetitions,
        gains=[s for s in seeds if a[s][metric] and not b[s][metric]],
        losses=[s for s in seeds if b[s][metric] and not a[s][metric]],
        both_safe_capture_success_n=len(both),median_delay_difference_s_among_both_success=float(np.median(times)) if times else None)
