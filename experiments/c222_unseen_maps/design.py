"""Fixed unseen seeds and geometry fingerprints; never instantiate a new world here."""
import hashlib
import json
import numpy as np

ARMS=['sector_bypass','coverage_metric_sector','radial_damped_sector']


def groups():
    result=[]
    for s,(label,agents,patterns) in enumerate([
        ('4to3',4,[[2],[0]]),('5to4',5,[[3],[1]]),('5to3',5,[[0,4],[1,3]])]):
        for batch in range(4):
            pattern=batch%2;times=[2.5,3.] if label=='5to3' else [2.5]
            result.append(dict(label=f'{label}_g{batch}',stratum=label,agents=agents,n=4,
                seed=22250000+s*1000+batch*16,indices=patterns[pattern],times=times,
                duration=max(times)+60.,windows=[30,45,60],gpu=2+pattern,
                cohort='c222_prospective',fault_pattern=f'{label}_p{pattern}'))
    return result


def fingerprint(obstacles,radii):
    """Rigid-motion/permutation invariant pair signature, rounded to 10 micrometres.

    This is a conservative duplicate-layout screen, not a proof of uniqueness
    against unavailable historical artifacts or arbitrary geometric isometries.
    """
    p=np.asarray(obstacles,dtype=np.float64);rr=np.asarray(radii,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or rr.shape!=(len(p),):raise ValueError('Expected (O,2), (O,)')
    if not np.isfinite(p).all() or not np.isfinite(rr).all():raise ValueError('Nonfinite map')
    pairs=[]
    for i in range(len(rr)):
        for j in range(i+1,len(rr)):
            a,b=sorted([round(float(rr[i]),5),round(float(rr[j]),5)])
            pairs.append((a,b,round(float(np.linalg.norm(p[i]-p[j])),5)))
    key=dict(radii=sorted(round(float(x),5) for x in rr),pairs=sorted(pairs))
    return hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).hexdigest()
