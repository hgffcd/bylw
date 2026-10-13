"""Score fresh closure runs using the frozen per-chapter judges."""
import argparse
import json
from pathlib import Path
from reproduce.closing_native import cases

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=sorted(cases()))
    p.add_argument('--run',action='append',required=True,metavar='ARM=OUTPUT_DIRECTORY')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    group=cases()[a.case];paths={}
    for text in a.run:
        arm,sep,path=text.partition('=')
        if not sep or not path or arm in paths:p.error('Each --run must be a unique ARM=OUTPUT_DIRECTORY')
        paths[arm]=Path(path)
    # Output metadata permits intentional n/seed/duration overrides while the
    # frozen success windows and fault schedule remain explicit.
    first=json.loads((next(iter(paths.values()))/'summary.json').read_text())
    group=dict(group,n=first['n'],duration=first['duration'])
    origin=max(group['times']) if group['times'] else 0.
    group['windows']=[w for w in group['windows'] if origin+w<=group['duration']+1e-8]
    if not group['windows']:p.error('Run is shorter than every registered scoring window')
    import numpy as np
    with np.load(next(iter(paths.values()))/'trace.npz') as z:group['seed']=int(z['seeds'][0])
    if group['chapter']=='4':
        from experiments.c233_fixed3_final_unseen.scoring import audit_group
        if not set(paths)<={'ordinary_predict','ordinary_single','ordinary_all','budget_all'}:
            p.error('The final-system scorer expects the four registered Chapter 4 arms')
        rows,checks=audit_group(group,paths)
    else:
        from experiments.c232_reconfiguration_control.scoring import audit
        rows=audit(group,paths);checks=dict(judge='C217 primary plus visible-hold sensitivity')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(dict(group=group,checks=checks,rows=rows),f,indent=2)
    print(f'Wrote {len(rows)} scored windows to {a.out}')

if __name__=='__main__':main()
