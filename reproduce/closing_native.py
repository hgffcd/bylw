"""Portable native execution for the final fixed-three and withdrawal comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from reproduce.provenance import ROOT,freeze_source

def cases():
    return json.loads((ROOT/'reproduce/closing_cases.json').read_text())['cases']

def build(a):
    g=cases()[a.case]
    if a.chapter!=g['chapter']:raise ValueError('Case belongs to another chapter')
    n=a.n if a.n is not None else g['n'];seed=a.seed_base if a.seed_base is not None else g['seed']
    duration=a.duration if a.duration is not None else g['duration']
    if n<1 or duration<=0:raise ValueError('Positive n and duration required')
    if a.chapter=='4':
        if a.arm not in ['ordinary_predict','ordinary_single','ordinary_all','budget_all','matched_all','smooth_all']:
            raise ValueError('Unsupported fixed-three arm')
        core,mode=a.arm.split('_');recovery='routed_transit' if core=='ordinary' else 'budget_transit'
        module='c233_fixed3_final_unseen' if core in ['ordinary','budget'] else 'c231_smooth_handoff'
        extra=['--search-mode',mode]
        if core in ['matched','smooth']:extra+=['--control-variant',core]
        method='active_polar'
    else:
        if a.arm not in ['active','retained']:raise ValueError('Chapter 5 arms are active or retained')
        recovery='radial_damped_sector';module='c232_reconfiguration_control';extra=[]
        method={'active':'active_polar','retained':'retained_slots'}[a.arm]
    if g['times'] and duration<max(g['times'])+5:raise ValueError('At least 5 post-withdrawal seconds required')
    command=[a.isaac_python,str(ROOT/'experiments'/module/'native_run.py'),'--out',str(a.out.resolve()),
        '--agents',str(g['agents']),'--n',str(n),'--seed-base',str(seed),'--duration',str(duration),
        '--scene','random_test','--scene-scale','1.6','--obstacle-count','6','--shape-profile','mixed',
        '--layout-profile','route_dense','--reconfig',method,'--recovery',recovery,
        '--failure-times',','.join(map(str,g['times'])),'--failure-indices',','.join(map(str,g['indices'])),
        '--map-protocol','valid','--opponent','routed',*extra,'--gpu',str(a.gpu),
        '--device',f'cuda:{a.gpu}' if getattr(a,'physical_device',False) else 'cuda:0','--headless']
    if getattr(a,'kit_args',None):command+=['--kit_args',a.kit_args]
    return command,dict(g,n=n,seed=seed,duration=duration)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chapter',choices=['4','5'],required=True)
    p.add_argument('--case',required=True,choices=sorted(cases()))
    p.add_argument('--arm',required=True)
    p.add_argument('--isaac-python',default=os.environ.get('ISAAC_PYTHON'))
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--physical-device',action='store_true',
        help='Use cuda:GPU without CUDA_VISIBLE_DEVICES remapping (for multi-GPU Isaac hosts)')
    p.add_argument('--kit-args',help='Isaac Kit arguments, e.g. portable runtime and installed asset root')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int);p.add_argument('--seed-base',type=int);p.add_argument('--duration',type=float)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if not a.isaac_python or a.gpu<0:p.error('Isaac Python and nonnegative GPU index required')
    if a.out.exists():p.error('Output exists; choose a fresh path')
    try:command,group=build(a)
    except ValueError as e:p.error(str(e))
    record=dict(command=command,CUDA_VISIBLE_DEVICES=None if a.physical_device else str(a.gpu),
        physical_device=a.physical_device,case=a.case,chapter=a.chapter,arm=a.arm,group=group,
        scope='Fresh rerun. No historical result data or pretrained weights are included. Exact cross-machine numerical equality is not asserted.')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
    if a.dry_run:return
    snapshot=a.out.resolve().with_name(a.out.name+'_source');freeze_source(snapshot)
    (snapshot/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONDONTWRITEBYTECODE='1',
        BYLW_P5_ENABLE_ISAAC_RUNTIME='1',PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'isaac_p5/source/bylw_isaac_p5')]))
    if a.physical_device:env.pop('CUDA_VISIBLE_DEVICES',None)
    runtime=a.out.resolve().parent/(a.out.name+'_runtime')
    for key,leaf in [('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','matplotlib'),('TORCH_HOME','torch'),('TMPDIR','tmp')]:
        d=runtime/leaf;d.mkdir(parents=True,exist_ok=True);env[key]=str(d)
    subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
