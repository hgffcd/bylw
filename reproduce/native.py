"""Run native Isaac evaluations using an explicitly selected installed Isaac Python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from reproduce.provenance import ROOT, freeze_source

ARMS = ['routed_transit', 'guarded_bypass', 'sector_bypass', 'time_matched_sector',
        'time_only_sector', 'metric_budget_sector', 'coverage_metric_sector',
        'radial_damped_sector', 'uniformity_sector', 'predictive_spread_sector']


def cases():
    return json.loads((ROOT / 'reproduce/scenarios.json').read_text())['cases']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['ch4', 'ch4-height', 'ch4-safety', 'ch4-random', 'ch5'], default='ch5')
    p.add_argument('--isaac-python', default=os.environ.get('ISAAC_PYTHON'))
    p.add_argument('--gpu', type=int, help='Physical GPU index; exposed as cuda:0 to the child')
    p.add_argument('--out', type=Path)
    p.add_argument('--case', default='recent_4to3_g0')
    p.add_argument('--arm', choices=ARMS, default='radial_damped_sector')
    p.add_argument('--n', type=int)
    p.add_argument('--seed-base', type=int)
    p.add_argument('--duration', type=float)
    p.add_argument('--methods', nargs='+', choices=['brake_predict', 'brake_search', 'staged_brake', 'all_search_staged'], default=['staged_brake', 'all_search_staged'])
    p.add_argument('--scenes', nargs='+', choices=['corner', 'corridor'], default=['corner', 'corridor'])
    p.add_argument('--vertical-mode', choices=['free', 'fixed'], default='free')
    p.add_argument('--safety-layer', choices=['on', 'off'], default='on')
    p.add_argument('--list-cases', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.list_cases:
        print(json.dumps(cases(), indent=2)); return
    if not a.isaac_python or a.gpu is None or a.out is None:
        p.error('--isaac-python (or ISAAC_PYTHON), --gpu and --out are required')
    if a.gpu < 0 or (a.n is not None and a.n < 1):
        p.error('GPU index must be nonnegative and n must be positive')
    out = a.out.resolve()
    if out.exists(): p.error(f'Output already exists: {out}; select a fresh path')
    module = {'ch4':'c166_unified_matrix', 'ch4-height':'c167_vertical_ablation',
              'ch4-safety':'c168_safety_ablation'}.get(a.task)
    if module:
        extra = ['--n', str(a.n or 16), '--seed-base', str(a.seed_base if a.seed_base is not None else 16200000),
                 '--duration', str(a.duration if a.duration is not None else 60),
                 '--methods', *a.methods, '--scenes', *a.scenes]
        if a.task == 'ch4-height': extra += ['--vertical-mode', a.vertical_mode]
        if a.task == 'ch4-safety': extra += ['--safety-layer', a.safety_layer]
    else:
        if a.task == 'ch4-random':
            g = dict(agents=3, n=4, seed=19650000, times=[], indices=[], duration=60)
        else:
            if a.case not in cases(): p.error(f'Unknown case: {a.case}; use --list-cases')
            g = cases()[a.case]
        duration = a.duration if a.duration is not None else g['duration']
        if duration <= 0 or (g['times'] and duration < max(g['times']) + 5):
            p.error('Duration must be positive and include at least 5 s after the last withdrawal')
        module = {'uniformity_sector':'c211_uniformity_handover',
                  'predictive_spread_sector':'c212_predictive_spread'}.get(a.arm, 'c210_radial_damping')
        extra = ['--agents', str(g['agents']), '--n', str(a.n or g['n']),
                 '--seed-base', str(a.seed_base if a.seed_base is not None else g['seed']),
                 '--duration', str(duration), '--scene', 'random_test',
                 '--obstacle-count', '6', '--shape-profile', 'mixed', '--layout-profile', 'route_dense',
                 '--reconfig', 'active_polar', '--recovery', a.arm, '--map-protocol', 'valid', '--opponent', 'routed',
                 '--failure-times', ','.join(map(str, g['times'])), '--failure-indices', ','.join(map(str, g['indices'])),
                 '--gpu', str(a.gpu)]
    command = [a.isaac_python, str(ROOT / 'experiments' / module / 'native_run.py'),
               '--out', str(out), '--scene-scale', '1.6', *extra, '--device', 'cuda:0', '--headless']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONDONTWRITEBYTECODE='1',
               BYLW_P5_ENABLE_ISAAC_RUNTIME='1',
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / 'isaac_p5/source/bylw_isaac_p5')]))
    record = {'command': command, 'CUDA_VISIBLE_DEVICES': str(a.gpu),
              'task': a.task, 'case': a.case if a.task == 'ch5' else None,
              'note': 'Fresh rerun; numerical equivalence with historical results is not asserted.'}
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if a.dry_run: return
    if a.task in ['ch4-random', 'ch5']:
        snapshot = out.parent / (out.name + '_source')
        freeze_source(snapshot)
        (snapshot / 'invocation.json').write_text(json.dumps(record, indent=2) + '\n')
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
