"""Score a freshly rerun Chapter 5 comparison with the original window audit."""
import argparse
import json
from pathlib import Path
from reproduce.native import cases
from experiments.c202_postfault_budget.scoring import audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, choices=sorted(cases()))
    p.add_argument('--run', action='append', required=True, metavar='ARM=OUTPUT_DIRECTORY')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    paths = {}
    for spec in a.run:
        arm, sep, directory = spec.partition('=')
        if not sep or not directory or arm in paths: p.error('Each --run must name a unique ARM=OUTPUT_DIRECTORY')
        paths[arm] = Path(directory)
    rows = audit(cases()[a.case], paths)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'Wrote {len(rows)} per-seed/per-window rows to {a.out}')


if __name__ == '__main__':
    main()
