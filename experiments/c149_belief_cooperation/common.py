import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def snapshot(out, metadata):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    hashes={}
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'PROTOCOL.md']
    files+=list((ROOT/'experiments/c146_loss_geometry').glob('*.py'))
    files += [ROOT/'experiments/c145_feasibility_reset'/n for n in ['planar.py','preview_evader.py']]
    for f in files:
        relative=f.relative_to(ROOT);dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest);hashes[str(relative)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (out/'protocol.json').write_text(json.dumps(dict(**metadata,source_sha256=hashes),indent=2)+'\n')

def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
