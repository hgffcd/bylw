"""Freeze current repository source without depending on historical run artifacts."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def freeze_source(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    hashes = {}
    for relative in manifest['files']:
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    (destination / 'manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return hashes


def snapshot_ch4(out, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    hashes = freeze_source(out / 'source')
    metadata['source_sha256'] = hashes
    metadata['native_source_sha256'] = {
        k: v for k, v in hashes.items() if k.startswith('isaac_p5/source/')
    }
    metadata['source_provenance'] = 'Current portable export, not the omitted C144 historical snapshot'
    (out / 'protocol.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
