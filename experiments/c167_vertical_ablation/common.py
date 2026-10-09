"""Portable source snapshot adapter; original controller/scoring code is unchanged."""
import json
from pathlib import Path
from reproduce.provenance import snapshot_ch4 as snapshot

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
