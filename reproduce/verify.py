"""Verify the code-only export against its recorded file hashes."""
import hashlib
import json
from reproduce.provenance import ROOT


def main():
    files = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())['files']
    mismatch = []
    for name, expected in files.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatch.append(name)
    if mismatch:
        raise SystemExit('Missing or changed source files:\n' + '\n'.join(mismatch))
    print(f'Verified {len(files)} source/configuration/documentation files.')


if __name__ == '__main__':
    main()
