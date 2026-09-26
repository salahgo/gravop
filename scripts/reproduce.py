"""Explicit reproduction targets with dependency preflight and failing exit codes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'reproduction.json'


def verify_hashes(root, manifest):
    """Verify content identities, rejecting missing files and empty manifests."""
    files = manifest.get('files', {})
    if not files: raise ValueError('empty hash manifest')
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError(f'path outside checkout: {name}')
        if not path.is_file(): raise ValueError(f'missing artifact: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f'changed artifact: {name}')
    return len(files)


def run_target(target, root=ROOT, execute=False):
    manifest = json.loads((root/'reproduction.json').read_text())
    spec = manifest['targets'][target]
    available = set()
    missing = []
    for step in spec['steps']:
        for name in step.get('requires', []):
            if name not in available and not (root/name).exists(): missing.append(name)
        available.update(step.get('produces', []))
    if missing: raise ValueError('missing prerequisites: '+', '.join(sorted(set(missing))))
    for step in spec['steps']:
        if 'hash_manifest' in step:
            path = root/step['hash_manifest']
            print('verify',path.relative_to(root),flush=True)
            if execute: print(verify_hashes(root,json.loads(path.read_text())),'hashes verified',flush=True)
        else:
            command = [sys.executable if c=='{python}' else c for c in step['command']]
            print(' '.join(command),flush=True)
            if execute: subprocess.run(command,cwd=root,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('target',choices=json.loads(MANIFEST.read_text())['targets'])
    p.add_argument('--execute',action='store_true',help='run the preflighted target; default prints the plan')
    args=p.parse_args()
    try: run_target(args.target,execute=args.execute)
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print(str(exc),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
