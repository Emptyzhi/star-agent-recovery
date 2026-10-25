"""Acquire pinned external dependencies separately from the published adapters."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def patch_text(raw, spec):
    text = raw.decode('utf-8-sig').replace('\r\n', '\n')
    if hashlib.sha256(text.encode()).hexdigest() != spec['base_sha256']:
        raise ValueError('upstream content does not match the recorded patch base')
    lines = text.splitlines(keepends=True)
    for edit in reversed(spec['changes']):
        lines[edit['start']:edit['end']] = edit['lines']
    result = ''.join(lines).encode()
    if hashlib.sha256(result).hexdigest() != spec['result_sha256']:
        raise ValueError('adapted content does not match the recorded result')
    return result


def extract_functions(raw, names):
    text = raw.decode('utf-8-sig')
    parsed = ast.parse(text)
    functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    lines = text.splitlines(keepends=True)
    return ('from __future__ import annotations\n\n' + '\n\n'.join(''.join(lines[functions[name].lineno - 1:functions[name].end_lineno]).rstrip() for name in names) + '\n').encode()


def safe_target(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('dependency target escapes the artifact directory')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', default=[], metavar='NAME=CHECKOUT',
                        help='use an existing exact-version checkout instead of downloading')
    args = parser.parse_args()
    supplied = dict(item.split('=', 1) for item in args.source)
    manifest = json.loads((ROOT / 'upstreams.json').read_text())
    if set(supplied) - manifest['upstreams'].keys():
        raise ValueError('unknown dependency name')
    sources = {}
    cache = ROOT / '.upstream-cache'
    for name, spec in manifest['upstreams'].items():
        folder = Path(supplied[name]).resolve() if name in supplied else cache / name
        if name not in supplied and not folder.exists():
            folder.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(folder)], check=True)
        # Every dependency must be an exact revision, not a moving branch.
        actual = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', spec['commit'] + '^{commit}'], text=True).strip()
        if actual != spec['commit']:
            raise ValueError('unexpected dependency revision')
        sources[name] = folder
    for asset in manifest['assets']:
        folder = sources[asset['upstream']]
        revision = manifest['upstreams'][asset['upstream']]['commit']
        target = safe_target(asset['target'])
        if asset.get('tree'):
            # Materialize tracked blobs only; exclude Git metadata and history.
            entries = subprocess.check_output(['git', '-C', str(folder), 'ls-tree', '-rz', revision]).split(b'\0')
            for entry in entries:
                if not entry:
                    continue
                metadata, relative = entry.split(b'\t', 1)
                mode, kind, object_id = metadata.decode().split()
                if 'files' in asset and relative.decode('utf-8') not in asset['files']:
                    continue
                if kind != 'blob' or mode == '120000':
                    raise ValueError('unsupported dependency entry')
                dest = target / relative.decode('utf-8')
                if not dest.resolve().is_relative_to(target):
                    raise ValueError('invalid dependency path')
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(subprocess.check_output(['git', '-C', str(folder), 'cat-file', 'blob', object_id]))
        else:
            raw = subprocess.check_output(['git', '-C', str(folder), 'show', revision + ':' + asset['source']])
            if 'functions' in asset:
                raw = extract_functions(raw, asset['functions'])
            if 'patch' in asset:
                raw = patch_text(raw, json.loads((ROOT / asset['patch']).read_text()))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    print('Pinned dependencies materialized. Generated files remain excluded from Git.')


if __name__ == '__main__':
    main()
