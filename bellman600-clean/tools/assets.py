"""Verify and transport frozen assets separately from Git history."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def manifest():
    return json.loads((ROOT/'asset_manifest.json').read_text())['files']

def verify():
    files=manifest();errors=[]
    for name,row in files.items():
        p=ROOT/name
        if not p.is_file():errors.append(f'Missing: {name}')
        elif p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:errors.append(f'Hash mismatch: {name}')
    if errors:raise SystemExit('\n'.join(errors)+'\nRestore the matching asset bundle with tools/assets.py unpack.')
    print(f'Verified {len(files)} frozen asset files.')

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','pack','unpack'])
    p.add_argument('bundle',nargs='?',type=Path,default=ROOT/'dist/bellman600-clean-assets.tar.gz')
    a=p.parse_args()
    if a.action=='verify':verify();return
    files=manifest()
    if a.action=='pack':
        verify();a.bundle.parent.mkdir(parents=True,exist_ok=True)
        if a.bundle.exists():raise SystemExit(f'Refusing to overwrite {a.bundle}')
        with tarfile.open(a.bundle,'w:gz',compresslevel=6) as tar:
            for name in sorted(files):tar.add(ROOT/name,arcname=name,recursive=False)
        digest=sha(a.bundle)
        a.bundle.with_suffix(a.bundle.suffix+'.sha256').write_text(digest+'  '+a.bundle.name+'\n')
        print(a.bundle);return
    # Accept only the exact regular files in the trusted manifest; no links or traversal.
    with tarfile.open(a.bundle,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(files) or {m.name for m in members}!=set(files):raise SystemExit('Asset bundle file set differs from manifest')
        for member in members:
            path=Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:raise SystemExit('Unsafe bundle member')
            if member.size!=files[member.name]['bytes']:raise SystemExit('Unexpected asset size')
        for member in members:
            destination=ROOT/member.name
            if destination.exists():
                if sha(destination)!=files[member.name]['sha256']:raise SystemExit(f'Refusing to overwrite changed asset: {member.name}')
                continue
            # Do not follow pre-existing directory symlinks outside this repository.
            if not destination.resolve().is_relative_to(ROOT):raise SystemExit('Asset destination escapes repository')
            destination.parent.mkdir(parents=True,exist_ok=True)
            stream=tar.extractfile(member)
            with destination.open('xb') as out:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):out.write(chunk)
    verify()

if __name__=='__main__':main()
