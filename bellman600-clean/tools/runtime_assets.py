"""Transport and verify the frozen problem/runtime, independently of trained artifacts."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def files():return json.loads((ROOT/'runtime_manifest.json').read_text())['files']
def verify():
 errors=[]
 for name,row in files().items():
  p=ROOT/name
  if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:errors.append(name)
 if errors:raise SystemExit('Missing/changed frozen runtime files: '+', '.join(errors)+'\nDownload bellman600-runtime.tar.gz; run python3 tools/runtime_assets.py unpack /path/to/bundle.')
 print(f'Verified {len(files())} fixed runtime files; no trained assets required.')
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','pack','unpack']);p.add_argument('bundle',nargs='?',type=Path,default=ROOT/'dist/bellman600-runtime.tar.gz');a=p.parse_args();manifest=files()
 if a.action=='verify':verify();return
 if a.action=='pack':
  verify();a.bundle.parent.mkdir(exist_ok=True)
  if a.bundle.exists():raise SystemExit('Refusing to overwrite '+str(a.bundle))
  with tarfile.open(a.bundle,'w:gz') as t:
   for name in sorted(manifest):t.add(ROOT/name,arcname=name,recursive=False)
  a.bundle.with_suffix(a.bundle.suffix+'.sha256').write_text(sha(a.bundle)+'  '+a.bundle.name+'\n');print(a.bundle);return
 with tarfile.open(a.bundle,'r:gz') as t:
  members=t.getmembers()
  if len(members)!=len(manifest) or {m.name for m in members}!=set(manifest):raise SystemExit('Unexpected bundle file set')
  for m in members:
   dest=ROOT/m.name
   if not m.isfile() or m.size!=manifest[m.name]['bytes'] or not dest.resolve().is_relative_to(ROOT):raise SystemExit('Unsafe archive member')
   if dest.exists() and sha(dest)!=manifest[m.name]['sha256']:raise SystemExit('Refusing to overwrite changed file: '+m.name)
  for m in members:
   dest=ROOT/m.name
   if dest.exists():continue
   dest.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as inp,dest.open('xb') as out:
    for b in iter(lambda:inp.read(8*1024*1024),b''):out.write(b)
 verify()
if __name__=='__main__':main()
