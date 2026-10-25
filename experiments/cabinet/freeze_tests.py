"""Freeze one registered cabinet-tests run (once; never overwrites): scene cabinet into tests/,
scene cabinet_<variant> into tests_<variant>/ (tests_plus10/, tests_plus20/).

    python3 experiments/cabinet/freeze_tests.py results/registered_<cabinet-tests run>

Copies suite_manifest.json, generation.json and suites/*.npz verbatim, checks every sha256 against
the manifest, the scene file and the generator, and records the source run in FROZEN.json.
"""
import argparse,hashlib,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    run=a.run.resolve();source=run/'tests';manifest=json.loads((source/'suite_manifest.json').read_text())
    generation=json.loads((source/'generation.json').read_text());assert generation['complete'] and not manifest['smoke'],'Not a complete full generation'
    scene=HERE/'scenes'/(manifest['scene']+'.yml');assert sha(scene)==manifest['scene_sha256'],'Scene file changed since generation'
    assert sha(HERE/'make_tests.py')==manifest['generator_sha256'],'Generator changed since generation'
    target=HERE/('tests'+manifest['scene'][len('cabinet'):]);assert manifest['scene'].startswith('cabinet') and not target.exists(),f'{target} is already frozen'
    (target/'suites').mkdir(parents=True)
    for name in ('suite_manifest.json','generation.json'):shutil.copyfile(source/name,target/name)
    for item in manifest['suites']:
        shutil.copyfile(source/'suites'/item['file'],target/'suites'/item['file'])
        assert sha(target/'suites'/item['file'])==item['sha256'],item['file']
    record=dict(source_run=run.name,scene=manifest['scene'],scene_sha256=manifest['scene_sha256'],generator_sha256=manifest['generator_sha256'],
                suite_manifest_sha256=sha(target/'suite_manifest.json'),total=manifest['total'],
                note='Frozen testing data: never resample or edit; evaluate.py checks these hashes.')
    (target/'FROZEN.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
