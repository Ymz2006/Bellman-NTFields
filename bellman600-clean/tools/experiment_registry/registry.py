#!/usr/bin/env python3
"""Local experiment registry: content identity, guarded execution and evidence discovery."""
import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import uuid

DEFAULT = Path(os.environ.get('NTFIELDS_REGISTRY', '~/.local/share/ntfields-experiments')).expanduser()
IGNORED = {'.git', '__pycache__', '.pytest_cache'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def process_token(pid):
    try:
        # Start time distinguishes a live runner from a reused PID.
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + stat[19]
    except (OSError, IndexError):
        return None


def prepare(spec_path):
    source = Path(spec_path).resolve()
    spec = json.loads(source.read_text())
    for key in ['name', 'kind', 'root', 'config', 'environment', 'inputs', 'command', 'required_outputs']:
        if key not in spec:
            raise ValueError('Missing spec field: ' + key)
    root = (source.parent / spec['root']).resolve()
    if not root.is_dir():
        raise ValueError('Missing working root: ' + str(root))
    cmd = spec['command']
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) for x in cmd):
        raise ValueError('command must be an argv string array (no shell)')
    if not any('{output}' in x for x in cmd):
        raise ValueError('command must include {output}; the runner allocates a fresh directory')
    if not isinstance(spec['inputs'], dict) or not spec['inputs'] or not spec['required_outputs']:
        raise ValueError('Declare nonempty inputs and required_outputs')
    for pattern in spec['required_outputs']:
        if Path(pattern).is_absolute() or '..' in Path(pattern).parts:
            raise ValueError('required_outputs must stay inside the output directory')
    hashes, paths = {}, {}
    for label, relative in sorted(spec['inputs'].items()):
        p = (root / relative).resolve()
        if not p.exists():
            raise ValueError('Missing input: ' + str(p))
        files = [p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and not IGNORED.intersection(x.parts))
        if not files:
            raise ValueError('Empty input: ' + str(p))
        for f in files:
            key = label if p.is_file() else label + '/' + f.relative_to(p).as_posix()
            if key in hashes:
                raise ValueError('Colliding logical input name: ' + key)
            hashes[key] = digest(f)
            paths[key] = str(f)
    identity = {k: spec[k] for k in ['kind', 'config', 'environment', 'command', 'required_outputs']}
    identity['validation'] = spec.get('validation', 'files')
    identity['inputs'] = hashes
    identity['inherited_environment'] = {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG']}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return {'spec': spec, 'root': str(root), 'identity': identity, 'fingerprint': fingerprint,
            'input_paths': paths, 'git_commit': git.stdout.strip() if git.returncode == 0 else None}


class Registry:
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.home / 'registry.sqlite3', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, fingerprint TEXT, name TEXT, state TEXT, data TEXT);
        CREATE INDEX IF NOT EXISTS by_fingerprint ON runs(fingerprint);
        CREATE TABLE IF NOT EXISTS discoveries(path TEXT PRIMARY KEY, data TEXT);
        ''')

    def rows(self):
        return [json.loads(x[0]) for x in self.db.execute('SELECT data FROM runs ORDER BY rowid')]

    def matches(self, fp):
        return [json.loads(x[0]) for x in self.db.execute('SELECT data FROM runs WHERE fingerprint=?', (fp,))]

    def historical_candidates(self, prep):
        checkpoint = prep['identity']['inputs'].get('checkpoint')
        if not checkpoint:
            return []
        return [json.loads(x[0]) for x in self.db.execute('SELECT data FROM discoveries')
                if json.loads(x[0]).get('checkpoint_sha256') == checkpoint]

    def save(self, row):
        with self.db:
            self.db.execute('UPDATE runs SET state=?, data=? WHERE id=?', (row['state'], json.dumps(row), row['id']))
        self.mirror(row)

    def mirror(self, row):
        folder = self.home / 'runs' / row['id']
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'record.json', row)

    def reserve(self, prep, reason=None, existing_output=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            matches = self.matches(prep['fingerprint'])
            candidates = self.historical_candidates(prep)
            if candidates and not existing_output and not (reason and reason.strip()):
                raise ValueError('Historical results use the same checkpoint; review before rerunning: ' + ', '.join(x['path'] for x in candidates) + '. Not proof of identical protocol; use --repeat-reason after review.')
            if matches and not (reason and reason.strip()):
                raise ValueError('Duplicate blocked: ' + ', '.join(x['id'] + ' [' + x['state'] + ']' for x in matches) + '. Inspect existing evidence; intentional retry requires --repeat-reason.')
            rid = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
            row = dict(prep, id=rid, name=prep['spec']['name'], state='reserved', created=now(),
                       host=platform.node(), pid=os.getpid(), process_token=process_token(os.getpid()),
                       repeat_reason=reason, prior_runs=[x['id'] for x in matches], artifacts=[])
            row['output'] = str(Path(existing_output).resolve()) if existing_output else str(Path(prep['root']) / 'results' / ('registered_' + rid))
            self.db.execute('INSERT INTO runs VALUES(?,?,?,?,?)', (rid, row['fingerprint'], row['name'], row['state'], json.dumps(row)))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.mirror(row)
        return row

    def discover(self, root):
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError('Missing discovery root: ' + str(root))
        known = {x['output'] for x in self.rows()}
        count = 0
        # Discover direct result groups. Never infer scientific completion from filenames.
        for folder in sorted(x for x in root.iterdir() if x.is_dir()):
            if str(folder) in known:
                continue
            markers = []
            for p in sorted(folder.rglob('*')):
                if p.is_file() and p.suffix.lower() in {'.json', '.md', '.pt', '.npz', '.log', '.csv'}:
                    markers.append({'path': str(p.relative_to(folder)), 'bytes': p.stat().st_size})
            if not markers:
                continue
            item = {'path': str(folder), 'name': folder.name, 'state': 'discovered_unverified',
                    'last_seen': now(), 'files': markers,
                    'note': 'Historical evidence location only; no complete identity or success claim. Review before rerunning.'}
            report = folder / 'evaluation.json'
            if report.is_file():
                try:
                    d = json.loads(report.read_text())
                    item['checkpoint_sha256'] = d.get('checkpoint_sha256')
                    item['reported_complete'] = d.get('complete')
                    item['reported_score_rows'] = len(d.get('scores', []))
                except (ValueError, AttributeError, TypeError):
                    item['report_parse_error'] = True
            with self.db:
                old = self.db.execute('SELECT data FROM discoveries WHERE path=?', (str(folder),)).fetchone()
                if old: item['notes'] = json.loads(old[0]).get('notes', [])
                self.db.execute('INSERT OR REPLACE INTO discoveries VALUES(?,?)', (str(folder), json.dumps(item)))
            count += 1
        return count


def artifacts(row):
    root = Path(row['output'])
    found = {}
    for pattern in row['spec']['required_outputs']:
        matches = [p for p in root.glob(pattern) if p.is_file()]
        if not matches:
            raise ValueError('Missing required output: ' + pattern)
        for p in matches:
            if not p.resolve().is_relative_to(root.resolve()):
                raise ValueError('Output symlink escapes experiment: ' + str(p))
            found[str(p.relative_to(root))] = {'sha256': digest(p), 'bytes': p.stat().st_size}
    if row['spec'].get('validation') == 'evaluation600':
        d = json.loads((root / 'evaluation.json').read_text())
        expected = {(t, s) for t in ['mixed', 'screwdriver', 'cutting', 'hammering', 'stirring', 'wiping'] for s in [0, 1, 2]}
        scores = d['scores']
        if not row['identity']['inputs'].get('checkpoint') or row['identity']['inputs']['checkpoint'] != d.get('checkpoint_sha256'):
            raise ValueError('Evaluation checkpoint hash differs from declared checkpoint input')
        if not d.get('complete') or len(scores) != 18 or {(x['task'], x['planning_seed']) for x in scores} != expected or any(x['n'] != 100 for x in scores):
            raise ValueError('Incomplete 600 x 3 planning-seed evaluation')
    elif row['spec'].get('validation') == 'training_steps':
        d = json.loads((root / 'model/status.json').read_text())
        if d.get('steps') != row['spec']['config']['steps']:
            raise ValueError('Training did not reach requested steps')
    elif row['spec'].get('validation', 'files') != 'files':
        raise ValueError('Unknown validation mode')
    return found


def execute(reg, prep, reason=None):
    row = reg.reserve(prep, reason)
    folder = reg.home / 'runs' / row['id']
    command = [x.replace('{output}', row['output']) for x in row['spec']['command']]
    row['executed_command'] = command
    write_json(folder / 'spec.json', row['spec'])
    # Preserve small source/config inputs; large weights/data retain their hashes and locations.
    sources = folder / 'inputs'
    sources.mkdir()
    for key, path in row['input_paths'].items():
        p = Path(path)
        if p.stat().st_size <= 1024 * 1024 and p.suffix in {'.py', '.json', '.yaml', '.yml', '.toml', '.sh'}:
            target = sources / (hashlib.sha256(key.encode()).hexdigest() + p.suffix)
            shutil.copyfile(p, target)
    child = None
    try:
        # Detect modifications between identity capture and launch.
        if any(digest(path) != row['identity']['inputs'][key] for key, path in row['input_paths'].items()):
            raise ValueError('Inputs changed before launch')
        row['state'] = 'running'
        row['started'] = now()
        reg.save(row)
        print(json.dumps({'id': row['id'], 'output': row['output'], 'log': str(folder / 'console.log')}, ensure_ascii=False), flush=True)
        with (folder / 'console.log').open('w') as log:
            child = subprocess.Popen(command, cwd=row['root'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, **row['spec']['environment'].get('variables', {})})
            row.update(child_pid=child.pid, child_token=process_token(child.pid))
            reg.save(row)
            row['returncode'] = child.wait()
        if row['returncode']:
            raise ValueError('Command failed with exit ' + str(row['returncode']))
        if any(digest(path) != row['identity']['inputs'][key] for key, path in row['input_paths'].items()):
            raise ValueError('Inputs changed during run; identity is invalid')
        row['artifacts'] = artifacts(row)
        row['state'] = 'succeeded'
        row['note'] = 'Command and declared output checks passed; not automatically a scientific conclusion or 52k qualification.'
    except BaseException as exc:
        row['state'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed'
        row['error'] = str(exc)
        if child and child.poll() is None:
            # Do not kill Docker containers or unrelated work automatically.
            row['state'] = 'needs_review'
            row['error'] += '; child may still run: inspect child PID/container before retry'
        row['ended'] = now()
        reg.save(row)
        raise
    row['ended'] = now()
    reg.save(row)
    print(json.dumps({'id': row['id'], 'state': row['state'], 'output': row['output']}))


def health(row):
    issues = []
    if row['state'] in {'reserved', 'running', 'needs_review'}:
        if row['host'] != platform.node():
            issues.append('remote host: process state unknown')
        else:
            parent = row.get('process_token') and process_token(row['pid']) == row['process_token']
            child = row.get('child_token') and process_token(row['child_pid']) == row['child_token']
            if not parent:
                issues.append('runner gone; child still alive' if child else 'runner/child gone; inspect Docker/output before retry')
    for name, info in (row.get('artifacts') or {}).items():
        path = Path(row['output']) / name
        if not path.is_file():
            issues.append('MISSING: ' + str(path))
        elif digest(path) != info['sha256']:
            issues.append('CHANGED: ' + str(path))
    return issues


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', type=Path, default=DEFAULT)
    sub = p.add_subparsers(dest='action', required=True)
    for action in ['check', 'run']:
        a = sub.add_parser(action); a.add_argument('spec', type=Path)
        if action == 'run': a.add_argument('--repeat-reason')
    a = sub.add_parser('link-existing'); a.add_argument('spec', type=Path); a.add_argument('output', type=Path); a.add_argument('--reason', required=True)
    a = sub.add_parser('list'); a.add_argument('--search', default='')
    a = sub.add_parser('show'); a.add_argument('id')
    a = sub.add_parser('discover'); a.add_argument('root', type=Path)
    sub.add_parser('doctor')
    a = sub.add_parser('export'); a.add_argument('output', type=Path)
    a = sub.add_parser('backup'); a.add_argument('output', type=Path); a.add_argument('--include-results', action='store_true'); a.add_argument('--include-inputs', action='store_true')
    a = sub.add_parser('note'); a.add_argument('id'); a.add_argument('text')
    a = sub.add_parser('recover'); a.add_argument('id'); a.add_argument('--reason', required=True)
    args = p.parse_args()
    reg = Registry(args.home)
    if args.action in ['check', 'run']:
        prep = prepare(args.spec)
        if args.action == 'run':
            execute(reg, prep, args.repeat_reason)
        else:
            matches = reg.matches(prep['fingerprint'])
            print(json.dumps({'fingerprint': prep['fingerprint'], 'exact_matches': matches, 'historical_same_checkpoint': reg.historical_candidates(prep),
                              'note': 'No exact match does not exclude historical unverified experiments; also search discoveries.'}, ensure_ascii=False, indent=2))
            return 3 if matches or reg.historical_candidates(prep) else 0
    elif args.action == 'link-existing':
        prep = prepare(args.spec)
        candidate = dict(prep, output=str(args.output.resolve()))
        checked = artifacts(candidate)
        snapshot = args.output / 'source_snapshot.json'
        if not snapshot.is_file(): raise ValueError('Historical source_snapshot.json required')
        historical = json.loads(snapshot.read_text())
        checked_sources = 0
        for name, path in prep['input_paths'].items():
            relative = os.path.relpath(path, prep['root'])
            if relative in historical:
                if prep['identity']['inputs'][name] != historical[relative]:
                    raise ValueError('Historical code mismatch: ' + relative)
                checked_sources += 1
        if not checked_sources: raise ValueError('No historical source hashes match declared inputs')
        row = reg.reserve(prep, existing_output=args.output)
        row.update(state='linked_existing', artifacts=checked, ended=now(),
                   identity_status='Declared association; matching saved source hashes and outputs checked, historical command/environment not automatically proven.',
                   notes=[{'time': now(), 'text': args.reason}], historical_source_checks=checked_sources)
        reg.save(row)
        print(json.dumps({'id': row['id'], 'state': row['state'], 'output': row['output']}))
    elif args.action == 'discover':
        print('Discovered groups:', reg.discover(args.root))
    elif args.action == 'list':
        rows = reg.rows() + [json.loads(x[0]) for x in reg.db.execute('SELECT data FROM discoveries ORDER BY path')]
        for row in rows:
            if args.search.lower() in json.dumps(row, ensure_ascii=False).lower():
                print('\t'.join([row.get('id', 'historical'), row['state'], row['name'], row.get('output', row.get('path', ''))]))
    elif args.action in ['show', 'note', 'recover']:
        matches = [x for x in reg.rows() if x['id'] == args.id]
        if not matches: raise ValueError('Unknown experiment ID')
        row = matches[0]
        if args.action == 'note':
            row.setdefault('notes', []).append({'time': now(), 'text': args.text})
            reg.save(row)
        elif args.action == 'recover':
            if row['host'] != platform.node(): raise ValueError('Recover on the original host')
            if (row.get('process_token') and process_token(row['pid']) == row['process_token']) or (row.get('child_token') and process_token(row['child_pid']) == row['child_token']):
                raise ValueError('Recorded runner/child is still alive')
            row['artifacts'] = artifacts(row)
            row['state'] = 'recovered_outputs'
            row.setdefault('notes', []).append({'time': now(), 'text': args.reason, 'boundary': 'Output checks passed after interruption; original exit status and unchanged inputs not proven.'})
            reg.save(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    elif args.action == 'doctor':
        problems = [{'id': row['id'], 'issues': health(row)} for row in reg.rows()]
        problems = [x for x in problems if x['issues']]
        for item in reg.db.execute('SELECT data FROM discoveries'):
            row = json.loads(item[0])
            missing = [f['path'] for f in row['files'] if not (Path(row['path']) / f['path']).is_file()]
            if missing: problems.append({'path': row['path'], 'missing_discovered_files': missing})
        print(json.dumps(problems, ensure_ascii=False, indent=2))
        return 1 if problems else 0
    elif args.action == 'export':
        if args.output.exists(): raise ValueError('Use a new export file')
        write_json(args.output, {'exported': now(), 'runs': reg.rows(), 'discoveries': [json.loads(x[0]) for x in reg.db.execute('SELECT data FROM discoveries')]})
        print(args.output)
    elif args.action == 'backup':
        if args.output.exists() or args.output.resolve().is_relative_to(reg.home):
            raise ValueError('Use a new backup file outside registry home')
        temp = reg.home / ('snapshot-' + uuid.uuid4().hex + '.sqlite3')
        archive_temp = args.output.with_name(args.output.name + '.tmp-' + uuid.uuid4().hex)
        try:
            with sqlite3.connect(temp) as dest: reg.db.backup(dest)
            with tarfile.open(archive_temp, 'w:gz') as tar:
                tar.add(temp, arcname='registry.sqlite3')
                if (reg.home / 'runs').exists(): tar.add(reg.home / 'runs', arcname='runs')
                if args.include_results:
                    for row in reg.rows():
                        output = Path(row['output'])
                        if output.exists(): tar.add(output, arcname='results/' + row['id'])
                if args.include_inputs:
                    mapping, copied = {}, set()
                    for row in reg.rows():
                        mapping[row['id']] = {}
                        for key, path in row['input_paths'].items():
                            sha = row['identity']['inputs'][key]
                            archived_source = Path(path)
                            if not archived_source.is_file() or digest(archived_source) != sha:
                                snapshot = reg.home / 'runs' / row['id'] / 'inputs' / (hashlib.sha256(key.encode()).hexdigest() + Path(path).suffix)
                                if not snapshot.is_file() or digest(snapshot) != sha:
                                    raise ValueError('Recorded input unavailable in live path or saved snapshot: ' + path)
                                archived_source = snapshot
                            mapping[row['id']][key] = {'original_path': path, 'archive_path': 'inputs/' + sha, 'copied_from': str(archived_source)}
                            if sha not in copied:
                                entry = tarfile.TarInfo('inputs/' + sha); entry.size = archived_source.stat().st_size
                                with archived_source.open('rb') as source: tar.addfile(entry, source)
                                copied.add(sha)
                    raw = json.dumps(mapping, ensure_ascii=False, indent=2).encode()
                    entry = tarfile.TarInfo('input_paths.json'); entry.size = len(raw)
                    tar.addfile(entry, io.BytesIO(raw))
            os.replace(archive_temp, args.output)
        finally:
            temp.unlink(missing_ok=True)
            archive_temp.unlink(missing_ok=True)
        print('Backup:', args.output, 'Registered results included.' if args.include_results else 'Metadata/source/log only; use --include-results for registered outputs.', 'Declared inputs included.' if args.include_inputs else 'Declared input assets are references; use --include-inputs to copy them.', 'Historical discovery directories are not copied.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(2)
