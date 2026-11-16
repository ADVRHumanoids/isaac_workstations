#!/usr/bin/env python3
"""Manage Isaac Lab training, checkpoints and local replay across SSH workstations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import readline
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import hashlib
import threading
import webbrowser
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Input, Log, OptionList, Static
from textual.widgets.option_list import Option

ROOT = Path(os.environ.get('HHCM_WORKSPACE', '.')).expanduser().resolve()
NAME = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9_-]*\Z')
DESTINATION = re.compile(r'[a-zA-Z0-9_][a-zA-Z0-9_.@-]*\Z')
STATUS_URL = os.environ.get('HHCM_MONITOR_URL', '')
DEFAULT_PLAY_ENVS = 10
STATS_LINE = re.compile(
    r'(\S+)\s+\[[^]]+\]\s+(\S+)\s+IP:\S+\s+'
    r'CPU:(\S+)\s+RAM:(\S+)\s+GPU:(\S+)\s+VRAM:(\S+).*?active_users:(.*)')


def color(text, code):
    if not sys.stdout.isatty() or 'NO_COLOR' in os.environ or os.environ.get('TERM') == 'dumb':
        return text
    return f'\033[{code}m{text}\033[0m'


class MonitorText(HTMLParser):
    """Keep the monitor's text and line breaks, without browser markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden = True
        elif tag == 'br' and not self.hidden:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = False
        elif tag in ('p', 'div', 'tr', 'pre') and not self.hidden:
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def monitor_text(payload):
    parser = MonitorText()
    parser.feed(payload)
    text = ''.join(parser.parts).replace('\xa0', ' ')
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text).strip()
    if not text:
        raise ValueError('The monitor returned no status text')
    return text


def inventory():
    global STATUS_URL
    config = json.loads((ROOT / 'machines.json').read_text())
    STATUS_URL = os.environ.get('HHCM_MONITOR_URL', config.get('monitor_url', ''))
    hosts = {}
    for name, host in config['hosts'].items():
        destination = host if '@' in host else f"{config['user']}@{host}"
        if not NAME.fullmatch(name) or not DESTINATION.fullmatch(destination):
            raise ValueError(f'Invalid machine entry: {name!r}')
        hosts[name] = destination
    return hosts


def new_machines(stats, hosts):
    known = {host.split('@')[-1].lower() for host in hosts.values()}
    return sorted(host for host in stats if host.lower() not in known
                  and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', host))


def add_machines(hostnames):
    path = ROOT / 'machines.json'
    config = json.loads(path.read_text())
    hosts = config['hosts']
    for host in new_machines(hostnames, hosts):
        alias = host.replace('.', '-')
        base, suffix = alias, 2
        while alias in hosts:
            alias, suffix = f'{base}-{suffix}', suffix + 1
        hosts[alias] = host
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=ROOT, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(config, indent=2) + '\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return inventory()


def ssh(host, interactive=False):
    return ['ssh', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=2',
            *([] if interactive else ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes']),
            host]


def status(hosts):
    print(color(f'\n{time.strftime("%H:%M:%S")} — Workstation Monitor: {STATUS_URL}', '1;36'))
    try:
        if not STATUS_URL:
            raise ValueError('Set monitor_url in machines.json or HHCM_MONITOR_URL to enable shared status')
        with urlopen(STATUS_URL, timeout=10) as response:
            text = monitor_text(response.read().decode('utf-8'))
    except (OSError, ValueError) as error:
        print(color(f'Monitor unavailable: {error}', '31'))
        print('No availability information received. This is not a free/busy result.')
        return 1
    print(compact_stats(text))
    return 0


def compact_stats(text):
    rows = [color('Machine                 State CPU   RAM     GPU   VRAM    Users', '1;36')]
    for line in text.splitlines():
        match = STATS_LINE.fullmatch(line)
        if not match:
            rows.append(line)  # Keep unfamiliar monitor output visible.
            continue
        host, state, cpu, ram, gpu, vram, users = match.groups()
        rows.append(f'{host:<23} {state}    {cpu:<5} {ram:<7} {gpu:<5} {vram:<7} {users}')
    return '\n'.join(rows)


def training_status(host):
    try:
        result = subprocess.run(
            [*ssh(host), 'ps -u "$(id -u)" -o comm=,args='],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return 'unreachable'
    if result.returncode:
        return 'unreachable'
    # shortcut: detects Python train.py entry points; extend when adding other trainers.
    runs = []
    for line in result.stdout.splitlines():
        command, _, args = line.strip().partition(' ')
        if command.startswith('python') and re.search(r'(?:^|[/\s])train\.py(?:\s|$)', args):
            names = re.findall(r'(?:^|\s)--run_name(?:=|\s+)(\S+)', args)
            runs.append(names[-1] if names else '(run name unavailable)')
    return 'TRAINING: ' + ', '.join(runs) if runs else 'idle'


RUN_PATH = re.compile(r'[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\Z')
FOLDER_TIME = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_')


def project_folder(name):
    if not NAME.fullmatch(name):
        raise ValueError('Invalid project name')
    folder = ROOT / name
    if not (folder / 'train.sh').is_file():
        raise ValueError(f'Project {name!r} has no train.sh')
    return folder


def saved_checkpoints(host, project, task=None, interactive=True):
    folder = project_folder(project)
    task_map = folder / 'run_tasks.json'
    legacy_tasks = json.loads(task_map.read_text()) if task_map.is_file() else {}
    selector = """from pathlib import Path
import json, sys
root = Path.home() / 'training' / sys.argv[1] / 'data/logs/rsl_rl'
task = sys.argv[2]
run_tasks = json.loads(sys.argv[3])
runs = []
for folder in root.glob('*/*'):
    task_file = folder / 'params/task.txt'
    recorded_task = (task_file.read_text().strip() if task_file.is_file()
                     else run_tasks.get(str(folder.relative_to(root))))
    if task and recorded_task != task:
        continue
    files = list(folder.glob('model_*.pt'))
    if files:
        latest = max(files, key=lambda p: p.stat().st_mtime_ns)
        path = str(latest.relative_to(root))
        runs.append((latest.stat().st_mtime_ns, path))
print(json.dumps([path for _, path in sorted(runs, reverse=True)]))
"""
    try:
        result = subprocess.run(
            [*ssh(host, interactive=interactive), shlex.join(['python3', '-c', selector, project, task or '', json.dumps(legacy_tasks)])],
            capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as error:
        raise ValueError(f'Cannot list training runs on {host}: connection timed out') from error
    if result.returncode:
        raise ValueError(f'Cannot list training runs on {host}: {result.stderr.strip()}')
    checkpoints = json.loads(result.stdout)
    if not isinstance(checkpoints, list) or any(
            not isinstance(path, str) or '..' in path or not re.fullmatch(
                r'[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/model_[0-9]+\.pt', path)
            for path in checkpoints):
        raise ValueError('Unexpected checkpoint list from workstation')
    return checkpoints


def saved_runs(hosts, project):
    folder = project_folder(project)
    print(color(f'Saved training runs — {project}', '1;36'), flush=True)
    print('Saved checkpoints do not imply training completed. Local means a cached checkpoint, not a complete fetch.')

    def listing(host):
        try:
            return saved_checkpoints(host, project, interactive=False), None
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            return [], str(error)

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(listing, hosts.values()))
    failed = False
    print(f'{"Machine":<12} {"Experiment / run":<65} {"Remote checkpoint":<20} Local checkpoint')
    for (name, host), (checkpoints, error) in zip(hosts.items(), results):
        remote = {str(Path(path).parent): Path(path).name for path in checkpoints}
        local = {}
        cache = folder / 'data/replays' / host / 'logs/rsl_rl'
        for path in cache.glob('*/*/model_*.pt'):
            if not re.fullmatch(r'model_[0-9]+\.pt', path.name):
                continue
            run = str(path.parent.relative_to(cache))
            if run not in local or int(path.stem[6:]) > int(Path(local[run]).stem[6:]):
                local[run] = path.name
        if error:
            failed = True
            print(f'{name}: remote unavailable: {error}')
        for run in sorted(remote.keys() | local.keys(), reverse=True):
            print(f'{name:<12} {run:<65} {remote.get(run, "unknown" if error else "—"):<20} {local.get(run, "—")}')
        if not remote and not local:
            print(f'{name}: ' + ('no local cached runs' if error else 'no saved runs'))
    return int(failed)


CLEAN_LIST_SCRIPT = """from pathlib import Path
import json, shutil, sys
root = Path.home() / 'training' / sys.argv[1] / 'data/logs/rsl_rl'
runs = {}
for folder in root.glob('*/*'):
    if folder.is_dir() and not folder.is_symlink():
        runs[str(folder.relative_to(root))] = {
            str(p.relative_to(folder)): p.stat().st_size
            for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}
usage = shutil.disk_usage(Path.home())
print(json.dumps({'runs': runs, 'free': usage.free, 'total': usage.total}))
"""
CLEAN_DELETE_SCRIPT = """from pathlib import Path
import json, re, shutil, sys
root = (Path.home() / 'training' / sys.argv[1] / 'data/logs/rsl_rl').resolve()
for run in json.loads(sys.argv[2]):
    target = root / run
    if (not re.fullmatch(r'[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+', run) or '..' in run
            or target.is_symlink() or not target.is_dir() or target.resolve().parent.parent != root):
        sys.exit('Refusing to delete ' + run)
    shutil.rmtree(target)
    if not any(target.parent.iterdir()):
        target.parent.rmdir()
    print('deleted', run, flush=True)
"""


def size_text(size):
    if size is None:
        return '—'
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1000 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1000


def remote_run_files(host, project):
    """Return ({run: {relative file: size}}, free bytes, total bytes) for the host's home disk."""
    result = subprocess.run([*ssh(host), shlex.join(['python3', '-c', CLEAN_LIST_SCRIPT, project])],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f'exit {result.returncode}')
    data = json.loads(result.stdout)
    runs = data['runs']
    if not isinstance(runs, dict) or any(
            '..' in run or not RUN_PATH.fullmatch(run) or not isinstance(files, dict) for run, files in runs.items()):
        raise ValueError('Unexpected run list from workstation')
    return runs, int(data['free']), int(data['total'])


def local_run_files(folder, host):
    cache = folder / 'data/replays' / host / 'logs/rsl_rl'
    return {str(run.relative_to(cache)): {str(p.relative_to(run)): p.stat().st_size
                                          for p in run.rglob('*') if p.is_file() and not p.is_symlink()}
            for run in cache.glob('*/*') if run.is_dir() and not run.is_symlink()}


def fetch_state(remote, local):
    """Compare file names and sizes: 'fetched' means every remote file is on this PC."""
    if remote is None:
        return 'local only' if local is not None else 'unknown'
    if local is None:
        return 'not fetched'
    return 'fetched' if all(local.get(name) == size for name, size in remote.items()) else 'partial'


def chosen_numbers(text, count):
    if text.strip().lower() == 'all':
        return list(range(1, count + 1))
    numbers = set()
    for part in text.replace(',', ' ').split():
        first, dash, last = part.partition('-')
        if not (first.isascii() and first.isdigit() and (not dash or (last.isascii() and last.isdigit()))):
            raise ValueError('Use numbers like 1 3 5-7, or all')
        numbers.update(range(int(first), int(last or first) + 1))
    if not numbers or min(numbers) < 1 or max(numbers) > count:
        raise ValueError('Choose run numbers from the list')
    return sorted(numbers)


def clean(name, host, project, run=None, where=None, yes=False, permanent=False):
    """Delete runs. where/yes/permanent answer the prompts (lab app jobs and scripts)."""
    folder = project_folder(project)
    cache = folder / 'data/replays' / host / 'logs/rsl_rl'
    print(f'Checking {name} ({host}) for {project} runs...', flush=True)
    try:
        remote, free, total = remote_run_files(host, project)
        remote_error = None
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        remote, free, total, remote_error = {}, None, None, str(error) or 'unreachable'
    local = local_run_files(folder, host)
    state = training_status(host) if remote_error is None else 'unreachable'
    active = state[len('TRAINING: '):].split(', ') if state.startswith('TRAINING: ') else []
    runs = sorted(remote.keys() | local.keys(), key=lambda run: (Path(run).name, run), reverse=True)
    only = run
    if only is not None:  # Already chosen in the lab app.
        if only not in runs:
            raise ValueError(f'{only} is not on {name} or this PC')
        runs = [only]

    print(color(f'\nRuns of {project} on {name} (newest first):', '1;36'))
    if remote_error:
        print(color(f'Remote unavailable: {remote_error}. Only local copies are shown and can be deleted.', '33'))
    else:
        print(f'{name} home disk: {size_text(free)} free of {size_text(total)}; '
              f'runs use {size_text(sum(sum(f.values()) for f in remote.values()))}.')
    print(f'Local copies on this PC: {size_text(sum(sum(f.values()) for f in local.values()))} in {cache}')
    if not runs:
        print('No runs found.')
        return 0 if remote_error is None else 1
    width = max(len('Experiment / run'), *(len(run) for run in runs))
    print(f'  {"#":>3}  {"Experiment / run":<{width}}  {"Remote":>9}  {"Local":>9}  State')
    rows = []
    for number, run in enumerate(runs, 1):
        remote_files = remote.get(run) if remote_error is None else None
        local_files = local.get(run)
        training = remote_files is not None and any(Path(run).name.endswith('_' + active_run) for active_run in active)
        status_text = fetch_state(remote_files, local_files) + (', TRAINING' if training else '')
        rows.append((run, remote_files, local_files, training, status_text))
        print(f'  {number:>3}  {run:<{width}}  {size_text(sum(remote_files.values()) if remote_files is not None else None):>9}  '
              f'{size_text(sum(local_files.values()) if local_files is not None else None):>9}  '
              + color(status_text, {'fetched': '32', 'partial': '33', 'not fetched': '33'}.get(status_text, '37')))
    print('Fetched = every remote file is on this PC with the same size (a play-cached checkpoint alone is partial).')

    choice = '1' if only is not None else input('Runs to delete (e.g. 1 3 5-7, all; blank cancels): ').strip()
    if not choice:
        return 0
    selected = [rows[number - 1] for number in chosen_numbers(choice, len(rows))]
    where = where or input('Delete [l]ocal copies, [r]emote originals, or [b]oth? (blank cancels): ').strip().lower()
    if where not in ('l', 'r', 'b'):
        return 0
    delete_local = [row for row in selected if where in 'lb' and row[2] is not None]
    delete_remote = [row for row in selected if where in 'rb' and row[1] is not None]
    if where in 'rb' and remote_error:
        raise ValueError(f'{name} is unreachable: remote runs cannot be deleted now')
    if any(row[3] for row in delete_remote):
        raise ValueError('A selected run is still training on ' + name + '; stop it first')
    if not delete_local and not delete_remote:
        print('Nothing to delete at the chosen location.')
        return 0
    # Losing the only complete copy: remote not fully fetched, or a local copy with no remote original.
    # Unknown remote state (unreachable) counts as possibly the only copy.
    unrecoverable = ([row[0] for row in delete_remote if not row[4].startswith('fetched') or row in delete_local]
                     + [row[0] for row in delete_local if row[1] is None])
    print(color('\nAbout to delete:', '1;33'))
    for run, remote_files, local_files, _, _ in delete_remote:
        print(f'  remote {name}:{run}  ({size_text(sum(remote_files.values()))})')
    for run, _, local_files, _, _ in delete_local:
        print(f'  local  {run}  ({size_text(sum(local_files.values()))})')
    if unrecoverable:
        print(color('These runs will be gone permanently (no complete copy remains):', '1;31'))
        for run in dict.fromkeys(unrecoverable):
            print(f'  {run}')
        if yes and not permanent:
            raise ValueError('Nothing deleted: this removes the last complete copy; confirm permanent deletion in the app')
        if not permanent and input("Type 'delete' to confirm: ").strip() != 'delete':
            print('Cancelled; nothing deleted.')
            return 0
    elif not yes and input('Confirm? [y/N] ').strip().lower() != 'y':
        print('Cancelled; nothing deleted.')
        return 0

    failed = 0
    if delete_remote:
        result = subprocess.run([*ssh(host, interactive=True), shlex.join(
            ['python3', '-c', CLEAN_DELETE_SCRIPT, project, json.dumps([row[0] for row in delete_remote])])])
        if result.returncode:
            print(f'Remote deletion stopped (exit {result.returncode}); lines above show what was deleted.')
            failed = 1
    root = cache.resolve()
    for run, *_ in delete_local:
        target = cache / run
        if target.is_symlink() or target.resolve().parent.parent != root:
            raise ValueError('Refusing to delete ' + str(target))
        shutil.rmtree(target)
        if not any(target.parent.iterdir()):
            target.parent.rmdir()
        print('deleted local', run)
    return failed


def selected_replay(host, project, purpose='replay on this PC', task=None, run=None):
    checkpoints = saved_checkpoints(host, project, task=task)
    if run is not None:  # Already chosen in the lab app: use that run's latest checkpoint.
        chosen = [path for path in checkpoints if str(Path(path).parent) == run]
        if not chosen:
            raise ValueError(f'{run} has no saved checkpoint on {host}'
                             + (f' for task {task}' if task else ''))
        return chosen[0]
    if not checkpoints:
        print(f'No saved training runs on {host} for {project}'
              + (f' matching task {task}.' if task else '.'))
        return None
    print(color(f'Saved training runs on {host} (newest first):', '1;36'))
    print('Each entry uses its latest saved checkpoint; a run may still be training.')
    for number, path in enumerate(checkpoints, 1):
        print(f'  {number}. {path}')
    choice = input(f'Run number to {purpose} (blank cancels): ').strip()
    if not choice:
        return None
    if not choice.isascii() or not choice.isdigit() or not 1 <= int(choice) <= len(checkpoints):
        raise ValueError('Choose a run number from the list')
    return checkpoints[int(choice) - 1]


def copy_run(host, project, run, upload=False, source=None):
    """rsync one run folder from a workstation into this PC's replay cache, or upload source to it."""
    if not DESTINATION.fullmatch(host) or '..' in run or not RUN_PATH.fullmatch(run):
        raise ValueError('Invalid workstation or run')
    remote = f'training/{project}/data/logs/rsl_rl/{run}'
    shell = shlex.join(ssh(host, interactive=True)[:-1])
    if upload:
        print(f'Uploading the run to {host}:~/{remote}', flush=True)
        made = subprocess.run([*ssh(host, interactive=True), shlex.join(['mkdir', '-p', remote])])
        if made.returncode:
            return made.returncode
        paths = [str(source) + '/', f'{host}:{remote}/']
    else:
        local = project_folder(project) / 'data/replays' / host / 'logs/rsl_rl' / run
        local.mkdir(parents=True, exist_ok=True)
        print(f'Copying the complete run to {local}', flush=True)
        paths = [f'{host}:{remote}/', str(local) + '/']
    return subprocess.run(['rsync', '-a', '--partial', '--info=progress2', '-s', '-e', shell, *paths]).returncode


ACTIVE_RUNS_SCRIPT = r"""from pathlib import Path
import json, os, re, sys
active = {}
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        if os.stat('/proc/' + pid).st_uid != os.getuid():
            continue
        args = [a.decode() for a in Path('/proc', pid, 'cmdline').read_bytes().split(b'\0')[:-1]]
    except (OSError, UnicodeDecodeError):
        continue
    script = next((i for i, a in enumerate(args) if re.search(r'(^|/)train\.py$', a)), None)
    if script is None or not os.path.basename(args[0]).startswith('python'):
        continue
    rest = args[script + 1:]
    names = [rest[i + 1] for i, a in enumerate(rest[:-1]) if a == '--run_name'] + [
        a.split('=', 1)[1] for a in rest if a.startswith('--run_name=')]
    if names:
        active[names[-1]] = rest
"""


ARGS_SCRIPT = ACTIVE_RUNS_SCRIPT + """root = Path.home() / 'training' / sys.argv[1] / 'data/logs/rsl_rl'
run = root / sys.argv[2]
saved = run / 'params/command.json'
saved = json.loads(saved.read_text()) if saved.is_file() else None
task = run / 'params/task.txt'
task = task.read_text().strip() if task.is_file() else None
running = None
for name, args in active.items():
    # The active process writes to the newest folder ending in its run name.
    same = sorted(p.name for p in run.parent.iterdir() if p.is_dir() and p.name.endswith('_' + name))
    if same and same[-1] == run.name:
        running = args
print(json.dumps({'saved': saved, 'active': running, 'task': task}))
"""
RESUME_OPTIONS = ('--resume', '--load_run', '--checkpoint', '--max_iterations')
RESUME_OVERRIDES = ('agent.resume=', 'agent.load_run=', 'agent.load_checkpoint=', 'agent.max_iterations=')


def resume_arguments(args, run_folder, checkpoint, iterations):
    """Replace any resume/iteration settings in the original train.py arguments."""
    kept, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        option = arg.split('=', 1)[0]
        if option in RESUME_OPTIONS:
            skip = option != '--resume' and '=' not in arg
            continue
        if arg.startswith(RESUME_OVERRIDES):
            continue
        kept.append(arg)
    return [*kept, '--resume', '--load_run', run_folder, '--checkpoint', checkpoint,
            '--max_iterations', str(iterations)]


def remaining_iterations(local_run, checkpoint):
    """Iterations left to the original target: first saved iteration + max_iterations - checkpoint."""
    agent = (local_run / 'params/agent.yaml').read_text()
    match = re.search(r'^max_iterations: (\d+)$', agent, re.M)
    if not match:
        raise ValueError('agent.yaml has no max_iterations')
    saved = [int(p.stem[6:]) for p in local_run.glob('model_*.pt') if re.fullmatch(r'model_[0-9]+\.pt', p.name)]
    return min(saved) + int(match.group(1)) - int(Path(checkpoint).stem[6:])


def move(source_name, source, target_name, target, project, run=None, yes=False, train_args=None,
         extra_iterations=None):
    """yes/train_args/extra_iterations answer the prompts (lab app jobs and scripts)."""
    folder = project_folder(project)
    if source == target:
        raise ValueError('Choose two different workstations')
    target_state = training_status(target)
    if target_state != 'idle':
        raise ValueError(f'{target_name} is {target_state}; choose an idle workstation')
    checkpoint = selected_replay(source, project, purpose=f'move to {target_name}', run=run)
    if checkpoint is None:
        return 0
    run = str(Path(checkpoint).parent)
    result = subprocess.run([*ssh(source), shlex.join(['python3', '-c', ARGS_SCRIPT, project, run])],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise ValueError(f'Cannot read run settings on {source_name}: {result.stderr.strip()}')
    info = json.loads(result.stdout)
    active = info['active']
    args = active or info['saved']
    if not (isinstance(args, list) and all(isinstance(arg, str) for arg in args)):
        print(color('This run did not save its command line (runs started before command.json existed).', '33'))
        default = shlex.join(['--task', info['task']]) if info['task'] else ''
        if train_args is not None:
            typed = train_args
        elif yes:
            raise ValueError('This run did not save its arguments; enter them in the app before moving')
        else:
            typed = input(f'Full train.py arguments to resume with [{default}]: ').strip() or default
        if not typed:
            return 0
        args = shlex.split(typed)
        args += [] if '--run_name' in args or any(a.startswith('--run_name=') for a in args) else \
            ['--run_name', Path(run).name[20:]]
    print(color(f'\nMove {run}', '1;36'))
    print(f'  1. ' + (f'Stop the running training on {source_name} (SIGINT; checkpoints are kept)'
                      if active else f'Not training on {source_name}; nothing to stop'))
    print(f'  2. Copy the run {source_name} -> this PC -> {target_name}')
    print(f'  3. Resume on {target_name} from its latest checkpoint after the copy')
    print('  The original run stays on ' + source_name + '; remove it later with clean.')
    print('  Arguments: ' + shlex.join(args))
    if not yes and input('Move this run? [y/N] ').strip().lower() != 'y':
        return 0

    if active:
        if action(source, 'stop', project, []):
            raise ValueError(f'Could not stop training on {source_name}; nothing was moved')
        wait = ('for i in $(seq 60); do docker container inspect ' + shlex.quote(project)
                + ' >/dev/null 2>&1 || exit 0; sleep 2; done; exit 1')
        if subprocess.run([*ssh(source, interactive=True), wait]).returncode:
            raise ValueError(f'Training on {source_name} did not exit within 2 minutes; nothing was moved')
    if copy_run(source, project, run):
        raise ValueError(f'Copy from {source_name} failed; run move again to continue the copy')
    local = folder / 'data/replays' / source / 'logs/rsl_rl' / run
    models = [p for p in local.glob('model_*.pt') if re.fullmatch(r'model_[0-9]+\.pt', p.name)]
    if not models:
        raise ValueError('The run has no checkpoint to resume from')
    latest = max(models, key=lambda p: int(p.stem[6:])).name
    iterations = remaining_iterations(local, latest)
    if iterations <= 0:
        if extra_iterations is not None:
            typed = str(extra_iterations)
        elif yes:
            print(f'{latest} already reached the target. Run copied to this PC; nothing was started on {target_name}.')
            print('To continue training, move again and enter extra iterations in the app.')
            return 0
        else:
            typed = input(f'{latest} already reached the target. Extra iterations to train (blank stops here): ').strip()
        if not typed:
            print(f'Run copied to this PC; nothing was started on {target_name}.')
            return 0
        if not typed.isascii() or not typed.isdigit() or int(typed) < 1:
            raise ValueError('Enter a positive number of iterations')
        iterations = int(typed)
    if copy_run(target, project, run, upload=True, source=local):
        raise ValueError(f'Copy to {target_name} failed; the run is safe on {source_name} and this PC')
    args = resume_arguments(args, Path(run).name, latest, iterations)
    environment = os.environ.copy()
    environment.pop('TRAINING_SSH_CONTROL_PATH', None)
    command = ['bash', str(Path(__file__).with_name('pipeline.sh')), 'deploy', str(folder), target, *args]
    print(f'Resuming from {latest} for {iterations} more iterations.')
    print('Command:', shlex.join(command), flush=True)
    return subprocess.run(command, env=environment).returncode


def action(host, command, project, extra, run=None):
    folder = project_folder(project)
    if command == 'stop':
        if extra:
            raise ValueError('stop takes only a machine and project')
        print(f'Stopping training container {project} on {host}. Saved results are preserved.', flush=True)
        stop = '''set -eu
case $(docker info --format '{{json .SecurityOptions}}') in
  *rootless*) ;;
  *) echo 'Select rootless Docker on this host.' >&2; exit 1 ;;
esac
docker stop --signal SIGINT --timeout 30 ''' + shlex.quote(project)
        return subprocess.run([*ssh(host, interactive=True), stop]).returncode
    if command == 'fetch':
        if extra:
            raise ValueError('fetch takes only a machine and project')
        if not DESTINATION.fullmatch(host):
            raise ValueError('Invalid workstation destination')
        checkpoint = selected_replay(host, project, purpose='copy to this PC', run=run)
        if checkpoint is None:
            return 0
        result = copy_run(host, project, str(Path(checkpoint).parent))
        if result == 0:
            print('Run copied. The remote original is preserved.')
        return result
    environment = os.environ.copy()
    environment.pop('TRAINING_SSH_CONTROL_PATH', None)
    environment.pop('TRAINING_REPLAY_CHECKPOINT', None)
    if command == 'play':
        iteration_parser = argparse.ArgumentParser(add_help=False)
        iteration_parser.add_argument('--iteration', type=int)
        iteration_parser.add_argument('--env', '--envs', dest='env', type=int)
        iteration_args, extra = iteration_parser.parse_known_args(extra)
        if iteration_args.iteration is not None and iteration_args.iteration < 0:
            raise ValueError('Iteration must be nonnegative')
        if iteration_args.env is not None and iteration_args.env < 1:
            raise ValueError('--env must be at least 1')
        explicit_envs = any(arg == '--num_envs' or arg.startswith('--num_envs=') for arg in extra)
        if iteration_args.env is not None and explicit_envs:
            raise ValueError('Use either --env or --num_envs, not both')
        if not explicit_envs:
            # play.sh defaults to 1 robot; a later --num_envs overrides it.
            extra = ['--num_envs', str(iteration_args.env or DEFAULT_PLAY_ENVS), *extra]
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--task')
        replay_args, _ = parser.parse_known_args(extra)
        checkpoint = selected_replay(host, project, task=replay_args.task, run=run)
        if checkpoint is None:
            return 0
        if iteration_args.iteration is not None:
            checkpoint = str(Path(checkpoint).with_name(f'model_{iteration_args.iteration}.pt'))
            print(f'Requested checkpoint: {checkpoint}', flush=True)
        environment['TRAINING_REPLAY_CHECKPOINT'] = checkpoint
    script = folder / ('play.sh' if command == 'play' else 'train.sh')
    args = ['bash', str(script), host, *extra]
    print('Command:', shlex.join(args), flush=True)
    # On Ctrl+C subprocess.run would kill the script at once, skipping its cleanup (play.sh
    # stops the viewer container). The script gets Ctrl+C too, so just keep waiting for it.
    # A handler (unlike SIG_IGN) is reset in the child, so the script still sees Ctrl+C.
    previous = signal.signal(signal.SIGINT, lambda *_: print('Stopping…', flush=True))
    try:
        return subprocess.run(args, env=environment).returncode
    finally:
        signal.signal(signal.SIGINT, previous)


def projects():
    return sorted(path.name for path in ROOT.iterdir()
                  if NAME.fullmatch(path.name) and (path / 'train.sh').is_file())


def selected_training(host):
    available = projects()
    if not available:
        raise ValueError('No projects found: add a project folder containing train.sh.')
    print(color('Available projects: ' + ', '.join(available), '1;36'))
    default = available[0]
    project = input(f'Project [{default}]: ').strip() or default
    folder = project_folder(project)
    run_name = input('Run name (blank cancels): ').strip()
    if not run_name:
        return
    extra = ['--run_name', run_name, 'agent.resume=false']
    extra += shlex.split(input('Extra arguments (blank uses train.sh defaults): '))
    print(color(f'\nDestination: {host}:~/training/{project}', '1;33'))
    print('Command:', shlex.join(['bash', str(folder / 'train.sh'), host, *extra]))
    print('This mirrors source/assets, prepares the image if missing, and starts training.')
    print('Existing results are preserved. W&B/video settings come from train.sh.')
    print('Check the machine is free first: contact the current user before using an occupied machine.')
    if input('Launch on this machine? [y/N] ').strip().lower() != 'y':
        return
    result = action(host, 'train', project, extra)
    if result:
        print(f'Launch command failed (exit {result}); see the pipeline output above.')


ACTIONS = ('train', 'play', 'fetch', 'stop', 'clean', 'move')


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog='''Commands:
  app     Open the terminal interface (also used when no command is given).
  status  Show shared workstation utilization; uses monitor_url from machines.json or HHCM_MONITOR_URL.
  runs    List saved runs on all machines, or on the selected machine.
  train   Launch training on a machine; "train MACHINE login" prepares W&B login.
  play    Download a checkpoint from a machine and replay it locally in Docker.
  fetch   Download a complete run, keeping the remote original.
  stop    Stop the project's remote training container, keeping saved results.
  clean   Delete selected local or remote runs; asks for confirmation.
  move    Transfer a run to another machine and resume training there.

Examples (replace MACHINE and TARGET with aliases from machines.json):
  %(prog)s
  %(prog)s --project kyon_training
  %(prog)s runs MACHINE
  %(prog)s train MACHINE --task Your-Task-v0 --run_name first_run
  %(prog)s play MACHINE
  %(prog)s move MACHINE TARGET

Extra train/play arguments are passed to your project's train.sh/play.sh.
Saved runs and confirmations are chosen interactively.''')
    parser.add_argument('command', nargs='?', default='app', choices=('app', 'status', 'runs', *ACTIONS),
                        help='Action to perform (default: app; descriptions below)')
    parser.add_argument('machine', nargs='?', help='SSH machine alias from machines.json; move also needs a target alias')
    parser.add_argument('--project', metavar='NAME',
                        help='Training folder inside the workspace, e.g. kyon_training; defaults to the first folder with train.sh')
    parser.add_argument('--workspace', type=Path, default=ROOT, metavar='PATH',
                        help='Folder containing machines.json and project folders (default: HHCM_WORKSPACE, otherwise current directory)')
    args, extra = parser.parse_known_args()
    removed = {'--run', '--yes', '--where', '--permanent', '--train-args', '--extra-iterations'}
    if any(arg.split('=', 1)[0] in removed for arg in extra):
        parser.error('Choose runs, confirmations and resume settings inside the app')
    answers = json.loads(os.environ.pop('HHCM_JOB_ANSWERS', '{}'))
    if extra and args.command not in ('train', 'play', 'move'):
        parser.error('Extra arguments are allowed only with train or play')
    ROOT = args.workspace.expanduser().resolve()
    hosts = inventory()
    args.project = args.project or next(iter(projects()), 'project')
    if args.command == 'app':
        return run_app(hosts, project_folder(args.project).name)
    if args.command == 'runs':
        if args.machine is not None and args.machine not in hosts:
            parser.error('Choose a machine: ' + ', '.join(hosts))
        return saved_runs({args.machine: hosts[args.machine]} if args.machine else hosts, args.project)
    if args.command == 'status':
        return status(hosts)
    if args.machine not in hosts:
        parser.error('Choose a machine: ' + ', '.join(hosts))
    if args.command == 'clean':
        return clean(args.machine, hosts[args.machine], args.project, run=answers.get('run'), where=answers.get('where'),
                     yes=answers.get('yes', False), permanent=answers.get('permanent', False))
    if args.command == 'move':
        if len(extra) != 1 or extra[0] not in hosts:
            parser.error('Usage: hhcm_isaac_workstations.py move <from> <to> [--project name]; machines: ' + ', '.join(hosts))
        return move(args.machine, hosts[args.machine], extra[0], hosts[extra[0]], args.project, run=answers.get('run'),
                    yes=answers.get('yes', False), train_args=answers.get('train_args'),
                    extra_iterations=answers.get('extra_iterations'))
    return action(hosts[args.machine], args.command, args.project, extra, run=answers.get('run'))


STATS_SECONDS = 10
RUNS_SECONDS = 30

# One SSH call per machine: runs (size, latest checkpoint, file manifest hash, task,
# saved or live arguments), free home-disk space and the run names training now.
APP_SCRIPT = ACTIVE_RUNS_SCRIPT + r"""import hashlib, shutil
root = Path.home() / 'training' / sys.argv[1] / 'data/logs/rsl_rl'
runs = []
for folder in root.glob('*/*'):
    if not folder.is_dir() or folder.is_symlink():
        continue
    files = sorted((str(p.relative_to(folder)), p.stat().st_size)
                   for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
    models = [int(m.group(1)) for name, _ in files for m in [re.fullmatch(r'model_([0-9]+)\.pt', name)] if m]
    task = folder / 'params/task.txt'
    command = folder / 'params/command.json'
    try:
        command = json.loads(command.read_text()) if command.is_file() else None
    except ValueError:
        command = None
    runs.append({'run': str(folder.relative_to(root)), 'size': sum(size for _, size in files),
                 'hash': hashlib.sha1(json.dumps(files).encode()).hexdigest(),
                 'checkpoint': f'model_{max(models)}.pt' if models else None,
                 'task': task.read_text().strip() if task.is_file() else None, 'command': command})
# A live run writes to the newest folder ending in its run name.
for name, args in active.items():
    same = [r for r in runs if Path(r['run']).name.endswith('_' + name)]
    if same:
        newest = max(same, key=lambda r: Path(r['run']).name)
        newest['training'] = True
        newest['command'] = newest['command'] or args
print(json.dumps({'runs': runs, 'free': shutil.disk_usage(Path.home()).free, 'active': sorted(active)}))
"""


def manifest_hash(files):
    return hashlib.sha1(json.dumps(sorted(files.items())).encode()).hexdigest()


def machine_snapshot(host, project):
    try:
        result = subprocess.run([*ssh(host), shlex.join(['python3', '-c', APP_SCRIPT, project])],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
        if result.returncode:
            return {'error': result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unreachable'}
        data = json.loads(result.stdout)
        for run in data['runs']:
            if '..' in run['run'] or not RUN_PATH.fullmatch(run['run']):
                return {'error': 'unexpected run list'}
        return data
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        return {'error': str(error) or 'unreachable'}


def availability():
    """{hostname: (state, cpu, ram, gpu, vram, users)} from the shared /global_stats monitor."""
    if not STATUS_URL:
        raise ValueError('Set monitor_url in machines.json or HHCM_MONITOR_URL to enable shared status')
    with urlopen(STATUS_URL, timeout=10) as response:
        text = monitor_text(response.read().decode('utf-8'))
    machines = {}
    for line in text.splitlines():
        match = STATS_LINE.fullmatch(line.strip())
        if match:
            host, state, cpu, ram, gpu, vram, users = match.groups()
            state = {'🟩': 'free', '🟥': 'busy'}.get(state, state)
            names = re.findall(r"'([^'\[]+)\[", users)
            machines[host] = (state, cpu, ram, gpu, vram, ', '.join(names) or '—')
    if not machines:
        raise ValueError('The monitor returned no workstation statistics')
    return machines


def run_rows(hosts, snapshots, folder):
    """Every run on each machine and in this PC's cache, newest first."""
    rows = []
    for name, host in hosts.items():
        snapshot = snapshots.get(name, {'error': 'not checked yet'})
        local = local_run_files(folder, host)
        remote = {run['run']: run for run in snapshot.get('runs', [])}
        for run in remote.keys() | local.keys():
            info, files = remote.get(run), local.get(run)
            if info and info.get('training'):
                state = 'training'
            elif 'error' in snapshot:
                state = 'remote unknown'
            elif info is None:
                state = 'local only'
            elif files is None:
                state = 'not fetched'
            else:
                state = 'fetched' if manifest_hash(files) == info['hash'] else 'partial'
            checkpoint = info['checkpoint'] if info else max(
                (f for f in files if re.fullmatch(r'model_[0-9]+\.pt', f)), key=lambda f: int(f[6:-3]), default=None)
            rows.append({'machine': name, 'host': host, 'run': run, 'state': state,
                         'checkpoint': checkpoint, 'remote': info['size'] if info else None,
                         'local': sum(files.values()) if files is not None else None,
                         'task': info.get('task') if info else None,
                         'command': info.get('command') if info else None})
    # Timestamped run folders first (newest on top); anything else sinks to the bottom.
    return sorted(rows, key=lambda row: (bool(FOLDER_TIME.match(Path(row['run']).name)),
                                         Path(row['run']).name, row['machine']), reverse=True)


ANSI = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[()][A-Za-z0-9]|\x1b[=>]')
LINK = re.compile(r'Open (http://\S+)')
PERCENT = re.compile(r'(\d{1,3})%')
KEEP_LINES = 3000


class Job:
    """One background CLI action; its output is collected line by line for the panel."""

    def __init__(self, number, kind, title, command, key=None, notify=lambda job, committed: None):
        self.number, self.kind, self.title, self.command, self.key = number, kind, title, command, key
        self.notify = notify
        self.lines, self.live, self.progress, self.link = [], '', None, None
        self.code, self.process, self.pending, self.signals, self.last_notice = None, None, '', 0, 0.0

    @property
    def running(self):
        return self.process is not None and self.code is None

    def start(self):
        environment = dict(os.environ, PYTHONUNBUFFERED='1')
        environment.pop('TRAINING_SSH_CONTROL_PATH', None)
        # A new session has no controlling terminal: SSH cannot prompt inside the app's screen.
        self.process = subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, start_new_session=True, env=environment,
                                        cwd=ROOT)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for chunk in iter(lambda: self.process.stdout.read1(65536), b''):
            self.feed(chunk.decode('utf-8', 'replace'))
        if self.pending.strip('\r'):
            self.add(self.pending.split('\r')[-1] or self.pending.strip('\r'))
        self.pending, self.live = '', ''
        self.code = self.process.wait()
        self.progress = None
        self.notify(self, True)

    def feed(self, text):
        """Split output into lines; carriage-return updates (rsync, dd) only replace the live line."""
        text = (self.pending + text).replace('\r\n', '\n')
        *done, self.pending = text.split('\n')
        for line in done:
            self.add(next((part for part in reversed(line.split('\r')) if part), ''))
        parts = self.pending.split('\r')
        if len(parts) > 1:  # Keep only the latest overwrite (and a trailing \r, which may start \r\n).
            self.pending = parts[-1] if parts[-1] else parts[-2] + '\r'
        self.live = ANSI.sub('', self.pending.rstrip('\r'))
        match = PERCENT.search(self.live)
        if match:
            self.progress = int(match.group(1))
        now = time.monotonic()
        if done or now - self.last_notice > 0.2:
            self.last_notice = now
            self.notify(self, bool(done))

    def add(self, line):
        line = ANSI.sub('', line)
        link = LINK.search(line)
        if link:
            self.link = link.group(1)
        self.lines.append(line)
        if len(self.lines) > KEEP_LINES:
            del self.lines[:len(self.lines) - KEEP_LINES]

    def interrupt(self):
        """First press: SIGINT (like Ctrl+C); again: SIGTERM."""
        if not self.running:
            return
        self.signals += 1
        try:
            os.killpg(self.process.pid, signal.SIGINT if self.signals == 1 else signal.SIGTERM)
        except ProcessLookupError:
            pass

    def label(self):
        if self.running:
            status = f'{self.progress}%' if self.progress is not None else '…'
        else:
            status = '✓' if self.code == 0 else f'✗ {self.code}'
        return f'[{self.number}] {self.title} {status}'


class Ask(ModalScreen):
    """One dialog for every question: text, input fields, a choice list and/or a typed confirmation."""
    DEFAULT_CSS = """
    Ask { align: center middle; }
    Ask > Vertical { width: 100; max-width: 95%; height: auto; max-height: 90%;
                     border: thick $accent; background: $surface; padding: 1 2; }
    Ask .title { text-style: bold; color: $accent; margin-bottom: 1; }
    Ask .body { margin-bottom: 1; }
    Ask .warning { color: $error; text-style: bold; margin-bottom: 1; }
    Ask OptionList { height: auto; max-height: 14; margin-bottom: 1; }
    Ask Input { margin-bottom: 1; }
    Ask Horizontal { height: auto; align-horizontal: right; }
    Ask Button { margin-left: 2; }
    """
    BINDINGS = [Binding('escape', 'cancel', 'cancel')]

    def __init__(self, title, body='', fields=(), choices=(), word=None, warning='', ok='OK'):
        super().__init__()
        self.title_text, self.body, self.fields, self.choices = title, body, list(fields), list(choices)
        self.word, self.warning, self.ok = word, warning, ok

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(self.title_text), classes='title')
            if self.body:
                yield Static(Text(self.body), classes='body')
            if self.warning:
                yield Static(Text(self.warning), classes='warning')
            if self.choices:
                yield OptionList(*(Option(Text(label), id=value, disabled=disabled)
                                   for value, label, disabled in self.choices))
            for key, label, default in self.fields:
                yield Static(Text(label))
                yield Input(value=default, id=f'field-{key}', select_on_focus=False)  # Typing extends defaults.
            if self.word:
                yield Static(Text(f"Type '{self.word}' to confirm:"))
                yield Input(id='word')
            if not self.choices:
                with Horizontal():
                    yield Button('Cancel', id='cancel')
                    yield Button(self.ok, id='ok', variant='error' if self.word else 'primary',
                                 disabled=bool(self.word))

    def on_mount(self):
        targets = list(self.query('OptionList, Input'))
        (targets[0] if targets else self.query_one('#ok', Button)).focus()
        for box in self.query(Input):
            box.cursor_position = len(box.value)

    def on_input_changed(self, event):
        if self.word and event.input.id == 'word':
            self.query_one('#ok', Button).disabled = event.value.strip() != self.word

    def on_input_submitted(self, _):
        if not (self.word and self.query_one('#ok', Button).disabled):
            self.submit()

    def on_option_list_option_selected(self, event):
        self.dismiss(event.option.id)

    def on_button_pressed(self, event):
        self.submit() if event.button.id == 'ok' else self.dismiss(None)

    def submit(self):
        # Plain confirmations return True; forms return their values (never an empty, falsy dict).
        values = {key: self.query_one(f'#field-{key}', Input).value for key, _, _ in self.fields}
        self.dismiss(values or True)

    def action_cancel(self):
        self.dismiss(None)


STATE_STYLE = {'training': 'bold green', 'fetched': 'green', 'partial': 'yellow', 'not fetched': 'yellow',
               'local only': 'cyan', 'remote unknown': 'dim'}
JOB_VERB = {'fetch': '⇣ fetching', 'move': '⇄ moving', 'play': '▷ playing', 'delete': '✗ deleting'}


def pasted_training(text, hosts, project):
    """Split a pasted train command into (machine or None, run name or None, train.py arguments).

    Accepts bare train.py arguments or a whole `[python3] hhcm_isaac_workstations.py train <machine> [project] ...` /
    `train.sh <machine> ...` line, with --run_name anywhere; agent.resume defaults to false.
    """
    words = shlex.split(text)
    machine = None
    while words and Path(words[0]).name in ('python', 'python3', 'hhcm_isaac_workstations.py', 'hhcm_isaac_workstations', 'lab.py', 'train', 'train.sh', 'bash'):
        words.pop(0)
    if words and words[0] in hosts:
        machine = words.pop(0)
    if words and words[0] == project:
        words.pop(0)
    name, args = None, []
    while words:
        word = words.pop(0)
        if word == '--run_name' and words:
            name = words.pop(0)
        elif word.startswith('--run_name='):
            name = word.split('=', 1)[1]
        elif word == '--project' and words:
            words.pop(0)
        else:
            args.append(word)
    if not any(word.startswith('agent.resume=') for word in args):
        args.insert(0, 'agent.resume=false')
    return machine, name, args


class LabApp(App):
    TITLE = 'HHCM Isaac Workstations'
    CSS = """
    #title { height: 1; padding: 0 1; background: $primary; color: $text; text-style: bold; }
    .label { height: 1; padding: 0 1; color: $accent; text-style: bold; }
    #machines { height: auto; max-height: 14; }
    #runs { height: 1fr; }
    #info { height: auto; min-height: 1; padding: 0 1; background: $panel; }
    #filter { display: none; }
    #filter.shown { display: block; }
    #jobs { height: 14; display: none; border-top: solid $accent; }
    #jobs.shown { display: block; }
    #jobtabs { height: auto; max-height: 4; overflow-y: auto; padding: 0 1; color: $accent; }
    #joblog { height: 1fr; }
    #jobline { height: 1; padding: 0 1; color: $success; }
    """
    BINDINGS = [
        Binding('t', 'train', 'train'), Binding('p', 'play', 'play'), Binding('f', 'fetch', 'fetch'),
        Binding('s', 'stop', 'stop'), Binding('m', 'move', 'move'), Binding('d', 'delete', 'delete'),
        Binding('slash', 'filter', 'filter'), Binding('r', 'reload', 'refresh'),
        Binding('j', 'next_job', 'next job'), Binding('x', 'stop_job', 'stop job'),
        Binding('v', 'open_link', 'open viewer'), Binding('o', 'toggle_jobs', 'output'),
        Binding('q', 'quit', 'quit'), Binding('escape', 'close_filter', show=False),
    ]

    def __init__(self, hosts, project):
        super().__init__()
        self.hosts, self.project = hosts, project
        self.folder = project_folder(project)
        self.stats, self.stats_error, self.snapshots = {}, None, {}
        self.rows, self.filter_text, self.paused = [], '', False
        self.jobs, self.current, self.shown_lines = [], None, 0
        self.offered_machines = set()

    def main(self, selector, kind=None):
        """Widgets of the main screen (App.query_one would search an open dialog instead)."""
        return self.screen_stack[0].query_one(selector, kind) if kind else self.screen_stack[0].query_one(selector)

    def compose(self) -> ComposeResult:
        yield Static(id='title')
        with Vertical():
            yield Static('MACHINES', classes='label')
            yield DataTable(id='machines', cursor_type='row', zebra_stripes=True)
            yield Static('RUNS', classes='label')
            yield Input(placeholder='Filter runs (machine, run name or state); Esc closes', id='filter')
            yield DataTable(id='runs', cursor_type='row', zebra_stripes=True)
            yield Static(id='info')
        with Vertical(id='jobs'):
            yield Static(id='jobtabs')
            yield Log(id='joblog', highlight=False)
            yield Static(id='jobline')
        yield Footer()

    def on_mount(self):
        self.main('#machines', DataTable).add_columns(
            'Machine', 'State', 'CPU', 'RAM', 'GPU', 'VRAM', 'Users', 'Your job', 'Free disk')
        self.state_column = self.main('#runs', DataTable).add_columns(
            'Machine', 'Run', 'Checkpoint', 'Remote', 'Local', 'State')[-1]
        self.main('#runs', DataTable).focus()
        self.draw()
        self.load_stats()
        self.load_runs()
        self.set_interval(STATS_SECONDS, self.load_stats)
        self.set_interval(RUNS_SECONDS, self.load_runs)
        self.set_interval(1, self.draw_title)

    # -- data ---------------------------------------------------------------------------
    @work(thread=True, exclusive=True, group='stats')
    def load_stats(self):
        if self.paused:
            return
        try:
            stats, error = availability(), None
        except (OSError, ValueError) as exc:
            stats, error = self.stats, f'monitor unavailable: {exc}'
        self.call_from_thread(self.set_stats, stats, error)

    def set_stats(self, stats, error):
        self.stats, self.stats_error = stats, error
        self.draw()
        new = [host for host in new_machines(stats, self.hosts) if host not in self.offered_machines]
        if not error and new and len(self.screen_stack) == 1:
            self.offered_machines.update(new)
            self.push_screen(Ask('New workstations found',
                                 'Add these to machines.json using your configured SSH user?\n\n'
                                 + '\n'.join(new), ok='Add workstations'),
                             lambda answer: self.accept_machines(new) if answer else None)

    def accept_machines(self, hostnames):
        try:
            self.hosts = add_machines(hostnames)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.offered_machines.difference_update(hostnames)
            self.notify(f'Could not save workstations: {exc}', severity='error')
            return
        self.draw()
        self.load_runs()
        self.notify('Workstations saved to machines.json.')

    @work(thread=True, exclusive=True, group='runs')
    def load_runs(self):
        if self.paused:
            return
        names = list(self.hosts)
        with ThreadPoolExecutor(max_workers=10) as pool:
            snapshots = dict(zip(names, pool.map(lambda n: machine_snapshot(self.hosts[n], self.project), names)))
        rows = run_rows(self.hosts, snapshots, self.folder)
        self.call_from_thread(self.set_runs, snapshots, rows)

    def set_runs(self, snapshots, rows):
        self.snapshots, self.rows = snapshots, rows
        self.draw()

    # -- drawing ------------------------------------------------------------------------
    def draw_title(self):
        checked = list(self.snapshots.values())
        reachable = sum('error' not in s for s in checked)
        status = f'{reachable}/{len(self.hosts)} reachable' if checked else 'connecting…'
        running = sum(job.running for job in self.jobs)
        if running:
            status += f' · {running} job{"s" * (running > 1)} running'
        if self.stats_error:
            status += f' · {self.stats_error}'
        self.main('#title', Static).update(
            f'HHCM Training Lab · {self.project} · {time.strftime("%H:%M:%S")} · {status}')

    def draw(self):
        self.draw_title()
        machines = self.main('#machines', DataTable)
        selected = self.selected_key(machines)
        machines.clear()
        for name, host in self.hosts.items():
            state, cpu, ram, gpu, vram, users = self.stats.get(host.split('@')[-1], ('unknown',) + ('—',) * 5)
            snapshot = self.snapshots.get(name, {})
            if 'error' in snapshot:
                job, disk = Text('unreachable', 'yellow'), '—'
            elif snapshot:
                job = Text('▶ ' + ', '.join(snapshot['active']), 'bold green') if snapshot['active'] else '—'
                disk = size_text(snapshot['free'])
            else:
                job, disk = Text('…', 'dim'), '…'
            machines.add_row(name, Text(state, {'free': 'green', 'busy': 'red'}.get(state, 'dim')),
                             cpu, ram, gpu, vram, users, job, disk, key=name)
        self.restore(machines, selected)

        runs = self.main('#runs', DataTable)
        selected = self.selected_key(runs)
        runs.clear()
        needle = self.filter_text.lower()
        for row in self.rows:
            if needle and needle not in f"{row['machine']} {row['run']} {row['state']}".lower():
                continue
            key = f"{row['machine']}|{row['run']}"
            runs.add_row(row['machine'], row['run'], row['checkpoint'] or '—',
                         size_text(row['remote']), size_text(row['local']),
                         self.state_cell(row, key), key=key)
        if not self.rows:
            runs.add_row('', Text('loading runs…' if not self.snapshots else 'no runs found', 'dim'), '', '', '', '')
        self.restore(runs, selected)
        self.draw_info()

    def state_cell(self, row, key):
        job = next((job for job in reversed(self.jobs) if job.running and job.key == key), None)
        if job:
            percent = f' {job.progress}%' if job.progress is not None else '…'
            return Text(JOB_VERB.get(job.kind, job.kind) + percent, 'bold magenta')
        return Text(('▶ ' if row['state'] == 'training' else '') + row['state'], STATE_STYLE.get(row['state'], ''))

    @staticmethod
    def selected_key(table):
        if not table.row_count:
            return None
        try:
            return table.coordinate_to_cell_key((table.cursor_row, 0)).row_key.value
        except Exception:  # Cursor outside the table while it is empty.
            return None

    @staticmethod
    def restore(table, key):
        if key is None:
            return
        try:
            table.move_cursor(row=table.get_row_index(key), animate=False)
        except Exception:  # Row disappeared (deleted or filtered out).
            pass

    def draw_info(self):
        row = self.selected_run()
        info = self.main('#info', Static)
        if row is None:
            info.update('Select a run. Train uses the selected machine; other actions use the selected run.')
            return
        command = row['command'] or []
        overrides = [arg for arg in command if '=' in arg and not arg.startswith('-')]
        details = [f"{row['machine']} · {row['run']}", row['task'] or 'task unknown']
        details.append(' '.join(overrides) if overrides else ('no overrides' if command else 'arguments not saved'))
        info.update(Text(' · '.join(details)))

    def on_data_table_row_highlighted(self, event):
        if event.data_table.id == 'runs':
            self.draw_info()

    def on_input_changed(self, event):
        if event.input.id == 'filter':
            self.filter_text = event.value
            self.draw()

    def on_input_submitted(self, event):
        if event.input.id == 'filter':
            self.main('#runs', DataTable).focus()

    # -- selection ----------------------------------------------------------------------
    def selected_run(self):
        key = self.selected_key(self.main('#runs', DataTable))
        if not key or '|' not in key:
            return None
        machine, run = key.split('|', 1)
        return next((row for row in self.rows if row['machine'] == machine and row['run'] == run), None)

    def selected_machine(self):
        if self.focused is self.main('#machines', DataTable):
            return self.selected_key(self.main('#machines', DataTable))
        row = self.selected_run()
        return row['machine'] if row else None

    def need_run(self, remote=False):
        row = self.selected_run()
        if row is None:
            self.notify('Select a run first.', severity='warning')
        elif remote and row['state'] == 'local only':
            self.notify('This run exists only on this PC.', severity='warning')
            return None
        return row

    def machine_summary(self, machine):
        state, _, _, gpu, vram, users = self.stats.get(self.hosts[machine].split('@')[-1], ('unknown',) + ('—',) * 5)
        return f'{machine}: {state}, GPU {gpu}, VRAM {vram}, users {users}'

    # -- jobs ---------------------------------------------------------------------------
    def start_job(self, kind, title, machine, args, key=None, **answers):
        """Run `hhcm_isaac_workstations.py <args>` in the background; password-only machines get the full terminal."""
        command = [sys.executable, str(Path(__file__).resolve()), *args, '--project', self.project, '--workspace', str(ROOT)]
        if answers:
            command = ['env', 'HHCM_JOB_ANSWERS=' + json.dumps(answers), *command]
        if 'error' in self.snapshots.get(machine, {}):
            self.in_terminal(title, command)
            return None
        job = Job(len(self.jobs) + 1, kind, title, command, key=key,
                  notify=lambda job, committed: self.call_from_thread(self.job_changed, job, committed))
        self.jobs.append(job)
        job.start()
        self.show_job(job)
        if key:
            self.draw()
        return job

    def show_job(self, job):
        self.current = job
        self.main('#jobs').add_class('shown')
        log = self.main('#joblog', Log)
        log.clear()
        log.write_lines(job.lines)
        self.shown_lines = len(job.lines)
        self.draw_jobs()
        self.main('#jobtabs', Static).scroll_home(animate=False)

    def job_changed(self, job, committed):
        if job is self.current:
            if committed:
                new = job.lines[self.shown_lines:] if len(job.lines) >= self.shown_lines else job.lines
                self.main('#joblog', Log).write_lines(new)
                self.shown_lines = len(job.lines)
        self.draw_jobs()
        if job.key:
            try:  # Progress in the run's row without rebuilding the tables.
                row = next(r for r in self.rows if f"{r['machine']}|{r['run']}" == job.key)
                self.main('#runs', DataTable).update_cell(job.key, self.state_column, self.state_cell(row, job.key))
            except Exception:  # Row filtered out or gone.
                pass
        if not job.running:
            if job.link and job.kind == 'play':
                job.link = None
            self.notify(f"{job.title}: {'done' if job.code == 0 else f'failed (exit {job.code})'}",
                        severity='information' if job.code == 0 else 'error')
            self.load_runs()
        elif job.link and job.kind == 'play' and not getattr(job, 'link_shown', False):
            job.link_shown = True
            self.notify(f'Viewer ready at {job.link} — press v to open it.', timeout=10)

    def draw_jobs(self):
        tabs = Text()
        for job in sorted(self.jobs, key=lambda job: (job is not self.current, not job.running)):
            style = 'reverse' if job is self.current else ('bold' if job.running else 'dim')
            tabs.append(f' {job.label()} ', style)
            tabs.append('\n')
        self.main('#jobtabs', Static).update(tabs)
        job = self.current
        self.main('#jobline', Static).update(Text(job.live if job and job.running else ''))

    def in_terminal(self, title, command):
        """Fallback for machines that need a password: run the same command in the real terminal."""
        self.paused = True
        try:
            with self.suspend():
                print(color(f'\n── {title} ──', '1;36'), flush=True)
                try:
                    code = subprocess.run(command, cwd=ROOT).returncode
                    print(f'\nFinished with exit code {code}.' if code else '\nDone.')
                except KeyboardInterrupt:
                    print('\nInterrupted. A detached training job keeps running.')
                try:
                    input('\nPress Enter to return to the lab… ')
                except (KeyboardInterrupt, EOFError):
                    pass
        finally:
            self.paused = False
        self.load_runs()

    def running_job(self, kind):
        return next((job for job in self.jobs if job.running and job.kind == kind), None)

    def action_next_job(self):
        if not self.jobs:
            self.notify('No jobs yet.')
            return
        index = self.jobs.index(self.current) if self.current in self.jobs else -1
        self.show_job(self.jobs[(index + 1) % len(self.jobs)])

    def action_stop_job(self):
        job = self.current
        if not job or not job.running:
            job = self.running_job('play')
            if job:
                self.show_job(job)
        if not job or not job.running:
            self.notify('The shown job is not running.')
            return
        job.interrupt()
        self.notify(f'Sent {"Ctrl+C" if job.signals == 1 else "terminate"} to {job.title}. '
                    + ('Training already started on a workstation keeps running.' if job.kind == 'train' else ''))

    def action_toggle_jobs(self):
        self.main('#jobs').toggle_class('shown')

    def action_open_link(self):
        job = next((job for job in reversed(self.jobs) if job.running and job.link), None)
        if job is None:
            self.notify('No viewer link yet.')
        else:
            webbrowser.open(job.link)

    # -- actions (questions first, then a background job) ---------------------------------
    async def ask(self, *args, **kwargs):
        return await self.push_screen_wait(Ask(*args, **kwargs))

    @work(group='ask')
    async def action_train(self):
        machine = self.selected_machine()
        answer = await self.ask(f'Train on {machine or "?"}', (self.machine_summary(machine) if machine else '')
                                + '\nPaste train.py arguments or a whole "hhcm_isaac_workstations.py train <machine> ..." line; '
                                '--run_name and the machine are taken from it.', fields=[
            ('extra', 'Arguments or pasted command', ''), ('name', 'Run name (blank = --run_name from the paste)', '')])
        if not answer:
            return
        pasted, name, args = pasted_training(answer['extra'], self.hosts, self.project)
        machine = pasted or machine
        name = answer['name'].strip() or name
        if machine is None:
            self.notify('Select a machine, or paste a command that names one.', severity='warning')
            return
        if not name:
            self.notify('Give a run name (field or --run_name).', severity='warning')
            return
        args = ['--run_name', name, *args]
        state = self.stats.get(self.hosts[machine].split('@')[-1], ('?',))[0]
        if not await self.ask(
                f'Launch on {machine}?', f'train.sh {machine} {shlex.join(args)}\n\nMirrors source/assets, prepares the '
                'image if missing, then starts training. W&B/video settings come from train.sh.',
                warning='' if state == 'free' else f'{self.machine_summary(machine)} — contact the current user first.',
                ok='Launch'):
            return
        self.start_job('train', f'train {machine} {name}', machine, ['train', machine, *args])

    @work(group='ask')
    async def action_play(self):
        row = self.need_run(remote=True)
        if not row:
            return
        replay = self.running_job('play')
        if replay:
            self.show_job(replay)
            self.notify('A replay is already running; stop it with x first.', severity='warning')
            return
        answer = await self.ask(f"Play {row['run']} ({row['machine']})", 'Replays the latest checkpoint on this PC.',
                                fields=[('extra', 'Play arguments, e.g. --env 4 --iteration 1000 (blank = defaults)', '')])
        if answer is None:
            return
        self.start_job('play', f"play {row['machine']} {Path(row['run']).name}", row['machine'],
                       ['play', row['machine'], *shlex.split(answer['extra'])], run=row['run'],
                       key=f"{row['machine']}|{row['run']}")

    def action_fetch(self):
        row = self.need_run(remote=True)
        if row:
            self.start_job('fetch', f"fetch {row['machine']} {Path(row['run']).name}", row['machine'],
                           ['fetch', row['machine']], run=row['run'], key=f"{row['machine']}|{row['run']}")

    @work(group='ask')
    async def action_stop(self):
        machine = self.selected_machine()
        active = self.snapshots.get(machine or '', {}).get('active')
        if not active:
            self.notify(f'Nothing of yours is training on {machine}.' if machine else 'Select a machine first.',
                        severity='warning')
            return
        if await self.ask(f'Stop training on {machine}?', f"{', '.join(active)}\n\nSends Ctrl+C to the container. "
                          'Saved checkpoints are kept; the newest iterations since the last save may be lost.',
                          ok='Stop'):
            self.start_job('stop', f'stop {machine}', machine, ['stop', machine])

    @work(group='ask')
    async def action_move(self):
        row = self.need_run(remote=True)
        if not row:
            return
        choices = []
        for name, host in self.hosts.items():
            if name == row['machine']:
                continue
            state = self.stats.get(host.split('@')[-1], ('unknown',))[0]
            mine = self.snapshots.get(name, {}).get('active')
            unreachable = 'error' in self.snapshots.get(name, {})
            note = 'unreachable' if unreachable else ('your job running' if mine else state)
            choices.append((name, f'{name:<10} {note}', bool(mine or unreachable)))
        choices.sort(key=lambda choice: (choice[2], 'free' not in choice[1]))
        target = await self.ask(f"Move {row['run']} from {row['machine']} to…", choices=choices)
        if not target:
            return
        answers = {'run': row['run'], 'yes': True}
        if not row['command'] and row['state'] != 'training':
            answer = await self.ask('Training arguments', 'This run did not save its command line. Enter the full '
                                    'train.py arguments to resume with (overrides used originally are unknown).',
                                    fields=[('args', 'train.py arguments', f"--task {row['task']}" if row['task'] else '')])
            if not answer or not answer['args'].strip():
                return
            answers['train_args'] = answer['args'].strip()
        steps = [f"1. {'Stop the training on ' + row['machine'] if row['state'] == 'training' else 'Nothing to stop'}",
                 f"2. Copy {row['machine']} → this PC → {target}",
                 f'3. Resume on {target} from the latest checkpoint for the remaining iterations',
                 f"The original stays on {row['machine']}; delete it later with d."]
        answer = await self.ask(f'Move to {target}?', '\n'.join(steps), ok='Move', fields=[
            ('iterations', 'Extra iterations if the target was reached (blank = copy only)', '')])
        if answer:
            typed = answer['iterations'].strip()
            if typed and (not typed.isascii() or not typed.isdigit() or int(typed) < 1):
                self.notify('Enter a positive number of iterations.', severity='warning')
                return
            answers['extra_iterations'] = int(typed) if typed else None
            self.start_job('move', f"move {row['machine']}→{target} {Path(row['run']).name}", row['machine'],
                           ['move', row['machine'], target], key=f"{row['machine']}|{row['run']}", **answers)

    @work(group='ask')
    async def action_delete(self):
        row = self.need_run()
        if not row:
            return
        remote_ok = row['remote'] is not None and row['state'] not in ('training', 'remote unknown')
        local_ok = row['local'] is not None
        choices = [('l', f"Local copy on this PC ({size_text(row['local'])})", not local_ok),
                   ('r', f"Remote original on {row['machine']} ({size_text(row['remote'])})", not remote_ok),
                   ('b', 'Both', not (local_ok and remote_ok))]
        note = ' (still training: stop it first)' if row['state'] == 'training' else ''
        where = await self.ask(f"Delete {row['run']}{note}", f"State: {row['state']}", choices=choices)
        if not where:
            return
        # Conservative: the app's table state can only under-report 'fetched', never over-report it.
        last_copy = (where == 'b' or (where == 'r' and row['state'] != 'fetched')
                     or (where == 'l' and row['remote'] is None))
        label = {'l': 'the local copy', 'r': f"the original on {row['machine']}", 'b': 'both copies'}[where]
        if not await self.ask(f'Delete {label}?', row['run'], word='delete' if last_copy else None,
                              warning='No complete copy will remain: this cannot be undone.' if last_copy else '',
                              ok='Delete'):
            return
        self.start_job('delete', f"delete {row['machine']} {Path(row['run']).name}", row['machine'],
                       ['clean', row['machine']], key=f"{row['machine']}|{row['run']}",
                       run=row['run'], where=where, yes=True, permanent=last_copy)

    def action_reload(self):
        self.load_stats()
        self.load_runs()
        self.notify('Refreshing…', timeout=2)

    def action_filter(self):
        box = self.main('#filter', Input)
        box.add_class('shown')
        box.focus()

    def action_close_filter(self):
        box = self.main('#filter', Input)
        box.value, self.filter_text = '', ''
        box.remove_class('shown')
        self.main('#runs', DataTable).focus()
        self.draw()

    @work(group='ask')
    async def action_quit(self):
        running = [job for job in self.jobs if job.running]
        if running and not await self.ask(
                'Quit and stop running jobs?', '\n'.join(job.title for job in running)
                + '\n\nThey get Ctrl+C. Training already running on a workstation is not affected.', ok='Quit'):
            return
        for job in running:
            job.interrupt()
        self.exit()


def run_app(hosts, project):
    LabApp(hosts, project).run()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nExited. This does not stop a detached training job.')
        sys.exit(130)
    except (ValueError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
    except EOFError:
        print('\nError: this step needs an answer; run it in a terminal or use the app.',
              file=sys.stderr)
        sys.exit(1)
