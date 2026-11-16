import asyncio
import json
import os
import shutil
import signal
import sys
import time
import subprocess
import tempfile
from pathlib import Path
import unittest
import unittest.mock
from unittest.mock import patch

import hhcm_isaac_workstations as lab


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / 'kyon_training').mkdir()
        (root / 'kyon_training/train.sh').touch()
        root_patch = patch.object(lab, 'ROOT', root)
        root_patch.start()
        self.addCleanup(root_patch.stop)


class LabTests(WorkspaceTest):
    def test_workspace_selects_its_own_inventory_and_default_project(self):
        (lab.ROOT / 'machines.json').write_text('{"user":"user","hosts":{"gpu":"gpu.example.org"}}')
        with patch.object(sys, 'argv', ['app', '--workspace', str(lab.ROOT)]), patch.object(lab, 'run_app') as run:
            lab.main()
        run.assert_called_once_with({'gpu': 'user@gpu.example.org'}, 'kyon_training')

    def test_monitor_is_optional(self):
        with patch.object(lab, 'STATUS_URL', ''), patch.object(lab, 'urlopen') as fetch:
            with self.assertRaisesRegex(ValueError, 'monitor_url'):
                lab.availability()
        fetch.assert_not_called()

    def test_workspace_monitor_configuration_and_environment_override(self):
        (lab.ROOT / 'machines.json').write_text(json.dumps({
            'user': 'user', 'hosts': {'gpu': 'gpu'}, 'monitor_url': 'http://workspace/global_stats'}))
        with patch.object(lab, 'STATUS_URL', ''), patch.dict(os.environ):
            os.environ.pop('HHCM_MONITOR_URL', None)
            lab.inventory()
            self.assertEqual(lab.STATUS_URL, 'http://workspace/global_stats')
            os.environ['HHCM_MONITOR_URL'] = 'http://override/global_stats'
            lab.inventory()
            self.assertEqual(lab.STATUS_URL, 'http://override/global_stats')

    @patch('hhcm_isaac_workstations.inventory', return_value={'ted': 'user@ted'})
    @patch('hhcm_isaac_workstations.project_folder', return_value=Path('kyon_training'))
    @patch('hhcm_isaac_workstations.run_app', return_value=0)
    def test_no_command_opens_app(self, run_app, project, inventory):
        for command in (['app.py'], ['app.py', 'app']):
            with patch('hhcm_isaac_workstations.sys.argv', command):
                self.assertEqual(lab.main(), 0)
        self.assertEqual(run_app.call_count, 2)

    @patch('hhcm_isaac_workstations.saved_checkpoints', return_value=['exp/b/model_9.pt', 'exp/a/model_5.pt'])
    @patch('builtins.input')
    def test_selected_replay_uses_preselected_run_without_prompt(self, ask, _):
        self.assertEqual(lab.selected_replay('user@ted', 'kyon_training', run='exp/a'), 'exp/a/model_5.pt')
        ask.assert_not_called()
        with self.assertRaises(ValueError):
            lab.selected_replay('user@ted', 'kyon_training', run='exp/missing')

    @patch('hhcm_isaac_workstations.sys.stdout.isatty', return_value=True)
    @patch.dict('hhcm_isaac_workstations.os.environ', {'TERM': 'xterm'}, clear=True)
    def test_color_respects_terminal_and_no_color(self, terminal):
        self.assertEqual(lab.color('Connected', '32'), '\033[32mConnected\033[0m')
        with patch.dict('hhcm_isaac_workstations.os.environ', {'NO_COLOR': ''}):
            self.assertEqual(lab.color('Connected', '32'), 'Connected')
        terminal.return_value = False
        self.assertEqual(lab.color('Connected', '32'), 'Connected')


    def test_projects_are_discovered_from_folders_with_train_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'container').mkdir()
            (root / 'kyon_jump').mkdir()
            (root / 'kyon_jump' / 'train.sh').touch()
            with patch('hhcm_isaac_workstations.ROOT', root), patch('builtins.print') as output, \
                    patch('builtins.input', side_effect=['', '']):
                lab.selected_training('user@ted')
            output.assert_any_call('Available projects: kyon_jump')

    def test_compact_stats_keeps_usage_and_users(self):
        text = ('workstation [0.7s] 🟥 IP:192.0.2.1 CPU:5% RAM:20.64% '
                'GPU:28% VRAM:52.79% disk:[91%] top_mem_user:user active_users:[\'user[0:18:17]\']')
        compact = lab.compact_stats(text)
        for value in ('workstation', '🟥', '5%', '20.64%', '28%', '52.79%', 'user[0:18:17]'):
            self.assertIn(value, compact)
        self.assertNotIn('IP:', compact)
        self.assertNotIn('disk:', compact)
        self.assertIn('unexpected output', lab.compact_stats('unexpected output'))


    @patch('hhcm_isaac_workstations.selected_replay', return_value='exp/trial/model_29999.pt')
    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_play_selects_requested_iteration(self, run, selected):
        run.return_value.returncode = 0
        self.assertEqual(lab.action('user@frank', 'play', 'kyon_training', [
            '--iteration', '10000', '--num_envs', '8', '--task', 'task-id']), 0)
        selected.assert_called_once_with('user@frank', 'kyon_training', task='task-id', run=None)
        self.assertEqual(run.call_args.kwargs['env']['TRAINING_REPLAY_CHECKPOINT'],
                         'exp/trial/model_10000.pt')
        self.assertEqual(run.call_args.args[0][-4:], ['--num_envs', '8', '--task', 'task-id'])
        self.assertNotIn('--iteration', run.call_args.args[0])
        run.reset_mock()
        with self.assertRaisesRegex(ValueError, 'Iteration must be nonnegative'):
            lab.action('user@frank', 'play', 'kyon_training', ['--iteration', '-1'])
        run.assert_not_called()

    @patch('hhcm_isaac_workstations.action', return_value=0)
    @patch('builtins.input', side_effect=['', 'airtime_fresh', '--max_iterations 10000', 'y'])
    def test_selected_training_launches_directly(self, _, action):
        lab.selected_training('user@ted')
        self.assertEqual(action.call_count, 1)
        self.assertEqual(action.call_args.args, (
            'user@ted', 'train', 'kyon_training',
            ['--run_name', 'airtime_fresh', 'agent.resume=false', '--max_iterations', '10000']))

    @patch('hhcm_isaac_workstations.action')
    @patch('builtins.input', side_effect=['', 'trial', '', 'n'])
    def test_cancel_does_not_connect_or_launch(self, _, action):
        lab.selected_training('user@ted')
        action.assert_not_called()


    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_training_status_checks_python_training_not_other_python(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = 'python python -u scripts/rsl_rl/train.py --run_name trial\n'
        self.assertEqual(lab.training_status('user@ted'), 'TRAINING: trial')
        args = run.call_args.args[0]
        self.assertIn('BatchMode=yes', args)
        self.assertIn('StrictHostKeyChecking=yes', args)
        self.assertIn('ps -u "$(id -u)"', args[-1])
        run.return_value.stdout = 'python python scripts/rsl_rl/play.py\nbash bash -c train.py\n'
        self.assertEqual(lab.training_status('user@ted'), 'idle')
        run.return_value.returncode = 255
        self.assertEqual(lab.training_status('user@ted'), 'unreachable')
        run.side_effect = subprocess.TimeoutExpired('ssh', 5)
        self.assertEqual(lab.training_status('user@ted'), 'unreachable')

    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_training_status_lists_runs_and_uses_last_run_name(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = (
            'python python scripts/rsl_rl/train.py --run_name default --run_name=limits_F_normalized\n'
            'python python scripts/rsl_rl/train.py --run_name limits_C_physical_slip05\n'
            'python python scripts/rsl_rl/train.py\n')
        self.assertEqual(lab.training_status('user@ted'),
                         'TRAINING: limits_F_normalized, limits_C_physical_slip05, (run name unavailable)')

    def test_monitor_keeps_status_text(self):
        self.assertEqual(lab.monitor_text('<pre>ted: <span>BUSY</span>\nfred: FREE &amp; ready<br>GPU 0</pre>'),
                         'ted: BUSY\nfred: FREE & ready\nGPU 0')

    def test_monitor_ignores_scripts_and_controls(self):
        self.assertEqual(lab.monitor_text('<style>hidden</style><script>hidden</script><pre>fred\x1b: FREE</pre>'),
                         'fred: FREE')

    def test_empty_status_is_not_available(self):
        with self.assertRaises(ValueError):
            lab.monitor_text('<pre></pre>')

    @patch('hhcm_isaac_workstations.subprocess.run')
    @patch('hhcm_isaac_workstations.urlopen')
    @patch('hhcm_isaac_workstations.STATUS_URL', 'http://monitor.example/global_stats')
    def test_status_uses_website_without_ssh(self, fetch, run):
        fetch.return_value.__enter__.return_value.read.return_value = b'<pre>ted: BUSY</pre>'
        with patch('builtins.print') as output:
            self.assertEqual(lab.status({'ted': 'user@ted'}), 0)
        fetch.assert_called_once_with(lab.STATUS_URL, timeout=10)
        output.assert_any_call(lab.compact_stats('ted: BUSY'))
        run.assert_not_called()

    @patch('hhcm_isaac_workstations.urlopen', side_effect=OSError('DNS unavailable'))
    def test_unreachable_monitor_is_not_free(self, _):
        with patch('builtins.print') as output:
            self.assertEqual(lab.status({}), 1)
        self.assertIn('Monitor unavailable', output.call_args_list[1].args[0])

    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_train_preserves_arguments(self, run):
        run.return_value.returncode = 0
        extra = ['--run_name', 'air time', 'agent.resume=false']
        lab.action('user@ted', 'train', 'kyon_training', extra)
        self.assertEqual(run.call_args.args[0][-3:], extra)
        self.assertEqual(run.call_args.args[0][2], 'user@ted')
        self.assertNotIn('TRAINING_SSH_CONTROL_PATH', run.call_args.kwargs['env'])
        run.assert_called_once()

    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_launch_without_shared_connection(self, run):
        run.return_value.returncode = 0
        for command in ('train', 'play'):
            with self.subTest(command=command), patch('hhcm_isaac_workstations.selected_replay', return_value='exp/run/model_10.pt'), patch.dict(
                    lab.os.environ, {'TRAINING_SSH_CONTROL_PATH': '/stale/socket'}):
                self.assertEqual(lab.action('user@ted', command, 'kyon_training', []), 0)
                self.assertNotIn('TRAINING_SSH_CONTROL_PATH', run.call_args.kwargs['env'])
        self.assertEqual(run.call_count, 2)

    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_invalid_project_cannot_run(self, run):
        with self.assertRaises(ValueError):
            lab.action('user@ted', 'train', '../other', [])
        run.assert_not_called()

    @patch('hhcm_isaac_workstations.saved_runs', return_value=0)
    @patch('hhcm_isaac_workstations.inventory', return_value={'ted': 'user@ted'})
    def test_direct_runs_all_or_one_machine(self, inventory, saved):
        for suffix in ([], ['ted']):
            with patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'runs', *suffix]):
                self.assertEqual(lab.main(), 0)
            saved.assert_called_with({'ted': 'user@ted'}, 'kyon_training')
        with patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'runs', 'missing']), \
                patch('sys.stderr'), self.assertRaises(SystemExit):
            lab.main()

    @patch('hhcm_isaac_workstations.saved_checkpoints')
    def test_saved_runs_combines_remote_local_and_offline_copies(self, listing):
        # Machines are listed in parallel threads, so answer by host rather than by call order.
        answers = {'user@ted': ['exp/new/model_100.pt', 'exp/remote/model_50.pt'],
                   'user@offline': subprocess.TimeoutExpired('ssh', 15), 'user@empty': []}

        def answer(host, *args, **kwargs):
            if isinstance(answers[host], Exception):
                raise answers[host]
            return answers[host]
        listing.side_effect = answer
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cache = folder / 'data/replays'
            for path in ('user@ted/logs/rsl_rl/exp/new/model_9.pt',
                         'user@ted/logs/rsl_rl/exp/new/model_10.pt',
                         'user@ted/logs/rsl_rl/exp/new/model_invalid.pt',
                         'user@offline/logs/rsl_rl/exp/copied/model_20.pt'):
                checkpoint = cache / path
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
            with patch('hhcm_isaac_workstations.project_folder', return_value=folder), patch('builtins.print') as output:
                self.assertEqual(lab.saved_runs(
                    {'ted': 'user@ted', 'offline': 'user@offline', 'empty': 'user@empty'}, 'project'), 1)
            lines = [' '.join(str(call.args[0]).split()) for call in output.call_args_list]
            self.assertIn('ted exp/new model_100.pt model_10.pt', lines)
            self.assertIn('ted exp/remote model_50.pt —', lines)
            self.assertIn('offline exp/copied unknown model_20.pt', lines)
            self.assertIn('empty: no saved runs', lines)
            self.assertTrue(any('remote unavailable' in line for line in lines))
            for call in listing.call_args_list:
                self.assertEqual(call.kwargs, {'interactive': False})

    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_saved_checkpoints_overview_uses_bounded_noninteractive_ssh(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = '[]'
        self.assertEqual(lab.saved_checkpoints('user@ted', 'kyon_training', interactive=False), [])
        self.assertIn('BatchMode=yes', run.call_args.args[0])
        self.assertIn('StrictHostKeyChecking=yes', run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs['timeout'], 15)
        run.side_effect = subprocess.TimeoutExpired('ssh', 15)
        with self.assertRaisesRegex(ValueError, 'connection timed out'):
            lab.saved_checkpoints('user@ted', 'kyon_training')

    @patch('hhcm_isaac_workstations.subprocess.run')
    @patch('builtins.input', return_value='2')
    def test_replay_selects_saved_run(self, _, run):
        run.return_value.returncode = 0
        run.return_value.stdout = '["exp/new/model_99.pt", "exp/old/model_50.pt"]'
        self.assertEqual(lab.selected_replay('user@ted', 'kyon_training'), 'exp/old/model_50.pt')
        args = run.call_args.args[0]
        self.assertEqual(args[-2], 'user@ted')
        self.assertIn('data/logs/rsl_rl', args[-1])

    def test_replay_filters_recorded_and_legacy_tasks(self):
        import json
        import shlex
        (lab.ROOT / 'kyon_training/run_tasks.json').write_text(json.dumps({
            'exp/legacy_a': 'Task-A', 'exp/legacy_b': 'Task-B', 'exp/legacy_c': 'Task-C'}))
        with patch('hhcm_isaac_workstations.subprocess.run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = '[]'
            lab.selected_replay('user@ted', 'kyon_training', task='Task-B')
            command = shlex.split(run.call_args.args[0][-1])
        mapping = json.loads(command[5])
        self.assertEqual(len(mapping), 3)
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            root = home / 'training/kyon_training/data/logs/rsl_rl'
            for name in [*mapping, 'exp/unknown', 'exp/new']:
                folder = root / name
                folder.mkdir(parents=True)
                (folder / 'model_1499.pt').touch()
            params = root / 'exp/new/params'
            params.mkdir()
            (params / 'task.txt').write_text('Task-New\n')
            code = command[2].replace('Path.home()', f'Path({str(home)!r})')

            def listing(task):
                result = subprocess.run(
                    [lab.sys.executable, '-c', code, 'kyon_training', task, command[5]],
                    capture_output=True, text=True, check=True)
                return set(json.loads(result.stdout))

            for name, task in mapping.items():
                with self.subTest(task=task):
                    self.assertEqual(listing(task), {name + '/model_1499.pt'})
            self.assertEqual(listing('Task-New'), {'exp/new/model_1499.pt'})
            self.assertEqual(listing('Missing'), set())
            self.assertEqual(len(listing('')), 5)
            legacy_name, legacy_task = next(iter(mapping.items()))
            params = root / legacy_name / 'params'
            params.mkdir()
            (params / 'task.txt').write_text('Task-Corrected\n')
            self.assertEqual(listing(legacy_task), set())
            self.assertEqual(listing('Task-Corrected'), {legacy_name + '/model_1499.pt'})

    @patch('hhcm_isaac_workstations.selected_replay', return_value='exp/old/model_50.pt')
    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_play_robot_count(self, run, _):
        run.return_value.returncode = 0
        script = str(lab.ROOT / 'kyon_training/play.sh')
        for extra, expected in (
                ([], ['--num_envs', '10']),
                (['--env', '3'], ['--num_envs', '3']),
                (['--envs=5', '--task', 'T'], ['--num_envs', '5', '--task', 'T']),
                (['--num_envs', '4'], ['--num_envs', '4'])):
            with self.subTest(extra=extra):
                lab.action('user@ted', 'play', 'kyon_training', extra)
                self.assertEqual(run.call_args.args[0], ['bash', script, 'user@ted', *expected])
        run.reset_mock()
        for extra in (['--env', '0'], ['--env', '2', '--num_envs', '4']):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                lab.action('user@ted', 'play', 'kyon_training', extra)
        run.assert_not_called()

    @patch('hhcm_isaac_workstations.selected_replay', return_value=None)
    def test_play_passes_task_to_selection(self, selected):
        lab.action('user@ted', 'play', 'kyon_training',
                   ['--num_envs', '8', '--task', 'Task-B'])
        selected.assert_called_once_with('user@ted', 'kyon_training', task='Task-B', run=None)

    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_replay_handles_empty_failed_and_invalid_listing(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = '[]'
        with patch('builtins.input') as ask:
            self.assertIsNone(lab.selected_replay('user@ted', 'kyon_training'))
        ask.assert_not_called()
        run.return_value.stdout = '["../bad/model_0.pt"]'
        with self.assertRaisesRegex(ValueError, 'Unexpected checkpoint'):
            lab.selected_replay('user@ted', 'kyon_training')
        run.return_value.returncode = 255
        run.return_value.stderr = 'host unreachable'
        with self.assertRaisesRegex(ValueError, 'Cannot list training'):
            lab.selected_replay('user@ted', 'kyon_training')

    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_replay_cancel_and_invalid_choice(self, run):
        run.return_value.returncode = 0
        run.return_value.stdout = '["exp/run/model_99.pt"]'
        for choice in ('0', '2', '-1', 'abc', '²'):
            with self.subTest(choice=choice), patch('builtins.input', return_value=choice):
                with self.assertRaisesRegex(ValueError, 'Choose a run number'):
                    lab.selected_replay('user@ted', 'kyon_training')
        with patch('builtins.input', return_value=''):
            self.assertIsNone(lab.selected_replay('user@ted', 'kyon_training'))

    @patch('hhcm_isaac_workstations.selected_replay', return_value='exp/old/model_50.pt')
    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_play_downloads_selection_and_runs_locally(self, run, selected):
        run.return_value.returncode = 0
        self.assertEqual(lab.action('user@ted', 'play', 'kyon_training', ['--num_envs', '4']), 0)
        self.assertEqual(run.call_args.args[0], [
            'bash', str(lab.ROOT / 'kyon_training/play.sh'), 'user@ted', '--num_envs', '4'])
        self.assertEqual(run.call_args.kwargs['env']['TRAINING_REPLAY_CHECKPOINT'], 'exp/old/model_50.pt')
        selected.return_value = None
        run.reset_mock()
        self.assertEqual(lab.action('user@ted', 'play', 'kyon_training', []), 0)
        run.assert_not_called()

    @patch('hhcm_isaac_workstations.selected_replay', return_value='exp/trial/model_50.pt')
    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_fetch_copies_complete_run_and_preserves_remote(self, run, selected):
        with tempfile.TemporaryDirectory() as temporary, patch('hhcm_isaac_workstations.project_folder', return_value=Path(temporary)):
            run.return_value.returncode = 0
            self.assertEqual(lab.action('user@ted', 'fetch', 'kyon_training', []), 0)
            destination = Path(temporary) / 'data/replays/user@ted/logs/rsl_rl/exp/trial'
            self.assertTrue(destination.is_dir())
            args = run.call_args.args[0]
            self.assertEqual(args[-2:], [
                'user@ted:training/kyon_training/data/logs/rsl_rl/exp/trial/', str(destination) + '/'])
            self.assertEqual(args[0], 'rsync')
            self.assertNotIn('--remove-source-files', args)
            run.assert_called_once()
            run.return_value.returncode = 23
            self.assertEqual(lab.action('user@ted', 'fetch', 'kyon_training', []), 23)
            selected.return_value = None
            run.reset_mock()
            self.assertEqual(lab.action('user@ted', 'fetch', 'kyon_training', []), 0)
            run.assert_not_called()

    @patch('hhcm_isaac_workstations.selected_replay')
    @patch('hhcm_isaac_workstations.subprocess.run')
    def test_fetch_rejects_extra_arguments_and_invalid_host(self, run, selected):
        for host, extra in [('user@ted', ['--delete']), ('-bad', [])]:
            with self.subTest(host=host), self.assertRaises(ValueError):
                lab.action(host, 'fetch', 'kyon_training', extra)
        selected.assert_not_called()
        run.assert_not_called()

    @patch('hhcm_isaac_workstations.action', return_value=0)
    @patch('hhcm_isaac_workstations.inventory', return_value={'ted': 'user@ted'})
    def test_direct_stop_uses_selected_project(self, inventory, action):
        with patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'stop', 'ted', '--project', 'kyon_training']):
            self.assertEqual(lab.main(), 0)
        action.assert_called_once_with('user@ted', 'stop', 'kyon_training', [], run=None)

    def test_stop_targets_only_project_and_requires_rootless_docker(self):
        with patch('hhcm_isaac_workstations.subprocess.run') as run:
            run.return_value.returncode = 7
            self.assertEqual(lab.action('user@ted', 'stop', 'kyon_training', []), 7)
            args = run.call_args.args[0]
            self.assertEqual(args[-2], 'user@ted')
            script = args[-1]
            run.reset_mock()
            with self.assertRaises(ValueError):
                lab.action('user@ted', 'stop', 'kyon_training', ['--all'])
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            docker = folder / 'docker'
            docker.write_text('''#!/bin/bash
if [[ $1 == info ]]; then
    echo "$TEST_SECURITY_OPTIONS"
else
    printf '%s\\n' "$@" > "$TEST_STOP_ARGS"
    exit "${TEST_STOP_EXIT:-0}"
fi
''')
            docker.chmod(0o755)
            target = folder / 'stop-args'
            env = dict(lab.os.environ, PATH=f'{folder}:' + lab.os.environ['PATH'],
                       TEST_STOP_ARGS=str(target), TEST_SECURITY_OPTIONS='["name=rootless"]')
            result = subprocess.run(['bash', '-c', script], env=env, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_text().splitlines(),
                             ['stop', '--signal', 'SIGINT', '--timeout', '30', 'kyon_training'])
            target.unlink()
            env['TEST_SECURITY_OPTIONS'] = '[]'
            result = subprocess.run(['bash', '-c', script], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())
            env.update(TEST_SECURITY_OPTIONS='["name=rootless"]', TEST_STOP_EXIT='1')
            result = subprocess.run(['bash', '-c', script], env=env, capture_output=True)
            self.assertEqual(result.returncode, 1)

    def test_old_commands_are_removed(self):
        self.assertEqual(lab.ACTIONS, ('train', 'play', 'fetch', 'stop', 'clean', 'move'))
        for command in ('connect', 'disconnect', 'login', 'logs', 'attach', 'refresh', 'monitor', 'menu'):
            with self.subTest(command=command), patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', command]), \
                    patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                lab.main()
            self.assertEqual(error.exception.code, 2)

    def test_fetch_state_compares_every_remote_file(self):
        remote = {'model_10.pt': 5, 'events.out.tfevents.1': 3}
        self.assertEqual(lab.fetch_state(remote, None), 'not fetched')
        self.assertEqual(lab.fetch_state(remote, {'model_10.pt': 5}), 'partial')
        self.assertEqual(lab.fetch_state(remote, {'model_10.pt': 4, 'events.out.tfevents.1': 3}), 'partial')
        self.assertEqual(lab.fetch_state(remote, {**remote, 'extra': 1}), 'fetched')
        self.assertEqual(lab.fetch_state(None, {'model_10.pt': 5}), 'local only')

    def test_chosen_numbers(self):
        self.assertEqual(lab.chosen_numbers('3 1,5-6', 6), [1, 3, 5, 6])
        self.assertEqual(lab.chosen_numbers('all', 3), [1, 2, 3])
        for bad in ('0', '7', 'x', '2-', '-1'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                lab.chosen_numbers(bad, 6)

    def clean_project(self, root):
        (root / 'kyon_training').mkdir()
        (root / 'kyon_training/train.sh').touch()
        run = root / 'kyon_training/data/replays/user@ted/logs/rsl_rl/exp/2026-01-01_00-00-00_a'
        run.mkdir(parents=True)
        (run / 'model_10.pt').write_bytes(b'12345')
        return run

    @patch('hhcm_isaac_workstations.training_status', return_value='idle')
    def test_clean_deletes_fetched_local_copy_after_confirmation(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.clean_project(root)
            remote = {'exp/2026-01-01_00-00-00_a': {'model_10.pt': 5}}
            with patch('hhcm_isaac_workstations.ROOT', root), patch('hhcm_isaac_workstations.remote_run_files', return_value=(remote, 10, 20)), \
                    patch('hhcm_isaac_workstations.subprocess.run') as ssh_run, patch('builtins.print'), \
                    patch('builtins.input', side_effect=['1', 'l', 'y']):
                self.assertEqual(lab.clean('ted', 'user@ted', 'kyon_training'), 0)
            self.assertFalse(run.exists())
            self.assertFalse(run.parent.exists())
            ssh_run.assert_not_called()

    @patch('hhcm_isaac_workstations.training_status', return_value='idle')
    def test_clean_requires_typed_confirmation_for_unfetched_remote(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.clean_project(root)
            remote = {'exp/2026-02-01_00-00-00_b': {'model_10.pt': 5}}
            with patch('hhcm_isaac_workstations.ROOT', root), patch('hhcm_isaac_workstations.remote_run_files', return_value=(remote, 10, 20)), \
                    patch('hhcm_isaac_workstations.subprocess.run') as ssh_run, patch('builtins.print'), \
                    patch('builtins.input', side_effect=['1', 'r', 'y']):
                lab.clean('ted', 'user@ted', 'kyon_training')
            ssh_run.assert_not_called()
            ssh_run.return_value.returncode = 0
            with patch('hhcm_isaac_workstations.ROOT', root), patch('hhcm_isaac_workstations.remote_run_files', return_value=(remote, 10, 20)), \
                    patch('hhcm_isaac_workstations.subprocess.run', return_value=ssh_run.return_value) as ssh_run, \
                    patch('builtins.print'), patch('builtins.input', side_effect=['1', 'r', 'delete']):
                self.assertEqual(lab.clean('ted', 'user@ted', 'kyon_training'), 0)
            command = ssh_run.call_args.args[0][-1]
            self.assertIn(lab.CLEAN_DELETE_SCRIPT.strip().splitlines()[0], command)
            self.assertIn('exp/2026-02-01_00-00-00_b', command)

    @patch('hhcm_isaac_workstations.training_status', return_value='TRAINING: b')
    def test_clean_refuses_remote_run_still_training(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.clean_project(root)
            remote = {'exp/2026-02-01_00-00-00_b': {'model_10.pt': 5}}
            with patch('hhcm_isaac_workstations.ROOT', root), patch('hhcm_isaac_workstations.remote_run_files', return_value=(remote, 10, 20)), \
                    patch('hhcm_isaac_workstations.subprocess.run') as ssh_run, patch('builtins.print'), \
                    patch('builtins.input', side_effect=['1', 'r']), self.assertRaises(ValueError):
                lab.clean('ted', 'user@ted', 'kyon_training')
            ssh_run.assert_not_called()

    def test_remote_delete_script_refuses_paths_outside_logs(self):
        with tempfile.TemporaryDirectory() as home:
            logs = Path(home) / 'training/kyon_training/data/logs/rsl_rl/exp/run_a'
            logs.mkdir(parents=True)
            (logs / 'model_1.pt').touch()
            env = {'HOME': home, 'PATH': '/usr/bin:/bin'}
            for bad in ('../../x', 'exp', 'exp/missing'):
                result = subprocess.run(['python3', '-c', lab.CLEAN_DELETE_SCRIPT, 'kyon_training', json.dumps([bad])],
                                        env=env, capture_output=True)
                self.assertNotEqual(result.returncode, 0, bad)
            self.assertTrue(logs.exists())
            result = subprocess.run(['python3', '-c', lab.CLEAN_DELETE_SCRIPT, 'kyon_training', json.dumps(['exp/run_a'])],
                                    env=env, capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertFalse(logs.parent.exists())
            self.assertTrue((Path(home) / 'training/kyon_training/data/logs/rsl_rl').exists())

    def test_remote_list_script_reports_files_and_disk(self):
        with tempfile.TemporaryDirectory() as home:
            logs = Path(home) / 'training/kyon_training/data/logs/rsl_rl/exp/run_a/params'
            logs.mkdir(parents=True)
            (logs / 'task.txt').write_text('abc')
            result = subprocess.run(['python3', '-c', lab.CLEAN_LIST_SCRIPT, 'kyon_training'],
                                    env={'HOME': home, 'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
            data = json.loads(result.stdout)
            self.assertEqual(data['runs'], {'exp/run_a': {'params/task.txt': 3}})
            self.assertGreater(data['total'], 0)

    def test_resume_arguments_replace_previous_resume_and_iterations(self):
        args = ['--task', 'T', '--max_iterations', '30000', '--resume', '--load_run=old', '--checkpoint', 'model_1.pt',
                'agent.resume=false', 'agent.load_run=x', 'env.rewards.a.weight=-1', '--run_name', 'r']
        self.assertEqual(lab.resume_arguments(args, '2026-01-01_00-00-00_r', 'model_500.pt', 29500),
                         ['--task', 'T', 'env.rewards.a.weight=-1', '--run_name', 'r', '--resume',
                          '--load_run', '2026-01-01_00-00-00_r', '--checkpoint', 'model_500.pt', '--max_iterations', '29500'])

    def test_remaining_iterations_counts_from_first_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'params').mkdir()
            (run / 'params/agent.yaml').write_text('seed: 1\nmax_iterations: 1000\n')
            for number in (0, 50, 400):
                (run / f'model_{number}.pt').touch()
            self.assertEqual(lab.remaining_iterations(run, 'model_400.pt'), 600)
            (run / 'model_0.pt').unlink()  # A resumed run starts at its loaded iteration.
            self.assertEqual(lab.remaining_iterations(run, 'model_400.pt'), 650)

    def test_active_runs_script_filters_processes_and_keeps_arguments(self):
        commands = [b'python3\0/work/train.py\0--run_name\0old\0--run_name=new\0x=a b\0',
                    b'bash\0/work/train.py\0--run_name=other\0',
                    b'python3\0/work/play.py\0--run_name=other\0',
                    b'python3\0/work/train.py\0', b'\xff\0', PermissionError()]
        state = {}
        with patch('os.listdir', return_value=['self', *map(str, range(len(commands)))]), \
                patch('os.stat', return_value=unittest.mock.Mock(st_uid=lab.os.getuid())), \
                patch.object(Path, 'read_bytes', side_effect=commands):
            exec(lab.ACTIVE_RUNS_SCRIPT, state)
        self.assertEqual(state['active'], {'new': ['--run_name', 'old', '--run_name=new', 'x=a b']})

    def test_args_script_reads_saved_command(self):
        with tempfile.TemporaryDirectory() as home:
            params = Path(home) / 'training/kyon_training/data/logs/rsl_rl/exp/2026-01-01_00-00-00_r/params'
            params.mkdir(parents=True)
            (params / 'command.json').write_text(json.dumps(['--task', 'T', 'a=b c']))
            (params / 'task.txt').write_text('T\n')
            result = subprocess.run(['python3', '-c', lab.ARGS_SCRIPT, 'kyon_training', 'exp/2026-01-01_00-00-00_r'],
                                    env={'HOME': home, 'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), {'saved': ['--task', 'T', 'a=b c'], 'active': None, 'task': 'T'})

    def move_run(self, root, info, inputs, target_state='idle'):
        (root / 'kyon_training').mkdir()
        (root / 'kyon_training/train.sh').touch()
        run = 'exp/2026-01-01_00-00-00_r'
        local = root / 'kyon_training/data/replays/user@frank/logs/rsl_rl' / run

        def copy(host, project, name, upload=False, source=None):
            if not upload:
                (local / 'params').mkdir(parents=True, exist_ok=True)
                (local / 'params/agent.yaml').write_text('max_iterations: 1000\n')
                for number in (0, 300):
                    (local / f'model_{number}.pt').touch()
            else:
                self.assertEqual(source, local)
            return 0

        calls = []

        def run_command(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(info), stderr='')

        with patch('hhcm_isaac_workstations.ROOT', root), patch('hhcm_isaac_workstations.training_status', return_value=target_state), \
                patch('hhcm_isaac_workstations.selected_replay', return_value=run + '/model_300.pt'), \
                patch('hhcm_isaac_workstations.copy_run', side_effect=copy) as copied, patch('hhcm_isaac_workstations.action', return_value=0) as action, \
                patch('hhcm_isaac_workstations.subprocess.run', side_effect=run_command), patch('builtins.print'), \
                patch('builtins.input', side_effect=inputs):
            result = lab.move('frank', 'user@frank', 'ted', 'user@ted', 'kyon_training')
        return result, calls, copied, action

    def test_move_stops_active_run_copies_and_resumes_on_target(self):
        with tempfile.TemporaryDirectory() as directory:
            info = {'saved': None, 'active': ['--task', 'T', '--run_name', 'r', '--max_iterations', '1000'], 'task': 'T'}
            result, calls, copied, action = self.move_run(Path(directory), info, ['y'])
        self.assertEqual(result, 0)
        action.assert_called_once_with('user@frank', 'stop', 'kyon_training', [])
        self.assertEqual([call.kwargs.get('upload', False) for call in copied.call_args_list], [False, True])
        launch = calls[-1]
        self.assertEqual(Path(launch[1]).name, 'pipeline.sh')
        self.assertEqual(launch[4:], ['user@ted', '--task', 'T', '--run_name', 'r', '--resume', '--load_run',
                                      '2026-01-01_00-00-00_r', '--checkpoint', 'model_300.pt', '--max_iterations', '700'])

    def test_move_asks_for_arguments_of_old_runs_and_can_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            info = {'saved': None, 'active': None, 'task': 'T'}
            result, calls, copied, action = self.move_run(Path(directory), info, ['--task T x=1', 'n'])
        self.assertEqual(result, 0)
        action.assert_not_called()
        copied.assert_not_called()

    def test_move_refuses_busy_target(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            self.move_run(Path(directory), {}, [], target_state='TRAINING: other')

    @patch('hhcm_isaac_workstations.move', return_value=0)
    @patch('hhcm_isaac_workstations.inventory', return_value={'frank': 'user@frank', 'ted': 'user@ted'})
    def test_move_command_line(self, _, move):
        with patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'move', 'frank', 'ted']):
            self.assertEqual(lab.main(), 0)
        move.assert_called_once_with('frank', 'user@frank', 'ted', 'user@ted', 'kyon_training', run=None,
                                     yes=False, train_args=None, extra_iterations=None)
        move.reset_mock()
        with patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'move', 'frank', 'ted']):
            with patch.dict(os.environ, HHCM_JOB_ANSWERS=json.dumps({
                    'run': 'exp/r', 'yes': True, 'train_args': '--task T x=1', 'extra_iterations': 500})):
                lab.main()
        move.assert_called_once_with('frank', 'user@frank', 'ted', 'user@ted', 'kyon_training', run='exp/r',
                                     yes=True, train_args='--task T x=1', extra_iterations=500)
        with patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'move', 'frank']), patch('sys.stderr'), self.assertRaises(SystemExit):
            lab.main()

    @patch('hhcm_isaac_workstations.inventory', return_value={'frank': 'user@frank'})
    def test_app_answers_reach_actions(self, _):
        with patch('hhcm_isaac_workstations.action', return_value=0) as action, \
                patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'play', 'frank', '--env', '4']), \
                patch.dict(os.environ, HHCM_JOB_ANSWERS=json.dumps({'run': 'exp/r'})):
            lab.main()
        action.assert_called_once_with('user@frank', 'play', 'kyon_training', ['--env', '4'], run='exp/r')
        with patch('hhcm_isaac_workstations.clean', return_value=0) as clean, patch('hhcm_isaac_workstations.sys.argv', [
                'hhcm_isaac_workstations.py', 'clean', 'frank']), patch.dict(os.environ, HHCM_JOB_ANSWERS=json.dumps({
                    'run': 'exp/r', 'where': 'b', 'yes': True, 'permanent': True})):
            lab.main()
        clean.assert_called_once_with('frank', 'user@frank', 'kyon_training', run='exp/r', where='b',
                                      yes=True, permanent=True)
        # Training options still pass through unchanged.
        with patch('hhcm_isaac_workstations.action', return_value=0) as action, \
                patch('hhcm_isaac_workstations.sys.argv', ['hhcm_isaac_workstations.py', 'train', 'frank', '--run_name', 'x', 'agent.resume=false']):
            lab.main()
        action.assert_called_once_with('user@frank', 'train', 'kyon_training', ['--run_name', 'x', 'agent.resume=false'],
                                       run=None)

    def test_removed_cli_options_are_rejected(self):
        for option in ('--run', '--yes', '--where', '--permanent', '--train-args', '--extra-iterations'):
            for command in ('app', 'train', 'play', 'clean', 'move'):
                with self.subTest(option=option, command=command), patch('sys.stderr'), \
                        patch.object(sys, 'argv', ['lab.py', command, 'frank', option]), \
                        self.assertRaises(SystemExit) as error:
                    lab.main()
                self.assertEqual(error.exception.code, 2)

    @patch('hhcm_isaac_workstations.training_status', return_value='idle')
    def test_clean_unattended_refuses_last_copy_without_permanent(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.clean_project(root)
            remote = {'exp/2026-02-01_00-00-00_b': {'model_10.pt': 5}}
            with patch('hhcm_isaac_workstations.ROOT', root), patch('hhcm_isaac_workstations.remote_run_files', return_value=(remote, 10, 20)), \
                    patch('hhcm_isaac_workstations.subprocess.run') as ssh_run, patch('builtins.print'), \
                    patch('builtins.input', side_effect=AssertionError('prompted')):
                with self.assertRaises(ValueError):
                    lab.clean('ted', 'user@ted', 'kyon_training', run='exp/2026-02-01_00-00-00_b', where='r', yes=True)
                ssh_run.assert_not_called()
                ssh_run.return_value.returncode = 0
                self.assertEqual(lab.clean('ted', 'user@ted', 'kyon_training', run='exp/2026-02-01_00-00-00_b',
                                           where='r', yes=True, permanent=True), 0)
                self.assertIn('exp/2026-02-01_00-00-00_b', ssh_run.call_args.args[0][-1])




lab_app = lab


class LabAppTests(WorkspaceTest):
    def test_discovered_machines_validate_deduplicate_and_preserve_config(self):
        path = lab.ROOT / 'machines.json'
        config = {'user': 'alice', 'monitor_url': 'http://monitor/global_stats',
                  'hosts': {'ted': 'other@TED', 'new-ws': 'existing'}}
        path.write_text(json.dumps(config))
        self.assertEqual(lab.new_machines(['ted', 'new.ws', '-bad', 'user@host', 'bad/path'],
                                         lab.inventory()), ['new.ws'])
        hosts = lab.add_machines(['ted', 'new.ws', 'new-ws', '-bad'])
        saved = json.loads(path.read_text())
        self.assertEqual(saved, {**config, 'hosts': {**config['hosts'],
                                                   'new-ws-2': 'new-ws', 'new-ws-3': 'new.ws'}})
        self.assertEqual(hosts['new-ws-3'], 'alice@new.ws')
        lab.add_machines(['new.ws'])
        self.assertEqual(json.loads(path.read_text()), saved)
        with patch.object(Path, 'replace', side_effect=OSError('read only')):
            with self.assertRaises(OSError):
                lab.add_machines(['another'])
        self.assertEqual(json.loads(path.read_text()), saved)
        self.assertEqual(sorted(p.name for p in lab.ROOT.iterdir()), ['kyon_training', 'machines.json'])

    def test_new_machine_prompt_cancel_accept_and_save_error(self):
        async def scenario():
            path = lab.ROOT / 'machines.json'
            path.write_text(json.dumps({'user': 'alice', 'hosts': {'ted': 'ted'}}))
            with patch.object(lab.LabApp, 'load_stats'), patch.object(lab.LabApp, 'load_runs') as reload_runs:
                app = lab.LabApp(lab.inventory(), 'kyon_training')
                async with app.run_test(size=(140, 40)) as pilot:
                    stats = {'ted': ('free',) + ('—',) * 5, 'new.ws': ('free',) + ('—',) * 5}
                    app.set_stats(stats, None)
                    await pilot.pause()
                    self.assertIsInstance(app.screen, lab.Ask)
                    await pilot.click('#cancel')
                    app.set_stats(stats, None)
                    self.assertEqual(len(app.screen_stack), 1)
                    self.assertNotIn('new.ws', path.read_text())
                    app.offered_machines.clear()
                    app.set_stats(stats, None)
                    await pilot.pause()
                    await pilot.click('#ok')
                    self.assertEqual(app.hosts['new-ws'], 'alice@new.ws')
                    self.assertEqual(app.main('#machines').row_count, 2)
                    self.assertEqual(json.loads(path.read_text())['hosts']['new-ws'], 'new.ws')
                    self.assertGreaterEqual(reload_runs.call_count, 2)
                    with patch.object(lab, 'add_machines', side_effect=OSError('read only')):
                        app.offered_machines.add('another')
                        app.accept_machines(['another'])
                    self.assertNotIn('another', app.hosts)
                    self.assertNotIn('another', app.offered_machines)
        asyncio.run(scenario())

    def test_background_job_uses_standalone_script_and_workspace(self):
        app = lab.LabApp({'gpu': 'user@gpu'}, 'kyon_training')
        app.snapshots = {'gpu': {'error': 'password required'}}
        with patch.object(app, 'in_terminal') as terminal:
            app.start_job('fetch', 'download', 'gpu', ['fetch', 'gpu'])
        self.assertEqual(terminal.call_args.args[1], [
            sys.executable, str(Path(lab.__file__).resolve()), 'fetch', 'gpu',
            '--project', 'kyon_training', '--workspace', str(lab.ROOT)])

    def test_job_answers_are_internal_and_not_command_options(self):
        app = lab.LabApp({'gpu': 'user@gpu'}, 'kyon_training')
        app.snapshots = {'gpu': {'error': 'password required'}}
        with patch.object(app, 'in_terminal') as terminal:
            app.start_job('fetch', 'download', 'gpu', ['fetch', 'gpu'], run='exp/a b')
        command = terminal.call_args.args[1]
        self.assertEqual(command[0], 'env')
        self.assertEqual(json.loads(command[1].split('=', 1)[1]), {'run': 'exp/a b'})
        self.assertNotIn('--run', command)

    def remote_home(self, home):
        run = Path(home) / 'training/kyon_training/data/logs/rsl_rl/exp/2026-01-02_00-00-00_b'
        (run / 'params').mkdir(parents=True)
        (run / 'params/task.txt').write_text('Task-B\n')
        (run / 'params/command.json').write_text(json.dumps(['--task', 'Task-B', 'x.y=1']))
        (run / 'model_0.pt').write_bytes(b'1')
        (run / 'model_50.pt').write_bytes(b'12')
        return run

    def test_app_script_lists_runs_and_hash_matches_local_copy(self):
        with tempfile.TemporaryDirectory() as home:
            run = self.remote_home(home)
            result = subprocess.run(['python3', '-c', lab_app.APP_SCRIPT, 'kyon_training'],
                                    env={'HOME': home, 'PATH': '/usr/bin:/bin'}, capture_output=True, text=True)
            data = json.loads(result.stdout)
            self.assertEqual(data['active'], [])
            self.assertGreater(data['free'], 0)
            [info] = data['runs']
            self.assertEqual((info['run'], info['checkpoint'], info['task'], info['size']),
                             ('exp/2026-01-02_00-00-00_b', 'model_50.pt', 'Task-B', 3 + len(json.dumps(['--task', 'Task-B', 'x.y=1'])) + len('Task-B\n')))
            files = {str(p.relative_to(run)): p.stat().st_size for p in run.rglob('*') if p.is_file()}
            self.assertEqual(lab_app.manifest_hash(files), info['hash'])

    def project(self, root):
        (root / 'kyon_training').mkdir()
        (root / 'kyon_training/train.sh').touch()
        return root / 'kyon_training'

    def test_run_rows_states(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.project(Path(directory))
            cache = folder / 'data/replays/user@ted/logs/rsl_rl'
            for run, files in {'exp/2026-01-03_00-00-00_full': {'model_1.pt': b'a'},
                               'exp/2026-01-02_00-00-00_part': {'model_1.pt': b'a'},
                               'exp/2026-01-01_00-00-00_mine': {'model_7.pt': b'a'},
                               'exp/params': {'agent.yaml': b'a'}}.items():
                (cache / run).mkdir(parents=True)
                for name, content in files.items():
                    (cache / run / name).write_bytes(content)
            remote = lambda run, files, **extra: {'run': run, 'size': 9, 'checkpoint': 'model_1.pt', 'task': 'T',
                                                  'command': None, 'hash': lab_app.manifest_hash(files), **extra}
            snapshots = {'ted': {'free': 1, 'active': ['live'], 'runs': [
                remote('exp/2026-01-03_00-00-00_full', {'model_1.pt': 1}),
                remote('exp/2026-01-02_00-00-00_part', {'model_1.pt': 1, 'model_2.pt': 1}),
                remote('exp/2026-01-04_00-00-00_remote', {}),
                remote('exp/2026-01-05_00-00-00_live', {}, training=True)]}}
            rows = lab_app.run_rows({'ted': 'user@ted'}, snapshots, folder)
            self.assertEqual([(Path(r['run']).name, r['state']) for r in rows], [
                ('2026-01-05_00-00-00_live', 'training'), ('2026-01-04_00-00-00_remote', 'not fetched'),
                ('2026-01-03_00-00-00_full', 'fetched'), ('2026-01-02_00-00-00_part', 'partial'),
                ('2026-01-01_00-00-00_mine', 'local only'), ('params', 'local only')])
            self.assertEqual(rows[4]['checkpoint'], 'model_7.pt')
            rows = lab_app.run_rows({'ted': 'user@ted'}, {'ted': {'error': 'unreachable'}}, folder)
            self.assertEqual({r['state'] for r in rows}, {'remote unknown'})

    def test_availability_parses_monitor_lines(self):
        line = ("workstation [0.7s]  🟩  IP:10.0.0.1   CPU:4%    RAM:18.04%   GPU:1%    VRAM:49.69%  "
                "disk:['91.65%']  top_mem_user:a 1%  top_vram_users:a 1%  dl:1%  wl:1%  "
                "active_users:['alice[2:17:09]', 'bob[1 day, 0:00:01]']")
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = line.encode()
        with patch('hhcm_isaac_workstations.STATUS_URL', 'http://monitor.example/global_stats'), \
                patch('hhcm_isaac_workstations.urlopen', return_value=response):
            self.assertEqual(lab_app.availability(), {
                'workstation': ('free', '4%', '18.04%', '1%', '49.69%', 'alice, bob')})
            response.__enter__.return_value.read.return_value = b'<pre>no statistics</pre>'
            with self.assertRaisesRegex(ValueError, 'no workstation statistics'):
                lab_app.availability()

    def test_job_output_lines_progress_and_link(self):
        job = lab_app.Job(1, 'fetch', 't', [], notify=lambda job, committed: None)
        job.feed('Copying\n  1,000  10%  1MB/s\r  5,000  50%')
        self.assertEqual((job.lines, job.live, job.progress), (['Copying'], '  5,000  50%', 50))
        job.feed('\r  9,000  90%\r')  # Trailing \r may be the start of \r\n.
        self.assertEqual(job.live, '  9,000  90%')
        job.feed('\n\x1b[32mOpen http://127.0.0.1:5555 when Viser reports ready\x1b[0m\r\nend\n')
        self.assertEqual(job.lines, ['Copying', '  9,000  90%', 'Open http://127.0.0.1:5555 when Viser reports ready', 'end'])
        self.assertEqual(job.link, 'http://127.0.0.1:5555')

    def test_job_runs_in_background_and_can_be_interrupted(self):
        events = []
        script = ('import sys, time\nprint("start", flush=True)\nsys.stdout.write("x 40%\\r"); sys.stdout.flush()\n'
                  'try:\n    time.sleep(30)\nexcept KeyboardInterrupt:\n    print("stopped"); sys.exit(130)\n')
        job = lab_app.Job(1, 'play', 't', [sys.executable, '-c', script], notify=lambda job, done: events.append(done))
        job.start()
        for _ in range(100):
            if job.progress == 40:
                break
            time.sleep(0.05)
        self.assertTrue(job.running)
        self.assertIsNone(job.process.stdin)
        job.interrupt()
        for _ in range(100):
            if not job.running:
                break
            time.sleep(0.05)
        self.assertEqual(job.code, 130)
        self.assertEqual(job.lines[0], 'start')
        self.assertIn('stopped', job.lines)
        self.assertTrue(events[-1])

    def test_pasted_training_commands(self):
        hosts = {'dlv': 'u@dlv'}
        paste = lambda text: lab_app.pasted_training(text, hosts, 'kyon_training')
        self.assertEqual(paste('train dlv kyon_training --task T --run_name n --video agent.resume=false'),
                         ('dlv', 'n', ['--task', 'T', '--video', 'agent.resume=false']))
        self.assertEqual(paste('--task T --run_name=n'), (None, 'n', ['agent.resume=false', '--task', 'T']))
        self.assertEqual(paste(''), (None, None, ['agent.resume=false']))

    def app_scenario(self, keys_and_checks):
        hosts = {'ted': 'user@ted', 'frank': 'user@frank', 'bill': 'user@bill'}
        rows = [{'machine': 'ted', 'host': 'user@ted', 'run': 'exp/2026-01-02_00-00-00_b', 'state': 'training',
                 'checkpoint': 'model_50.pt', 'remote': 3, 'local': None, 'task': 'Task-B',
                 'command': ['--task', 'Task-B', 'x.y=1']},
                {'machine': 'frank', 'host': 'user@frank', 'run': 'exp/2026-01-01_00-00-00_a', 'state': 'not fetched',
                 'checkpoint': 'model_9.pt', 'remote': 5, 'local': None, 'task': 'Task-A', 'command': None}]
        snapshots = {'ted': {'free': 10, 'active': ['b'], 'runs': []}, 'frank': {'free': 10, 'active': [], 'runs': []},
                     'bill': {'free': 10, 'active': [], 'runs': []}}

        async def scenario():
            with tempfile.TemporaryDirectory() as directory, patch('hhcm_isaac_workstations.ROOT', Path(directory)), \
                    patch('hhcm_isaac_workstations.availability', return_value={'ted': ('free',) + ('1%',) * 4 + ('—',)}), \
                    patch('hhcm_isaac_workstations.machine_snapshot', side_effect=lambda host, project: snapshots[
                        next(n for n, h in hosts.items() if h == host)]), \
                    patch('hhcm_isaac_workstations.run_rows', return_value=rows):
                self.project(Path(directory))
                app = lab_app.LabApp(hosts, 'kyon_training')
                started = []
                app.start_job = lambda kind, title, machine, args, key=None, **answers: started.append((kind, machine, args, answers))
                async with app.run_test(size=(140, 40)) as pilot:
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    await keys_and_checks(app, pilot, started)

        asyncio.run(scenario())

    def test_app_fetch_filter_and_info(self):
        async def check(app, pilot, started):
            self.assertEqual(app.query_one('#runs').row_count, 2)
            self.assertIn('x.y=1', str(app.query_one('#info').render()))
            await pilot.press('f')
            self.assertEqual(started[-1], ('fetch', 'ted', ['fetch', 'ted'], {'run': 'exp/2026-01-02_00-00-00_b'}))
            await pilot.press('slash', 'f', 'r', 'a', 'n', 'k')
            self.assertEqual(app.query_one('#runs').row_count, 1)
            await pilot.press('escape')
            self.assertEqual(app.query_one('#runs').row_count, 2)
        self.app_scenario(check)

    def test_jobs_include_old_replay_and_refresh_background_status(self):
        app = lab_app.LabApp({'ted': 'user@ted'}, 'kyon_training')
        app.jobs = [lab_app.Job(i, 'play' if i == 1 else 'fetch', f'job {i}', [])
                    for i in range(1, 10)]
        for job in app.jobs:
            job.code = 0
        replay = app.jobs[0]
        replay.process, replay.code = unittest.mock.Mock(), None
        app.current = app.jobs[-1]
        tabs, live = unittest.mock.Mock(), unittest.mock.Mock()
        with patch.object(app, 'main', side_effect=lambda selector, cls: tabs if selector == '#jobtabs' else live):
            app.draw_jobs()
            labels = tabs.update.call_args.args[0].plain.splitlines()
            self.assertEqual(len(labels), 9)
            self.assertIn('[9]', labels[0])
            self.assertIn('[1]', labels[1])
            self.assertTrue(all(any(job.label() in line for line in labels) for job in app.jobs))
            replay.code = 130
            with patch.object(app, 'notify'), patch.object(app, 'load_runs'):
                app.job_changed(replay, True)
            self.assertIn(replay.label(), tabs.update.call_args.args[0].plain)

    def test_stop_finds_active_replay_when_shown_job_finished(self):
        app = lab_app.LabApp({'ted': 'user@ted'}, 'kyon_training')
        replay = lab_app.Job(1, 'play', 'play ted', [])
        replay.process = unittest.mock.Mock()
        finished = lab_app.Job(2, 'fetch', 'fetch ted', [])
        finished.code = 0
        app.jobs = [replay, finished]
        with patch.object(app, 'show_job') as show, patch.object(app, 'notify'), \
                patch.object(replay, 'interrupt') as interrupt:
            for shown in (finished, None):
                app.current = shown
                app.action_stop_job()
                show.assert_called_with(replay)
            self.assertEqual(interrupt.call_count, 2)
            app.current = finished
            with patch.object(app, 'need_run', return_value={'run': 'test', 'machine': 'ted'}), \
                    patch.object(app, 'ask', new_callable=unittest.mock.AsyncMock) as ask:
                asyncio.run(lab_app.LabApp.action_play.__wrapped__(app))
                show.assert_called_with(replay)
                ask.assert_not_called()
                replay.code = 130
                ask.return_value = None
                asyncio.run(lab_app.LabApp.action_play.__wrapped__(app))
                ask.assert_awaited_once()

    def test_stop_preserves_selected_running_job(self):
        app = lab_app.LabApp({'ted': 'user@ted'}, 'kyon_training')
        replay = lab_app.Job(1, 'play', 'play ted', [])
        fetch = lab_app.Job(2, 'fetch', 'fetch ted', [])
        replay.process = fetch.process = unittest.mock.Mock()
        app.jobs = [replay, fetch]
        app.current = fetch
        with patch.object(app, 'notify'), patch.object(replay, 'interrupt') as stop_replay, \
                patch.object(fetch, 'interrupt') as stop_fetch:
            app.action_stop_job()
            stop_fetch.assert_called_once_with()
            stop_replay.assert_not_called()

    def test_app_delete_last_copy_needs_typed_word(self):
        async def check(app, pilot, started):
            await pilot.press('down', 'd')
            await pilot.pause()
            options = app.screen.query_one('OptionList')
            self.assertTrue(options.get_option('l').disabled)  # no local copy
            self.assertEqual(options.highlighted, 1)  # first enabled choice: 'r'
            await pilot.press('enter')
            await pilot.pause()
            ok = app.screen.query_one('#ok')
            self.assertTrue(ok.disabled)
            await pilot.press(*'delete')
            self.assertFalse(ok.disabled)
            await pilot.press('enter')
            await pilot.pause()
            self.assertEqual(started[-1], ('delete', 'frank', [
                'clean', 'frank'], {'run': 'exp/2026-01-01_00-00-00_a', 'where': 'r', 'yes': True, 'permanent': True}))
        self.app_scenario(check)

    def test_app_train_form_and_cancel_paths(self):
        async def check(app, pilot, started):
            await pilot.press('t')
            await pilot.pause()
            await pilot.press(*'--num_envs 8', 'tab', *'trial', 'enter')  # form
            await pilot.pause()
            await pilot.press('enter')  # confirm (OK has focus)
            await pilot.pause()
            self.assertEqual(started[-1], ('train', 'ted', [
                'train', 'ted', '--run_name', 'trial', 'agent.resume=false', '--num_envs', '8'], {}))
            await pilot.press('t')
            await pilot.pause()
            app.screen.query_one('#field-extra').insert_text_at_cursor(
                'python3 lab.py train bill kyon_training --task T --run_name pasted x=1 agent.resume=true')
            await pilot.press('enter', 'enter')  # form, confirm
            await pilot.pause()
            self.assertEqual(started[-1], ('train', 'bill', [
                'train', 'bill', '--run_name', 'pasted', '--task', 'T', 'x=1', 'agent.resume=true'], {}))
            await pilot.press('s')
            await pilot.pause()
            await pilot.press('escape')  # cancelled: nothing started
            await pilot.pause()
            self.assertEqual(len(started), 2)
        self.app_scenario(check)

    def test_app_move_asks_target_and_arguments_for_old_runs(self):
        async def check(app, pilot, started):
            await pilot.press('down', 'm')
            await pilot.pause()
            options = app.screen.query_one('OptionList')
            self.assertEqual([options.get_option_at_index(i).id for i in range(options.option_count)], ['bill', 'ted'])
            self.assertTrue(options.get_option('ted').disabled)  # your job is running there
            await pilot.press('enter')  # bill
            await pilot.pause()
            self.assertEqual(app.screen.query_one('#field-args').value, '--task Task-A')
            await pilot.press(*' x=1', 'enter')
            await pilot.pause()
            await pilot.press(*'500', 'enter')  # confirm with extra iterations for a completed run
            await pilot.pause()
            self.assertEqual(started[-1], ('move', 'frank', [
                'move', 'frank', 'bill'], {'run': 'exp/2026-01-01_00-00-00_a', 'yes': True,
                                        'train_args': '--task Task-A x=1', 'extra_iterations': 500}))
        self.app_scenario(check)

class PortableTest(unittest.TestCase):
    def test_startup_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "trial"
            project.mkdir()
            (project / "code").mkdir()
            shutil.copy(Path(__file__).with_name("pipeline.sh"), project)
            binary = root / "bin"
            binary.mkdir()
            docker = binary / "docker"
            docker.write_text('''#!/bin/bash
case "$1" in
  info) echo rootless ;;
  run)
    if [[ $SCENARIO == early_exit ]]; then exit 17; fi
    if [[ $SCENARIO == learning ]]; then echo 'Learning iteration 0/1500'; fi
    for ((i=0; i<30; i++)); do
      [[ -e "$HOME/aborted" ]] && exit 134
      sleep 0.1
    done ;;
  top) echo 'diagnostic process list' ;;
  kill) touch "$HOME/aborted" ;;
  stop) : ;;
esac
''')
            docker.chmod(0o755)
            env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
                       HOME=str(root), OMNI_KIT_ACCEPT_EULA="YES",
                       TRAINING_STARTUP_TIMEOUT="1")
            # Historical iterations must not make a new hung launch look healthy.
            (project / "data").mkdir()
            (project / "data/train.log").write_text("Learning iteration 1499/1500\n")
            for scenario, expected in (("hung", 134), ("learning", 0), ("early_exit", 17)):
                with self.subTest(scenario=scenario):
                    (root / "aborted").unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", str(project / "pipeline.sh"), "run", "--task", "task"],
                        env=dict(env, SCENARIO=scenario), capture_output=True,
                        text=True, timeout=10)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertEqual((root / "aborted").exists(), scenario == "hung")
                    self.assertEqual("ERROR: No learning iteration" in result.stdout,
                                     scenario == "hung")
            logs = list((project / "data/startup").glob("*.log"))
            self.assertEqual(len(logs), 3)
            self.assertTrue(any("diagnostic process list" in p.read_text() for p in logs))

    def test_launch_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            portable = project / "training/container"
            portable.mkdir(parents=True)
            shutil.copy(Path(__file__).with_name("pipeline.sh"), portable)
            for directory in ("source", "scripts", "assets"):
                (project / directory).mkdir(parents=True, exist_ok=True)
            (project / "source/task.py").write_text("task")
            (project / "source/.env").write_text("secret")
            (project / "assets/robot.usd").write_text("asset")
            workspace = project / "training/kyon_training"
            workspace.mkdir()
            for name in ("source", "scripts", "assets"):
                (workspace / name).symlink_to(project / name, target_is_directory=True)
            home = root / "host"
            data = home / "training/kyon_training/data"
            data.mkdir(parents=True)
            (data / "checkpoint").write_text("keep")
            binary = root / "bin"
            binary.mkdir()
            stubs = {
                "ssh": '''#!/bin/bash
printf '%q ' "$@" >> "$HOME/ssh-calls"
printf '\\n' >> "$HOME/ssh-calls"
while [[ $1 == -* ]]; do
  case "$1" in -l|-p|-o|-S|-O|-L) shift 2 ;; *) shift ;; esac
done
shift
[[ $# == 0 ]] && exit 0
cd "$HOME"
exec /bin/bash -c "$*"
''',
                "tmux": '''#!/bin/bash
if [[ $1 == has-session ]]; then exit "${BUSY_SESSION:-1}"; fi
printf '%s' "${@: -1}" > "$HOME/tmux-command"
session=$4
printf '%s' "$session" > "$HOME/tmux-session"
mkdir -p "$HOME/training/$session/data"
if [[ ${STARTUP_FAIL:-} == 1 ]]; then
  echo 'simulated launch error' >&2
  exit 1
else
  launch_log=$(printf '%s' "${@: -1}" | sed -n 's/^env TRAINING_LAUNCH_LOG=\\([^ ]*\\) .*/\\1/p')
  if [[ ${LEARNING_FAIL:-} == 1 ]]; then
    echo '[Error] PhysX failed to allocate memory 268435456 bytes' > "$launch_log"
    for i in {1..100}; do echo '[Error] No physics scene created' >> "$launch_log"; done
    echo "Exception: Failed to create simulation view" >> "$launch_log"
  else
    echo 'Learning iteration 0/20' >> "$launch_log"
  fi
fi
''',
                "docker": '''#!/usr/bin/python3
import json, os, pathlib, sys
home = pathlib.Path.home()
args = sys.argv[1:]
if args[0] == 'info':
    print(os.environ.get('DOCKER_INFO', '["rootless"]'))
elif args[:2] == ['container', 'inspect']:
    sys.exit(int(os.environ.get('BUSY_CONTAINER', '1')))
elif args[:2] == ['image', 'inspect']:
    sys.exit(0 if (home / 'image-loaded').exists() else 1)
elif args[:2] == ['image', 'save']:
    print('image')
elif args[:2] == ['image', 'load']:
    if os.environ.get('FAIL_LOAD'): sys.exit(9)
    assert sys.stdin.read() == 'image\\n'
    (home / 'image-loaded').touch()
elif args[0] == 'run':
    (home / 'docker-args').write_text(json.dumps(args))
    if '-dit' in args: print('replay-container')
elif args[0] == 'port':
    assert args[1:] == ['replay-container', '8080/tcp']
    print('127.0.0.1:49152')
elif args[0] == 'attach':
    # The test has no terminal, as in a lab app job: attach to output only.
    assert args[1:] == ['--no-stdin', '--sig-proxy=false', 'replay-container']
    (home / 'docker-attach').touch()
    if os.environ.get('ATTACH_SECONDS'):  # a replay still running; ignores Ctrl+C like docker attach
        import signal, time
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        time.sleep(float(os.environ['ATTACH_SECONDS']))
elif args[:2] == ['rm', '-f']:
    assert args[2:] == ['kyon_training-viewer']
    (home / 'docker-rm').touch()
elif args[0] == 'stop':
    assert args[1:] == ['replay-container']
    (home / 'docker-stop').touch()
else:
    sys.exit(1)
''',
            }
            for name, body in stubs.items():
                path = binary / name
                path.write_text(body)
                path.chmod(0o755)
            env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
                       HOME=str(home), OMNI_KIT_ACCEPT_EULA="YES")

            def remote(*args):
                return subprocess.run(["bash", str(portable / "pipeline.sh"), "deploy", str(workspace), *args],
                                      env=env, capture_output=True, text=True)

            payload = "a name; $(touch SHOULD_NOT_EXIST)"
            result = remote("user@host", "--run_name", payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('STARTED: first learning iteration confirmed', result.stdout)
            self.assertNotIn('sleep 2', (portable / 'pipeline.sh').read_text())
            for stage in ('[1/5] SSH connected.', '[2/5] Remote checks passed.',
                          '[3/5] Files copied.', '[4/5] Image stream completed', '[5/5] Starting training'):
                self.assertIn(stage, result.stdout)
            calls = (home / 'ssh-calls').read_text().splitlines()
            self.assertIn('-M -S ', calls[0])
            for call in calls[1:]:
                if '-O exit' not in call:
                    self.assertIn('BatchMode=yes', call)
            folder = home / "training/kyon_training"
            self.assertEqual((folder / "code/source/task.py").read_text(), "task")
            self.assertFalse((folder / "code/source/.env").exists())
            self.assertTrue((folder / "assets/robot.usd").exists())
            self.assertTrue((home / "image-loaded").exists())
            self.assertFalse(list(root.rglob("*.tar.gz")))
            result = subprocess.run(["bash", "-c", (home / "tmux-command").read_text()],
                                    cwd=folder, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = json.loads((home / "docker-args").read_text())
            self.assertEqual(arguments[:2], ["run", "--rm"])
            self.assertEqual(arguments[arguments.index("--name") + 1], "kyon_training")
            self.assertEqual(arguments[-2:], ["--run_name", payload])
            self.assertIn(f"type=bind,source={folder}/data,target=/data", arguments)
            self.assertEqual(os.readlink(folder / "code/logs"), "../data/logs")
            self.assertFalse((folder / "SHOULD_NOT_EXIST").exists())
            logs = data / "logs/rsl_rl/task"
            (logs / "old/params").mkdir(parents=True)
            (logs / "empty").mkdir()
            (logs / "old/model_10.pt").write_text("old weights")
            (logs / "old/model_20.pt").write_text("latest weights")
            (logs / "old/params/env.yaml").write_text("saved settings")
            os.utime(logs / "old/model_10.pt", (1, 1))
            (home / "ssh-calls").write_text("")
            replay = subprocess.run(
                ["bash", str(portable / "pipeline.sh"), "play", str(workspace), "user@host",
                 "--viz", "viser", "--run_name", payload],
                env=env, capture_output=True, text=True)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            downloaded = workspace / "data/replays/user@host/logs/rsl_rl/task/old"
            self.assertEqual((downloaded / "model_20.pt").read_text(), "latest weights")
            self.assertFalse((downloaded / "model_10.pt").exists())
            self.assertEqual((downloaded / "params/env.yaml").read_text(), "saved settings")
            arguments = json.loads((home / "docker-args").read_text())
            self.assertEqual(arguments[arguments.index("--name") + 1], "kyon_training-viewer")
            self.assertIn("127.0.0.1::8080", arguments)
            self.assertIn("http://127.0.0.1:49152", replay.stdout)
            self.assertTrue((home / "docker-attach").exists())
            self.assertTrue((home / "docker-stop").exists())
            self.assertFalse((home / "docker-rm").exists())
            self.assertEqual(arguments[-2:], ["--run_name", payload])
            leftover = subprocess.run(
                ["bash", str(portable / "pipeline.sh"), "play", str(workspace), "user@host", "--viz", "viser"],
                env={**env, "BUSY_CONTAINER": "0"}, capture_output=True, text=True)
            self.assertEqual(leftover.returncode, 0, leftover.stderr)
            self.assertIn("Stopping the previous replay viewer", leftover.stdout)
            self.assertTrue((home / "docker-rm").exists())
            # Stopping a running replay as the lab app does (Ctrl+C to the job's process
            # group, through lab.py) must still stop the viewer container, promptly.
            (home / "docker-stop").unlink()
            (home / "docker-attach").unlink()
            job = subprocess.Popen(
                [sys.executable, "-c", "import subprocess, sys; sys.exit(subprocess.Popen(sys.argv[1:]).wait())",
                 "bash", str(portable / "pipeline.sh"), "play", str(workspace), "user@host", "--viz", "viser"],
                env={**env, "ATTACH_SECONDS": "30"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True)
            for _ in range(200):
                if (home / "docker-attach").exists():
                    break
                time.sleep(0.05)
            time.sleep(0.3)
            os.killpg(job.pid, signal.SIGINT)
            job.wait(timeout=10)
            self.assertTrue((home / "docker-stop").exists())
            self.assertIn("websockets==13.1", arguments[arguments.index("-ec") + 1])
            self.assertNotIn("-L ", (home / "ssh-calls").read_text())
            self.assertNotIn("docker", (home / "ssh-calls").read_text())
            self.assertFalse((folder / "SHOULD_NOT_EXIST").exists())
            replay = subprocess.run(
                ["bash", str(portable / "pipeline.sh"), "play", str(workspace), "user@host"],
                env={**env, "TRAINING_REPLAY_CHECKPOINT": "task/old/model_10.pt"},
                capture_output=True, text=True)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual((downloaded / "model_10.pt").read_text(), "old weights")
            arguments = json.loads((home / "docker-args").read_text())
            self.assertIn("/data/logs/rsl_rl/task/old/model_10.pt", arguments)
            # Saved task must override project defaults, including normalized models.
            task = 'Template-Kyon-Velocity-TargetLimits-Slip05-Normalized-Airtime-v0'
            (logs / 'old/params/task.txt').write_text(task + '\n')
            replay = subprocess.run(
                ['bash', str(portable / 'pipeline.sh'), 'play', str(workspace), 'user@host',
                 '--task', 'Template-Kyon-Velocity-v0'],
                env=env, capture_output=True, text=True)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            arguments = json.loads((home / 'docker-args').read_text())
            self.assertEqual(arguments[-2:], ['--task', task])
            (logs / 'old/params/task.txt').unlink()
            (downloaded / 'params/task.txt').unlink()
            (workspace / 'run_tasks.json').write_text(json.dumps({'task/old': task}))
            replay = subprocess.run(
                ['bash', str(portable / 'pipeline.sh'), 'play', str(workspace), 'user@host'],
                env=env, capture_output=True, text=True)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            arguments = json.loads((home / 'docker-args').read_text())
            self.assertEqual(arguments[-2:], ['--task', task])
            # No checkpoint must stop before launching Docker or reusing a cached file.
            shutil.rmtree(data / "logs/rsl_rl")
            (home / "docker-args").unlink()
            replay = subprocess.run(
                ["bash", str(portable / "pipeline.sh"), "play", str(workspace), "user@host"],
                env=env, capture_output=True, text=True)
            self.assertNotEqual(replay.returncode, 0)
            self.assertFalse((home / "docker-args").exists())
            (folder / "code/source/stale.py").touch()
            login = remote("user@host", "login")
            self.assertEqual(login.returncode, 0, login.stderr)
            self.assertIn('Image already present', login.stdout)
            self.assertIn('No training was started.', login.stdout)
            self.assertFalse((folder / "code/source/stale.py").exists())
            self.assertEqual((data / "checkpoint").read_text(), "keep")
            arguments = json.loads((home / "docker-args").read_text())
            self.assertIn("-it", arguments)
            self.assertIn("uv pip install wandb==0.22.3", arguments[-1])
            self.assertIn("exec wandb login", arguments[-1])
            # The terminal manager lends its connection; deployment must not close it.
            (home / 'ssh-calls').write_text('')
            env['TRAINING_SSH_CONTROL_PATH'] = str(root / 'shared-socket')
            shared = remote('user@host', 'login')
            self.assertEqual(shared.returncode, 0, shared.stderr)
            shared_calls = (home / 'ssh-calls').read_text()
            self.assertIn('-O check', shared_calls)
            self.assertNotIn('-O exit', shared_calls)
            self.assertNotIn('-M -S', shared_calls)
            env.pop('TRAINING_SSH_CONTROL_PATH')
            # A failed SDK installation must prevent the login prompt.
            for name, body in {"uv": "exit 19", "wandb": "touch SHOULD_NOT_LOGIN"}.items():
                stub = binary / name
                stub.write_text("#!/bin/bash\n" + body + "\n")
                stub.chmod(0o755)
            failed_install = subprocess.run(["bash", "-ec", arguments[-1]],
                                            cwd=folder, env=env, capture_output=True)
            self.assertEqual(failed_install.returncode, 19)
            self.assertFalse((folder / "SHOULD_NOT_LOGIN").exists())
            # A failed tmux launch must still be reported without waiting for learning.
            env["STARTUP_FAIL"] = "1"
            failure = remote("user@host", "--task", "task")
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("Stopped during: training startup", failure.stderr)
            self.assertIn("simulated launch error", failure.stderr)
            env.pop("STARTUP_FAIL")
            # Historical learning must not hide this launch failure or bury its cause.
            (folder / 'data/train.log').write_text('Learning iteration 999/1000\n')
            env['LEARNING_FAIL'] = '1'
            failure = remote('user@host', '--task', 'task')
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn('FAILED: training exited before', failure.stderr)
            self.assertIn('failed to allocate memory 268435456 bytes', failure.stderr)
            self.assertIn('Exception: Failed to create simulation view', failure.stderr)
            self.assertNotIn('STARTED:', failure.stdout)
            env.pop('LEARNING_FAIL')
            (project / "source/task.py").write_text("updated")
            for setting, value in (("BUSY_SESSION", "0"), ("BUSY_CONTAINER", "0"),
                                   ("DOCKER_INFO", "[]"), ("OMNI_KIT_ACCEPT_EULA", "")):
                env[setting] = value
                failure = remote("user@host", "--task", "task")
                self.assertNotEqual(failure.returncode, 0)
                if setting != "OMNI_KIT_ACCEPT_EULA":
                    self.assertIn("Stopped during: remote checks", failure.stderr)
                self.assertEqual((folder / "code/source/task.py").read_text(), "task")
                env.pop(setting)
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            jump = workspace.with_name("kyon_jump")
            jump.mkdir()
            for name in ("source", "scripts", "assets"):
                (jump / name).symlink_to(project / name, target_is_directory=True)
            result = subprocess.run(
                ["bash", str(portable / "pipeline.sh"), "deploy", str(jump), "user@host", "--task", "jump"],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((home / "tmux-session").read_text(), "kyon_jump")
            jump_folder = home / "training/kyon_jump"
            subprocess.run(["bash", "-c", (home / "tmux-command").read_text()],
                           cwd=jump_folder, env=env, capture_output=True, check=True)
            arguments = json.loads((home / "docker-args").read_text())
            self.assertEqual(arguments[arguments.index("--name") + 1], "kyon_jump")
            self.assertTrue((jump_folder / "code/source/task.py").exists())
            self.assertEqual((data / "checkpoint").read_text(), "keep")
            invalid = workspace.with_name("bad.name")
            invalid.mkdir()
            result = subprocess.run(
                ["bash", str(portable / "pipeline.sh"), "deploy", str(invalid), "user@host", "login"],
                env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            (home / "image-loaded").unlink()
            env["FAIL_LOAD"] = "1"
            failure = remote("user@host", "login")
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("Stopped during: Isaac image setup", failure.stderr)
            self.assertNotIn("[5/5]", failure.stdout)
            self.assertNotEqual(remote("-oProxyCommand=bad", "login").returncode, 0)


if __name__ == '__main__':
    unittest.main()
