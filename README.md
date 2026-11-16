# hhcm_isaac_workstations

A terminal app for training Isaac Lab policies on SSH workstations. Launch and
stop training, inspect runs, download checkpoints, replay locally, and move a run
to another workstation. Results survive container exit. Deleting the last copy
requires explicit confirmation.

This is a local terminal app, not a hosted web service. Each user runs their own
copy and supplies SSH access, training code, robot assets and an NVIDIA GPU.

## Files

| File | Purpose |
|---|---|
| `hhcm_isaac_workstations.py` | Terminal interface, CLI, SSH jobs and run management |
| `pipeline.sh` | Deployment, remote Docker runner and local replay |
| `Dockerfile` | Build the Isaac Lab image, including its viewer fix |
| `test_workstations.py` | CLI, interface and simulated deployment regression tests |
| `.gitignore` | Keep local configuration, credentials and results out of Git |
| `README.md` | Installation, configuration and usage |

## Install

Use Linux, Python 3.10+, Bash, OpenSSH, rsync and Docker. Training workstations
also need tmux, rootless Docker and NVIDIA Container Toolkit with GPU access.
Local replay also needs NVIDIA GPU access through Docker, even with `--device cpu`.
SSH keys are recommended. Connect to each workstation manually once to verify
its host key; background checks enforce known host keys.

From this repository:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'textual==2.1.2'
export HHCM_APP="$PWD"
docker build -t hhcm-isaaclab:v3.0.0-beta2 .
```

The image build downloads Isaac Lab v3.0.0-beta2 and Isaac Sim 6.0.0.1. It is large
and needs internet access and sufficient disk space. NVIDIA software remains
subject to NVIDIA's terms. After reviewing and accepting those terms, set
`OMNI_KIT_ACCEPT_EULA=YES`. The app never accepts them on your behalf.
To use an already compatible image, set `HHCM_ISAAC_IMAGE` to its tag instead.

## Configure your workspace

Keep the workspace outside this repository. For example:

```text
workspace/
  machines.json
  my_robot/
    train.sh
    play.sh
    source/       # Isaac Lab extension source (a symlink is also supported)
    scripts/      # rsl_rl/train.py and rsl_rl/play.py
    assets/       # robot assets
```

Create `machines.json` with your own accounts and hostnames:

```json
{"user": "your-user", "hosts": {"gpu1": "gpu1.example.org", "gpu2": "other-user@gpu2.example.org"}}
```

Project names must contain letters, digits, underscores or hyphens. The supplied
pipeline expects an Isaac Lab extension at `source/isaac_lab_tutorial` and scripts
at `scripts/rsl_rl/{train,play}.py`. For another extension directory, export
`HHCM_ISAAC_EXTENSION=your_extension`. The container exposes assets at `/assets`
and sets `KYON_ASSET_DIR=/assets`; your project must use the appropriate asset path.
Your training/play scripts must support the Isaac Lab RSL-RL arguments and save
checkpoints as `logs/rsl_rl/<experiment>/<run>/model_<iteration>.pt`.

Use this `my_robot/train.sh` (add your project's default arguments after `"$host"`
and before `"$@"`, if desired):

```bash
#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
host=${1:?Supply an SSH destination}
shift
exec bash "${HHCM_APP:?Set HHCM_APP}/pipeline.sh" deploy "$project" "$host" "$@"
```

Use this `my_robot/play.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
host=${1:?Supply an SSH destination}
shift
exec bash "${HHCM_APP:?Set HHCM_APP}/pipeline.sh" play "$project" "$host" \
  --num_envs 1 --device cpu --viz viser --real-time agent.device=cpu "$@"
```

Set `HHCM_WORKSPACE` to your workspace's absolute path. Set `HHCM_APP` to this
repository's absolute path. Keep both variables in the terminal environment used
to start the app. No dependency on the original HHCM training folder is required.

```bash
export HHCM_WORKSPACE=/absolute/path/to/workspace
```

Optionally add `"monitor_url": "http://your-monitor:9423/global_stats"` to
`machines.json`, or override it with `HHCM_MONITOR_URL`, for a compatible HTML endpoint
with hostname, state, CPU, RAM, GPU, VRAM and active_users fields. Without it,
shared utilization is unknown; your own jobs and runs are still queried over SSH.
An idle account does not prove a shared workstation is free: coordinate with its users.

## Run

```bash
export OMNI_KIT_ACCEPT_EULA=YES  # only after accepting NVIDIA's terms
python "$HHCM_APP/hhcm_isaac_workstations.py" --project my_robot
```

The first project is selected if `--project` is omitted. The workspace defaults
to the current directory; `--workspace /path/to/workspace` overrides the environment.
The app footer shows keyboard shortcuts. Training accepts a pasted command or
training arguments; supply your task and run name, such as
`--task Your-Task-v0 --run_name first_run --num_envs 256`.

Choose runs, copy locations, deletion confirmations and resume settings in the app.
The Move dialog accepts extra iterations for runs that already reached their target.

CLI commands use the machine aliases in `machines.json`:

```bash
python "$HHCM_APP/hhcm_isaac_workstations.py" train gpu1 --project my_robot login
python "$HHCM_APP/hhcm_isaac_workstations.py" train gpu1 --project my_robot --task Your-Task-v0 --run_name first_run
python "$HHCM_APP/hhcm_isaac_workstations.py" runs --project my_robot
python "$HHCM_APP/hhcm_isaac_workstations.py" play gpu1 --project my_robot
python "$HHCM_APP/hhcm_isaac_workstations.py" --help
```

`login` prepares W&B credentials on the workstation without starting training.
Use `WANDB_MODE=disabled` for training without W&B. Credentials remain in the
project's remote data directory. Replay does not need W&B login.

Deployment mirrors `source`, `scripts` and `assets` to
`~/training/<project>/` on the workstation, including removal of obsolete files.
It refuses to overwrite an active project. It transfers the Docker image only
when its tag is missing remotely; rebuilds using the same tag require a manual
remote refresh or a new tag. Training runs in tmux; closing the app leaves an
already launched remote job running. Startup failures are reported with logs.

Results live remotely under `~/training/<project>/data/`. Downloaded runs live
under `<workspace>/<project>/data/replays/<destination>/`. Replay downloads the
selected checkpoint and parameters, then starts a local viewer bound to localhost.
Open the URL printed by the viewer. Stopping replay stops its local container.

## Check

```bash
python -m unittest test_workstations -v
bash -n pipeline.sh
```

The tests simulate SSH, Docker and tmux; they exercise transfers with real rsync.
They do not run Isaac Sim or require workstation access.

No license has been selected yet. Choose a license before distributing the code
for reuse; robot assets and NVIDIA components have separate ownership and terms.
