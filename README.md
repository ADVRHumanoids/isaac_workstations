# Isaac Workstations

Terminal app to train, monitor, replay and move Isaac Lab runs across SSH workstations.

Requires Linux, Python 3.10+, SSH, rsync and Docker. GPU machines also need tmux,
rootless Docker and NVIDIA Container Toolkit. Supply your own training code and assets.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install textual==2.1.2
export HHCM_APP="$PWD"
docker build -t hhcm-isaaclab:v3.0.0-beta2 .
```

Copy the workstation list into your workspace, then set `user` to your SSH username:

```bash
cp machines.example.json /path/to/workspace/machines.json
```

The supplied hosts and monitor require access to the HHCM network.
The app checks the monitor for new workstations and asks before adding them to `machines.json`.

Your workspace contains the machine list and your Isaac Lab projects, separate
from this app:

```text
workspace/
├── machines.json
└── my_robot/
    ├── source/              # Robot environments, rewards and training settings
    ├── scripts/rsl_rl/
    │   ├── train.py         # Training code
    │   └── play.py          # Replay code
    ├── assets/              # Robot models
    ├── train.sh             # Connects training to this app
    └── play.sh              # Connects replay to this app
```

Use the Python code and assets from your existing Isaac Lab project.
Create `train.sh` inside that project with:

```bash
exec bash "$HHCM_APP/pipeline.sh" deploy "$(dirname "$(realpath "$0")")" "$@"
```

Create `play.sh` alongside it with:

```bash
exec bash "$HHCM_APP/pipeline.sh" play "$(dirname "$(realpath "$0")")" "$@" --num_envs 1 --device cpu --viz viser --real-time agent.device=cpu
```

`deploy` prepares the selected workstation and starts training in Docker.
`play` opens a saved policy locally. The app calls these launchers for you.

After accepting NVIDIA's terms, launch:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
python hhcm_isaac_workstations.py --workspace /path/to/workspace
```

Use the app for all actions, or `--help` for CLI commands.
