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

Each project folder needs `source/`, `scripts/rsl_rl/{train,play}.py`, `assets/`
and these launchers:

```bash
# train.sh
exec bash "$HHCM_APP/pipeline.sh" deploy "$(dirname "$(realpath "$0")")" "$@"
# play.sh
exec bash "$HHCM_APP/pipeline.sh" play "$(dirname "$(realpath "$0")")" "$@" --num_envs 1 --device cpu --viz viser --real-time agent.device=cpu
```

The pipeline loads the training package in `source/isaac_lab_tutorial`.
If your package has a different name, set `HHCM_ISAAC_EXTENSION` to that name.

After accepting NVIDIA's terms, launch:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
python hhcm_isaac_workstations.py --workspace /path/to/workspace
```

Use the app for all actions, or `--help` for CLI commands.
