#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Usage: pipeline.sh deploy|run|play [arguments]}
shift
case "$mode" in
deploy)
container_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project=${1:?Usage: pipeline.sh deploy project-folder user@host login | training arguments}
shift
cd -- "$project"
project_name=${PWD##*/}
if [[ ! $project_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Project folder names must contain only letters, digits, underscores or hyphens." >&2
  exit 1
fi

host=${1:?Usage: pipeline.sh deploy project-folder user@host login | user@host [training arguments]}
shift
image=${HHCM_ISAAC_IMAGE:-hhcm-isaaclab:v3.0.0-beta2}

if [[ ! $host =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ ]]; then
  echo 'Supply user@host or an SSH alias.' >&2
  exit 1
fi
if [[ ${OMNI_KIT_ACCEPT_EULA:-} != YES ]]; then
  echo 'Accept NVIDIA terms, then export OMNI_KIT_ACCEPT_EULA=YES.' >&2
  exit 1
fi
if [[ $# == 0 ]]; then
  echo 'Supply login or training arguments, for example --task Template-Kyon-Velocity-v0.' >&2
  exit 1
fi

echo "Destination: $host:~/training/$project_name"
if [[ $1 == login ]]; then
  echo 'Action: prepare files/image, then open W&B login. No training will start.'
else
  printf 'Action: prepare files/image, then start training with arguments: '
  printf '%q ' "$@"
  echo
fi
echo 'Source/assets will be mirrored, including deletions. Results/credentials are preserved.'
echo 'SSH may ask for your workstation password once. W&B separately asks for an API key.'

connection=$(mktemp -d /tmp/kyon-ssh.XXXXXXXX)
phase='SSH connection'
cleanup() {
  result=$?
  trap - EXIT
  if [[ -z ${TRAINING_SSH_CONTROL_PATH:-} ]]; then
    ssh -S "$connection/socket" -O exit -- "$host" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$connection"
  if (( result != 0 )); then
    echo "Stopped during: $phase (exit $result). Earlier completed steps remain on $host." >&2
  fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo '[1/5] Connecting to workstation...'
if [[ -n ${TRAINING_SSH_CONTROL_PATH:-} ]]; then
  ssh -S "$TRAINING_SSH_CONTROL_PATH" -O check -- "$host"
else
  ssh -M -S "$connection/socket" -o ControlPersist=no -fN -- "$host"
fi
# Later operations must reuse this connection or fail without another password prompt.
ssh_options=( -S "${TRAINING_SSH_CONTROL_PATH:-$connection/socket}" -o ControlMaster=no -o BatchMode=yes )
printf -v RSYNC_RSH '%q ' ssh "${ssh_options[@]}"
export RSYNC_RSH
echo '[1/5] SSH connected.'

phase='remote checks'
echo '[2/5] Checking rootless Docker, rsync, tmux and existing jobs...'
# Refuse to overwrite source belonging to an active job.
ssh "${ssh_options[@]}" -- "$host" "project_name='$project_name';" '
  set -eu
  command -v rsync >/dev/null
  command -v tmux >/dev/null
  case $(docker info --format "{{json .SecurityOptions}}") in
    *rootless*) ;;
    *) echo "Select rootless Docker on this host." >&2; exit 1 ;;
  esac
  if tmux has-session -t "=$project_name" 2>/dev/null || docker container inspect "$project_name" >/dev/null 2>&1; then
    echo "This project already has a session or container. Finish that job first." >&2
    exit 1
  fi
  mkdir -p "$HOME/training/$project_name/code" "$HOME/training/$project_name/assets"
'

echo '[2/5] Remote checks passed.'
phase='code and asset copy'
echo '[3/5] Mirroring source/scripts/assets and copying pipeline.sh (transfer progress below)...'
# Only project code and assets are mirrored; data and credentials stay on the host.
rsync --info=progress2 -rlt --delete --safe-links \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.egg-info/' \
  --exclude='.env*' --exclude='.netrc' --exclude='logs/' --exclude='wandb/' \
  "$(realpath source)" "$(realpath scripts)" "$host:training/$project_name/code/"
rsync --info=progress2 -rlt --delete --safe-links assets/ "$host:training/$project_name/assets/"
rsync --info=progress2 -t "$container_dir/pipeline.sh" "$host:training/$project_name/"

echo '[3/5] Files copied.'
phase='Isaac image setup'
echo "[4/5] Checking image $image on $host..."
# shortcut: the base tag is fixed; reload it manually if its contents change.
if ssh "${ssh_options[@]}" -- "$host" "docker image inspect '$image' >/dev/null 2>&1"; then
  echo '[4/5] Image already present; no image transfer needed.'
else
  echo 'Image missing. Streaming local image → compression → SSH → remote Docker load.'
  echo 'No temporary archive. Byte count below measures compressed data sent; total size is unknown.'
  echo 'Remote Docker still needs space for the image. Ctrl+C interrupts the stream.'
  docker image save "$image" | gzip -1 | dd bs=4M status=progress | \
    ssh "${ssh_options[@]}" -- "$host" 'bash -o pipefail -c "gzip -dc | docker image load"'
  echo '[4/5] Image stream completed and remote Docker load succeeded.'

fi

# Preserve arguments containing spaces or shell characters across SSH and tmux.
printf -v command '%q ' env OMNI_KIT_ACCEPT_EULA=YES \
  "WANDB_MODE=${WANDB_MODE:-online}" "HHCM_ISAAC_IMAGE=$image" "HHCM_ISAAC_EXTENSION=${HHCM_ISAAC_EXTENSION:-isaac_lab_tutorial}" bash pipeline.sh run "$@"
phase='W&B login'
if [[ $1 == login ]]; then
  echo '[5/5] Opening W&B login. Enter your W&B API key when requested (not your SSH password).'
  ssh -t "${ssh_options[@]}" -- "$host" "cd \"\$HOME/training/$project_name\" && $command"
  echo '[5/5] W&B login command completed. No training was started.'
else
  phase='training launch'
  echo '[5/5] Starting training in remote tmux...'
  printf -v quoted_command '%q' "$command"
  phase='training startup'
  ssh "${ssh_options[@]}" -- "$host" "project_name='$project_name'
    cd \"\$HOME/training/$project_name\" || exit
    mkdir -p data/startup || exit
    launch_log=\$(mktemp data/startup/launch-XXXXXXXX.log) || exit
    tmux new-session -d -s \"\$project_name\" \"env TRAINING_LAUNCH_LOG=\$launch_log \"$quoted_command || exit
  " '
    echo "Waiting for the first learning iteration; Ctrl+C leaves the detached job running."
    echo "Launch log: $PWD/$launch_log"
    started=$SECONDS
    while true; do
      if grep -q "Learning iteration [0-9]" "$launch_log"; then
        echo "STARTED: first learning iteration confirmed."
        exit 0
      fi
      if ! tmux has-session -t "=$project_name" 2>/dev/null; then
        # Recheck after exit so a short completed run still counts as started.
        if grep -q "Learning iteration [0-9]" "$launch_log"; then
          echo "STARTED: learning iteration confirmed; session has already exited."
          exit 0
        fi
        echo "FAILED: training exited before its first learning iteration." >&2
        echo "First errors from this launch:" >&2
        awk '\''/\[Error\]|ERROR|Error:|Exception:|Traceback/ {print; if (++n == 8) exit}'\'' "$launch_log" >&2
        echo "Last 30 lines from this launch:" >&2
        tail -n 30 "$launch_log" >&2
        exit 1
      fi
      if (( SECONDS - started >= 930 )); then
        echo "UNCONFIRMED: no learning iteration after 930s; detached session still exists." >&2
        tail -n 30 "$launch_log" >&2
        exit 124
      fi
      if (( (SECONDS - started) % 15 == 0 )); then
        echo "Still waiting for learning startup ($((SECONDS - started))s)..."
      fi
      sleep 1
    done
  '
  echo "Inspect the job: ssh -t $host 'tmux attach -t $project_name'"
fi
echo "Console log on $host: ~/training/$project_name/data/train.log"
;;
run)
cd -- "$(dirname -- "${BASH_SOURCE[0]:-.}")"
image=${HHCM_ISAAC_IMAGE:-hhcm-isaaclab:v3.0.0-beta2}
project_name=${PWD##*/}
if [[ ! $project_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Invalid project folder name." >&2
  exit 1
fi

if [[ ${OMNI_KIT_ACCEPT_EULA:-} != YES ]]; then
  echo 'Accept NVIDIA terms, then export OMNI_KIT_ACCEPT_EULA=YES.' >&2
  exit 1
fi
if [[ $(docker info --format '{{json .SecurityOptions}}') != *rootless* ]]; then
  echo 'Select rootless Docker first.' >&2
  exit 1
fi

options=()
container_name=$project_name
if [[ ${1:-} == play ]]; then
  shift
  container_name=$project_name-viewer
  options=( --publish 127.0.0.1:8080:8080 )
  command=( -it "$image" bash -ec '
    uv pip install websockets==13.1
    uv pip install --no-deps -e "/workspace/source/$HHCM_ISAAC_EXTENSION"
    checkpoint=$(python -c "from pathlib import Path; files=list(Path(\"/data/logs/rsl_rl\").glob(\"*/*/model_*.pt\")); assert files, \"No saved checkpoint found in this project\"; print(max(files, key=lambda p: p.stat().st_mtime_ns))")
    echo "Playing checkpoint: $checkpoint"
    exec python scripts/rsl_rl/play.py --checkpoint "$checkpoint" "$@"
  ' bash "$@" )
elif [[ ${1:-} == login ]]; then
  echo "Starting W&B login container; waiting for W&B instructions..."
  command=( -it "$image" bash -ec '
    echo "Preparing W&B 0.22.3 for current API keys (download cached)..."
    uv pip install wandb==0.22.3
    exec wandb login
  ' )
else
  startup_timeout=${TRAINING_STARTUP_TIMEOUT:-900}
  if [[ ! $startup_timeout =~ ^[1-9][0-9]*$ ]]; then
    echo 'TRAINING_STARTUP_TIMEOUT must be a positive number of seconds.' >&2
    exit 1
  fi
  mkdir -p data/startup
  launch_log=${TRAINING_LAUNCH_LOG:-"data/startup/$(date -u +%Y%m%dT%H%M%SZ)-$$.log"}
  touch "$launch_log"
  exec > >(tee -a data/train.log "$launch_log") 2>&1
  # Watch only this launch; older learning output must not hide a stalled startup.
  (
    deadline=$((SECONDS + startup_timeout))
    while (( SECONDS < deadline )); do
      if grep -q 'Learning iteration [0-9]' "$launch_log"; then
        exit 0
      fi
      sleep 1
    done
    if grep -q 'Learning iteration [0-9]' "$launch_log"; then exit 0; fi
    echo "ERROR: No learning iteration after ${startup_timeout}s; stopping $container_name. Diagnostics: $launch_log and data/kit-logs/."
    docker top "$container_name" || true
    # SIGABRT requests a Python traceback even when a native startup call is stuck.
    docker kill --signal SIGABRT "$container_name" || true
    docker stop --time 10 "$container_name" || true
  ) &
  startup_watchdog=$!
  trap 'kill "$startup_watchdog" 2>/dev/null || true; wait "$startup_watchdog" 2>/dev/null || true' EXIT
  command=( "$image" bash -ec '
    echo "Preparing W&B 0.22.3 for current API keys (download cached)..."
    uv pip install wandb==0.22.3
    echo "Registering mounted project (no dependencies installed)..."
    uv pip install --no-deps -e "/workspace/source/$HHCM_ISAAC_EXTENSION"
    echo "Starting training; console output is saved in /data/train.log."
    exec python scripts/rsl_rl/train.py "$@"
  ' bash "$@" )
fi

mkdir -p data/{logs,wandb,cache,config,home,kit-logs}
ln -sfnT ../data/logs code/logs

# Root inside rootless Docker maps to your SSH account; data survives --rm.
docker run --rm --name "$container_name" "${options[@]}" --user 0:0 --gpus all --shm-size 2g --ulimit core=0 --init \
  --mount "type=bind,source=$PWD/code,target=/workspace" \
  --mount "type=bind,source=$PWD/assets,target=/assets,readonly" \
  --mount "type=bind,source=$PWD/data,target=/data" \
  --mount "type=bind,source=$PWD/data/kit-logs,target=/opt/IsaacLab/env_isaaclab/lib/python3.12/site-packages/isaacsim/kit/logs" \
  -w /workspace -e OMNI_KIT_ACCEPT_EULA=YES -e HOME=/data/home \
  -e "HHCM_ISAAC_EXTENSION=${HHCM_ISAAC_EXTENSION:-isaac_lab_tutorial}" -e KYON_ASSET_DIR=/assets -e WANDB_DIR=/data/wandb -e "WANDB_MODE=${WANDB_MODE:-online}" \
  -e UV_CACHE_DIR=/data/cache/uv -e XDG_CACHE_HOME=/data/cache -e XDG_CONFIG_HOME=/data/config \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 "${command[@]}"
;;
play)
project=${1:?Usage: pipeline.sh play project-folder user@host [play arguments]}
host=${2:?Usage: pipeline.sh play project-folder user@host [play arguments]}
shift 2
cd -- "$project"
project_name=${PWD##*/}
image=${HHCM_ISAAC_IMAGE:-hhcm-isaaclab:v3.0.0-beta2}
if [[ ! $project_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ || ! $host =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ ]]; then
  echo 'Use a valid project folder and user@host or SSH alias.' >&2
  exit 1
fi
if [[ ${OMNI_KIT_ACCEPT_EULA:-} != YES ]]; then
  echo 'Accept NVIDIA terms, then export OMNI_KIT_ACCEPT_EULA=YES.' >&2
  exit 1
fi
# Fail locally before requesting workstation authentication or transferring data.
docker image inspect "$image" >/dev/null
for folder in source scripts assets; do
  test -d "$folder" || { echo "Missing project folder: $folder" >&2; exit 1; }
done
ssh_options=()
if [[ -n ${TRAINING_SSH_CONTROL_PATH:-} ]]; then
  ssh_options=( -S "$TRAINING_SSH_CONTROL_PATH" -o BatchMode=yes )
fi
printf -v ssh_command '%q ' ssh "${ssh_options[@]}"
echo "[1/3] Finding saved checkpoint on $host (files only; no remote replay)."
echo 'SSH may request your workstation password for selection and transfer. No W&B login is needed.'
selector='from pathlib import Path; import sys; root=Path.home()/"training"/sys.argv[1]/"data/logs/rsl_rl"; files=list(root.glob("*/*/model_*.pt")); assert files, "No saved checkpoint found"; selected=next((p for p in files if str(p.relative_to(root)) == sys.argv[2]), None) if sys.argv[2] else max(files, key=lambda p:p.stat().st_mtime_ns); assert selected, "Selected checkpoint not found"; print(selected.relative_to(root))'
printf -v command '%q ' python3 -c "$selector" "$project_name" "${TRAINING_REPLAY_CHECKPOINT:-}"
checkpoint=$(ssh "${ssh_options[@]}" -- "$host" "$command")
if [[ ! $checkpoint =~ ^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/model_[0-9]+\.pt$ || $checkpoint == *..* ]]; then
  echo "Unexpected checkpoint path: $checkpoint" >&2
  exit 1
fi
run=${checkpoint%/*}
cache="$PWD/data/replays/$host"
mkdir -p "$cache/logs/rsl_rl/$run" "$cache/home" "$cache/cache" "$cache/config"
echo "[2/3] Copying $checkpoint and saved parameters to $cache"
rsync -a --info=progress2 -s -e "$ssh_command" \
  --include="/${checkpoint##*/}" --include='/params/***' --exclude='*' \
  "$host:training/$project_name/data/logs/rsl_rl/$run/" "$cache/logs/rsl_rl/$run/"
test -s "$cache/logs/rsl_rl/$checkpoint"
task=$(python3 - "$cache/logs/rsl_rl/$run/params/task.txt" "$PWD/run_tasks.json" "$run" <<'PY'
import json, re, sys
from pathlib import Path
record, mapping = map(Path, sys.argv[1:3])
task = record.read_text().strip() if record.is_file() else (
    json.loads(mapping.read_text()).get(sys.argv[3], '') if mapping.is_file() else '')
if task and (not isinstance(task, str) or not re.fullmatch(r'[a-zA-Z0-9_:.\-]+', task)):
    raise SystemExit('Invalid saved task ID')
print(task)
PY
)
if [[ -n $task ]]; then
  echo "Using training task: $task"
  set -- "$@" --task "$task"
else
  echo 'No saved task ID; using replay arguments. Check that --task matches training.' >&2
fi
echo '[3/3] Starting replay on this PC.'
echo 'Ctrl+C stops local replay. The workstation is no longer needed.'
# One viewer at a time: replace one left running by an earlier replay that did not clean up.
if docker container inspect "$project_name-viewer" >/dev/null 2>&1; then
  echo "Stopping the previous replay viewer ($project_name-viewer)."
  docker rm -f "$project_name-viewer" >/dev/null
fi
container=$(docker run --rm -dit --pull=never --name "$project_name-viewer" \
  --gpus all --shm-size 2g --init --publish 127.0.0.1::8080 \
  --mount "type=bind,source=$(realpath source),target=/workspace/source" \
  --mount "type=bind,source=$(realpath scripts),target=/workspace/scripts" \
  --mount "type=bind,source=$(realpath assets),target=/assets,readonly" \
  --mount "type=bind,source=$cache,target=/data" \
  --mount "type=bind,source=$cache/logs,target=/workspace/logs" \
  -w /workspace -e OMNI_KIT_ACCEPT_EULA=YES -e HOME=/data/home \
  -e "HHCM_ISAAC_EXTENSION=${HHCM_ISAAC_EXTENSION:-isaac_lab_tutorial}" -e KYON_ASSET_DIR=/assets -e WANDB_MODE=disabled -e UV_CACHE_DIR=/data/cache/uv \
  -e XDG_CACHE_HOME=/data/cache -e XDG_CONFIG_HOME=/data/config \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  "$image" bash -ec '
    uv pip install websockets==13.1
    uv pip install --no-deps -e "/workspace/source/$HHCM_ISAAC_EXTENSION"
    checkpoint=$1
    shift
    exec python scripts/rsl_rl/play.py "$@" --checkpoint "$checkpoint"
  ' bash "/data/logs/rsl_rl/$checkpoint" "$@")
trap 'docker stop "$container" >/dev/null 2>&1 || true' EXIT
address=$(docker port "$container" 8080/tcp)
echo "Open http://$address when Viser reports ready (use this link instead of its internal port)."
# Follow the output in the background and wait, so Ctrl+C (or the lab app's stop)
# interrupts the wait at once and the EXIT trap stops the container. A foreground
# attach would only forward the signal and keep the script from cleaning up.
docker attach --no-stdin --sig-proxy=false "$container" &
trap 'exit 130' INT TERM HUP
wait $! || true
;;
*) echo "Unknown pipeline mode: $mode" >&2; exit 1 ;;
esac
