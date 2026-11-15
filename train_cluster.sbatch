#!/bin/bash -l
# Convert the data (if needed) and train ACT on one NHR alex A40.
#   sbatch --export=NONE,TRAIN_CONFIG=train_config_wam train_cluster.sbatch put_bottle_in_shelf
# TRAIN_CONFIG names <TRAIN_CONFIG>.yml, e.g. train_config_wam / train_config_all_wam (ACT + future tactile
# latent prediction; head camera / head + wrist cameras) or train_config / train_config_all (plain ACT).
# The data is exported with the cameras of task_settings.json, plus the wrist camera when the config uses it.
#
# Paths (override with --export):
#   REPO          this repository                                   ($HOME/Projects/act-wam)
#   UNIVTAC_DATA  raw UniVTAC episodes, <task>/<config>/*.hdf5      ($HOME/Projects/UniVTAC/data)
#   WS            workspace: $WS/act/data (ACT episodes, linked as ./data), $WS/act/ckpt, $WS/act/venv
# The tactile encoder checkpoint is read from checkpoints/ (see the README).
# wandb: online through the NHR proxy (needs ~/.netrc from `wandb login`); WANDB_PROJECT= disables it.
#SBATCH --job-name=act-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=act-train-%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

REPO=${REPO:-$HOME/Projects/act-wam}
if [ $# -gt 0 ] && [[ $1 != --* ]]; then TASK_NAME=$1; shift; fi
TASK_NAME=${TASK_NAME:?usage: sbatch train_cluster.sbatch <task>}
TASK_CONFIG=${TASK_CONFIG:-clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-100}
TRAIN_CONFIG=${TRAIN_CONFIG:-train_config_wam}
SEED=${SEED:-0}
UNIVTAC_DATA=${UNIVTAC_DATA:-$HOME/Projects/UniVTAC/data}
WS=${WS:-$(ws_find univtac)}
PY=${PY:-$WS/act/venv/bin/python}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-cong_fu}
export WANDB_PROJECT=${WANDB_PROJECT-act_univtac}
export WANDB_NAME=${WANDB_NAME:-act_${TASK_NAME}_${TRAIN_CONFIG}}

SETTINGS=sim-${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}
CKPT_DIR=$WS/act/ckpt/act-${TASK_NAME}/${TASK_CONFIG}-${EXPERT_DATA_NUM}/${TRAIN_CONFIG}

cd "$REPO"
mkdir -p "$WS/act/data"
[ -e data ] || ln -s "$WS/act/data" data
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "[act] task=$TASK_NAME config=$TRAIN_CONFIG ckpt=$CKPT_DIR git=$(git rev-parse --short HEAD 2>/dev/null || echo none)"

# One job converts a task; a second job of the same task (baseline and WAM) waits for it.
# SIM_TASK_CONFIGS.json is shared by all tasks, so its update is serialized too.
(
    flock 9
    META=data/sim-${TASK_NAME}/${TASK_CONFIG}-${EXPERT_DATA_NUM}/metadata.json
    CAMERAS=()
    if grep -qE "^\s*-\s*cam_wrist\s*$" "${TRAIN_CONFIG}.yml"; then CAMERAS=(--visual-cameras head wrist); fi
    if python3 -c "import json,sys; sys.exit(0 if '$SETTINGS' in json.load(open('SIM_TASK_CONFIGS.json')) else 1)" 2>/dev/null \
        && [ -f "$META" ] && { [ ${#CAMERAS[@]} -eq 0 ] || grep -q cam_wrist "$META"; }; then
        echo "[act] dataset exists"
    else
        "$PY" process_data.py "$TASK_NAME" "$TASK_CONFIG" "$EXPERT_DATA_NUM" --raw-root "$UNIVTAC_DATA" \
            ${CAMERAS[@]+"${CAMERAS[@]}"}
    fi
) 9>"$WS/act/data/.lock"

if [ -f "$CKPT_DIR/policy_last.ckpt" ]; then
    echo "$CKPT_DIR already holds a finished run; move it away first"; exit 1
fi
mkdir -p "$CKPT_DIR"
"$PY" imitate_episodes.py --task_name "$SETTINGS" --ckpt_dir "$CKPT_DIR" \
    --config_path "./${TRAIN_CONFIG}.yml" --seed "$SEED"
