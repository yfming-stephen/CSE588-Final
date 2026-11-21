#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 {base|normal|attack|dev} TRIALS RUN_NAME [TASK_ID ...]" >&2
  exit 2
fi

split=$1
trials=$2
run_name=$3
shift 3
if [[ ! $run_name =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "RUN_NAME must be one directory name" >&2
  exit 2
fi
if [[ ! $trials =~ ^[1-9][0-9]*$ ]]; then
  echo "TRIALS must be a positive integer" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export TAU2_DATA_DIR="$root/data"
python_bin=${G0_PYTHON:-python}
model=gpt-4.1-2025-04-14
case ${MILESTONE_ARM:-g0} in
  g0)
    agent=llm_agent
    agent_args='{"temperature":0.0}'
    ;;
  g1)
    agent=g1_replan
    agent_args='{"temperature":0.0,"guard_llm":"gpt-4.1-2025-04-14","guard_llm_args":{"temperature":0.0},"guard_prompt_version":"v3","guard_evidence_rule":true,"guard_max_replans":3,"guard_on_error":"block"}'
    ;;
  *)
    echo "MILESTONE_ARM must be g0 or g1" >&2
    exit 2
    ;;
esac
args=()
if [[ $# -gt 0 ]]; then
  args=(--task-ids "$@")
fi
if [[ -f "data/simulations/$run_name/results.json" ]]; then
  "$python_bin" - "data/simulations/$run_name/results.json" "$agent" "$trials" <<'PY_CHECKPOINT'
import json
import sys

with open(sys.argv[1]) as handle:
    info = json.load(handle)["info"]
if info["agent_info"]["implementation"] != sys.argv[2] or info["num_trials"] != int(sys.argv[3]):
    raise SystemExit("Checkpoint agent or trial count differs. Use a new run name.")
PY_CHECKPOINT
fi
exec "$python_bin" -m tau2.cli run --domain airline --agent "$agent" \
  --task-set-name airline_g0_milestone --task-split-name "$split" \
  --num-trials "$trials" --save-to "$run_name" --auto-resume \
  --agent-llm "$model" --user-llm "$model" \
  --agent-llm-args "$agent_args" --user-llm-args '{"temperature":0.0}' \
  --seed 300 --timeout 240 --max-concurrency "${G0_CONCURRENCY:-2}" \
  ${args[@]+"${args[@]}"}
