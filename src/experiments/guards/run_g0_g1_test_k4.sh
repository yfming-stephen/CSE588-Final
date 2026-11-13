#!/bin/bash
# G0 vs G1 (replan mode, v3 judge) on the airline test split, 4 trials per task.
# Run from the repo root. Both arms run side by side at concurrency 1 each
# (2 sims total, which stays under a 450k TPM OpenAI limit); afterwards any
# infrastructure-error sims are re-run with --auto-resume, up to 4 passes.
# Re-running the script only fills in missing sims.
set -u
TAU2=.venv/bin/tau2
PY=.venv/bin/python
LOG=data/simulations/run_logs
mkdir -p "$LOG"

COMMON=(--domain airline --task-split-name test --num-trials 4 --max-concurrency 1
        --agent-llm gpt-4.1-2025-04-14
        --user-llm gpt-4.1-2025-04-14 --user-llm-args '{"temperature": 0.0}'
        --seed 300 --auto-resume)
G1ARGS='{"temperature": 0.0, "guard_llm": "gpt-4.1-2025-04-14", "guard_llm_args": {"temperature": 0.0}, "guard_prompt_version": "v3", "guard_evidence_rule": true, "guard_max_replans": 3, "guard_on_error": "block"}'

scored() {  # number of scored (non-infrastructure-error) sims in a run
  $PY - "$1" <<'EOF'
import sys
from pathlib import Path
from tau2.guards.summarize import load_simulations
p = Path("data/simulations") / sys.argv[1] / "results.json"
sims = load_simulations(p) if p.exists() else []
print(sum(s.get("termination_reason") != "infrastructure_error" for s in sims))
EOF
}

for attempt in 1 2 3 4; do
  echo "=== pass $attempt $(date '+%H:%M:%S')"
  $TAU2 run "${COMMON[@]}" --agent llm_agent --agent-llm-args '{"temperature": 0.0}' \
    --save-to g0_test_k4 > "$LOG/g0_test_k4.pass$attempt.log" 2>&1 &
  $TAU2 run "${COMMON[@]}" --agent g1_replan --agent-llm-args "$G1ARGS" \
    --save-to g1_replan_test_k4 > "$LOG/g1_replan_test_k4.pass$attempt.log" 2>&1 &
  wait
  g0=$(scored g0_test_k4); g1=$(scored g1_replan_test_k4)
  echo "after pass $attempt: G0 $g0/80, G1 $g1/80 scored"
  if [ "$g0" -ge 80 ] && [ "$g1" -ge 80 ]; then echo "DONE"; exit 0; fi
  if grep -q "no credits remaining" "$LOG"/*.pass$attempt.log; then
    echo "OpenAI credits exhausted; add credits and run this script again."; exit 1
  fi
  sleep 60
done
echo "INCOMPLETE after 4 passes"
exit 1
