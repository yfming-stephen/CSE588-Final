#!/bin/sh
set -eu
cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv('.env')
key = os.getenv('OPENAI_API_KEY', '').strip()
if not key or key.startswith('<'):
    raise SystemExit('please fill in OPENAI_API_KEY in .env')
PY
exec .venv/bin/tau2 run \
  --domain airline \
  --agent-llm gpt-4.1-2025-04-14 \
  --user-llm gpt-4.1-2025-04-14 \
  --num-trials 1 --num-tasks 1 --max-concurrency 1 "$@"
