# Guards (CSE 588)

Pre-execution guards that check an agent's database-writing tool calls before
they run. G1 is implemented here; G2 and G3 plug into the same wrapper.

| Arm | What it is | CLI `--agent` |
|-----|------------|---------------|
| G0 | Plain `LLMAgent`, policy in the prompt only | `llm_agent` |
| G1 | `LLMAgent` + LLM judge (PolicyGuard-style) | `g1_replan`, `g1_feedback`, `g1_notify`, `g1_terminate` |

Experiment scripts, results and audits: `src/experiments/guards/README.md`.

## How it works

`GuardedAgent` (`base.py`) wraps G0's `LLMAgent` unchanged. When the inner
agent proposes a call to a guarded tool (default: every tool with
`mutates_state=True`; on airline the 6 write tools), the guard is asked first.

* **Allowed** calls are returned to the orchestrator and executed as usual.
* **Blocked** calls are never returned. The blocked call and the guard's reply
  stay in the inner agent's private history only, so they never reach the
  trajectory.

This matters because tau2 v1.0.1 scores a run by replaying every mutating
call in the trajectory into a fresh DB, and it has no notion of a rejected
call. A blocked call left in the trajectory would be executed during scoring.
`tests/test_guards` checks that a blocked write leaves the scored DB unchanged
under both strict and non-strict replay.

### Block modes

| Mode | Agent sees | User sees | Turn |
|------|------------|-----------|------|
| `replan` | generic tool error, no reason | whatever the agent says next | agent retries (up to `guard_max_replans`) |
| `feedback` | tool error listing the unmet conditions | whatever the agent says next | agent retries |
| `notify` | tool error with reasons (in history) | a guard-written message listing the unmet conditions | ends |
| `terminate` | — | a closing message | conversation ends (`AGENT_STOP`, still scored) |

The proposal's three strategies are `terminate`, `replan` and `notify`;
`feedback` is the common middle ground (agent gets the reason, user does not).

In `replan` and `feedback` the agent may retry up to `guard_max_replans` times
after a block, so `guard_max_replans + 1` blocked attempts in a row. The count
is per agent step (one reply to a user message or tool result) and starts over
once a step goes through, for example a read-only call. If every retry is
blocked, the agent replies "I'm sorry, but I'm unable to complete that
request."

### The G1 judge

`LLMJudgeGuard` (`llm_judge.py`) sends the judge the policy, the real
conversation so far (user and agent messages, tool calls and tool results,
each in an escaped `<message role=...>` tag so user text cannot pose as a tool
result) and the proposed call. It returns `allow`/`deny`, the unmet policy
conditions and a short rationale. It is stateless: it rereads the transcript
for every call and keeps no fact store.

By default the judge prompt includes PolicyGuard's evidence rule (trust only
facts confirmed by tool results, not user claims). Without it G1 would be a
strawman. `guard_evidence_rule: false` removes the rule, which gives the G1
counterpart of the G3-trust-all ablation.

If the judge call fails (after litellm's own retries), or the judge returns
unparseable output twice, the call is blocked and the error is logged. That is
the default `guard_on_error: "block"`; `"allow"` lets the call through instead.

#### Prompt versions (`guard_prompt_version`)

| Version | What changed | Why |
|---------|--------------|-----|
| `v1` | first prompt; used for the first test run | — |
| `v2` | confirmation counts once the user agreed; no invented rules; judge only this call | v1 blocked correct writes after the user had said "yes" |
| `v3` (default) | v2 plus a cited per-condition check: every record fact must be quoted from a tool result, agent statements are not evidence, dates are computed against the policy's current time; the reply includes `checks` | v2 stopped checking eligibility and let forbidden cancellations through |

Offline on the 32 write attempts from G0's train-split runs (2 repeats each):

| | false blocks of gold writes | catches on no-write tasks | judge flips |
|---|---|---|---|
| v1 | 11% (14) | 56% (9) | 3% |
| v2 | 7% (14) | 39% (9) | 9% |
| v3 | 7% (14) | 50% (9) | 3% |

v3's one "false block" on train is task 39, whose gold cancels a reservation
the policy does not allow cancelling. The samples are small; treat these as
directional.

**Test-split caveat.** The v1 -> v2 -> v3 changes were prompted by guard
decisions seen in a k=1 G1 run on the *test* split (tasks 22, 24, 29, 37, 48),
before the train-split data existed. Tuning and the table above use train only, but the
final test-split numbers for G1 should be reported with this caveat (or with
those tasks reported separately).

### Repeated checks

The wrapper caches decisions per simulation, keyed on the call (name and
arguments, not its id) plus the exact transcript, so the same question gets
the same answer instead of a fresh sample from the judge. Failed checks are not
cached; cached decisions are logged with `cached: true`.

This does **not** stop a replanning agent from getting a denied call through
after doing something else first. In task 24 of the first test run the judge
denied a forbidden cancellation, the agent ran three read-only tools, retried
the identical call, and the judge allowed it. The transcript had changed, so
the guard was asked again, as it is now too. That is a property of a stateless
judge (G1), and a concrete case for a stateful guard (G3).

### Parallel tool calls

Calls in one step run in order, so each guarded call is judged with the calls
before it in the same step shown as pending (an assistant message with
`PENDING_NOTE` and those calls, no results). Two identical writes in one step
are therefore two different questions. If any guarded call in a step is
blocked, the whole step is withheld and the other calls get a "not executed"
error.

### Stopping

Only the guard's own `terminate` message ends a run (`raw_data["guard"]
["terminated"]`). Agent text containing `###STOP###` does not, matching G0's
`LLMAgent`, which never stops on its own.

## Running

```bash
tau2 run --domain airline --task-split-name test --num-trials 4 --max-concurrency 2 \
  --agent g1_replan --agent-llm gpt-4.1-2025-04-14 \
  --agent-llm-args '{"temperature": 0.0, "guard_llm": "gpt-4.1-2025-04-14",
                     "guard_llm_args": {"temperature": 0.0}, "guard_prompt_version": "v3",
                     "guard_evidence_rule": true, "guard_max_replans": 3,
                     "guard_on_error": "block"}' \
  --user-llm gpt-4.1-2025-04-14
```

Always pass `--task-split-name` (the default split `base` includes the test
tasks). `--agent-llm-args` replaces tau2's default `{"temperature": 0.0}`, so
keep `temperature` in it. Keep total concurrency low (about 2-3 sims across all
runs on one key): above that, OpenAI rate limits turn sims into
`infrastructure_error`, which tau2 drops from the metrics.

Guard options go in `--agent-llm-args` with a `guard_` prefix. They are removed
before the agent's own LLM call. tau2 records `--agent-llm-args` as passed, so
pin every option you care about, as above. Independently, every logged guard
decision carries the resolved settings in `raw_data["guard"]["config"]`
(mode, max_replans, on_guard_error, guarded tools, judge model, judge args,
prompt version, evidence rule). Commit before an experiment run: results.json
records the git commit, and uncommitted code is not identified by it.

| Key | Default | Meaning |
|-----|---------|---------|
| `guard_llm` | `gpt-4.1-2025-04-14` | judge model (fixed across agent models) |
| `guard_llm_args` | `{"temperature": 0.0}` | judge LLM args |
| `guard_evidence_rule` | `true` | include the "trust only tool results" rule |
| `guard_prompt_version` | `"v3"` | judge prompt: `v1`, `v2` or `v3` |
| `guard_tools` | all DB-writing tools | list of tool names to guard |
| `guard_max_replans` | `3` | retries after a block in `replan`/`feedback`, per agent step (see "Block modes") |
| `guard_on_error` | `"block"` | `"block"` or `"allow"` when the judge fails |

Qwen3-8B behind vLLM as the agent, with the API judge:

```bash
--agent-llm hosted_vllm/Qwen/Qwen3-8B \
--agent-llm-args '{"api_base": "http://<host>:8000/v1", "temperature": <...>,
                   "guard_llm": "gpt-4.1-2025-04-14"}'
```

Use the `hosted_vllm/` prefix, not `openai/`, so your OpenAI key is not sent
to the vLLM server, and set Qwen's sampling parameters explicitly. This path
has not been run against a real vLLM server yet.

## Reading the logs

Every assistant message that involved a guard decision carries
`raw_data["guard"]`: mode, every decision (tool call, allow, violated
conditions, rationale, latency, judge cost/usage, error), and the blocked
attempts. The message's `cost`/`usage` include all agent attempts in that turn,
blocked ones too. Judge cost is kept separate (`guard_cost`) and is **not** in
tau2's "Avg Cost/Conversation".

```bash
python -m tau2.guards.summarize data/simulations/<run>/results.json        # table + totals
python -m tau2.guards.summarize data/simulations/<run>/results.json --json # everything
```

Reward and DB match still come from tau2. Attack success rate should be
computed separately from the executed writes; it is not `1 - reward`.

## Comparing judges offline

`offline_eval.py` re-asks the judge about write attempts already recorded in a
results file, with exactly the transcript the guard saw (or would have seen,
for a G0 run). Every variant is judged on identical inputs and only judge calls
cost money, so it is the cheap way to tune prompts. Tune on train-split runs.

```bash
python -m tau2.guards.offline_eval data/simulations/g0_train_k1 \
  --versions v1 v3 --repeats 2 --out judge_eval.json
```

Attempts are labeled `gold` (exact gold action: should be allowed),
`forbidden` (the task has no gold write: should be blocked) or `other`
(unlabeled). It prints deny rates per label, how often repeats disagree, cost,
and every attempt where the first and last version disagree.

## Adding G2 / G3

Subclass `Guard`, implement `check(tool_call, transcript, context) ->
GuardDecision`, and register a factory like `create_g1_agent` in `factory.py`.
Everything else (block modes, logging, the replay-safe wrapper) is shared, so
the arms differ only in the decision.

## Known limits

* G1 is stateless: after a denial, a retry with anything new in the
  transcript (even a read-only tool result) is judged afresh and can be
  allowed (see "Repeated checks").
* If one call in a parallel tool-call step is blocked, the whole step is
  blocked; the other calls get a "not executed" error.
* `notify` exposes the unmet conditions to the user, which is the feedback
  channel an adaptive attacker would use.
* The judge sees only what is in the transcript. It cannot query the DB itself.
* A judge reply with two disagreeing verdicts (e.g. a draft and a final
  answer) is treated as unparseable: retried once, then blocked.
