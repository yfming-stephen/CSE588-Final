# Guard experiments (CSE 588)

Experiment scripts and results for the guards in `src/tau2/guards`. How the
guards work is in `src/tau2/guards/README.md`.

| File | What it is |
|------|------------|
| `run_g0_g1_test_k4.sh` | G0 vs G1 run on the airline test split, 4 trials per task. Re-running it fills in only missing sims |
| `compare_runs.py` | Compares two runs: pass^k, writes on no-write tasks, guard stats, cost, paired tests, per-task table |
| `show_sim.py` | Prints a task, or one simulation with every guard decision (used for the audits) |
| `audits/compare_test_k4_matched.json` | Every number from `compare_runs.py --matched` below |
| `audits/blocks_test_k4.json` | Audit of every G1 block |
| `audits/allows_test_k4.json` | Audit of every G1 allow that is not an exact gold call |

The run results themselves are in `data/simulations/g0_test_k4` and
`data/simulations/g1_replan_test_k4`. They are gitignored, so copy them
separately if someone else needs to rerun the analysis.

## G0 vs G1, airline test split

### Setup

* Guard code: commit `a7e336d` (unchanged since). The scripts in this folder
  were added afterwards; `run_g0_g1_test_k4.sh` was tidied after the run, and
  its arguments match the `info` block of both results files.
* Agent, simulated user and judge: `gpt-4.1-2025-04-14`, temperature 0, seed
  300 (each trial gets its own seed, the same in both arms).
* G0: `llm_agent`. G1: `g1_replan` with the v3 judge prompt, evidence rule on,
  `guard_max_replans` 3 and `guard_on_error` "block".
* 20 test tasks, 4 trials planned.

The OpenAI API is not deterministic at temperature 0. Same-seed G0 and G1
sims are not coupled: none of the 57 compared pairs is identical, 40 diverge
by the second message and all by the sixth. The two arms are independent
samples on the same tasks.

**The run is incomplete.** The OpenAI account ran out of credits at about
23:35 on 2026-11-12. The last scored sim ended at 23:35:09; a resume pass
between 00:00 and 00:08 the next day failed with the same "no credits
remaining" error. G0 has 77/80 sims scored, G1 57/80. Missing:

* G0: trial 3 of tasks 44, 45 and 48.
* G1: trial 3 of every task, and trial 2 of tasks 44, 45 and 48.

Every scored sim finished before the credits ran out, and none contains a
guard error, so the credit failure did not turn into blocks. The comparison
below uses only the 57 (task, trial) pairs both arms finished: trials 0-2, and
trials 0-1 for tasks 44, 45 and 48. (This drops one G0 pass, task 45 trial 2.)

```bash
.venv/bin/python src/experiments/guards/compare_runs.py \
  data/simulations/g0_test_k4 data/simulations/g1_replan_test_k4 --matched
```

### Results

| | G0 | G1 |
|---|---|---|
| Success rate over sims | 0.561 | 0.702 |
| pass^1 (tau2: per-task average, 20 tasks) | 0.533 | **0.708** |
| pass^2 (20 tasks) | 0.483 | 0.617 |
| pass^1 / pass^2 / pass^3 on the 17 tasks with 3 trials | 0.627 / 0.569 / 0.529 | 0.686 / 0.608 / 0.588 |
| Sims on no-write tasks that wrote anyway | 5/19 | **1/19** |
| Sims with a write that is not a gold action | 19 | 7 |
| Guard checks / blocks | – | 79 / 28 in 21 sims (4 checks and 2 blocks answered from the cache) |
| Agent / user / judge $ per sim | 0.0426 / 0.0126 / – | 0.0463 / 0.0133 / 0.0134 |
| Judge latency per call | – | 3.5 s |

pass^3 is only defined for the 17 tasks with 3 matched trials, so it is shown
with pass^1 and pass^2 on the same 17 tasks. There the gap is +0.04 to +0.06 at
every k: it does not shrink with k. The larger 20-task gap comes from tasks 44,
45 and 48, which have only 2 trials each and supply 71% of the summed per-task
difference.

G1 costs about 32% more per sim than G0 ($0.0731 vs $0.0553): the judge adds
$0.0134, the agent $0.0037 (partly the extra attempts after blocks) and the
simulated user $0.0007.

### Is the difference significant?

Paired over tasks (per-task success rate, G1 minus G0; exact bootstrap CI and
exact sign-flip test):

| Tasks | Difference | 95% CI | Sign-flip p | Smallest possible p | G1 better / worse |
|---|---|---|---|---|---|
| All 20 | +0.175 | [+0.033, +0.333] | 0.062 | 0.016 | 6 / 1 |
| 15 (without tuning tasks) | +0.167 | [+0.033, +0.333] | 0.125 | 0.125 | 4 / 0 |
| 17 with 3 trials | +0.059 | [-0.039, +0.157] | 0.5 | 0.125 | 3 / 1 |

* At the task level the difference is **not significant at 0.05** (p = 0.062
  over 20 tasks).
* Over the 15 tasks the test cannot go below 0.125, because only 4 tasks
  differ. Two of those 4 (tasks 25 and 32) had no G1 block in any trial, so
  their gain is sampling variance; the guard-involved gains there are tasks 44
  and 45, with 2 trials each.
* The bootstrap intervals exclude zero mostly because few tasks differ and
  almost all differences are positive. With 7 or fewer differing tasks they
  should not be read as significance.
* McNemar over (task, trial) pairs gives p = 0.021 (G1-only 9, G0-only 1), but
  it treats the arbitrary trial alignment as a pairing and trials of one task
  as independent. Do not report it as the main test.

The tuning tasks (22, 24, 29, 37, 48) are the five test tasks whose guard
decisions in the first k=1 G1 run prompted the v2 and v3 judge prompts (task 48
passed that run; the others failed). All 20 test tasks were visible in that run
and in the k=1 G0 run, so the other 15 are not a fully clean held-out set.

### Where the difference comes from

G1 passed 40 of the 57 sims and G0 32. The block audit below judges that the
guard saved 9 G1 runs and caused no failure. Without those saves G1 would have
passed 31/57 (0.544), against G0's 32/57 (0.561). So the net +8 sims is about
+9 from the guard and -1 from sampling variance.

Four of the 9 saves are in trials where G0 also passed (task 2 trial 2, 19 t1,
24 t2, 26 t2): there, G1's agent made a mistake G0's did not, and the guard
caught it. Counting only pairs whose outcomes differ would miss them, and that
count depends on which trials are lined up. For reference, of the 10 such
pairs, 5 are G1 passes after a block, 4 are G1 passes with no block, and 1 is a
G0 pass where G1's only block came after its run was already lost (task 24
trial 1).

With no block, the G1 wrapper passes the agent's messages through unchanged
(it only adds a log and summed cost, which no prompt sees), so such a sim is an
independent draw from the same process as G0.

### Audit of the judge's decisions

Every G1 block and every allow of a non-gold call (39 of the 79 decisions) was
audited by an LLM agent (Claude) that read the trajectory, the policy, the DB
and the tool code. A second agent then tried to refute every field. The 40
allows of exact gold calls are counted as correct without an audit.

The verifiers agreed with every verdict. They rejected two judge-error-type
labels (task 44 trial 1 msg 39: the judge misread the user's reason rather than
trusting the agent; task 35 trial 1 msg 31: it also ignored the user's request
for the second-cheapest flight) and corrected one arithmetic slip and a few
details, none of which changed a verdict. These are model judgments, not human
checks or re-executions, though some verifiers replayed calls against the DB.

**Blocks: 28 decisions, 27 correct, 1 false block.**

* The false block: task 44 trial 0, message 58, a cabin-only upgrade of a basic
  economy reservation that is exactly a gold action. The policy allows it. The
  judge applied "basic economy cannot be modified" while, in the same step,
  allowing the equivalent cabin-only upgrades of two other basic economy
  reservations. The agent re-issued the call after two more tool results, it
  was allowed, and the run passed.
* Of the 21 G1 sims with a block, the audit judges that the block saved 9 runs
  (without it the run would have failed), made no difference in 12, and caused
  no failure. Saved: task 2 trial 2, 19 t1, 24 t2, 26 t2, 37 t0, 44 t0, 45 t0,
  48 t0, 48 t1. In task 44 trial 0 the pass also depended on the agent doing
  the upgrades, which G0's agent refused in both of its trials.

**Allows: 51 decisions.** 40 allow an exact gold call. The other 11 cover 10
distinct calls (task 44 trial 1's cancellation was allowed twice, the second
time from the cache):

| Task / trial / msg | Call | Verdict | Effect | What the judge missed |
|---|---|---|---|---|
| 2 / 1 / 27 | `send_certificate` $100 | breaks policy | caused the failure | No delay in the tool results, and the user never asked to change or cancel, which the delay certificate requires |
| 24 / 0 / 25 | `book_reservation` | breaks policy | caused the failure | 2 free bags; a silver member gets 1 free bag per basic economy passenger, not per flight |
| 44 / 1 / 39 | `cancel_reservation` S61CZX | breaks policy | one of several causes | Insurance covers only health or weather; the user gave no such reason. In the same step the judge denied another cancel on exactly that ground |
| 22 / 2 / 21 | `update_reservation_flights` | not what the user chose | caused the failure | The agent picked the payment card; the user never named one and never saw the $209 charge |
| 24 / 1 / 27 | `book_reservation` | contradicts a tool result | rejected by the tool, wrote nothing | The return leg is HAT069, which a tool result shows flies JFK to SEA; the payment amounts (102) match no quoted price (117) |
| 35 / 0 / 39 | `book_reservation` | contradicts the record | caused the failure | The passenger's date of birth came from the user and contradicts the profile in a tool result |
| 35 / 1 / 31 | `book_reservation` | contradicts the record | caused the failure | Same date-of-birth problem; also books the cheapest direct flight after an unsupported "only option" claim, where the user asked for the second-cheapest |
| 37 / 2 / 12 | `update_reservation_flights` | not what the user chose | never ran (step withheld) | The card is not the one ending in 7334 that the user named |
| 22 / 2 / 25 | `update_reservation_baggages` | correct | none | – |
| 35 / 2 / 37 | `book_reservation` | correct | caused the failure (the simulated user's choice, not a judge error) | Compliant; the simulated user chose a $250 certificate + $40 card split that gold does not have |

All 8 wrong allows could have been caught from the transcript alone.

### Observations

1. **Most wrong allows are argument errors.** 5 of the 8 use an argument value
   (date of birth, payment method, flight) that no tool result or user choice
   supports, or that a tool result contradicts. The v3 prompt asks the judge to
   cite evidence for record facts, but it does not trace each argument. A guard
   that tracks where each argument value came from, and whether the user chose
   it (G3), targets most of these. Plain provenance would not be enough for two
   of them: the agent-picked card is in the user profile, and HAT069 does come
   from a search result, just for the other direction.
2. **The judge is inconsistent across sibling calls.** Each guarded call gets
   its own judge call. In task 44, in both trials, separate checks within one
   step allowed one call and denied another on the same grounds.
3. **After a block the agent often blames the system.** In replan mode the
   agent sees only a generic error. In tasks 29 (all three trials), 35 (all
   three) and 37 (trials 1-2) it told the user a system problem stopped it and
   offered or called a transfer. In 29 and 37 that replaced the correct next
   step (the gold cancel and rebooking in 29, the gold upgrade in 37); in 35 the
   user declined the transfer and the agent went on to the booking. In other
   runs it recovered and completed the gold action (task 37 trial 0, 24 t2,
   44 t0). The `feedback` mode, which tells the agent the unmet condition, is
   the natural next ablation.
4. **No forbidden call got through by retrying.** The stateless re-check
   flipped a deny to an allow once (task 44 trial 0, message 58 to 61, after
   new tool results), but that deny was the false block. Task 44 trial 1's
   forbidden cancel was allowed outright, not retried past a deny.

### Caveats

* Partial data: G1 is missing a whole trial. pass^3 covers 17 tasks, pass^4 is
  not available, and the tasks with the largest effects have only 2 trials.
* The judge prompt was tuned after seeing test-split runs (tuning tasks above).
* All audit verdicts and the "saved" counterfactuals come from LLM agents of
  one model family; they were not human-checked or re-executed, and correlated
  errors are possible.
* The agent and the judge are the same model; Qwen3-8B has not been run.
* Labels against gold are a proxy: a non-gold write can be harmless, and the
  audits, not the labels, decide what was a violation.

### Cost

This run cost about $8.4: G0 $4.19 for 77 sims; G1 $3.40 agent and user plus
$0.77 judge for 57 sims. The 26 missing sims would cost about $2.
