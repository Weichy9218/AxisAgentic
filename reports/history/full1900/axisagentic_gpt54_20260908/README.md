# axisagentic_gpt54_20260908

AxisAgentic answering the same 1,900 resolved questions as `serper_gpt54_20260907`,
under the same information boundary, scored by the same scorer.

    run          AxisAgentic/logs/offline_forecast/axis_strict_full  (4 shards × 8 concurrent)
    export       /tmp/axis_export/axis_strict_full                   (galaxy-shaped run dir)
    model        gpt-5.4 · as-of judge alice/gpt-5.4-mini
    provider     serper · temporal_policy strict · ΔT 7d · observation_time 2026-09-01
    tool budget  50 per task

Nothing here is scored by AxisAgentic's own code. The run is exported into the
shape `scripts/eval/score_offline.py` reads, and every page below is produced by
the same unmodified script that produced the reference arm's page. A comparison
where each side computes its own score measures the scorers as much as the
agents.

## Which page answers which question

| Page | Question |
|---|---|
| `results.html` | What did this arm score, by type, calibration and effort? |
| `leakage.html` | What did the filtering achieve, **measured the way the reference arm is measured**? |
| `leakage_native.html` | The same question, measured from **this runtime's own artifacts** |
| `cohort.html` | How does it compare with `serper_gate3` across 7 areas and 65 subareas? |
| `runtime.html` | **How much work** did each side do to reach its score? |
| `audit_time_filtering.txt` | The five-stage boundary audit, run on this export |

There are two leakage pages because there are two honest ways to ask the
question and they are worth having side by side. `leakage.html` comes from
`render_leakage_report.py` unmodified — the same script, on the same kind of
input, as the reference arm's page — so its numbers are comparable to galaxy's
by construction. `leakage_native.html` comes from `render_axis_leakage_report.py`,
which scans the `content` field of all 53,101 tool calls: 61 MB of exactly the
characters that entered the model's context. The first is comparable; the second
is closer to the thing being audited. They agree (below), which is the point.

`runtime.html` has no counterpart in the reference folder. It exists because the
two runtimes are not the same shape — different tool inventories, 28.8 tool calls
per task against 17.0, 809 seconds per task against 208 — and without that page
the score gap reads as a difference in reasoning quality, which is not where most
of it comes from.

The mechanism itself — how the boundary is encoded provider-side, what the two
arithmetic gates test, which model the judge uses and with what prompt — is in
`../filtering.html`, generated from the implementation and shared by both arms.

## Headline

    axisagentic     0.7206    accuracy 0.6834    1,899 scored · 1 excluded
    serper_gate3    0.6557    accuracy 0.5317    1,900 scored · 0 excluded

Paired on the 1,899 both answered:

| Type | n | axisagentic | serper_gate3 | Δ |
|---|---|---|---|---|
| BINARY | 689 | 0.8397 | 0.8045 | +0.0352 |
| MULTIPLE_CHOICE | 761 | 0.7055 | 0.6087 | +0.0969 |
| NUMERIC | 449 | 0.5632 | 0.5071 | +0.0560 |
| **all** | **1,899** | **0.7206** | **0.6557** | **+0.0649** |

The gap holds in all three types and in all seven areas, so it is not one type
carrying the mean. It is largest on MULTIPLE_CHOICE, which is where the scoring
rule has the most room: a choice score is `1 − BS/2` over the whole imputed
option vector, so a better-calibrated probability moves it even when the chosen
option is the same.

**What this pair does and does not control.** Both arms use serper, both run the
three-gate as-of screen, both use gpt-5.4 at ΔT=7. What differs is the runtime:
tool inventory, context management, tool budget, and — see below — how much
evidence each one lost to a judge outage. This is a runtime comparison, not a
controlled ablation of any single design choice.

### Calibration

Under-confident across the range and over-confident at the top: claimed 0.255 →
observed 0.376, claimed 0.436 → 0.535, claimed 0.960 → 0.871. Binarised Brier
0.1940 over 1,450 discrete questions. `results.html` has the full curve; the
Brier column is a diagnostic, not the headline.

### The one excluded question

`UlhphdcuPz` (MULTIPLE_CHOICE, SpaceX IPO closing market cap) submitted
`$1.75B - $2T` against an option list reading `$1.75T - $2T`. The model wrote B
where the option says T. That is the model's answer, not a bridge defect, and it
is left excluded rather than repaired — `n_excluded` is the count the validity
principle exists to protect.

It is worth naming why the reference arm has zero: galaxy submits through a
`submit_forecast` tool that validates the string against the option list and
makes the model try again. AxisAgentic accepts whatever is in the box. One
question out of 1,900 is the size of that difference here.

## Filtering, on this arm's evidence

    0 unscreened search results            33,454 packets, every one stamped
    170,871 units screened, 144,661 kept   84.7%
    Gate 1 provider-stated date   12,252
    Gate 2 list-shaped card          124
    Gate 3 small-model verdict     9,205
    2,745 post-cutoff dates found, 294 of them unexplained, across 156 questions (8.2%)

Beside the reference arm:

| | axisagentic | serper_gate3 |
|---|---|---|
| search packets scanned | 33,454 | 14,090 |
| units screened / kept | 170,871 / 144,661 (84.7%) | 74,101 / 68,021 (91.8%) |
| Gate 1 / 2 / 3 | 12,252 / 124 / 9,205 | 3,291 / 44 / 2,728 |
| post-cutoff dates, total | 2,745 | 1,497 |
| — the model's own note | 0 | 519 (34.7%) |
| — the question's own horizon | 2,278 (83.0%) | 211 (14.1%) |
| — document clock or schedule | 38 (1.4%) | 609 (40.7%) |
| — a date inside a URL slug | 135 (4.9%) | 52 (3.5%) |
| — **residual (upper bound)** | **294 (10.7%), 156 questions** | **106 (7.1%), 50 questions** |

**The two residual rates are not directly comparable, and the difference is in
our favour on the strictness axis and against us on the volume axis.** galaxy's
notebook holds artifacts *written to a workspace* — what the agent chose to keep.
This export's notebook holds the tool results themselves: exactly the bytes that
entered the model's context, including everything it read once and ignored. That
is the stricter corpus for a leakage scan, and it is 2.4× larger, which is most
of why 8.2% of questions carry a residual date against 2.5%. The zero in "the
model's own note" is the same fact from the other side: AxisAgentic has no
note-taking tool, so there is no channel in which the model can write a date of
its own.

What the residual is and is not: a post-cutoff date is not a leak, and a leak
need not carry a date. The classification is deliberately conservative — only
structurally certain document clocks and schedules are excluded, anything
ambiguous stays in the residual — so it is an upper bound on the date-visible
part, not a leak rate.

### The two scans agree, which is what validates the mirror

`leakage.html` reads a notebook mirror written by the exporter; `leakage_native.html`
reads the run directly and shares no code with it past the date parser. If the
mirror dropped or duplicated evidence, the two would diverge.

| | via the mirror | native scan |
|---|---|---|
| questions covered | 1,882 | 1,882 |
| search-channel units / kept | 170,871 / 144,661 | 170,871 / 144,661 |
| residual post-cutoff dates | 294, on 156 questions (8.2%) | 281, on 148 questions (7.9%) |

The search channel reproduces exactly: same denominator, same kept count. The
residual differs by 13 occurrences out of 294. The two corpora are not
byte-identical — the mirror preserves each search result as its JSON packet,
the native scan reads the `content` string as rendered into the conversation —
and the 13 have not been attributed one by one. Both figures are upper bounds on
the same quantity, and the page published as comparable to galaxy's is the larger
of the two.

**One independent check the native page can run and the mirror cannot.** It
re-parses the publication date of every card that reached context, using the
gate's own parser (`agentic/tools/as_of/provider_dates.py`) rather than a second
implementation that would disagree somewhere and render each disagreement as a
leak. Of 104,403 cards that resolve to a specific day, **0** are dated at or
after their question's `T_cut`. 38,176 more carry a date field in no known
format and 2,082 are relative ("1 week ago", deliberately not resolved to a day);
both fall through to Gate 3 rather than being trusted.

### Judge verdicts: `leakage.html` shows galaxy's store, not ours

`render_leakage_report.py` reads the verdict store at `ROOT/.cache/as_of_verdicts`
where ROOT is the repository the script lives in. Run against this export it
still reads Milkyway's store, so the "小模型这道门做了什么" section of
`leakage.html` describes **galaxy's** judge. `leakage_native.html` and
`runtime.html` read ours; `judge_verdicts.json` carries both, produced by
importing their `judge_verdicts()` function and pointing it at each cache in turn.

| verdict | axisagentic | serper_gate3 |
|---|---|---|
| `null` — nothing settled, pass | 58,640 (60.6%) | 12,558 (70.1%) |
| a date — compared in code | 32,005 (33.1%) | 4,501 (25.1%) |
| `unknown` — settled, undatable, block | 6,153 (6.4%) | 862 (4.8%) |
| total | 96,798 | 17,921 |

Same judge model, same prompt, and `T_cut` appears in neither — the comparison
stays in code. The store is content-addressed and shared across runs, so these
are verdict *distributions*, a superset of any single run's blocks.

## The defect that belongs beside the score

    axisagentic    7,965 units blocked on 211 questions   (both channels, 1,882 scanned)
                   4,629 units blocked on 194 questions   (search channel only, as leakage.html counts it)
    serper_gate3      17 units blocked on  11 questions

The as-of screen is fail-closed: a unit sent to the judge whose verdict does not
come back is blocked. That is the correct default and it is why nothing reached
context past that path, but it means those questions were answered with evidence
removed for a reason unrelated to the evidence. Ours is two orders of magnitude
worse than the reference arm's.

It is a bounded outage rather than a systemic one. 7,909 of the 7,965 blocks
(99.3%) fall in four clock hours — 2026-09-08 02:00, 03:00, 12:00 and 13:00 —
and every other hour of the run is in single digits. `leakage_native.html` has
the hour-by-hour table.

The affected subset scored **0.6729** against **0.7265** for the rest — 0.054
below, on 217 of 1,951 traces. Removing it would move the headline by roughly
+0.006, so it does not explain the +0.065 gap, but it is a run defect and it is
reported rather than netted out. `audit_time_filtering.txt` ends in FAIL for
exactly this reason; the exit status is recorded at the bottom of that file.

## Two things the shared scripts cannot say about this run

**Audit stage 1 reports zero and means "not applicable."** Stage 1 asks whether
the runtime put `T_cut` on each `search_web` call by reading tool calls out of
`<task>/main_agent.jsonl`. This export writes no agent transcript — that would
mean fabricating another runtime's private format — so stage 1 reads 0 calls.
The question it asks is answered instead by stages 2–4, which read the packets:
33,454 of 33,454 carry a compiled provider bound, 0 name another task's `T_cut`,
0 carry a lower bound on the wire, and 0 cards dated at or after their own
`T_cut` reached context.

**18 questions are invisible to the leakage scan.** The trace writer hit
`OSError: File name too long` on 18 of 1,969 tasks (percent-encoded JSON task ids
past the 255-byte path limit), so 1,882 of the 1,900 scored questions have a
notebook. Their scores are unaffected — those come from `benchmark_results.jsonl`
— but their evidence is not in the 33,454 packets above. Directory names are
truncated at exactly 255 bytes, the way galaxy truncates them, so both arms'
scans cover the same task set.

## Effort

|  | axisagentic | serper_gate3 |
|---|---|---|
| tool calls / task | 28.8 (p50 26, p90 50, max 51) | 17.0 |
| model calls / task | 20.5 turns | 8.4 LLM calls |
| wall clock / task | 809 s (p50 538, p90 1,890) | 208 s |
| throughput, measured | 124 tasks/h | 530 tasks/h |
| input tokens / task | 217,988 | 109,964 |
| output tokens / task | 9,862 | 1,722 |
| tool budget bound | 333 of 1,900 scored (17.5%) hit 50 | `forced_tool_cap` false on all 1,969 |

The last row is the one asymmetry most likely to be misread. The same integer
means different things on the two sides: galaxy declares 30 and its own
`forced_tool_cap` flag is true on **zero** of 1,969 questions, so 30 is a safety
net there. AxisAgentic declares 50 and roughly one question in six reaches it, so
50 is a live constraint here. (`runtime.html` reports 376 of 1,951 — the same
count over all traces including the 69 whose questions were pruned from the
scored set. And `results.html` shows "被迫在上限提交 0 题" for this arm: that
reads a `forced_tool_cap` field this exporter does not populate, so it is an
absent field, not a measurement. The 333 above is counted from the tool calls
themselves.)

Scraping is 58.0% of per-task time, search 14.5%, model inference and
orchestration the remaining 27.5%. The input-token gap is a context-management
difference, not a longer prompt: galaxy writes retrieved pages into a workspace
and reads them back with `read_artifact`, compressing at 128k characters, while
AxisAgentic leaves every tool result in the conversation. That one choice
explains both the turn count and the token count.

`runtime.html` derives all of these from each run's own artifacts and states
where a field is missing rather than estimating it — the prompt-cache figure is
recorded by galaxy (50,996 tokens/task, 46% of its input) and not by AxisAgentic,
so the cost comparison stops at input and output totals.

## Regenerating

```bash
A=/home/dataset-local/wcy/AxisAgentic
M=/home/dataset-local/wcy/Milkyway-Harness-Agent
E="env -u PYTHONPATH PYTHONNOUSERSITE=1"; V="$M/.venv-test/bin/python"
D=$M/data/benchmarks/offline_0901/offline_forecast_20260401_20260829.jsonl
X=/tmp/axis_export/axis_strict_full
R=$M/reports/offline_0901/axisagentic_gpt54_20260908

# 1. export the run into the shape the scorer reads, with the notebook mirror
#    the leakage scan needs. Exits non-zero if any live retrieval reached the
#    model without the screen having run against that task's T_cut.
cd $A && source .envs/axis_agentic_env.sh
python3 -m recipe.web_search.runners.export_galaxy_run_dir \
  --run-dir $A/logs/offline_forecast/axis_strict_full --data-path $D --out-dir $X \
  --temporal-policy strict --delta-days 7 --observation-time 2026-09-01 \
  --search-provider serper --max-tool-calls 50 --emit-notebook

# 2. score this arm on its own; results.html and leakage.html read this file
cd $M
$E $V scripts/eval/score_offline.py --ground-truth $D \
  --run-dir $X --label axisagentic \
  --out $R/metrics.json --per-task $R/per_task_scores.jsonl

$E $V scripts/eval/render_metrics_html.py --metrics $R/metrics.json --arm 0 \
  --out $R/results.html
$E $V scripts/eval/render_leakage_report.py --run-dir $X \
  --metrics $R/metrics.json --out $R/leakage.html

# 2b. the same question asked of the run itself, sharing no code with the mirror
$E $V scripts/eval/render_axis_leakage_report.py \
  --run-dir $A/logs/offline_forecast/axis_strict_full --export-dir $X \
  --metrics $R/metrics.json --out $R/leakage_native.html

# 3. the pair; cohort.html needs a two-arm metrics file, not the one above
$E $V scripts/eval/score_offline.py --ground-truth $D \
  --run-dir $X                                        --label axisagentic \
  --run-dir $M/log/offline/20260907_dt7_1969_serper_gpt54 --label serper_gate3 \
  --out $M/reports/offline_0901/compare_axisagentic_galaxy/metrics.json \
  --per-task $M/reports/offline_0901/compare_axisagentic_galaxy/per_task_scores.jsonl

$E $V scripts/eval/render_cohort_report.py \
  --metrics $M/reports/offline_0901/compare_axisagentic_galaxy/metrics.json \
  --out $R/cohort.html

# 4. our own judge store, and the run-shape page
$E $V /tmp/axis_judge_store.py $R/judge_verdicts.json
$E $V /tmp/render_axis_runtime.py --axis-run $A/logs/offline_forecast/axis_strict_full \
  --metrics $R/metrics.json --per-task $R/per_task_scores.jsonl \
  --ref-run $M/log/offline/20260907_dt7_1969_serper_gpt54 \
  --judge-store $R/judge_verdicts.json --out $R/runtime.html \
  --label axisagentic --max-tool-calls 50

# 5. the boundary audit. Exits 1 on a FAIL verdict; on this run the FAIL is the
#    judge outage above, which is a finding about the run, so the output is kept.
$E $V scripts/eval/audit_time_filtering.py $X > $R/audit_time_filtering.txt
```

Before quoting a number from a re-run, read stage 3 of the audit:
`blocked_judge_unavailable` must be 0. It is not 0 here, and the size of it is
reported above rather than left in the file.

## What is not committed

This directory is gitignored, as is `log/`, `/tmp/axis_export/`, and the
AxisAgentic run it scores. The scripts that build these pages are tracked; the
pages are not. Two of the five are not yet checked in anywhere:
`scripts/eval/render_axis_leakage_report.py` is untracked in this repository, and
`/tmp/axis_judge_store.py` and `/tmp/render_axis_runtime.py` have no home in
either repository. The exporter that feeds all of them,
`recipe/web_search/runners/export_galaxy_run_dir.py`, is tracked in AxisAgentic.
