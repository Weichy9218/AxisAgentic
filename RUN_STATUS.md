> **这份文档描述的是 `axis_strict_full`，该运行作为分数来源已经退役。**
> 它跑在 `temperature 0.2` / `reasoning_effort medium` / 摘要模型 `deepseek-v4-pro` 下，
> 并且 19.3% 的题触到了当时 50 的工具上限，四处设置与现在不同。文中的分数不可与新结果并排。
> 保留它是因为里面的运行史、网关基准和故障分析都是实测，而且 `site:` 归因的原始材料出自这次运行。
> 当前结果看 `reports/README.md` 与 `reports/sample200_overview.html`。

# AxisAgentic on offline_0901 — current state

The run finished 2026-09-08 20:14. All 1,969 tasks completed; the report is at
`Milkyway-Harness-Agent/reports/offline_0901/axisagentic_gpt54_20260908/`, whose
README is the place to read results. This file keeps the operational history:
what broke, what was changed and why, and what the numbers below were measured on.

    axisagentic    0.7206   accuracy 0.6834   1,899 scored · 1 excluded
    serper_gate3   0.6557   accuracy 0.5317   1,900 scored · 0 excluded

## Where the run is

Four shards, `logs/offline_forecast/axis_strict_full/shard{1..4}/`, finalized
2026-09-08 at 493 + 492 + 492 + 492 = **1,969 rows**. Logs at
`/tmp/full_shard{1..4}.log`. Sustained rate after the 10:29 resume was ~440
tasks/h; the run-wide average is lower because it spans a five-hour crash window.

## The crash, and the fix

All four shards died together between 05:05 and 05:15 with
``OSError: [Errno 36] File name too long``. `BatchEvaluator.log_result` used the
raw `task_id` as a filename, and some FutureX rows carry a 190-character
percent-encoded JSON query string as their id. Only this dataset has ids of that
shape, which is why nothing caught it earlier.

Fixed in `agentic/evaluation/evaluator.py::_safe_filename_stem`: sanitize, cap at
200 bytes, append a digest of the full id so two long ids sharing a prefix cannot
collide. The complete id is still written inside the record — the filename is an
index, not the data. Pinned by `tests/test_eval_result_filenames.py`.

## Speed: where the time actually goes

Mid-run, over 1161 finished traces, per task and serial:

    scrape            431s   65.3%     p50 163s, p90 1309s  <- heavily right-skewed
    web_search        122s   18.4%
    model inference   107s   16.2%     19.8 turns x 5.4s

Over the finished run the shares settle at scrape 58.0%, search 14.5%, inference
and orchestration 27.5%, on 809 s per task (p50 538, p90 1,890) against the
reference arm's 208 s — 124 tasks/h against 530.

**The bottleneck is page fetching, not the model.** Tuning concurrency or moving
to a faster model gateway only touches the 16%. Levers worth applying to the
*next* run, in order:

1. **Cap the scrape timeout.** p90 is 22 minutes for one page, which buys
   nothing. A 60s ceiling removes most of the tail. Costs: some slow-but-valid
   pages become failures.
2. **Stop retrying scrape failures to exhaustion.** A Serper 500 currently
   retries four times and rarely succeeds on retry.
3. **`judge_page_prose: false`** would cut ~46 judge requests per task (the page
   channel averages 4.3 chunks per page). Not recommended: it weakens the filter
   and breaks alignment with the comparison arm. If speed ever forces it, the
   whole run must use the same setting — a run half-filtered one way and half the
   other is not one dataset.

They were not applied mid-run: with under two hours left, a restart would have
discarded the ~32 in-flight tasks and re-filled the pipeline for less than it
saved.

## What changed and why (all measured, not guessed)

**Gateway zgci -> yihui.** Same model (`gpt-5.4`), same `reasoning_effort:
medium`, same prompt shape. Benchmarked side by side:

    gateway   c=8 p50   c=16 p50   c=32 sustained
    yihui     4.32s     4.44s      2.62s p50, 5.10 calls/s, 0 failures over 128
    zgci      6.92s     7.83s      12.1s p95 at 16-way

yihui is also the gateway the reference arm used, so this removes a variable
rather than adding one. `GPT_sub2api` remains dead (four keys, 502/429/503, zero
successes across three separate probes hours apart).

**2 shards -> 4, concurrency 8 each.** Matches the reference arm's 4 x 8 = 32.
The 32-way sustained check above is what says one yihui key carries it.

**Tool budget 50.** The number was never the thing to match. Measured over the
reference arm's own 1966 records: mean 14.4, p50 12, p90 26, max 53, and its
declared cap of 30 binds on 3.2% of tasks. A cap of 30 here bound 59%, so the
same integer described two different experiments. Against the smoke run (the
only observation of this agent with no tool budget) a cap of 50 leaves 87%
unbound, the same regime. `max_turns` went to 70 so turns do not silently become
the new binding constraint.

**Serper date normalization** (`agentic/tools/as_of/provider_dates.py`). Gate 1's
parser is ISO-first; Serper serves `"Jun 17, 2026"` and `"1 week ago"`. Without
normalization every English-dated card looked undated and cost a judge call.
After: `blocked_provider_date` per search call 0.00 -> 0.24, judge requests 2.07
-> 1.26 (-39%), same boundary enforced. Relative forms deliberately stay undated:
"1 week ago" spans 5-9 days, and resolving it against a clock near a boundary
manufactures a false `known_before`.

## Filtering alignment with galaxy

The requirement is that both systems filter the same way. Audited on 630 tasks:

**Gate 1 is correct.** Three independent checks:

    blocks              3.80 per task   (reference arm: 1.66)
    cards inspected     55,884
    post-cutoff cards that reached the model:  0

The higher rate is expected — `judge_page_prose: true` means the page channel
blocks too. 66.3% of cards now resolve to a day; 33% are formats the parser does
not know and fall through to Gate 3, which is safe but pays a judge call. That
residue is the next thing worth optimizing.

**Gate 3 is meaningful and not replaceable by a date rule.** Replayed on a
controlled mixed set of 13 passages, 13/13 matched intent. Three results are the
ones that matter:

* It blocked three passages that carry **no date at all** ("the committee
  decided to lower the rate", "the incumbent has conceded"). It returned
  `unknown` — settled but undatable — which blocks. No regex can see these.
* It passed a schedule naming a post-cutoff date ("decision on 2026-07-06") by
  returning `null`. A date rule would block it, and over-blocking that class is
  what moved directional accuracy 0.933 -> 0.667 upstream.
* It pushed period claims to the period's end: "Q2 revenue" -> `2026-06-30`,
  "July 2026 CPI" -> `2026-07-31`, neither being a date present in the text.

Reproduce with `/tmp/eval_gate3.py`.

## A judge outage blocked evidence on ~11% of questions, in four clock hours

The Milkyway README gates on this before a score may be quoted:
`blocked_judge_unavailable` must be 0. Ours is not. Final counts, over the whole
run, in three scopes because three pages count it three ways:

    8,189 units / 217 traces    both channels, all 1,951 traces
    7,965 units / 211 questions both channels, the 1,882 with a notebook
    4,629 units / 194 questions search channel only — what leakage.html reports
       17 units /  11 questions galaxy, for scale

It is a bounded outage, not a systemic one. 7,909 of the 7,965 (99.3%) fall in
four clock hours: 2026-09-08 02, 03, 12 and 13. Every other hour is single digits.

`screen.py` fails closed: a unit the judge did not return a verdict for is
blocked, not passed. That is the right default and it is why nothing leaked, but
it means those questions were answered with evidence withheld for a reason
unrelated to the evidence. The affected subset scored **0.6729** against
**0.7265** for the rest, so removing it would move the headline by about +0.006
— it does not explain the +0.065 gap, and it is reported rather than netted out.

`audit_time_filtering.py` *can* now answer this for our arm: `--emit-notebook`
writes the tool results into `<task>/notebook/art-NNNN.search.json`, so galaxy's
unmodified audit and leakage scripts run on the export. The earlier note here
saying they could not is superseded. The audit exits 1 on this run, and the FAIL
is this outage.

## Where the results are

Built 2026-09-08 by `/tmp/axis_build_report.py` and `/tmp/axis_finish_report.py`.
The report folder is in the Milkyway repository, beside the arm it is compared
against, because the pages are rendered by that repository's scripts:

    reports/offline_0901/axisagentic_gpt54_20260908/
        README.md               what the numbers are and what they do not control
        metrics.json            this arm alone, scored by score_offline.py
        per_task_scores.jsonl
        results.html            by type, calibration, effort
        leakage.html            galaxy's renderer, on the notebook mirror
        leakage_native.html     our own renderer, on the run's 61 MB of tool output
        cohort.html             7 areas / 65 subareas, paired with serper_gate3
        runtime.html            how much work each side did
        run_shape.json  judge_verdicts.json  audit_time_filtering.txt

    reports/offline_0901/compare_axisagentic_galaxy/metrics.json   the paired block

That README carries the exact commands. Four facts about the build worth keeping
here because they are properties of *this* runtime, not of the report:

* **`--emit-notebook`** is what makes galaxy's leakage and audit scripts run on
  our export. Without it they find no notebook directories and report a clean
  zero from having found nothing.
* **The export normalises LaTeX in stored answers.** A run that finished before
  `normalize_boxed_answer` existed still has `\boxed{BK\ Hacken}` on disk. Four
  answers in this run; without the fix they score 0 as `output_not_in_options`.
* **Verification gate passed**: 1,900 rows exported, all `usable`, 0 boundary
  failures, 0 unstamped live results, `n_missing_from_dataset` 69 (the pruned
  numeric rows, expected). `n_excluded` is 1, not 0 — task `UlhphdcuPz` submitted
  `$1.75B - $2T` against the option `$1.75T - $2T`. That is a model typo, left
  unrepaired. galaxy has 0 because its `submit_forecast` tool validates against
  the option list and forces a retry; this agent accepts whatever is in the box.
* **18 of 1,969 traces were lost** to the long-filename `OSError` before the fix
  landed, so 1,882 of the 1,900 scored questions have a notebook. Scores are
  unaffected (they come from `benchmark_results.jsonl`); only the leakage scan
  loses those 18.

## The ground truth moved on 2026-09-08, and so did the numeric rule

Both changes landed upstream while this run was in flight. Neither is optional
and neither is selected by a flag; the committed scorer is the rule.

`offline_forecast_20260401_20260829.jsonl` was pruned at 15:33 from 1969 rows to
1900. The 69 removed rows are numeric tasks whose price series could not be
recovered (`reason: history_unavailable`, ledger in `pruned_numeric_ledger.json`)
and they now live in `.pruned.jsonl`. Our shards still run all 1969, so the
exporter drops those 69 and the scorer's `n_run_tasks_outside_ground_truth`
should be 0. Any earlier number carrying `n_ground_truth: 1969` cannot be placed
beside a number from the 1900-row file.

Milkyway commit f1ed6c8 replaced the asinh integrated Brier for NUMERIC with the
paper's Eq. 13, `s = max(0, 1 - ((v_hat - v) / (3 sigma(V) + eps))^2)`, where V
is the target-day value plus the seven most recent historical values. This is
what made the 69 history-less tasks unscoreable, which is why they were pruned.
It also removes the saturation problem noted below: on the galaxy arm the same
run goes from a numeric score near 0.98 under asinh to ~0.51 under Eq. 13, and
overall from 0.7752 to 0.6557.

**galaxy baseline on the current file and rule.** An earlier note here read
`scored 1897 / excluded 3 (run_missing_task) / 0.6566`; that was measured against
an export missing three directories. Final, on the full 1,900-row file:

    scored 1900   excluded 0   score 0.6557   acc 0.5317

Paired against AxisAgentic on the 1,899 both answered:

    BINARY 0.8045 (n=689)   MULTIPLE_CHOICE 0.6087 (n=761)   NUMERIC 0.5071 (n=449)


## What must be stated in any report

These are real asymmetries, not noise, and a score difference that ignores them
will be misread as runtime quality.

* **Tool inventory.** galaxy has 12 tools including `read_artifact` (2927 calls),
  `read`/`grep`/`find`/`bash` (1646), and `fit_timeseries_forecast` (122, a
  deterministic fit for NUMERIC questions). AxisAgentic has 2: `web_search` and
  `scrape_and_extract_info`. galaxy can re-read evidence it already fetched;
  this agent must search again.
* **Model calls per task.** Final, from both runs' artifacts: galaxy 8.4 LLM
  calls and 17.0 tool calls; AxisAgentic 20.5 turns and 28.8 tool calls. The gap
  is mostly the point above, not reasoning quality.
* **Prompt caching.** galaxy reports **50,996** cached tokens per task, 46% of
  its input. (An earlier note here said 76,390 / 49%; wrong field.) AxisAgentic
  does record token usage — see the gaps section — but not a cache field, so the
  cost comparison stops at input and output totals: 217,988 / 9,862 per task
  against galaxy's 109,964 / 1,722.
* **Retrieval source.** Both Serper, so this one *is* aligned. The reference
  arm's published 0.81268 was an Exa run and is not the right comparison; the
  serper run at `log/offline/20260907_dt7_1969_serper_gpt54` is.
* **NUMERIC scale insensitivity.** This applied to the asinh integrated Brier at
  `scale=1.0`, where predicting 100000 against a true 50000 scored 0.999995 and
  80% of numeric tasks scored above 0.99. Eq. 13 replaced that rule on
  2026-09-08; the saturation is gone and numeric now lands near 0.51 on galaxy.
  Still report `by_type`, never the headline alone.
* **galaxy's 17 `judge_unavailable` blocks** over 11 tasks (0.56%) from a 33s
  judge outage, against our 7,965 units over 211 questions. Report both; the
  numbers and the hour-by-hour breakdown are in the outage section above.

## Known gaps in my own analysis

* The token claim above is now corrected, not open: AxisAgentic traces **do**
  carry `token_usage` = `{input_tokens, output_tokens, total_tokens}`. My earlier
  note that the field was unreadable was a wrong field name on my side.
* The tool-call comparison I first reported ("reference arm uses 12") came from a
  single sample row. The real distribution is mean 14.4 / p90 26 / max 53. Any
  earlier note repeating the 12 figure as the distribution is wrong.
* The "cap of 30 binds on 3.2% of tasks" figure above counted galaxy tasks with
  ≥30 tool calls. galaxy's own `forced_tool_cap` flag is false on **all 1,969**,
  so 30 never actually truncated a task there. On our side the budget of 50 is a
  live constraint: 333 of the 1,900 scored questions (17.5%) reach it, counted
  from the tool calls themselves because this exporter does not populate a
  `forced_tool_cap` field.
* The residual post-cutoff dates differ by 13 occurrences between the mirror scan
  (294, 156 questions) and the native scan (281, 148 questions). The two corpora
  are not byte-identical and the 13 have not been attributed one by one. Both are
  upper bounds; the page published as comparable to galaxy's is the larger.
