# sample200

从 offline_0901 的 1,900 题里分层抽出的 200 题，三个模型跑同一个 runtime、同一批题、
同一个记分脚本，结果都在这个目录里。页面是 `results.html`。

    gpt-5.4         0.6902    准确率 0.5878
    gpt-5.6-sol     0.7243    准确率 0.6959
    qwen3.5-397b    0.6786    准确率 0.6014
    194 题配对分母，标准误 0.0202

## 为什么有三个臂

原本只有 gpt-5.6-sol 和 qwen3.5-397b 两个臂。gpt-5.6-sol 的训练快照是 2026-07-09，
这 200 题里有 123 题在那之前就已结算，它的分数里可能有一部分来自记忆而不是预测。

判断这件事需要一个证明记不住答案的参照物。gpt-5.4 的快照是 2026-03-05，比题集窗口的
第一天还早 27 天，题集里 0 题落在它的快照内。它的成绩从 1,900 题全量运行里按这 200 个
task_id 取出来，构成第三个臂。

## 结论

gpt-5.6-sol 在快照覆盖的那一半上确实分更高，但 gpt-5.4 也高，而且高出 0.1557。
它记不住任何一题，所以"前半分数更高"是题目的性质，不是记忆的证据。

扣掉这个共同的难度差之后，gpt-5.6-sol 相对 gpt-5.4 的领先在两半之间只变化 +0.0298，
95% 区间 −0.0560 到 +0.1156，跨过 0。**这批题量既不能证实也不能排除参数化泄漏**，
它只排除了那条最容易被误读的证据。要把分辨力压到 0.03，需要约 1,500 道配对题。
更省的路是闭卷探测：同样这两半题，不给检索工具，前后半的准确率差就是泄漏本身。

真正拉开的是工作量。gpt-5.6-sol 单题 166 秒、9 轮、11.5 次工具调用，gpt-5.4 是
560 秒、21 轮、27 次，qwen3.5-397b 是 541 秒、16 轮、18 次。而且在更难的后半，
另外两个臂都明显加大投入，gpt-5.6-sol 几乎不动。

## 文件

| 文件 | 内容 |
|---|---|
| `results.html` | 三臂对比的完整页面，八节 |
| `metrics.json` | 三个臂的 own 与 paired 统计、run_shape |
| `per_task_scores.jsonl` | 每臂每题一行 |

工作量数字不在 `metrics.json` 的分位数里，来自三个运行目录各自的 `task_summary.jsonl`，
页面按 194 题的子集重算。

## 重新生成

```bash
E="env -u PYTHONPATH PYTHONNOUSERSITE=1"; V=".venv-test/bin/python"
D=data/benchmarks/offline_0901/sample200/offline_forecast_sample200.jsonl

$E $V scripts/eval/score_offline.py --ground-truth $D \
  --run-dir /tmp/axis_export/axis_strict_full   --label axis_gpt54 \
  --run-dir /tmp/axis_export/sample200_gpt56    --label axis_gpt56_sol \
  --run-dir /tmp/axis_export/sample200_qwen     --label axis_qwen35_397b \
  --out reports/offline_0901/sample200/metrics.json \
  --per-task reports/offline_0901/sample200/per_task_scores.jsonl

$E $V scripts/eval/render_sample200_report.py \
  --report-dir reports/offline_0901/sample200 --ground-truth $D
```

题集本身由 `scripts/benchmark/sample_offline_subset.py --n 200 --seed 20260908` 生成，
它同时写出 `sample200/labels.jsonl`。记分脚本按数据集同目录查找这个文件，没有它，
`by_subarea` 和 `by_scoring_group` 会静默为空。

## 这一页控制不了什么

gpt-5.4 那一臂的 config 不同：temperature 0.2、top_p 0.95、reasoning_effort medium、
max_output_tokens 12800、抓取失败会重试 5xx。所以它作为"快照干净"的参照物可靠，
快照日期与 config 无关；把它的分数或工作量直接当第三个模型的成绩读不可靠。

qwen3.5-397b 的快照未知。网关回显 `Qwen3.5-397B-A17B`，不带日期，模型自报的训练截止
不能当证据。它在参数化泄漏这一项上没有证据也没有反证。
