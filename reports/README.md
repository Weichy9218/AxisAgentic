# AxisAgentic 在 offline_0901 上的全部运行

每一行都是从该次运行自己的 `run_metadata.json` 和记分脚本的 `metrics.json` 里读出来的，
不是手写的。新增一臂或重新记分之后跑 `python3 build_index.py` 即可更新。

记分一律用 Milkyway 的 `scripts/eval/score_offline.py`，两个 runtime 共用一个实现，
所以分数之间可比；galaxy 自己的报告留在 Milkyway 仓库，跨 runtime 的对比也在那边。

## 总表

| 运行 | 模型 | 数据 | skill | fit 工具 | 分数 | 准确率 | 记分/剔除 |
|---|---|---|---|---|---|---|---|
| `axis_strict_full` | gpt-5.4 | offline_0901 全量 1,969 题 | — | — | 0.7206 | 0.6834 | 1899 / 1 |
| `s2_d_skill2` | gpt-5.6-sol | S2 · 200 题分层抽样，与 S1 不相交（seed 20260910） | forecast_v2.md | — | 0.7123 | 0.7451 | 200 / 0 |
| `s2_e_tool` | gpt-5.6-sol | S2 · 200 题分层抽样，与 S1 不相交（seed 20260910） | — | 是 | 0.7214 | 0.6993 | 200 / 0 |
| `s2_f_skill2tool` | gpt-5.6-sol | S2 · 200 题分层抽样，与 S1 不相交（seed 20260910） | forecast_v2.md | 是 | 0.7275 | 0.7386 | 200 / 0 |
| `s2_noskill` | gpt-5.6-sol | S2 · 200 题分层抽样，与 S1 不相交（seed 20260910） | — | — | 0.7063 | 0.7059 | 200 / 0 |
| `s2_skill` | gpt-5.6-sol | S2 · 200 题分层抽样，与 S1 不相交（seed 20260910） | forecast_v1.md | — | 0.7109 | 0.7516 | 200 / 0 |
| `sample200_gpt56` | gpt-5.6-sol | S1 · 200 题分层抽样（seed 20260908） | — | — | 0.7239 | 0.6863 | 200 / 0 |
| `sample200_gpt56_siteretry` | gpt-5.6-sol | S1 · 200 题分层抽样（seed 20260908） | — | — | 0.7324 | 0.6405 | 200 / 0 |
| `sample200_qwen` | qwen3.5-397b-a17b | S1 · 200 题分层抽样（seed 20260908） | — | — | 0.6801 | 0.6040 | 195 / 5 |

## 逐个运行

### `axis_strict_full`

第一次全量：AxisAgentic 对 galaxy 的 runtime 对比

```
模型            gpt-5.4
数据            offline_0901 全量 1,969 题
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/offline_forecast_shard1.jsonl
轨迹            1951 条
采样            temperature=0.2  top_p=0.95  reasoning_effort=medium
预算            max_output_tokens=12800  max_tool_calls=50  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-pro
skill           无
fit 工具        未启用
并发            8 / shard
跑于            2026-09-08 10:29
```

```
分数            0.7206      准确率 0.6834
记分 / 剔除     1899 / 1
分题型          BINARY 0.8397   MULTIPLE_CHOICE 0.7055   NUMERIC 0.5632
报告            reports/offline_0901/axisagentic_gpt54_20260908/
```

### `s2_d_skill2`

第二轮 gate：只加 v2 skill，隔离 prompt 的作用

```
模型            gpt-5.6-sol
数据            S2 · 200 题分层抽样，与 S1 不相交（seed 20260910）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200b/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=12800  max_tool_calls=100  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           forecast_v2.md
fit 工具        未启用
并发            10 / shard
跑于            2026-09-10 14:08
```

```
分数            0.7123      准确率 0.7451
记分 / 剔除     200 / 0
分题型          BINARY 0.8546   MULTIPLE_CHOICE 0.7542   NUMERIC 0.4201
报告            reports/offline_0901/s2_gate_round2/
```

### `s2_e_tool`

第二轮 gate：只加 fit 工具，隔离工具的作用

```
模型            gpt-5.6-sol
数据            S2 · 200 题分层抽样，与 S1 不相交（seed 20260910）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200b/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=12800  max_tool_calls=100  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           无
fit 工具        启用
并发            10 / shard
跑于            2026-09-10 14:08
```

```
分数            0.7214      准确率 0.6993
记分 / 剔除     200 / 0
分题型          BINARY 0.8295   MULTIPLE_CHOICE 0.7358   NUMERIC 0.5290
报告            reports/offline_0901/s2_gate_round2/
```

### `s2_f_skill2tool`

第二轮 gate：v2 skill 加 fit 工具

```
模型            gpt-5.6-sol
数据            S2 · 200 题分层抽样，与 S1 不相交（seed 20260910）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200b/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=12800  max_tool_calls=100  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           forecast_v2.md
fit 工具        启用
并发            10 / shard
跑于            2026-09-10 14:08
```

```
分数            0.7275      准确率 0.7386
记分 / 剔除     200 / 0
分题型          BINARY 0.8509   MULTIPLE_CHOICE 0.7699   NUMERIC 0.4636
报告            reports/offline_0901/s2_gate_round2/
```

### `s2_noskill`

S2 上的对照臂，skill 实验的基准

```
模型            gpt-5.6-sol
数据            S2 · 200 题分层抽样，与 S1 不相交（seed 20260910）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200b/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=16384  max_tool_calls=50  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           无
fit 工具        未启用
并发            8 / shard
跑于            2026-09-10 01:23
```

```
分数            0.7063      准确率 0.7059
记分 / 剔除     200 / 0
分题型          BINARY 0.8317   MULTIPLE_CHOICE 0.7408   NUMERIC 0.4527
报告            reports/offline_0901/s2_skill_test/
```

### `s2_skill`

S2 上的 v1 skill 臂

```
模型            gpt-5.6-sol
数据            S2 · 200 题分层抽样，与 S1 不相交（seed 20260910）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200b/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=16384  max_tool_calls=50  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           forecast_v1.md
fit 工具        未启用
并发            8 / shard
跑于            2026-09-10 01:33
```

```
分数            0.7109      准确率 0.7516
记分 / 剔除     200 / 0
分题型          BINARY 0.8640   MULTIPLE_CHOICE 0.7177   NUMERIC 0.4614
报告            reports/offline_0901/s2_skill_test/
```

### `sample200_gpt56`

抽样 200 题的模型对比臂（gpt-5.6-sol）

```
模型            gpt-5.6-sol
数据            S1 · 200 题分层抽样（seed 20260908）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=16384  max_tool_calls=50  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           无
fit 工具        未启用
并发            8 / shard
跑于            2026-09-08 23:49
```

```
分数            0.7239      准确率 0.6863
记分 / 剔除     200 / 0
分题型          BINARY 0.8733   MULTIPLE_CHOICE 0.6920   NUMERIC 0.5463
报告            reports/offline_0901/sample200/
```

### `sample200_gpt56_siteretry`

site: 梯子修复后的重测，量 harness 修复的效应

```
模型            gpt-5.6-sol
数据            S1 · 200 题分层抽样（seed 20260908）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=16384  max_tool_calls=50  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           无
fit 工具        未启用
并发            8 / shard
跑于            2026-09-09 23:51
```

```
分数            0.7324      准确率 0.6405
记分 / 剔除     200 / 0
分题型          BINARY 0.8849   MULTIPLE_CHOICE 0.6732   NUMERIC 0.5963
报告            reports/offline_0901/sample200_siteretry/
```

### `sample200_qwen`

抽样 200 题的模型对比臂（qwen3.5-397b）

```
模型            qwen3.5-397b-a17b
数据            S1 · 200 题分层抽样（seed 20260908）
                /home/dataset-local/wcy/AxisAgentic/data/offline_forecast/sample200/offline_forecast_sample200_shard1.jsonl
轨迹            199 条
采样            temperature=1.0  top_p=1.0  reasoning_effort=high
预算            max_output_tokens=16384  max_tool_calls=50  max_turns=70
上下文          max_context_length=262144
时间边界        strict  ΔT=7d  observation_time=2026-09-01
检索            serper   摘要模型 deepseek-v4-flash
skill           无
fit 工具        未启用
并发            12 / shard
跑于            2026-09-08 23:50
```

```
分数            0.6801      准确率 0.6040
记分 / 剔除     195 / 5
分题型          BINARY 0.8380   MULTIPLE_CHOICE 0.6294   NUMERIC 0.5179
报告            reports/offline_0901/sample200/
```

## 哪些能放在一起比

同一个题集、同一份 config、只差一个键的才是配对比较。跨题集的分数不能并排：
同一 cohort 按结算日期切两半，实测分数差就有 0.16。

```
S1 上的 harness 效应   sample200_gpt56  ->  sample200_gpt56_siteretry
S2 上的 v1 skill 效应  s2_noskill       ->  s2_skill
S2 上的第二轮 gate     s2_noskill       ->  s2_d_skill2 / s2_e_tool / s2_f_skill2tool
S1 上的模型对比        sample200_gpt56  vs   sample200_qwen  vs  axis_strict_full 的同 200 题
```

`axis_strict_full` 作为分数来源已经退役：它跑在 `temperature 0.2` 加 `reasoning_effort medium` 下，并且 19.3% 的题触到当时 50 的工具上限，三个变量都和现在不同。它的价值是提供了 `site:` 归因的原始材料。

生成于 2026-09-10 15:39。
