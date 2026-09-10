# 两个 runtime 的对齐审计

galaxy（Milkyway）和 AxisAgentic 在同一批题上比，所以哪些设置必须相同、哪些本来就该不同，
要写下来而不是靠记。这一页只记**测过的**，没测的标注为未测。

审计日期 2026-09-10，依据是 S1/S2 已完成运行的产物，不是读 config 推的。

## 一、必须相同的（它们不是 runtime 的一部分）

| 设置 | galaxy | AxisAgentic | 状态 |
|---|---|---|---|
| temperature | 1.0 | 1.0 | 对齐 |
| top_p | 1.0 | 1.0 | 对齐 |
| reasoning_effort（gpt） | high | high | 对齐 |
| 题集 | 见下 §4 | 见下 §4 | **要按轮次确认** |
| temporal_policy | strict | strict | 对齐 |
| delta_days | 7 | 7 | 对齐 |
| observation_time | 2026-09-01 | 2026-09-01 | 对齐 |
| search provider | serper | serper | 对齐 |
| 边界句进 prompt | `aligned_prompt: true` | `_forecast_boundary_line(t_cut)` | 等价 |
| 记分 | 同一个 `score_offline.py` | 同一个 | 对齐 |
| **qwen thinking** | **off** (`enable_thinking: false`) | **on**（实测 4174/4174） | **未对齐，见 §3** |

## 二、纸面不一致但实测不咬人的

`max_output_tokens`：galaxy 对 gpt 用 12800、对 qwen 用 4096；Axis 一律 16384。

实测单次生成的 output token（S1/S2 共约 10,000 次生成）：

```
gpt-5.6   中位 107   p95 275   p99 427    max 952
qwen      中位 149   p95 736   p99 1198   max 3,586
超过 12800 的生成 0 次；超过 4096 的 0 次
```

**从来没有一次生成逼近过任何一侧的上限**，所以这个差值至今没有影响过任何结果，不值得为它
改 config 并牺牲与 S1/S2 的可比性。但它是潜在的：如果将来某个臂的生成变长（例如 qwen 开
thinking 后写得更多），要重新查这张表再决定。

`window_token_budget`：galaxy 默认 1,000,000，Axis 声明 262,144。galaxy 实测上下文峰值
中位 12,690、最大 37,050，Axis 侧同样远低于上限，两边的压缩门都没触发过。这个差值目前
同样不咬人。

## 三、qwen 的 thinking：唯一真实的能力差

Axis 那一轮 qwen 的每条 assistant 消息都带 `reasoning_content`，galaxy 明确关掉了，
理由写在 config 里是"开了会让本来就受预算约束的 qwen 臂慢很多"。

这不是超参数微调，是让不让模型用它的推理模式。关掉测的是一个被削过的模型。

**建议两边都开**，三条理由：

1. 这是 qwen 的预期工作模式，关掉之后分数不能被读作模型能力。
2. 我们已有的 Axis qwen 结果（配对 0.6786）是开着测的，统一到开可以保住这个结果的可比性。
3. 决定性的一条：轨迹要用来微调 qwen3.5-397b。`reasoning_content` 是训练思考行为唯一的
   来源，关掉 thinking 等于把要训练的那个字段本身删掉。

代价是慢。Axis 侧开着 thinking 跑 200 题用了 11.2 小时。如果接受，galaxy 侧的
`max_tokens: 4096` 要一起抬高到 12800，因为 thinking 的 token 也计入输出预算，
而我们实测 qwen 开着 thinking 时最大单次生成已经到 3,586。

## 四、工具预算：整数不同，但两边都没被咬住

这是最容易被误读的一项。galaxy 声明 30、Axis 声明 50，但这两个数不可比：
galaxy 的 30 是**主 worker** 的上限，它的 `sub_agent_factor`（12 次）和
`sub_agent_access`（18 次）各自还有自己的预算，主 worker 调一次子 agent 只算 1 次。

实测触到上限的比例：

```
galaxy（1,969 题）                forced_tool_cap 全 false，中位 16，p75 19     0%
Axis S2 gpt-5.6 无 skill          中位  9  p95 34  max 51                     1.0%
Axis S2 gpt-5.6 有 skill          中位 11  p95 35  max 51                     2.0%
Axis S1 qwen                      中位 18  p95 45  max 52                     2.5%
Axis S1 gpt-5.4（旧 config）       中位 26  p95 51  max 52                    19.3%
```

**当前设置下两边都基本不受预算约束**，所以 50 对 30 不再是实质混杂。不要为了让整数相等
去改它——那会破坏与 S1/S2 的可比性，换不来任何已测到的收益。

那个 19.3% 是旧 config（`reasoning_effort: medium`）留下的。同一个模型换成 high 之后
中位从 26 掉到 9，所以 round 2 的 gpt-5.4 臂大概率不会重现它。**但要在报告里报出触顶
比例**：如果某个臂超过 5%，那个臂的分数就带着截断，必须并排说明。

## 五、题集：按轮次确认，不能想当然

```
S1  data/benchmarks/offline_0901/sample200/        v1 skill 从它蒸馏
S2  data/benchmarks/offline_0901/sample200b/       v1 skill 的测试集
S3  尚未抽                                          v2 skill 的测试集
```

对 Axis，一个题集一旦被用来写 skill 就不能再当测试集。对 galaxy 没有这个约束，
因为 galaxy 不读我们的 skill，所以 galaxy 在 S1 上跑是干净的。

**但要做 galaxy 对 Axis 的跨 runtime 比较，两边必须在同一个文件上。**
现在 galaxy 的 `offline_forecast_sample200_serper_gpt56.yaml` 指向 S1，
Axis 的 `s2_*` 指向 S2，这两组不能直接比。

## 六、本来就该不同的（这就是被比较的东西）

工具清单（galaxy 12 个加两个子 agent，Axis 2 到 3 个）、是否并行调用工具
（galaxy true，Axis false，因为它的执行循环是串行的，开了只会减少轮数不会真并行）、
上下文压缩策略、子 agent 递归。这些是 runtime 的身份，对齐它们等于取消实验。

并发（galaxy 6，Axis 8）不影响分数，但两边同时跑会叠加到网关上，排期时要算。
