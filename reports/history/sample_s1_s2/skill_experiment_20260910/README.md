# skill_experiment_20260910

「加了 skill 之后系统性能有没有提升」这个问题的完整证据。

打开 `index.html` 看结论和图。页面由 `build_report.py` 从下面这些原始产物重新计算，
没有手写数字，重跑脚本页面就更新。

```
index.html              主页面，八节
build_report.py         生成器
metrics.json            两臂完整统计，由 Milkyway 的 score_offline.py 产出
per_task_scores.jsonl   每臂每题一行
logs.tar.gz             三个 run 的原始轨迹，16 MB
logs/                   同样内容解压后 93 MB，被 gitignore，只在服务器上
```

`logs/` 里三个 run：`s2_noskill` 和 `s2_skill` 是这次实验的两臂，
`sample200_gpt56_siteretry` 是蒸馏 skill 用的那批（S1），一并放进来是因为 v1 的规则
都来自它。

## 三句话结论

skill 被读了也被执行了，查询里含 URL 或路径的从 283 次降到 3 次，遵从度接近完全。

分数没动，+0.0046，95% 区间 [−0.0295, +0.0387]。机制查清楚了不是噪声：skill 修的那类
错误，harness 的 `site:` 梯子已经自动兜住，真正被消灭的失败每题只有 0.22 次。

真正的发现在数值题：**照抄「信息截止日之前最后一个公布值」得 0.7190 分，模型自己答得
0.4429 分**，配对差 +0.2761，95% 区间 [+0.1391, +0.4131]，不跨 0。折算到 200 题总分
是 +0.065，远超这批题量 ±0.034 的噪声底。`skills/forecast_v2.md` 就是围绕这一条改的。

## 复现

```bash
# 记分（在 Milkyway 仓库下）
GT=data/benchmarks/offline_0901/sample200b/offline_forecast_sample200.jsonl
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-test/bin/python scripts/eval/score_offline.py \
  --ground-truth $GT \
  --run-dir /tmp/axis_export/s2_noskill --label s2_noskill \
  --run-dir /tmp/axis_export/s2_skill   --label s2_skill \
  --out reports/offline_0901/s2_skill_test/metrics.json \
  --per-task reports/offline_0901/s2_skill_test/per_task_scores.jsonl

# 渲染（在本目录下）
python3 build_report.py
```

## 这一页控制不了什么

没有验证集。v1 到 v2 的改动依据是分析而不是一次 gate 过的实验，SkillOpt-Lite 的循环每轮
都在 val 上判定接受或回滚，这里没有。要补上需要第三批题 S3 做测试，S2 退化为验证集。

锚点基线用的是数据集自带的真实序列，模型必须自己找到它。0.7190 是这条规则的上界，
不是模型照做就能拿到的分。
