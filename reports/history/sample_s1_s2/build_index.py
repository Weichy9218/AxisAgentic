#!/usr/bin/env python3
"""Build the AxisAgentic run index from the artifacts themselves.

Every column is read out of the run's own run_metadata.json and the scorer's
metrics.json, so the index cannot drift from what was actually executed. Re-run
after any new arm or re-scoring.

    python3 build_index.py
"""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

AX = Path("/home/dataset-local/wcy/AxisAgentic")
LOGS = AX / "logs/offline_forecast"
REPORTS = AX / "reports"

#: run dir -> the metrics file that scores it, and which label inside it.
SCORED_BY = {
    "axis_strict_full": ("offline_0901/axisagentic_gpt54_20260908/metrics.json", "axisagentic"),
    "sample200_gpt56": ("offline_0901/sample200/metrics.json", "axis_gpt56_sol"),
    "sample200_qwen": ("offline_0901/sample200/metrics.json", "axis_qwen35_397b"),
    "sample200_gpt56_siteretry": ("offline_0901/sample200_siteretry/metrics.json", "gpt56_siteretry"),
    "s2_noskill": ("offline_0901/s2_skill_test/metrics.json", "s2_noskill"),
    "s2_skill": ("offline_0901/s2_skill_test/metrics.json", "s2_skill"),
    "s2_d_skill2": ("offline_0901/s2_gate_round2/metrics.json", "D_skill2"),
    "s2_e_tool": ("offline_0901/s2_gate_round2/metrics.json", "E_tool"),
    "s2_f_skill2tool": ("offline_0901/s2_gate_round2/metrics.json", "F_both"),
}

#: One line saying what each run was for. Written by hand because intent is not
#: recoverable from an artifact.
PURPOSE = {
    "axis_strict_full": "第一次全量：AxisAgentic 对 galaxy 的 runtime 对比",
    "sample200_gpt56": "抽样 200 题的模型对比臂（gpt-5.6-sol）",
    "sample200_qwen": "抽样 200 题的模型对比臂（qwen3.5-397b）",
    "sample200_gpt56_siteretry": "site: 梯子修复后的重测，量 harness 修复的效应",
    "s2_noskill": "S2 上的对照臂，skill 实验的基准",
    "s2_skill": "S2 上的 v1 skill 臂",
    "s2_d_skill2": "第二轮 gate：只加 v2 skill，隔离 prompt 的作用",
    "s2_e_tool": "第二轮 gate：只加 fit 工具，隔离工具的作用",
    "s2_f_skill2tool": "第二轮 gate：v2 skill 加 fit 工具",
    "axis_nocut_smoke30": "接线阶段的冒烟，无时间边界",
    "axis_strict_smoke30": "接线阶段的冒烟，有时间边界",
}

DATASETS = {
    "offline_forecast_shard": "offline_0901 全量 1,969 题",
    "sample200/offline_forecast_sample200": "S1 · 200 题分层抽样（seed 20260908）",
    "sample200b/offline_forecast_sample200": "S2 · 200 题分层抽样，与 S1 不相交（seed 20260910）",
    "smoke30": "30 题冒烟",
}


def read_meta(run: str) -> dict | None:
    for p in sorted(glob.glob(f"{LOGS}/{run}/shard*/run_*/run_metadata.json")):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def dataset_name(path: str) -> str:
    for key, label in DATASETS.items():
        if key in (path or ""):
            return label
    return Path(path or "?").name


def n_traces(run: str) -> int:
    return len(glob.glob(f"{LOGS}/{run}/shard*/run_*/web-search-benchmark/*.json"))


def score_for(run: str) -> dict:
    entry = SCORED_BY.get(run)
    if not entry:
        return {}
    rel, label = entry
    p = REPORTS / rel
    if not p.exists():
        return {"pending": True}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for arm in d.get("arms") or []:
        if arm.get("label") == label:
            own = arm.get("own") or {}
            return {
                "score": own.get("score"),
                "accuracy": own.get("accuracy"),
                "n_scored": own.get("n_scored"),
                "n_excluded": own.get("n_excluded"),
                "by_type": {k: v.get("score") for k, v in (own.get("by_type") or {}).items()},
                "report": rel.rsplit("/", 1)[0],
            }
    return {}


def collect() -> list[dict]:
    rows = []
    for run in sorted(p.name for p in LOGS.iterdir() if p.is_dir()):
        meta = read_meta(run)
        if meta is None:
            continue
        a = meta.get("cli_args") or {}
        extra = a.get("request_extra_body_json") or "{}"
        try:
            extra = json.loads(extra) if isinstance(extra, str) else extra
        except Exception:
            extra = {}
        rows.append({
            "run": run,
            "purpose": PURPOSE.get(run, ""),
            "model": a.get("model"),
            "dataset": dataset_name(a.get("data_path", "")),
            "data_path": a.get("data_path", ""),
            "traces": n_traces(run),
            "temperature": a.get("temperature"),
            "top_p": a.get("top_p"),
            "reasoning_effort": extra.get("reasoning_effort"),
            "enable_thinking": extra.get("enable_thinking"),
            "max_output_tokens": a.get("max_output_tokens"),
            "max_tool_calls": a.get("max_tool_calls_per_task"),
            "max_turns": a.get("max_turns"),
            "max_context": a.get("max_context_length"),
            "concurrent": a.get("max_concurrent"),
            "provider": a.get("forecast_search_provider"),
            "delta_days": a.get("forecast_delta_days"),
            "observation_time": a.get("forecast_observation_time"),
            "policy": a.get("forecast_temporal_policy"),
            "skill": Path(a["skill_file"]).name if a.get("skill_file") else None,
            "fit_tool": str(a.get("fit_timeseries_enabled", "false")).lower() == "true",
            "summary_llm": a.get("summary_llm_model_name"),
            "created": (meta.get("created_at") or "")[:16].replace("T", " "),
            **score_for(run),
        })
    return rows


def fmt(v, digits=4):
    return "—" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def render(rows: list[dict]) -> str:
    live = [r for r in rows if "smoke" not in r["run"]]
    smoke = [r for r in rows if "smoke" in r["run"]]

    out = ["# AxisAgentic 在 offline_0901 上的全部运行",
           "",
           "每一行都是从该次运行自己的 `run_metadata.json` 和记分脚本的 `metrics.json` 里读出来的，",
           "不是手写的。新增一臂或重新记分之后跑 `python3 build_index.py` 即可更新。",
           "",
           "记分一律用 Milkyway 的 `scripts/eval/score_offline.py`，两个 runtime 共用一个实现，",
           "所以分数之间可比；galaxy 自己的报告留在 Milkyway 仓库，跨 runtime 的对比也在那边。",
           "",
           "## 总表",
           "",
           "| 运行 | 模型 | 数据 | skill | fit 工具 | 分数 | 准确率 | 记分/剔除 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in live:
        pend = r.get("pending")
        score = "跑分中" if pend else fmt(r.get("score"))
        acc = "—" if pend else fmt(r.get("accuracy"))
        cnt = "—" if pend else f"{r.get('n_scored', '—')} / {r.get('n_excluded', '—')}"
        out.append(f"| `{r['run']}` | {r['model']} | {r['dataset']} | "
                   f"{r['skill'] or '—'} | {'是' if r['fit_tool'] else '—'} | "
                   f"{score} | {acc} | {cnt} |")

    out += ["", "## 逐个运行", ""]
    for r in live:
        out += [f"### `{r['run']}`", ""]
        if r["purpose"]:
            out += [r["purpose"], ""]
        out += ["```",
                f"模型            {r['model']}",
                f"数据            {r['dataset']}",
                f"                {r['data_path']}",
                f"轨迹            {r['traces']} 条",
                f"采样            temperature={r['temperature']}  top_p={r['top_p']}"
                + (f"  reasoning_effort={r['reasoning_effort']}" if r['reasoning_effort'] else "")
                + (f"  enable_thinking={r['enable_thinking']}" if r['enable_thinking'] is not None else ""),
                f"预算            max_output_tokens={r['max_output_tokens']}  "
                f"max_tool_calls={r['max_tool_calls']}  max_turns={r['max_turns']}",
                f"上下文          max_context_length={r['max_context']}",
                f"时间边界        {r['policy']}  ΔT={r['delta_days']}d  "
                f"observation_time={r['observation_time']}",
                f"检索            {r['provider']}   摘要模型 {r['summary_llm']}",
                f"skill           {r['skill'] or '无'}",
                f"fit 工具        {'启用' if r['fit_tool'] else '未启用'}",
                f"并发            {r['concurrent']} / shard",
                f"跑于            {r['created']}",
                "```", ""]
        if r.get("pending"):
            out += ["记分尚未完成。", ""]
        elif r.get("score") is not None:
            bt = r.get("by_type") or {}
            out += ["```",
                    f"分数            {fmt(r['score'])}      准确率 {fmt(r.get('accuracy'))}",
                    f"记分 / 剔除     {r.get('n_scored')} / {r.get('n_excluded')}",
                    "分题型          " + "   ".join(f"{k} {fmt(v)}" for k, v in bt.items()),
                    f"报告            reports/{r.get('report')}/",
                    "```", ""]
        else:
            out += ["尚未记分。", ""]

    if smoke:
        out += ["## 冒烟运行（不用于任何结论）", ""]
        for r in smoke:
            out.append(f"- `{r['run']}` — {r['purpose']}，{r['traces']} 条轨迹，{r['model']}")
        out.append("")

    out += ["## 哪些能放在一起比",
            "",
            "同一个题集、同一份 config、只差一个键的才是配对比较。跨题集的分数不能并排：",
            "同一 cohort 按结算日期切两半，实测分数差就有 0.16。",
            "",
            "```",
            "S1 上的 harness 效应   sample200_gpt56  ->  sample200_gpt56_siteretry",
            "S2 上的 v1 skill 效应  s2_noskill       ->  s2_skill",
            "S2 上的第二轮 gate     s2_noskill       ->  s2_d_skill2 / s2_e_tool / s2_f_skill2tool",
            "S1 上的模型对比        sample200_gpt56  vs   sample200_qwen  vs  axis_strict_full 的同 200 题",
            "```",
            "",
            "`axis_strict_full` 作为分数来源已经退役：它跑在 `temperature 0.2` 加 "
            "`reasoning_effort medium` 下，并且 19.3% 的题触到当时 50 的工具上限，"
            "三个变量都和现在不同。它的价值是提供了 `site:` 归因的原始材料。",
            "",
            f"生成于 {datetime.now():%Y-%m-%d %H:%M}。",
            ""]
    return "\n".join(out)


if __name__ == "__main__":
    rows = collect()
    target = REPORTS / "README.md"
    target.write_text(render(rows), encoding="utf-8")
    print(f"wrote {target}")
    for r in rows:
        mark = "跑分中" if r.get("pending") else fmt(r.get("score"))
        print(f"  {r['run']:<28} {str(r['model']):<16} {r['traces']:>4} 条   {mark}")
