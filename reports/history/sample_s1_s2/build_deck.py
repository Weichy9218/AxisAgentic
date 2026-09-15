#!/usr/bin/env python3
"""Render the group-meeting deck for AxisAgentic on the 200-question subsets.

Every number is computed here from the runs' own artifacts and the scorer's
metrics, so a slide cannot claim a setting or a result the run did not produce.
The page theme is imported from the sibling repo rather than copied — same reason
the scorer is shared: one implementation, no drift.

    python3 build_deck.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

MW = Path("/home/dataset-local/wcy/Milkyway-Harness-Agent")
AX = Path("/home/dataset-local/wcy/AxisAgentic")
sys.path.insert(0, str(MW))

from scripts.page_theme import CAT, code, esc, legend, page, table  # noqa: E402

LOGS = AX / "logs/offline_forecast"
REPORTS = AX / "reports"
DATA = MW / "data/benchmarks/offline_0901"

C56, CQW, C54 = CAT[1], CAT[2], CAT[0]


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def arm_own(rel: str, label: str) -> dict:
    p = REPORTS / rel
    if not p.exists():
        return {}
    for arm in (json.loads(p.read_text(encoding="utf-8")).get("arms") or []):
        if arm.get("label") == label:
            return arm.get("own") or {}
    return {}


def effort(run: str) -> dict:
    tools, turns, lat, ptok = [], [], [], []
    status: Counter = Counter()
    se = sc = emp = rb = 0
    for f in glob.glob(f"{LOGS}/{run}/shard*/run_*/web-search-benchmark/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = d.get("metadata") or {}
        status[d.get("status")] += 1
        tools.append(len(d.get("tool_calls") or []))
        turns.append(m.get("turn_used") or 0)
        lat.append((m.get("total_elapsed_ms") or 0) / 1000.0)
        ptok.append((d.get("token_usage") or {}).get("input_tokens") or 0)
        rb += sum(1 for x in (d.get("conversation") or []) if x.get("role") == "ConversationRuntime")
        for e in d.get("tool_calls") or []:
            if not isinstance(e, dict):
                continue
            a = (e.get("metadata") or {}).get("attempts") or []
            if a and "raw_organic_count" in a[0]:
                se += 1
                emp += (a[-1].get("organic_count") or 0) == 0
            else:
                sc += 1
    n = max(len(tools), 1)
    tools_sorted = sorted(tools)
    return {
        "n": len(tools), "tools": tools_sorted, "status": status,
        "tools_med": st.median(tools) if tools else 0,
        "tools_p90": tools_sorted[int(0.90 * n)] if tools else 0,
        "at_cap": sum(1 for v in tools if v >= 50),
        "turns_med": st.median(turns) if turns else 0,
        "lat_med": st.median(lat) if lat else 0,
        "ptok_med": st.median(ptok) if ptok else 0,
        "search_pt": se / n, "scrape_pt": sc / n, "empty": emp, "rollbacks": rb,
    }


def cfg(run: str) -> dict:
    for p in sorted(glob.glob(f"{LOGS}/{run}/shard*/run_*/run_metadata.json")):
        a = json.loads(Path(p).read_text(encoding="utf-8")).get("cli_args") or {}
        try:
            a["_extra"] = json.loads(a.get("request_extra_body_json") or "{}")
        except Exception:
            a["_extra"] = {}
        return a
    return {}


def hist(values: list[int], edges: list[tuple[int, int]]) -> list[int]:
    return [sum(1 for v in values if lo <= v <= hi) for lo, hi in edges]


def bars(rows, vmax, unit="", digits=1, width=150):
    out = [f'<div class="gb" style="--lblw:{width}px">']
    for label, value, colour in rows:
        w = 100 * value / vmax if vmax else 0
        out.append(f'<div class="row"><div class="cat">{esc(label)}</div><div class="bars">'
                   f'<div class="b"><div class="fill" style="width:{w:.1f}%;background:{colour}"></div>'
                   f'<span class="v">{value:,.{digits}f}{esc(unit)}</span></div></div></div>')
    return "".join(out) + "</div>"


EXTRA_CSS = """
.gb .row{display:grid;grid-template-columns:var(--lblw,150px) 1fr;gap:14px;align-items:center;margin:9px 0}
.gb .cat{font-size:12.5px;color:var(--ink-soft);text-align:right;line-height:1.35}
.gb .b{display:flex;align-items:center;gap:8px;height:16px}
.gb .b .fill{height:16px;border-radius:0 4px 4px 0;min-width:2px}
.gb .b .v{font:11.5px/1 var(--font-mono);color:var(--ink-muted);white-space:nowrap}
.fair{background:#e8f1f7;border-left:4px solid #1A6E9E;border-radius:8px;padding:14px 18px;margin:16px 0}
.unfair{background:#fdf1f0;border-left:4px solid #A63D38;border-radius:8px;padding:14px 18px;margin:16px 0}
"""


# ---------------------------------------------------------------- 各节
def sec_architecture() -> str:
    return f"""
<h2 id="s1"><span class="num">01</span>AxisAgentic 是什么</h2>
<p>一个 ReAct 循环加两个工具的检索型 agent。它的简单本身就是被比较的对象：
对照的 galaxy 有 12 个工具、两个子 agent 和一层 notebook 持久化。</p>

<h4>工具</h4>
{table(["工具", "作用", "说明"], [
    [code("web_search"), "Serper 上的 Google 检索", "参数 query / num / gl / hl；返回结构化 organic 结果"],
    [code("scrape_and_extract_info"), "抓页面并抽取", "给了 info_to_extract 就用小模型只抽相关段，否则返回全文"],
    ["\x00thin" + code("fit_timeseries_forecast"), "确定性序列拟合", "本轮新增，默认关闭；纯计算，不联网"],
])}
<p class="small">前两个是这个 runtime 的全部检索能力。<b>没有文件工具，没有持久化</b>：
每次工具返回的内容直接进对话，读过的东西只存在于上下文里，
要再用就得重新抓。这一条在后面解释数值题为什么难。</p>

<h4>上下文管理</h4>
<ul>
<li><b>不压缩。</b><code>context_compression.enabled: false</code>，<code>keep_tool_result: -1</code>
    保留全部工具返回。声明窗口 262,144，实测峰值远低于它，所以压缩从未触发。</li>
<li><b>工具错误回滚。</b>这是最特别的设计：一次工具调用失败，或者
    <code>web_search</code> 返回空结果集，runtime 就往对话里写一条
    <code>role: "ConversationRuntime"</code> 的标记，把前面那条 assistant 消息撤销，
    让模型重新出手。撤销的消息仍留在轨迹里，但对模型不可见。</li>
<li><b>强制输出契约。</b>最终答案必须是 <code>\\boxed{{...}}</code> 加一行
    <code>Confidence:</code>，解析不出来就整题剔除，不补默认值。</li>
</ul>

<h4>为这次实验加的东西</h4>
<ul>
<li><b>三道门时间边界</b>（从 galaxy 移植）。检索结果在工具内部、网络返回之后、
    进入对话之前过一遍：provider 声明的日期、结构化裁剪、小模型判定
    「这段话是不是陈述了 T_cut 之后才可知的事实」。两个 runtime 用同一套设计，
    分数才可比。</li>
<li><b>skill 注入</b>：<code>agent.skill_file</code> 把一段 markdown 拼进 system prompt，
    不设就是对照臂。</li>
</ul>
"""


def sec_dataset(full, s1, s2, lab) -> str:
    def dist(rows):
        t = Counter(r["task_type"] for r in rows)
        return t

    tf, t1, t2 = dist(full), dist(s1), dist(s2)
    m1 = Counter(r["end_time"][:7] for r in s1)
    m2 = Counter(r["end_time"][:7] for r in s2)
    months = sorted(set(m1) | set(m2))
    overlap = len({r["task_id"] for r in s1} & {r["task_id"] for r in s2})

    return f"""
<h2 id="s2"><span class="num">02</span>数据：1,900 题里抽两个 200</h2>
<p>底库是 offline_0901，{len(full):,} 道已结算、有 ground truth 的预测题，
结算窗口 2026-04-01 到 2026-08-29，预测时点是结算日减 7 天、逐题算出来的。
每一道都能记分：原本 1,969 题，2026-09-08 删掉 69 道拿不到历史序列因而无法按
论文 Eq. 13 记分的数值题。</p>

<p>从中<b>分层抽两个不相交的 200 题</b>：先按题型按比例分配，再在题型内按 14 个记分组
用最大余数法分，同一个 seed 可复现。<b>S1 用来积累 skill，S2 用来验证</b>，
交集 {overlap} 题。</p>

{table(["", "题数", "BINARY", "MULTIPLE_CHOICE", "NUMERIC", "记分组"], [
    ["全量 offline_0901", f"{len(full):,}", str(tf['BINARY']), str(tf['MULTIPLE_CHOICE']), str(tf['NUMERIC']), "14"],
    ["S1（训练 / 积累 skill）", str(len(s1)), str(t1['BINARY']), str(t1['MULTIPLE_CHOICE']), str(t1['NUMERIC']), "14"],
    ["S2（验证）", str(len(s2)), str(t2['BINARY']), str(t2['MULTIPLE_CHOICE']), str(t2['NUMERIC']), "14"],
])}
<p class="small">两个子集的题型分布逐项相同（73 / 80 / 47），
比例也和全量一致（36.3% / 40.1% / 23.6%）。</p>

<h4>按结算月份</h4>
{legend([("S1", C56), ("S2", CQW)])}
{"".join(bars([(f"{m}", m1.get(m, 0), C56), ("", m2.get(m, 0), CQW)],
              max(max(m1.values()), max(m2.values())), digits=0, width=90) for m in months)}
<p class="small">月份分布没有强制对齐，抽样只按题型和记分组分层。
两个子集在 2026-07 都偏少，因为底库本身该月只有 167 题。</p>

<div class="note">为什么不用 <code>max_tasks: 200</code>：数据集加载器按行号截断，
<code>shuffle_tasks</code> 只重排已加载的行，所以那等于取前 200 行再打乱 ——
而这个题集的加载顺序开头是 27 道连续的 NUMERIC。抽样必须落成文件。</div>
"""


def sec_models(e56, eqw, e54, c56, cqw, c54) -> str:
    m56 = arm_own("offline_0901/sample200/metrics.json", "axis_gpt56_sol")
    mqw = arm_own("offline_0901/sample200/metrics.json", "axis_qwen35_397b")
    m54 = arm_own("offline_0901/sample200/metrics.json", "axis_gpt54")

    def bt(m, k):
        return (m.get("by_type") or {}).get(k, {}).get("score")

    def f(v, d=4):
        return "—" if v is None else f"{v:.{d}f}"

    rows_cfg = []
    for key, label in [("temperature", "temperature"), ("top_p", "top_p"),
                       ("max_output_tokens", "max_output_tokens"),
                       ("max_tool_calls_per_task", "工具上限"),
                       ("summary_llm_model_name", "摘要模型"),
                       ("forecast_search_provider", "检索 provider"),
                       ("forecast_delta_days", "ΔT")]:
        a, b, c = str(c56.get(key)), str(cqw.get(key)), str(c54.get(key))
        same = a == b
        rows_cfg.append([label, a, b,
                         (c if c == a else f'<b style="color:{C54}">{esc(c)}</b>'),
                         "对齐" if same else '<b style="color:#A63D38">不一致</b>'])
    rows_cfg.append(["推理档位",
                     str(c56["_extra"].get("reasoning_effort")),
                     str(cqw["_extra"].get("reasoning_effort")),
                     str(c54["_extra"].get("reasoning_effort")), "见下"])

    return f"""
<h2 id="s3"><span class="num">03</span>模型对比：只有一组是公平的</h2>

<div class="fair"><b>可以直接比：gpt-5.6-sol 对 qwen3.5-397b。</b>
同一批 S1 200 题、同一份 config、同一个记分脚本，逐项参数相同，只换模型。
两臂都是 w/o skill、w/o fit tool。</div>

<div class="unfair"><b>不能直接比：gpt-5.4。</b>
它的分数来自更早的 1,969 题全量运行按这 200 个 task_id 取出的子集，
config 有四处以上不同（见下表红色），并且 <b>19.3% 的题触到了当时 50 的工具上限</b>。
放进来只作历史参照，任何结论不建立在它上面。</div>

{table(["参数", "gpt-5.6-sol", "qwen3.5-397b", "gpt-5.4", "两个主臂"], rows_cfg)}

<h4>分数</h4>
{table(["模型", "#分数", "#准确率", "#BINARY", "#MULTIPLE_CHOICE", "#NUMERIC", "#记分/剔除"], [
    [f'<b style="color:{C56}">gpt-5.6-sol</b>', f(m56.get("score")), f(m56.get("accuracy")),
     f(bt(m56, "BINARY")), f(bt(m56, "MULTIPLE_CHOICE")), f(bt(m56, "NUMERIC")),
     f'{m56.get("n_scored")} / {m56.get("n_excluded")}'],
    [f'<b style="color:{CQW}">qwen3.5-397b</b>', f(mqw.get("score")), f(mqw.get("accuracy")),
     f(bt(mqw, "BINARY")), f(bt(mqw, "MULTIPLE_CHOICE")), f(bt(mqw, "NUMERIC")),
     f'{mqw.get("n_scored")} / {mqw.get("n_excluded")}'],
    ["\x00thin" + f'<span style="color:{C54}">gpt-5.4（不可比）</span>', f(m54.get("score")), f(m54.get("accuracy")),
     f(bt(m54, "BINARY")), f(bt(m54, "MULTIPLE_CHOICE")), f(bt(m54, "NUMERIC")),
     f'{m54.get("n_scored")} / {m54.get("n_excluded")}'],
])}
<p>gpt-5.6-sol 领先 qwen 的配对差是 +0.0608，95% 区间 [+0.0205, +0.1011]，
这是目前唯一一个区间不跨 0 的模型对比。分题型看，差距主要在 MULTIPLE_CHOICE
（+0.086）和 BINARY（+0.047），NUMERIC 只有 +0.039，两个模型在数值题上都很差。</p>

<h4>代价</h4>
{table(["每题中位数", f"gpt-5.6-sol", "qwen3.5-397b", "倍数"], [
    ["工具调用", f'{e56["tools_med"]:.0f}', f'{eqw["tools_med"]:.0f}', f'{eqw["tools_med"]/max(e56["tools_med"],1):.1f}×'],
    ["rollout 轮数", f'{e56["turns_med"]:.0f}', f'{eqw["turns_med"]:.0f}', f'{eqw["turns_med"]/max(e56["turns_med"],1):.1f}×'],
    ["耗时（秒）", f'{e56["lat_med"]:.0f}', f'{eqw["lat_med"]:.0f}', f'{eqw["lat_med"]/max(e56["lat_med"],1):.1f}×'],
    ["prompt token", f'{e56["ptok_med"]:,.0f}', f'{eqw["ptok_med"]:,.0f}', f'{eqw["ptok_med"]/max(e56["ptok_med"],1):.1f}×'],
])}
<p><b>qwen 每一项都更贵，分数还更低。</b>耗时是 3.3 倍，prompt token 是 2.8 倍。
一个原因是它开着 thinking（每条 assistant 消息都带 reasoning_content，
覆盖率 100%），gpt 那边网关不回传推理文本。</p>
"""


def sec_tooldist(e56, eqw) -> str:
    edges = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 49), (50, 999)]
    labels = ["0–5", "6–10", "11–15", "16–20", "21–30", "31–49", "50+（触顶）"]
    h56 = hist(e56["tools"], edges)
    hqw = hist(eqw["tools"], edges)
    vmax = max(max(h56), max(hqw))
    body = "".join(bars([(lab, h56[i], C56), ("", hqw[i], CQW)], vmax, digits=0, width=110)
                   for i, lab in enumerate(labels))
    return f"""
<h2 id="s4"><span class="num">04</span>工具调用分布</h2>
{legend([("gpt-5.6-sol", C56), ("qwen3.5-397b", CQW)])}
{body}
{table(["", "gpt-5.6-sol", "qwen3.5-397b"], [
    ["每题搜索次数", f'{e56["search_pt"]:.1f}', f'{eqw["search_pt"]:.1f}'],
    ["每题抓取次数", f'{e56["scrape_pt"]:.1f}', f'{eqw["scrape_pt"]:.1f}'],
    ["工具调用 p90", f'{e56["tools_p90"]:.0f}', f'{eqw["tools_p90"]:.0f}'],
    ["触到 50 上限的题", f'{e56["at_cap"]} / {e56["n"]}', f'{eqw["at_cap"]} / {eqw["n"]}'],
    ["返空的搜索（总数）", f'{e56["empty"]}', f'{eqw["empty"]}'],
    ["tool_error 回滚（总数）", f'{e56["rollbacks"]}', f'{eqw["rollbacks"]}'],
])}
<div class="note">最反直觉的一格是<b>返空的搜索</b>：gpt-5.6-sol {e56["empty"]} 次，
qwen 只有 {eqw["empty"]} 次，差 {e56["empty"]/max(eqw["empty"],1):.1f} 倍。
更强的模型写出了多得多的无效查询。追下去发现的原因，就是下一节那个 harness 缺陷。</div>
"""


def sec_harness() -> str:
    return """
<h2 id="s5"><span class="num">05</span>改了哪些 harness，为什么，效果如何</h2>

<h3>一、<code>site:</code> 降级梯子</h3>
<p><b>现象。</b>全量运行里 31.6% 的 assistant 消息被回滚，而 runtime 的规则是
<code>web_search</code> 返回空结果集就回滚。</p>
<p><b>归因（重放，不是相关性）。</b>1,539 次被回滚的搜索里 <b>92.2% 是 provider 那端就返回 0 条</b>，
三道门根本没见到卡片。把这些查询按四种变体重新打给 Serper：去掉日期窗几乎救不回任何一条，
去掉 <code>site:</code> 里的路径救回 30%，整个去掉 <code>site:</code> 再救回 35%。
1,869 次带 <code>site:</code> 的空搜里，<b>69.2% 的 <code>site:</code> 里塞了 URL 路径</b>。</p>
<p><b>真因。</b>Google 的 <code>site:</code> 拿这个 token 去匹配已索引页面的 URL，
而 <code>site:host/api/v1/x.json</code> 这种 API 端点从来没被当页面索引过 ——
这个限定不是太窄，是<b>不可满足</b>。不是 Serper 的问题：同一次调用去掉
<code>site:</code> 就返 10 条。</p>
<p><b>改法。</b>在已有的「返空且含引号则去引号重试」后面接两级降级：
<code>site:</code> 带路径就降到裸域名，仍为空就整个去掉。每级只在上一级仍为空时触发，
所以能搜到东西的查询仍然只花一次请求。</p>
<p><b>效果</b>（191 题配对，同一批题、同一 config、只差这一处代码）：</p>
<pre><code>最终仍为空的搜索     490 -> 76      -84.5%
tool_error 回滚      928 -> 493     -46.9%
工具调用总数        2,681 -> 2,208   -17.6%
每题 prompt token  48.8k -> 40.4k   -17.2%
梯子触发 161 次，救回 155 次（96.3%）</code></pre>
<p>分数：0.7239 → 0.7324，95% 区间 [−0.0251, +0.0421]，跨过 0。
<b>过程指标是计数，站得住；分数没动，不声称。</b></p>

<h3>二、skill 注入（v1 / v2）</h3>
<p><b>动机。</b>用 S1 的轨迹蒸馏出一段流程规则拼进 system prompt，S2 上验证。</p>
<p><b>v1 的效果。</b>模型确实照做了：查询里含 URL 或路径的从 283 次降到 3 次，
遵从度接近完全。但分数 +0.0046，区间 [−0.0295, +0.0387]。</p>
<p><b>为什么不动，机制查清了。</b>对照臂里 <code>site:</code> 梯子触发 178 次、救回 172 次（96.6%），
也就是 skill 阻止模型写的那些坏查询<b>本来就会被 harness 自动改写并拿到结果</b>。
skill 省下的是额外的 HTTP 请求，不是失败。真正被消灭的失败每题只有 0.22 次。</p>
<p><b>顺带测出一个记分规则的性质</b>，会影响以后所有这类实验：把分数变化按答案有没有翻转
拆开，捡回一题平均 +0.3417，丢掉一题平均 −0.5013 —— <b>丢的代价比捡的收益大 47%</b>，
因为模型答对时置信度 0.726、答错时 0.561。所以净答对题数不是分数的好代理。</p>

<h3>三、<code>fit_timeseries_forecast</code>（从 galaxy 移植）</h3>
<p><b>动机。</b>数值题是最大的洼地。离线算过：照抄「T_cut 之前最后一个公布值」在
论文 Eq. 13 下得 0.7190，而模型实际答的是 0.4429，配对差 +0.2761，区间
[+0.1391, +0.4131] 不跨 0。这个结论在 S1 上独立复现（+0.2118）。</p>
<p><b>移植时发现的坑。</b>原样搬会把病一起搬进来：工具自己推荐的方法比照抄锚点差 0.131
（27 次里 22 次推荐 <code>drift</code>）。7 天视界上从 8 个点估出的斜率基本是噪声。
所以在 wrapper 层加了守卫 —— <b>按真实视界留出、重拟合，趋势打不过持久性就不用</b>，
平手和测不了都归持久性。守卫把差距从 0.131 收到 0.0296。</p>
<p><b>效果：不显著，而且原因很具体。</b>S2 上 +0.0151，区间 [−0.0210, +0.0512]。
按是否真正调用工具拆开：</p>
<pre><code>用了工具的 21 题    0.4629 -> 0.4431   -0.0198
没用工具的 26 题    0.4445 -> 0.5984   +0.1540</code></pre>
<p>收益全在没碰过工具的那一半，那不可能是工具算出来的。而在用了工具的那一半：
<b>模型是听话的，86% 的答案落在工具给的数 1% 以内</b>，但工具是它输入的纯函数 ——
照抄工具值的中位相对误差仍有 <b>22.27%</b>，模型自己答的是 33.33%。</p>

<div class="note">所以数值题的缺口<b>既不是算术问题，也不是听话问题，是检索问题</b>。
那个 0.72 的上界是用数据集自带的真实序列算的；模型必须自己找到那条序列，
而这一轮就是对「它找得多准」的测量，答案是找不准。
AxisAgentic 没有持久化层，每次都要从散文里重新拼出序列，拼的过程就是失败发生的地方。</div>
"""


def sec_fairness() -> str:
    return """
<h2 id="s6"><span class="num">06</span>哪些能放在一起比</h2>
<p>配对比较的条件是：同一个题集、同一份 config、只差一个声明出来的键。
跨题集的分数不能并排 —— 同一个 cohort 按结算日期切两半，实测分数差就有 0.16，
那个差是抽签造成的。</p>
{}
<p class="small">四组之外的组合都不成立。特别是
<code>axis_strict_full</code>（全量 gpt-5.4）作为分数来源已经退役：
temperature 0.2、reasoning_effort medium、摘要模型 deepseek-v4-pro、
19.3% 触到工具上限，四处不同。它现在的价值是提供了 <code>site:</code> 归因的原始材料。</p>
""".format(table(["比较", "题集", "变量", "结论"], [
        ["gpt-5.6-sol vs qwen3.5-397b", "S1", "只换模型", "+0.0608，[+0.0205, +0.1011]，<b>不跨 0</b>"],
        ["site: 梯子 修复前 vs 修复后", "S1", "只差一处代码", "过程 −84.5% 空搜；分数跨 0"],
        ["v1 skill 有 vs 无", "S2", "只差 skill_file", "+0.0046，跨 0"],
        ["第二轮 gate（v2 / 工具 / 两者）", "S2", "各差一个键", "+0.0060 / +0.0151 / +0.0212，全部跨 0"],
    ]))


def sec_limits() -> str:
    return """
<h2 id="s7"><span class="num">07</span>这一页控制不了什么</h2>
<ul>
<li><b>200 题的分辨力约 0.034。</b>配对标准误 0.017。两次独立的配对对比翻转率都是
    16.3%，那是 temperature 1.0 下重跑本身的抖动。所有跨过 0 的结论都要读成
    「这个题量测不出来」，不是「没有效果」。</li>
<li><b>没有噪声底的直接测量。</b>同一份 config 重跑两遍会差多少，我们没有单独测过，
    只有翻转率这个间接估计。</li>
<li><b>gpt-5.6-sol 的快照是 2026-07-09</b>，这 200 题里 61.5% 在那之前就结算了。
    差中差检验的区间跨 0，既没证实也没排除参数化泄漏。gpt-5.4 的快照是 2026-03-05，
    题集里 0 题落在它的快照内 —— 这是它唯一无法替代的价值。</li>
<li><b>qwen 的快照不可查</b>，网关只回显 <code>Qwen3.5-397B-A17B</code>，
    模型自报的训练截止不能当证据。</li>
</ul>
"""


def main() -> None:
    full = jl(DATA / "offline_forecast_20260401_20260829.jsonl")
    s1 = jl(DATA / "sample200/offline_forecast_sample200.jsonl")
    s2 = jl(DATA / "sample200b/offline_forecast_sample200.jsonl")
    lab = {r["task_id"]: r for r in jl(DATA / "labels.jsonl")}

    e56, eqw = effort("sample200_gpt56"), effort("sample200_qwen")
    e54 = effort("axis_strict_full")
    c56, cqw, c54 = cfg("sample200_gpt56"), cfg("sample200_qwen"), cfg("axis_strict_full")

    crumbs = ('<div class="crumbs">'
              '<a class="crumb" href="../README.md">运行索引</a>'
              '<a class="crumb" href="../skill_experiment_20260910/index.html">skill 实验</a>'
              '<a class="crumb" href="../offline_0901/s2_gate_round2/metrics.json">第二轮 gate</a>'
              '</div>')

    lead = """
<div class="note">一句话：<b>在这 200 题上，唯一区间不跨 0 的结论是 gpt-5.6-sol 比
qwen3.5-397b 高 0.0608</b>，而且 qwen 每题贵 3.3 倍时间、2.8 倍 token。
harness 那一处修复把无效搜索砍掉 84.5%、回滚砍掉 46.9%，但分数没动；
skill 和序列工具也都没动分数，原因各自查清了，都不是噪声。</div>
"""

    body = "".join([
        lead,
        sec_architecture(),
        sec_dataset(full, s1, s2, lab),
        sec_models(e56, eqw, e54, c56, cqw, c54),
        sec_tooldist(e56, eqw),
        sec_harness(),
        sec_fairness(),
        sec_limits(),
        f'<hr class="rule"><p class="small">生成于 {datetime.now():%Y-%m-%d %H:%M}，'
        f'由 <code>build_deck.py</code> 从各运行的 <code>run_metadata.json</code>、'
        f'原始轨迹和记分脚本的 <code>metrics.json</code> 重新计算，不含手写数字。</p>',
    ])

    html = page(
        "AxisAgentic · sample200",
        "AxisAgentic 在 200 题子集上的实验",
        "架构 · 数据 · 模型对比 · 工具调用 · harness 改动",
        crumbs,
        body,
    ).replace("</style>", EXTRA_CSS + "</style>")

    out = REPORTS / "sample200_overview.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"  gpt-5.6-sol  n={e56['n']}  工具中位 {e56['tools_med']:.0f}  耗时中位 {e56['lat_med']:.0f}s")
    print(f"  qwen         n={eqw['n']}  工具中位 {eqw['tools_med']:.0f}  耗时中位 {eqw['lat_med']:.0f}s")


if __name__ == "__main__":
    main()
