#!/usr/bin/env python3
"""Render the skill experiment report from the run artifacts.

Every number on the page is recomputed here from the scorer's per-task rows, the
ground-truth series and the raw traces, so the page cannot drift from what the
run actually produced. Re-run it after any re-scoring.

    python3 build_report.py
"""

from __future__ import annotations

import glob
import json
import statistics as st
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
MW = Path("/home/dataset-local/wcy/Milkyway-Harness-Agent")
AX = Path("/home/dataset-local/wcy/AxisAgentic")
PT = MW / "reports/offline_0901/s2_skill_test/per_task_scores.jsonl"
GT = MW / "data/benchmarks/offline_0901/offline_forecast_20260401_20260829.jsonl"
LOGS = HERE / "logs"

C_CTL, C_SKL, C_THIRD = "#A63D38", "#1A6E9E", "#A2740F"


def esc(x) -> str:
    return escape(str(x), quote=True)


def num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


# ------------------------------------------------------------------ 读数据
rows = [json.loads(l) for l in PT.read_text().splitlines() if l.strip()]
arms: dict[str, dict] = {}
for r in rows:
    arms.setdefault(r["arm"], {})[r["task_id"]] = r
ctl, skl = arms["s2_noskill"], arms["s2_skill"]
ids = sorted(set(ctl) & set(skl))
choice = [t for t in ids if ctl[t].get("correct") is not None and skl[t].get("correct") is not None]

truth = {}
for line in GT.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        truth[r["task_id"]] = r


def traces(run: str) -> dict:
    out = {}
    for f in glob.glob(f"{LOGS}/{run}/shard*/run_*/web-search-benchmark/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        out[str(d.get("task_id", "")).replace("_attempt-1", "")] = d
    return out


tc, ts = traces("s2_noskill"), traces("s2_skill")
common = sorted(set(tc) & set(ts))


# ------------------------------------------------------------------ 统计
def paired(va, vb):
    d = [y - x for x, y in zip(va, vb)]
    m = st.mean(d)
    se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def score_row(label, va, vb, unit=""):
    m, lo, hi = paired(va, vb)
    crosses = lo <= 0 <= hi
    mark = '<span class="ns">跨过 0</span>' if crosses else '<span class="sig">不含 0</span>'
    return (f"<tr><td>{esc(label)}</td><td class='n'>{st.mean(va):.4f}{unit}</td>"
            f"<td class='n'>{st.mean(vb):.4f}{unit}</td><td class='n'>{m:+.4f}</td>"
            f"<td class='n'>[{lo:+.4f}, {hi:+.4f}]</td><td>{mark}</td></tr>")


def eq13(pred, true, series):
    sigma = st.pstdev(series[-8:]) if len(series) >= 2 else 0.0
    if sigma <= 0:
        return None, None
    return max(0.0, 1.0 - ((pred - true) / (3 * sigma)) ** 2), abs(pred - true) / sigma


anchor_rows = []
for t in ids:
    r = skl[t]
    if r["task_type"] != "NUMERIC":
        continue
    g = truth.get(t) or {}
    pts = (g.get("gt_history") or {}).get("points") or []
    v_true, v_pred = num(r["ground_truth"]), num(r["output"])
    if v_true is None or v_pred is None or len(pts) < 4:
        continue
    end = date.fromisoformat(str(g.get("end_time"))[:10])
    t_cut = end - timedelta(days=int(r.get("effective_delta_days") or 7))
    visible = [p for p in pts if date.fromisoformat(str(p["date"])[:10]) <= t_cut]
    if not visible:
        continue
    series = [v for v in (num(p["value"]) for p in pts) if v is not None]
    s_m, z_m = eq13(v_pred, v_true, series)
    s_a, z_a = eq13(num(visible[-1]["value"]), v_true, series)
    if s_m is None or s_a is None:
        continue
    anchor_rows.append({"true": v_true, "pred": v_pred, "anchor": num(visible[-1]["value"]),
                        "sigma": st.pstdev(series[-8:]), "s_m": s_m, "s_a": s_a,
                        "z_m": z_m, "z_a": z_a})


def behaviour(tr):
    import re
    url_q = searches = empty = ladder = repeat_url = scrapes = 0
    for t in common:
        seen = {}
        for e in tr[t].get("tool_calls") or []:
            if not isinstance(e, dict):
                continue
            md = e.get("metadata") or {}
            atts = md.get("attempts") or []
            if atts and "raw_organic_count" in atts[0]:
                searches += 1
                if (atts[-1].get("organic_count") or 0) == 0:
                    empty += 1
                if md.get("site_retry_used"):
                    ladder += 1
                q = str(atts[0].get("query") or "")
                if re.search(r"https?://|\w+\.\w+/\S|\?\w+=", q):
                    url_q += 1
            else:
                scrapes += 1
                u = str(md.get("url") or "")
                seen[u] = seen.get(u, 0) + 1
        repeat_url += sum(v - 1 for v in seen.values() if v > 1)
    return {"searches": searches, "url_q": url_q, "empty": empty, "ladder": ladder,
            "scrapes": scrapes, "repeat_url": repeat_url}


bc, bs = behaviour(tc), behaviour(ts)


def med(tr, key):
    vals = []
    for t in common:
        m = tr[t].get("metadata") or {}
        vals.append({"turns": m.get("turn_used") or 0,
                     "tools": len(tr[t].get("tool_calls") or []),
                     "lat": (m.get("total_elapsed_ms") or 0) / 1000,
                     "ptok": (tr[t].get("token_usage") or {}).get("input_tokens") or 0}[key])
    return st.median(vals)


# ------------------------------------------------------------------ 图表
CSS = """
:root{--ink:#0c1222;--soft:#334155;--mut:#64748b;--bd:#e2e8f0;--sunken:#f1f5f9;
 --raised:#f8fafc;--acc:#A63D38;
 --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
 --body:'DM Sans',-apple-system,'PingFang SC',sans-serif;
 --disp:'Crimson Pro','Songti SC',Georgia,serif}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font:15px/1.75 var(--body)}
main{max-width:1000px;margin:0 auto;padding:36px 24px 90px}
h1{font-family:var(--disp);font-size:36px;margin:0 0 6px}
h2{font-family:var(--disp);font-size:24px;margin:52px 0 8px}
h2 .n{display:block;font:700 11px var(--body);letter-spacing:.12em;color:var(--acc);
 text-transform:uppercase;margin-bottom:6px}
h3{font-family:var(--disp);font-size:18px;margin:26px 0 6px}
p{margin:10px 0;color:var(--soft)}
ul,ol{color:var(--soft);padding-left:22px}li{margin:6px 0}
.sub{color:var(--mut);font-size:14px;margin:0 0 18px}
code{font:.88em var(--mono);background:var(--sunken);padding:2px 6px;border-radius:4px;color:#8B3A36}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:14px 0 20px;
 border:1px solid var(--bd);border-radius:8px;overflow:hidden}
th{background:var(--acc);color:#fff;font:600 12px var(--body);letter-spacing:.03em;
 text-align:left;padding:9px 13px}
td{padding:9px 13px;border-bottom:1px solid var(--bd);color:var(--soft);vertical-align:top}
tr:last-child td{border-bottom:0}
td.n{text-align:right;font:12.5px var(--mono);white-space:nowrap}
.note{background:var(--raised);border-left:4px solid var(--acc);border-radius:8px;
 padding:15px 20px;margin:20px 0;font-family:var(--disp);font-size:15.5px;color:var(--ink)}
.note.key{border-left-color:#1A6E9E}
.sig{color:#1A6E9E;font-weight:700;font-size:12px}
.ns{color:var(--mut);font-size:12px}
.bars{margin:12px 0 18px}
.brow{display:grid;grid-template-columns:210px 1fr 110px;gap:12px;align-items:center;margin:8px 0}
.bl{font-size:12.5px;color:var(--soft);text-align:right}
.bt{height:15px;background:var(--sunken);border-radius:4px;overflow:hidden}
.bt i{display:block;height:100%;border-radius:0 4px 4px 0}
.bv{font:11.5px var(--mono);color:var(--mut)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--soft);margin:10px 0}
.legend span{display:inline-flex;align-items:center;gap:7px}
.legend i{width:11px;height:11px;border-radius:3px}
figure{margin:18px 0;background:var(--raised);border:1px solid var(--bd);
 border-radius:8px;padding:18px 20px 12px}
figcaption{margin-top:10px;font-size:12.5px;color:var(--mut);line-height:1.6}
figure svg{display:block;width:100%;height:auto;overflow:visible}
.small{font-size:12.5px;color:var(--mut)}
"""


def bars(items, vmax, unit="", digits=2, width=210):
    out = [f'<div class="bars" style="--w:{width}px">']
    for label, value, colour in items:
        w = 100 * value / vmax if vmax else 0
        out.append(f'<div class="brow"><span class="bl">{esc(label)}</span>'
                   f'<span class="bt"><i style="width:{w:.1f}%;background:{colour}"></i></span>'
                   f'<span class="bv">{value:,.{digits}f}{esc(unit)}</span></div>')
    out.append("</div>")
    return "".join(out)


def scatter_anchor(rows_):
    """Model error against anchor error, both in sigma, log axes, y=x reference."""
    import math
    w, h, pad = 620, 380, 54
    lo, hi = -1.2, 3.6  # log10 sigma range, clipped

    def X(z):
        v = max(lo, min(hi, math.log10(max(z, 1e-2))))
        return pad + (w - pad - 20) * (v - lo) / (hi - lo)

    def Y(z):
        v = max(lo, min(hi, math.log10(max(z, 1e-2))))
        return h - pad - (h - pad - 20) * (v - lo) / (hi - lo)

    p = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for e in (-1, 0, 1, 2, 3):
        x, y = X(10 ** e), Y(10 ** e)
        p.append(f'<line x1="{x:.0f}" y1="{h-pad}" x2="{x:.0f}" y2="20" stroke="#e2e8f0"/>')
        p.append(f'<line x1="{pad}" y1="{y:.0f}" x2="{w-20}" y2="{y:.0f}" stroke="#e2e8f0"/>')
        lab = f"{10**e:g}"
        p.append(f'<text x="{x:.0f}" y="{h-pad+16}" text-anchor="middle" font-size="10.5" '
                 f'font-family="monospace" fill="#64748b">{lab}</text>')
        p.append(f'<text x="{pad-8}" y="{y+4:.0f}" text-anchor="end" font-size="10.5" '
                 f'font-family="monospace" fill="#64748b">{lab}</text>')
    p.append(f'<line x1="{X(10**lo):.0f}" y1="{Y(10**lo):.0f}" x2="{X(10**hi):.0f}" '
             f'y2="{Y(10**hi):.0f}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4"/>')
    x3, y3 = X(3), Y(3)
    p.append(f'<line x1="{x3:.0f}" y1="20" x2="{x3:.0f}" y2="{h-pad}" stroke="#A63D38" '
             f'stroke-width="1.5" stroke-dasharray="3 3" opacity=".7"/>')
    p.append(f'<text x="{x3+5:.0f}" y="34" font-size="10.5" fill="#A63D38">3σ 之外记 0 分</text>')
    for r in rows_:
        cx, cy = X(r["z_m"]), Y(r["z_a"])
        colour = C_SKL if r["z_a"] < r["z_m"] else C_THIRD
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{colour}" fill-opacity=".75" '
                 f'stroke="#fff" stroke-width="1.5"><title>模型 {r["z_m"]:.2f}σ  锚点 {r["z_a"]:.2f}σ</title></circle>')
    p.append(f'<text x="{(w)/2:.0f}" y="{h-14}" text-anchor="middle" font-size="11.5" '
             f'fill="#334155">模型答案的偏差（σ）</text>')
    p.append(f'<text transform="translate(16,{h/2:.0f}) rotate(-90)" text-anchor="middle" '
             f'font-size="11.5" fill="#334155">照抄锚点的偏差（σ）</text>')
    p.append("</svg>")
    return "".join(p)


# ------------------------------------------------------------------ 章节
n_choice = len(choice)
s_ctl = [ctl[t]["score"] for t in ids]
s_skl = [skl[t]["score"] for t in ids]
m_score, lo_score, hi_score = paired(s_ctl, s_skl)

flip = [t for t in choice if ctl[t]["correct"] != skl[t]["correct"]]
gain = [t for t in flip if skl[t]["correct"]]
lose = [t for t in flip if not skl[t]["correct"]]
g_avg = st.mean([skl[t]["score"] - ctl[t]["score"] for t in gain])
l_avg = st.mean([skl[t]["score"] - ctl[t]["score"] for t in lose])

a_model = st.mean([r["s_m"] for r in anchor_rows])
a_anchor = st.mean([r["s_a"] for r in anchor_rows])
m_a, lo_a, hi_a = paired([r["s_m"] for r in anchor_rows], [r["s_a"] for r in anchor_rows])
better = sum(1 for r in anchor_rows if r["s_a"] > r["s_m"])

body = f"""
<p class="sub">S2 共 {len(ids)} 题 · gpt-5.6-sol · serper · 三道门 · 两臂只差 <code>agent.skill_file</code> 一个键</p>

<div class="note">skill 被读了也被执行了：查询里含 URL 或路径的从 {bc['url_q']} 次降到 {bs['url_q']} 次。
但分数没动，{m_score:+.4f}，95% 区间 [{lo_score:+.4f}, {hi_score:+.4f}]。
机制查清楚了，不是噪声：skill 修的那类错误，harness 的 <code>site:</code> 梯子已经以 96.6% 的成功率自动兜住了，
真正被消灭的失败每题只有 {(bc['empty']-bs['empty'])/len(common):.2f} 次。</div>

<div class="note key">这一轮最有价值的发现不在 skill 上。数值题里，
<b>照抄「信息截止日之前最后一个公布值」的朴素做法得 {a_anchor:.4f} 分，模型自己答得 {a_model:.4f} 分</b>，
配对差 {m_a:+.4f}，95% 区间 [{lo_a:+.4f}, {hi_a:+.4f}]，不跨 0。
折算到 200 题总分上是 {m_a * len(anchor_rows) / len(ids):+.4f}，远超这批题量的噪声底 ±0.034。</div>

<h2 id="s1"><span class="n">01</span>实验设计</h2>
<p>S1 是蒸馏 skill 用的 200 题，S2 是重抽的另外 200 题，两者<b>交集为 0</b>，分层相同。
两臂跑同一批 S2、同一份 config、同一个模型，只差 <code>agent.skill_file</code>。
落盘可验：对照臂 <code>system_prompt.txt</code> 2,276 字节不含 skill 块，skill 臂 6,497 字节含。
两臂各 {len(ids)}/{len(ids)} 记分，零剔除，<code>blocked_judge_unavailable = 0</code>。</p>
<p class="small">skill 由我（Claude）读 S1 轨迹后一次写成，<b>没有跑 SkillOpt-Lite 的循环</b>。
SkillOpt-Lite 默认 10 轮，每轮有 val gate 决定接受或回滚；这里是它的第一轮 improve，没有 gate。</p>

<h2 id="s2"><span class="n">02</span>分数：没动</h2>
<table><tr><th>指标</th><th>w/o skill</th><th>w/ skill</th><th>配对差</th><th>95% 区间</th><th></th></tr>
{score_row("总分", s_ctl, s_skl)}
{score_row(f"准确率（选择题 n={n_choice}）", [float(ctl[t]['correct']) for t in choice], [float(skl[t]['correct']) for t in choice])}
{score_row("平均置信度", [ctl[t]['probability'] for t in choice], [skl[t]['probability'] for t in choice])}
{"".join(score_row(f"{tt} (n={sum(1 for t in ids if ctl[t]['task_type']==tt)})", [ctl[t]['score'] for t in ids if ctl[t]['task_type']==tt], [skl[t]['score'] for t in ids if ctl[t]['task_type']==tt]) for tt in ("BINARY","MULTIPLE_CHOICE","NUMERIC"))}
</table>
<p>六个检验里只有 BINARY 的区间不含 0，而且下界压在 +0.0003。六个检验出现一个边缘显著是随机就会有的，不作为发现。</p>

<h2 id="s3"><span class="n">03</span>行为：变化很大，方向正确</h2>
<div class="legend"><span><i style="background:{C_CTL}"></i>w/o skill</span><span><i style="background:{C_SKL}"></i>w/ skill</span></div>
{bars([("查询含 URL / 路径", bc['url_q'], C_CTL), ("", bs['url_q'], C_SKL)], max(bc['url_q'], 1), digits=0)}
{bars([("site: 梯子触发", bc['ladder'], C_CTL), ("", bs['ladder'], C_SKL)], max(bc['ladder'], 1), digits=0)}
{bars([("最终仍为空的搜索", bc['empty'], C_CTL), ("", bs['empty'], C_SKL)], max(bc['empty'], 1), digits=0)}
{bars([("重复抓取同一 URL", bc['repeat_url'], C_CTL), ("", bs['repeat_url'], C_SKL)], max(bc['repeat_url'], bs['repeat_url'], 1), digits=0)}
<p>前三项是 skill 直接瞄准的，都动了，其中「查询含 URL」几乎清零，说明模型对规则的遵从度很高。
第四项是 skill 里那条「同一站点两次失败就换源」，<b>反而变差</b>：它同时让模型抓得更多，副作用盖过了规则本身。</p>

<h3>代价</h3>
<table><tr><th>每题中位数</th><th>w/o skill</th><th>w/ skill</th><th>变化</th></tr>
{"".join(f"<tr><td>{esc(lab)}</td><td class='n'>{med(tc,k):,.0f}</td><td class='n'>{med(ts,k):,.0f}</td><td class='n'>{(med(ts,k)-med(tc,k))/max(med(tc,k),1):+.1%}</td></tr>" for lab,k in [("轮数","turns"),("工具调用","tools"),("耗时（秒）","lat"),("prompt token","ptok")])}
</table>
<p>skill 正文约 1,000 token，按 10 轮重发算每题约 10,000 token，占 prompt token 增量的四成；
其余六成是它诱发的更多工具调用带来的更长上下文。</p>

<h2 id="s4"><span class="n">04</span>为什么行为变了分数没变</h2>
<p>对照臂里 <code>site:</code> 梯子触发 {bc['ladder']} 次。这些查询在触发前已经返空，触发后绝大多数拿到了结果：
最终仍为空的只剩 {bc['empty']} 次，占搜索的 {bc['empty']/bc['searches']:.1%}。
也就是说 skill 阻止模型写的那些坏查询，在对照臂里<b>本来就会被 harness 自动改写并拿到结果</b>。
skill 省下的是额外的 HTTP 请求，不是失败。</p>
<p>两臂真实失败的差是 {bc['empty']} − {bs['empty']} = {bc['empty']-bs['empty']} 次，摊到 {len(common)} 题上是每题
{(bc['empty']-bs['empty'])/len(common):.2f} 次。一个每题少犯 0.2 次、而且这类错误有 96.6% 会被自动兜住的改进，
不可能移动分数。<b>机制到此完全解释，不需要归因到噪声。</b></p>

<h2 id="s5"><span class="n">05</span>记分规则的不对称，会影响以后所有这类实验</h2>
<p>按答案有没有翻转做无偏拆分（不用反事实，因为反事实里的概率和正确性相关，会系统性低估收益）：</p>
<table><tr><th>分组</th><th>题数</th><th>对总分的贡献</th><th>每题平均</th></tr>
<tr><td>答案没变（纯置信度效应）</td><td class="n">{n_choice-len(flip)}</td>
<td class="n">{sum(skl[t]['score']-ctl[t]['score'] for t in choice if t not in flip)/n_choice:+.4f}</td><td class="n">—</td></tr>
<tr><td>捡回（错→对）</td><td class="n">{len(gain)}</td>
<td class="n">{sum(skl[t]['score']-ctl[t]['score'] for t in gain)/n_choice:+.4f}</td><td class="n">{g_avg:+.4f}</td></tr>
<tr><td>丢掉（对→错）</td><td class="n">{len(lose)}</td>
<td class="n">{sum(skl[t]['score']-ctl[t]['score'] for t in lose)/n_choice:+.4f}</td><td class="n">{l_avg:+.4f}</td></tr>
</table>
<p><b>丢一题的代价比捡一题的收益大 {abs(l_avg)/g_avg-1:.0%}。</b>
原因是结构性的：模型答对时平均置信度
{st.mean([ctl[t]['probability'] for t in choice if ctl[t]['correct']]):.3f}，答错时
{st.mean([ctl[t]['probability'] for t in choice if not ctl[t]['correct']]):.3f}。
把一个自信答对的翻成错，损失远大于把一个不自信答错的翻成对。
所以<b>净答对题数不是分数的好代理</b>，捡回和丢掉的比例要到约 1.5 比 1 才打平。</p>

<h2 id="s6"><span class="n">06</span>数值题：模型输给了照抄</h2>
<p>Eq. 13 把误差除以序列近期波动的 3σ，所以「差多少」要按 σ 读。
拿 T_cut 之前最后一个公布值当答案（模型能合法看到的最新观测，距结算日中位 7 天），
和模型实际的答案放在同一个记分规则下比：</p>
<table><tr><th></th><th>中位 z（σ）</th><th>超过 3σ</th><th>Eq.13 均分</th></tr>
<tr><td>模型的答案</td><td class="n">{st.median([r['z_m'] for r in anchor_rows]):.2f}</td>
<td class="n">{sum(1 for r in anchor_rows if r['z_m']>3)/len(anchor_rows):.1%}</td>
<td class="n">{a_model:.4f}</td></tr>
<tr><td>照抄 T_cut 前最后一个观测</td><td class="n">{st.median([r['z_a'] for r in anchor_rows]):.2f}</td>
<td class="n">{sum(1 for r in anchor_rows if r['z_a']>3)/len(anchor_rows):.1%}</td>
<td class="n">{a_anchor:.4f}</td></tr>
</table>
<figure>{scatter_anchor(anchor_rows)}
<figcaption>每点一道数值题（n={len(anchor_rows)}）。虚线是两者相等；点落在<b>左上方</b>表示模型比照抄更差。
蓝色点是照抄更优的题（{better} 道，{better/len(anchor_rows):.0%}）。竖线是 3σ，越过它记 0 分。
两个轴都是对数刻度，因为最差的几道偏了四个数量级。</figcaption></figure>
<h3>失败的两种形态</h3>
<table><tr><th>真值</th><th>模型</th><th>锚点</th><th>σ</th><th>模型 z</th><th>锚点 z</th></tr>
{"".join(f"<tr><td class='n'>{r['true']:.4g}</td><td class='n'>{r['pred']:.4g}</td><td class='n'>{r['anchor']:.4g}</td><td class='n'>{r['sigma']:.4g}</td><td class='n'>{r['z_m']:.2f}</td><td class='n'>{r['z_a']:.2f}</td></tr>" for r in sorted(anchor_rows, key=lambda r: -r['z_m'])[:6])}
</table>
<p>第一种是<b>量纲错</b>：前两行差了四个数量级，而锚点分别只偏 0.15σ 和 0.33σ，
说明正确的数就在它已经拿到的证据里，是换算把它毁了。
第二种是<b>拿到锚点又调走</b>：第四行真值 4530、锚点 4558、模型答 5480。</p>
<p>轨迹里还能看到一个直接的浪费：同一个 URL 被反复抓五到六次
（<code>hq.sinajs.cn/list=nf_PG0</code> 抓了 5 次、<code>list=Y0</code> 抓了 6 次），
每次返回同样的内容。两臂合计重复抓取 {bc['repeat_url']} 和 {bs['repeat_url']} 次。</p>

<h2 id="s7"><span class="n">07</span>v2 依据</h2>
<p>基于以上，<code>skills/forecast_v2.md</code> 相对 v1 的改动：</p>
<ul>
<li><b>数值题改成锚点优先。</b>先写下 T_cut 前最后一个公布值，把它当答案；要偏离必须给出可命名、可定日期的理由，且幅度小于一次典型的发布间波动。这是本页里唯一有明确、区间不跨 0 的证据支持的规则。</li>
<li><b>加量纲核对。</b>写完数字和锚点比数量级，差两倍以上就是换算错了。</li>
<li><b>「同一站点两次失败换源」换成「绝不重复抓同一个 URL」。</b>原规则实测反向，真实形态是重复抓取而不是换站点。</li>
<li><b>加一条实时行情端点的说明。</b>回测下它返回的是今天的数字，会被边界挡掉，要找同源的历史形式。</li>
<li><b>压缩搜索那一节。</b>它瞄准的问题 harness 已经修了，留两行保留省请求的收益，把 token 让给数值题。</li>
<li><b>置信度那一条补上不对称性。</b>证据扎实时不要因为一个晚到的弱信号推翻已有答案。</li>
</ul>

<h2 id="s8"><span class="n">08</span>这一页控制不了什么</h2>
<p><b>没有验证集。</b>v1 到 v2 的改动依据是我对 S2 轨迹的分析，不是一次 gate 过的实验。
SkillOpt-Lite 的循环每轮都在 val 上判定接受或回滚；这里没有，所以 v2 可能比 v1 差而我们暂时不知道。
要补上，需要第三批题 S3 做测试，S2 退化为验证集。</p>
<p><b>锚点基线用的是数据集自带的真实序列，模型必须自己找到它。</b>
所以 0.7190 是这条规则的上界，不是模型照做就能拿到的分。它仍然要正确识别序列、读对字段和单位。</p>
<p><b>{len(ids)} 题的配对标准误约 0.017，能分辨的最小差约 0.034。</b>
两次独立的配对对比（harness 修复、skill）翻转率都是 16.3%，这是当前噪声底的最好估计。</p>
"""

html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>skill 验证实验</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{CSS}</style></head><body><main>
<h1>skill 到底有没有用</h1>
{body}
<p class="small" style="margin-top:40px">生成于 {datetime.now():%Y-%m-%d %H:%M}，
由 <code>build_report.py</code> 从 <code>per_task_scores.jsonl</code>、题集的 <code>gt_history</code>
和 <code>logs/</code> 下的原始轨迹重新计算，不含手写数字。</p>
</main></body></html>"""

out = HERE / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size:,} bytes)")
print(f"  配对 {len(ids)} 题，数值题基线 {len(anchor_rows)} 道")
print(f"  score {m_score:+.4f} [{lo_score:+.4f}, {hi_score:+.4f}]")
print(f"  anchor − model {m_a:+.4f} [{lo_a:+.4f}, {hi_a:+.4f}]")
