#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctxbench — LLM 上下文画像工具

读你的模型调用日志，告诉你：每次请求带了多少上下文、处在什么水平、
有多少可能被浪费掉。

零依赖，单文件，Python 3.8+。

用法
----
    python ctxbench.py calls.csv
    python ctxbench.py calls.jsonl
    python ctxbench.py calls.csv --input prompt_tokens --model model
    cat calls.jsonl | python ctxbench.py -
    python ctxbench.py calls.csv --price-in 2 --price-out 8 --target 8000

输入格式
--------
自动识别 CSV / JSONL / JSON 数组。列名大小写与下划线不敏感，
常见命名都能自动匹配（input_tokens / prompt_tokens / input ...）。

设计原则
--------
1. 只读元数据，不碰 prompt / completion 正文
2. 不联网，不上传，纯本地计算
3. 结论给区间不给假精度
"""

import argparse
import csv
import io
import json
import math
import os
import sys
from collections import Counter, defaultdict

# ── 列名推断表 ──────────────────────────────────────────────
ALIASES = {
    "input": ["input_tokens", "prompt_tokens", "input", "prompt", "in_tok",
              "input_token", "prompt_token", "n_input_tokens", "inputtokens"],
    "output": ["output_tokens", "completion_tokens", "output", "completion",
               "out_tok", "output_token", "completion_token", "n_output_tokens"],
    "model": ["model", "model_name", "provided_model_name", "model_id",
              "engine", "deployment"],
    "time": ["timestamp", "time", "start_time", "created_at", "ts", "datetime",
             "request_time", "event_time"],
    "cost": ["cost", "total_cost", "price", "usd", "amount"],
}

BENCH = [
    (0,      8000,   "健康",   "在常见区间内，无需特别处理"),
    (8000,   20000,  "偏高",   "值得检查，通常有 30-50% 可压缩"),
    (20000,  50000,  "严重",   "大概率有 60% 以上是重复或冗余内容"),
    (50000,  float("inf"), "异常", "除非是长文档场景，否则几乎肯定存在设计问题"),
]


# ── 工具函数 ────────────────────────────────────────────────
def norm(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def pick(header, key):
    """在表头里找出该逻辑列对应的实际列名"""
    h = [norm(x) for x in header]
    for alias in ALIASES[key]:
        if alias in h:
            return header[h.index(alias)]
    # 退化匹配：包含关键字
    for alias in ALIASES[key]:
        for i, col in enumerate(h):
            if alias in col:
                return header[i]
    return None


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def human(n):
    n = float(n)
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:,.0f}{unit}" if unit == "" else f"{n:,.1f}{unit}"
        n /= 1000
    return f"{n:,.1f}T"


def money(x, cur="¥"):
    if x == 0:
        return f"{cur}0"
    ax = abs(x)
    if ax >= 10000:
        return f"{cur}{x/10000:,.1f}万"
    if ax >= 1:
        return f"{cur}{x:,.0f}"
    if ax >= 0.01:
        return f"{cur}{x:.3f}"
    return f"{cur}{x:.6f}"


def bar(frac, width=34):
    filled = int(round(max(0.0, min(1.0, frac)) * width))
    return "█" * filled + "·" * (width - filled)


# ── 读取 ────────────────────────────────────────────────────
def load(path):
    if path == "-":
        raw = sys.stdin.read()
    else:
        if not os.path.exists(path):
            sys.exit(f"文件不存在: {path}")
        if os.path.isdir(path):
            sys.exit(f"这是目录不是文件: {path}")
        try:
            with io.open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                raw = f.read()
        except OSError as e:
            sys.exit(f"读取失败: {e}")

    raw = raw.strip()
    if not raw:
        sys.exit("输入为空")

    # JSON 数组
    if raw[0] == "[":
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data, list(data[0].keys()) if data else []
        except json.JSONDecodeError:
            pass

    # JSONL：首行能解析成对象就按 JSONL
    first = raw.split("\n", 1)[0].strip()
    if first.startswith("{"):
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        if rows:
            keys = []
            for r in rows[:200]:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            return rows, keys

    # CSV
    sample = raw[:8192]
    delim = ","
    for cand in [",", "\t", ";", "|"]:
        if sample.count(cand) > sample.count(delim):
            delim = cand
    rdr = csv.DictReader(io.StringIO(raw), delimiter=delim)
    rows = list(rdr)
    return rows, list(rdr.fieldnames or [])


def to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(float(str(v).replace(",", "")))
    except (ValueError, TypeError):
        return None


# ── 主流程 ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="LLM 上下文画像工具：测量你的模型调用平均携带多少上下文",
        epilog="示例: python ctxbench.py calls.csv --price-in 2 --price-out 8")
    ap.add_argument("file", help="CSV / JSONL / JSON 文件，或 - 表示从 stdin 读")
    ap.add_argument("--input", help="input token 列名（默认自动识别）")
    ap.add_argument("--output", help="output token 列名")
    ap.add_argument("--model", help="模型列名")
    ap.add_argument("--price-in", type=float, default=2.0,
                    help="输入价格，元/百万 token（默认 2.0）")
    ap.add_argument("--price-out", type=float, default=8.0,
                    help="输出价格，元/百万 token（默认 8.0）")
    ap.add_argument("--target", type=float, default=8000,
                    help="目标上下文长度，用于估算可压缩量（默认 8000）")
    ap.add_argument("--currency", default="¥", help="货币符号")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而非报告")
    ap.add_argument("--quiet", action="store_true", help="只输出结论一行")
    a = ap.parse_args()

    rows, header = load(a.file)
    if not rows:
        sys.exit("没有解析到数据行")

    # 列定位
    col_in = a.input or pick(header, "input")
    col_out = a.output or pick(header, "output")
    col_model = a.model or pick(header, "model")

    if not col_in:
        sys.exit(
            "找不到 input token 列。\n"
            f"检测到的列: {', '.join(str(h) for h in header[:25])}\n"
            "请用 --input 指定，例如 --input prompt_tokens")

    # 抽取
    ins, outs, models = [], [], []
    for r in rows:
        v = to_int(r.get(col_in))
        if v is None or v < 0:
            continue
        ins.append(v)
        o = to_int(r.get(col_out)) if col_out else None
        outs.append(o if o is not None else 0)
        models.append(str(r.get(col_model, "unknown")) if col_model else "unknown")

    if not ins:
        sys.exit(f"列 '{col_in}' 里没有可用的数值")

    n = len(ins)
    ins_sorted = sorted(ins)
    avg = sum(ins) / n
    p50, p90, p99 = pct(ins_sorted, .5), pct(ins_sorted, .9), pct(ins_sorted, .99)
    mx = ins_sorted[-1]
    total_in = sum(ins)
    total_out = sum(outs)

    # 基准判定
    level, meaning = "健康", ""
    for lo, hi, name, desc in BENCH:
        if lo <= avg < hi:
            level, meaning = name, desc
            break

    # 浪费估算：超出目标的 token
    excess = sum(max(0, x - a.target) for x in ins)
    excess_ratio = excess / total_in if total_in else 0
    saved_cost = excess / 1e6 * a.price_in

    # 成本
    cost_in = total_in / 1e6 * a.price_in
    cost_out = total_out / 1e6 * a.price_out
    cost_all = cost_in + cost_out

    # 模型分布
    m_calls = Counter(models)
    m_tokens = defaultdict(int)
    for m, t in zip(models, ins):
        m_tokens[m] += t
    top_model, top_calls = m_calls.most_common(1)[0]
    top_share = top_calls / n

    # 直方图
    buckets = [(0, 2000), (2000, 8000), (8000, 20000),
               (20000, 50000), (50000, 100000), (100000, float("inf"))]
    hist = []
    for lo, hi in buckets:
        c = sum(1 for x in ins if lo <= x < hi)
        hist.append((lo, hi, c))

    # ── JSON 输出 ──────────────────────────────────────────
    result = {
        "calls": n,
        "avg_input": round(avg, 1),
        "p50_input": round(p50, 1), "p90_input": round(p90, 1),
        "p99_input": round(p99, 1), "max_input": mx,
        "avg_output": round(total_out / n, 1),
        "total_input": total_in, "total_output": total_out,
        "level": level, "meaning": meaning,
        "excess_tokens": excess,
        "excess_ratio": round(excess_ratio, 4),
        "est_saved_cost": round(saved_cost, 2),
        "est_total_cost": round(cost_all, 2),
        "input_cost_share": round(cost_in / cost_all, 4) if cost_all else 0,
        "top_model": top_model, "top_model_share": round(top_share, 4),
        "model_count": len(m_calls),
    }
    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if a.quiet:
        print(f"avg_input={int(avg)} level={level} excess={excess_ratio*100:.0f}%")
        return

    # ── 报告输出 ───────────────────────────────────────────
    W = 66
    print()
    print("═" * W)
    print("  ctxbench · LLM 上下文画像".center(W - 2))
    print("═" * W)

    print(f"\n  样本            {n:,} 次调用")
    print(f"  输入列          {col_in}")
    if col_out:
        print(f"  输出列          {col_out}")
    if col_model:
        print(f"  模型数          {len(m_calls)}")

    print(f"\n{'─'*W}")
    print("  上下文长度分布（input tokens）")
    print(f"{'─'*W}")
    print(f"  平均            {human(avg):>10}")
    print(f"  中位数          {human(p50):>10}")
    print(f"  P90             {human(p90):>10}")
    print(f"  P99             {human(p99):>10}")
    print(f"  最大            {human(mx):>10}")

    print()
    for lo, hi, c in hist:
        if c == 0:
            continue
        label = f"{human(lo)}–{human(hi)}" if hi != float("inf") else f"{human(lo)}+"
        print(f"  {label:>14}  {bar(c/n)}  {c/n*100:5.1f}%")

    print(f"\n{'─'*W}")
    print("  基准对照")
    print(f"{'─'*W}")
    print(f"  你的平均        {human(avg)}")
    print(f"  常见区间        1K–8K（健康） / 8K–20K（偏高） / >20K（严重）")
    print(f"  判定            {level} —— {meaning}")

    print(f"\n{'─'*W}")
    print("  成本测算")
    print(f"{'─'*W}")
    print(f"  输入占比        {cost_in/cost_all*100:.1f}%" if cost_all else "  输入占比        n/a")
    print(f"  样本总成本      {money(cost_all, a.currency)}"
          f"  (in {money(cost_in, a.currency)} / out {money(cost_out, a.currency)})")
    print(f"  单次均摊        {money(cost_all/n, a.currency)}")
    print(f"  计价参数        in {a.price_in}/M, out {a.price_out}/M")

    print(f"\n{'─'*W}")
    print("  可压缩空间")
    print(f"{'─'*W}")
    print(f"  目标上下文      {human(a.target)}")
    print(f"  超出目标的量    {human(excess)}  ({excess_ratio*100:.1f}% of 输入)")
    print(f"  本样本可省      {money(saved_cost, a.currency)}")
    if n:
        monthly = saved_cost / n * 300000
        print(f"  折算月度        {money(monthly, a.currency)}"
              f"  (假设月调用 30 万次，按本样本均值外推)")

    print(f"\n{'─'*W}")
    print("  模型分布")
    print(f"{'─'*W}")
    for m, c in m_calls.most_common(8):
        share = c / n
        print(f"  {m[:28]:<28} {share*100:5.1f}%  {human(m_tokens[m]):>9} tok")

    # ── 建议 ───────────────────────────────────────────────
    tips = []
    if avg >= 20000:
        tips.append(("上下文压缩", f"平均值 {human(avg)} 是健康上限的 {avg/8000:.1f} 倍",
                     "优先检查：是否每次都在拼接完整历史、是否把整篇文档塞进 prompt"))
    elif avg >= 8000:
        tips.append(("上下文压缩", f"平均值 {human(avg)} 偏高",
                     "检查历史对话是否做了摘要、system prompt 是否走了缓存"))
    if top_share > 0.7:
        tips.append(("模型路由", f"单模型占 {top_share*100:.0f}%，存在集中度风险",
                     "把分类/提取/改写这类短输出任务分流到小模型"))
    if total_out and total_in / total_out > 50:
        tips.append(("读多写少", f"输入是输出的 {total_in/total_out:.0f} 倍",
                     "这类场景压缩输入的收益最大，优先做"))
    if p99 > avg * 5:
        tips.append(("长尾治理", f"P99 是平均的 {p99/avg:.1f} 倍",
                     "极少数超长请求消耗了大量 token，可单独限流或截断"))

    if tips:
        print(f"\n{'─'*W}")
        print("  建议（按影响排序）")
        print(f"{'─'*W}")
        for i, (title, why, how) in enumerate(tips, 1):
            print(f"\n  {i}. {title}")
            print(f"     现象  {why}")
            print(f"     动作  {how}")

    print(f"\n{'═'*W}\n")


if __name__ == "__main__":
    main()
