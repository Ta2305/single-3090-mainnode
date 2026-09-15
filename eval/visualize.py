#!/usr/bin/env python3
"""
eval/run_eval.py の出力(結果JSON + event log群)を読み込み、
スライドに直接貼れる図(PNG)を生成する。

生成する図:
  1. delegation_breakdown.png — タスクごとのサブタスク割り振り先
     (メインLLM直接完了 / main / worker-1)
  2. latency_breakdown.png    — タスクごとの処理時間の内訳
     (main実行時間 / worker実行時間 / プランニングその他)
  3. pass_rate.png            — タスクごとの合否 (test_codeによる自動採点)
  4. summary.md               — 上記の元データを表にしたMarkdown

配色: main = blue #2a78d6, worker-1 = orange #eb6834,
  direct(委譲なし) = muted gray #898781, pass = green #0ca30c,
  fail = red #d03b3b。

図・summary.mdは、読み込んだresults.jsonと同じ実験ディレクトリ
(eval/results/<batch_id>[_label]/)内に生成される。過去の実験ディレクトリは
上書きされないので、プロンプト変更前後などをディレクトリ単位で見比べられる。

Usage:
    python -m eval.visualize eval/results/<batch_id>/results.json
    python -m eval.visualize eval/results/<batch_id>/
    python -m eval.visualize --latest   # eval/results/内の最新の実験ディレクトリを使う
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 日本語ラベルのための和文フォント指定（このマシンにNoto Sans CJK JPが入っている前提）。
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "sans-serif"]

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- 配色定義 ---
COLOR_MAIN = "#2a78d6"       # categorical slot 1 (blue)
COLOR_WORKER = "#eb6834"     # categorical slot 2 (orange)
COLOR_DIRECT = "#898781"     # muted (委譲なしの直接完了)
COLOR_OTHER = "#c3c2b7"      # baseline寄りの薄いグレー(プランニング等のその他時間)
COLOR_PASS = "#0ca30c"       # status good
COLOR_FAIL = "#d03b3b"       # status critical
COLOR_UNGRADED = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_results(results_file: Path) -> dict:
    with open(results_file, encoding="utf-8") as f:
        return json.load(f)


def load_events(event_log_file: Path) -> list[dict]:
    path = REPO_ROOT / event_log_file
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize_task(result: dict) -> dict:
    """1タスク分のevent logから、委譲内訳・レイテンシ内訳を集計する。"""
    events = load_events(Path(result["event_log_file"]))

    subtask_counts = defaultdict(int)  # {"main": n, "worker-1": m, ...}
    latency_by_executor = defaultdict(float)
    tokens_by_executor = defaultdict(int)
    n_results_ok = 0
    n_results_error = 0

    for ev in events:
        content = ev.get("content", {})
        if ev["event_type"] == "task_created":
            subtask_counts[content["assigned_to"]] += 1
        elif ev["event_type"] == "result_received":
            worker_id = content.get("worker_id", "unknown")
            latency_by_executor[worker_id] += content.get("latency_ms") or 0.0
            tokens_by_executor[worker_id] += content.get("tokens_used") or 0
            if content.get("status") == "ok":
                n_results_ok += 1
            else:
                n_results_error += 1

    total_subtasks = sum(subtask_counts.values())
    return {
        "task_id": result["task_id"],
        "source_file": result["source_file"],
        "status": result["status"],
        "iterations": result["iterations"],
        "wall_clock_s": result["wall_clock_s"],
        "passed": result.get("passed"),
        "graded": result.get("graded"),
        "subtask_counts": dict(subtask_counts),
        "total_subtasks": total_subtasks,
        "delegated": subtask_counts.get("worker-1", 0) > 0,
        "latency_by_executor_ms": dict(latency_by_executor),
        "tokens_by_executor": dict(tokens_by_executor),
        "n_results_ok": n_results_ok,
        "n_results_error": n_results_error,
    }


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_delegation_breakdown(summaries: list[dict], out_path: Path):
    task_ids = [s["task_id"] for s in summaries]
    y = range(len(task_ids))

    direct_vals = [0 if s["total_subtasks"] > 0 else 1 for s in summaries]
    main_vals = [s["subtask_counts"].get("main", 0) for s in summaries]
    worker_vals = [s["subtask_counts"].get("worker-1", 0) for s in summaries]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(task_ids) + 1.0))
    style_axes(ax)

    left = [0] * len(task_ids)
    bars_direct = ax.barh(y, direct_vals, left=left, color=COLOR_DIRECT, height=0.6,
                           label="委譲なし(メインLLMが直接complete)")
    left = [l + v for l, v in zip(left, direct_vals)]
    bars_main = ax.barh(y, main_vals, left=left, color=COLOR_MAIN, height=0.6,
                         label="main(サブタスクとして自己処理)")
    left = [l + v for l, v in zip(left, main_vals)]
    bars_worker = ax.barh(y, worker_vals, left=left, color=COLOR_WORKER, height=0.6,
                           label="worker-1(3060へ委譲)")

    for bars in (bars_direct, bars_main, bars_worker):
        for b in bars:
            if b.get_width() > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2,
                    str(int(b.get_width())), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold",
                )

    ax.set_yticks(list(y))
    ax.set_yticklabels(task_ids, color=INK_PRIMARY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("サブタスク数", color=INK_SECONDARY, fontsize=10)
    ax.set_title("タスクごとのサブタスク割り振り先", color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=1,
                        frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_extra_artists=(legend,), bbox_inches="tight")
    plt.close(fig)


def plot_latency_breakdown(summaries: list[dict], out_path: Path):
    task_ids = [s["task_id"] for s in summaries]
    y = range(len(task_ids))

    main_s = [s["latency_by_executor_ms"].get("main", 0) / 1000 for s in summaries]
    worker_s = [s["latency_by_executor_ms"].get("worker-1", 0) / 1000 for s in summaries]
    other_s = [
        max(0.0, s["wall_clock_s"] - (s["latency_by_executor_ms"].get("main", 0) +
                                       s["latency_by_executor_ms"].get("worker-1", 0)) / 1000)
        for s in summaries
    ]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(task_ids) + 1.0))
    style_axes(ax)

    left = [0] * len(task_ids)
    ax.barh(y, main_s, left=left, color=COLOR_MAIN, height=0.6, label="main LLM 実行時間")
    left = [l + v for l, v in zip(left, main_s)]
    ax.barh(y, worker_s, left=left, color=COLOR_WORKER, height=0.6, label="worker-1 実行時間")
    left = [l + v for l, v in zip(left, worker_s)]
    ax.barh(y, other_s, left=left, color=COLOR_OTHER, height=0.6, label="プランニング/その他")

    max_total = max(main_s[i] + worker_s[i] + other_s[i] for i in range(len(task_ids))) if task_ids else 1
    for i, s in enumerate(summaries):
        ax.text(s["wall_clock_s"] + max_total * 0.02, i,
                f"{s['wall_clock_s']:.1f}s", va="center", ha="left",
                color=INK_SECONDARY, fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(task_ids, color=INK_PRIMARY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("処理時間 (秒)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("タスクごとの処理時間の内訳", color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=1,
                        frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_extra_artists=(legend,), bbox_inches="tight")
    plt.close(fig)


def plot_pass_rate(summaries: list[dict], out_path: Path):
    task_ids = [s["task_id"] for s in summaries]
    y = range(len(task_ids))

    colors = []
    for s in summaries:
        if not s["graded"]:
            colors.append(COLOR_UNGRADED)
        elif s["passed"]:
            colors.append(COLOR_PASS)
        else:
            colors.append(COLOR_FAIL)

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(task_ids) + 1.2))
    style_axes(ax)
    ax.barh(y, [1] * len(task_ids), color=colors, height=0.6)

    for i, s in enumerate(summaries):
        label = "PASS" if s["passed"] else ("採点対象外" if not s["graded"] else "FAIL")
        ax.text(0.5, i, label, va="center", ha="center", color="white",
                fontsize=9, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(task_ids, color=INK_PRIMARY, fontsize=9)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.xaxis.grid(False)
    n_graded = sum(1 for s in summaries if s["graded"])
    n_passed = sum(1 for s in summaries if s["passed"])
    ax.set_title(
        f"タスクごとの合否(test_codeによる自動採点) — {n_passed}/{n_graded} PASS",
        color=INK_PRIMARY, fontsize=13, loc="left", pad=12,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def write_summary_md(summaries: list[dict], out_path: Path, run_batch_id: str):
    lines = [
        f"# 評価バッチ結果サマリー ({run_batch_id})",
        "",
        "| task_id | status | iterations | 委譲 | main数 | worker数 | 合否 | wall_clock(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        delegated = "○" if s["delegated"] else "×"
        passed = "PASS" if s["passed"] else ("N/A" if not s["graded"] else "FAIL")
        lines.append(
            f"| {s['task_id']} | {s['status']} | {s['iterations']} | {delegated} | "
            f"{s['subtask_counts'].get('main', 0)} | {s['subtask_counts'].get('worker-1', 0)} | "
            f"{passed} | {s['wall_clock_s']:.1f} |"
        )
    lines.append("")
    n_graded = sum(1 for s in summaries if s["graded"])
    n_passed = sum(1 for s in summaries if s["passed"])
    n_delegated = sum(1 for s in summaries if s["delegated"])
    lines.append(f"- 合否: {n_passed}/{n_graded} PASS")
    lines.append(f"- 委譲が発生したタスク: {n_delegated}/{len(summaries)}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_path", nargs="?",
        help="eval/results/<batch_id>/results.json、または実験ディレクトリそのもの",
    )
    parser.add_argument("--latest", action="store_true", help="eval/results/内の最新の実験ディレクトリを使う")
    parser.add_argument("--out-dir", default=None, help="省略時は実験ディレクトリ自身に出力")
    args = parser.parse_args()

    results_root = REPO_ROOT / "eval/results"
    if args.latest or not args.results_path:
        candidates = sorted(
            (p for p in results_root.glob("*/results.json")),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit("eval/results/に実験ディレクトリ(results.json)が見つかりません。先にrun_eval.pyを実行してください。")
        results_file = candidates[-1]
    else:
        path = Path(args.results_path)
        results_file = path / "results.json" if path.is_dir() else path

    data = load_results(results_file)
    run_batch_id = data["run_batch_id"]
    summaries = [summarize_task(r) for r in data["results"]]

    batch_dir = results_file.parent
    out_dir = Path(args.out_dir) if args.out_dir else batch_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_delegation_breakdown(summaries, out_dir / "delegation_breakdown.png")
    plot_latency_breakdown(summaries, out_dir / "latency_breakdown.png")
    plot_pass_rate(summaries, out_dir / "pass_rate.png")
    write_summary_md(summaries, out_dir / "summary.md", run_batch_id)

    print(f"Wrote figures + summary to {out_dir} (batch {run_batch_id})")


if __name__ == "__main__":
    main()
