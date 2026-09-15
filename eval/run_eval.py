#!/usr/bin/env python3
"""
発表・分析用の評価ランナー。

eval/tasks/*.yaml のタスクスイートを実際にPlanner経由(main.pyと同じ経路)で
実行し、以下を1回のバッチとして記録する:
  - 各タスクの最終ステータス・イテレーション数・出力
  - test_code による合否判定（entry_point/test_codeがある場合のみ）
  - サブタスクがmain/どのワーカーに割り振られたか、レイテンシ・トークン数
    （個別のevent log jsonlとして保存。集計はeval/visualize.py側で行う）

外部ベンチマーク(HumanEval/MBPP/APPS/LiveCodeBench)からタスクを一部だけ
サンプリングして混ぜることもできる(eval/benchmarks/参照)。

結果は eval/results/<timestamp>[_<label>]/ という実験ディレクトリ単位で保存される
(results.json, events/*.jsonl)。同じ結果に対する図はeval/visualize.pyが同じ
ディレクトリ内に生成する。過去の実験ディレクトリは自動では消えないので、
プロンプト変更前後などの比較がしやすい。

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --label prompt-improved
    python -m eval.run_eval --task-ids phase0-add-two-numbers phase1b-lru-cache
    python -m eval.run_eval --limit 3
    python -m eval.run_eval --no-local-tasks --humaneval 5 --mbpp 5
    python -m eval.run_eval --apps 3 --apps-difficulty introductory
    python -m eval.run_eval --livecodebench 3 --livecodebench-difficulty easy
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from orchestrator import EventLog, WorkerClient, Planner

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_FILES = [
    REPO_ROOT / "eval/tasks/phase0_smoke.yaml",
    REPO_ROOT / "eval/tasks/phase1_functions.yaml",
    REPO_ROOT / "eval/tasks/phase1b_composite.yaml",
]

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def load_local_tasks(task_files) -> list[dict]:
    tasks = []
    for path in task_files:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for t in data.get("tasks", []):
            t["source_file"] = str(Path(path).relative_to(REPO_ROOT))
            t.setdefault("grading_mode", "assert")
            t.setdefault("stdio_tests", None)
            tasks.append(t)
    return tasks


def build_prompt(task: dict) -> str:
    """
    タスクYAMLのprompt をそのまま使う（test_codeは採点専用のオラクルなので、
    モデルに「テストも書いて」と追加指示するのはあえてしない）。
    実際に試したところ、単発関数タスクに"also write tests"を追加すると、
    分解を誘発するどころか「1つのcompleteに実装+多数のテストを詰め込む」
    方向に転び、guided_json応答がmax_tokensで途中切断され
    No valid JSON found（パース失敗・abort）を多発させた。
    実装+テストの分解を見たい場合は、プロンプト自体にそれを明示した
    phase1b_composite.yamlのタスクを使う。
    """
    return task["prompt"].strip()


def extract_code(final_output: str) -> str:
    """final_outputからコード片を取り出す。```で囲まれていればその中身、なければ全体。"""
    match = CODE_BLOCK_RE.search(final_output)
    if match:
        return match.group(1)
    return final_output


def grade_task(task: dict, final_output: str) -> dict:
    """
    test_codeを使って合否判定する。サブプロセスで実行し、
    生成コードの構文エラー・無限ループ等がevalプロセス自体に影響しないようにする。
    """
    test_code = task.get("test_code")
    if not test_code or not final_output:
        return {"graded": False, "passed": None, "grade_error": None}

    code = extract_code(final_output)
    test_fn_match = re.search(r"def (test_\w+)\(", test_code)
    test_fn_name = test_fn_match.group(1) if test_fn_match else "test_main"

    harness = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code) + f"\n\n{test_fn_name}()\nprint('__EVAL_PASS__')\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        harness_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, harness_path],
            capture_output=True,
            text=True,
            timeout=task.get("timeout_s", 60),
        )
        passed = proc.returncode == 0 and "__EVAL_PASS__" in proc.stdout
        error = None if passed else ((proc.stderr or proc.stdout)[-2000:])
        return {"graded": True, "passed": passed, "grade_error": error}
    except subprocess.TimeoutExpired:
        return {"graded": True, "passed": False, "grade_error": "timeout"}
    finally:
        Path(harness_path).unlink(missing_ok=True)


def grade_stdio_task(task: dict, final_output: str) -> dict:
    """
    標準入出力形式(APPS/LiveCodeBenchのstdio-based問題)の合否判定。
    生成されたプログラムを`stdio_tests`の各ケースについてサブプロセスとして実行し、
    標準入力を渡して標準出力を期待値と突き合わせる。全ケースに合格した場合のみPASS。

    既知の制約: 出力の厳密な文字列一致で判定するため、正解が複数通りありうる
    問題(同値だが順序等が異なる出力も正解とみなされる問題)では、公式解答でさえ
    不合格になることがある(実際にAPPS introductoryの公式解答で確認済み)。
    厳密な採点には問題ごとのカスタムチェッカーが必要だが、このプロトタイプでは
    簡略化して厳密一致のみをサポートする。
    """
    stdio_tests = task.get("stdio_tests") or []
    if not stdio_tests or not final_output:
        return {"graded": False, "passed": None, "grade_error": None}

    code = extract_code(final_output)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        prog_path = f.name

    per_test_timeout = max(task.get("timeout_s", 60) / len(stdio_tests), 5)
    try:
        for i, tc in enumerate(stdio_tests):
            try:
                proc = subprocess.run(
                    [sys.executable, prog_path],
                    input=tc["input"],
                    capture_output=True,
                    text=True,
                    timeout=per_test_timeout,
                )
            except subprocess.TimeoutExpired:
                return {"graded": True, "passed": False, "grade_error": f"test case {i}: timeout"}

            actual = proc.stdout.strip()
            expected = tc["output"].strip()
            if actual != expected:
                error = (
                    f"test case {i} failed: expected {expected!r}, got {actual!r} "
                    f"(stderr: {(proc.stderr or '')[-500:]})"
                )
                return {"graded": True, "passed": False, "grade_error": error}
        return {"graded": True, "passed": True, "grade_error": None}
    finally:
        Path(prog_path).unlink(missing_ok=True)


async def run_one_task(planner: Planner, task: dict, max_iterations: int) -> dict:
    prompt = build_prompt(task)
    start = datetime.utcnow()
    try:
        exec_result = await planner.execute(prompt, max_iterations=max_iterations)
    except Exception as e:
        return {
            "task_id": task["id"],
            "source_file": task["source_file"],
            "prompt": prompt,
            "status": "crashed",
            "error": str(e),
            "iterations": 0,
            "final_output": "",
            "wall_clock_s": (datetime.utcnow() - start).total_seconds(),
            "graded": False,
            "passed": None,
            "grade_error": None,
        }
    wall_clock_s = (datetime.utcnow() - start).total_seconds()
    grading_mode = task.get("grading_mode", "assert")
    if grading_mode == "stdio":
        grade = grade_stdio_task(task, exec_result.final_output)
    else:
        grade = grade_task(task, exec_result.final_output)
    return {
        "task_id": task["id"],
        "source_file": task["source_file"],
        "prompt": prompt,
        "status": exec_result.status,
        "iterations": exec_result.iterations,
        "final_output": exec_result.final_output,
        "wall_clock_s": wall_clock_s,
        **grade,
    }


def load_benchmark_tasks(args) -> list[dict]:
    """--humaneval/--mbpp/--apps/--livecodebench で指定された分だけ外部ベンチマークから読み込む。"""
    tasks = []
    if args.humaneval:
        from eval.benchmarks import humaneval
        tasks += humaneval.load_tasks(n=args.humaneval, seed=args.benchmark_seed)
    if args.mbpp:
        from eval.benchmarks import mbpp
        tasks += mbpp.load_tasks(n=args.mbpp, seed=args.benchmark_seed)
    if args.apps:
        from eval.benchmarks import apps
        tasks += apps.load_tasks(n=args.apps, seed=args.benchmark_seed, difficulty=args.apps_difficulty)
    if args.livecodebench:
        from eval.benchmarks import livecodebench
        tasks += livecodebench.load_tasks(
            n=args.livecodebench, seed=args.benchmark_seed, difficulty=args.livecodebench_difficulty
        )
    for t in tasks:
        t.setdefault("grading_mode", "assert")
    return tasks


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="発表・分析用の評価ランナー")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/workers.yaml"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "eval/results"),
                        help="実験ディレクトリ(eval/results/<batch_id>[_label]/)を作る親ディレクトリ")
    parser.add_argument("--label", default=None,
                        help="実験ディレクトリ名にタイムスタンプの後ろへ付けるラベル(例: prompt-improved)")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--main-vllm-base-url",
        default=os.environ.get("MAIN_VLLM_BASE_URL")
        or f"http://localhost:{os.environ.get('MAIN_VLLM_PORT', 8000)}",
    )
    parser.add_argument(
        "--main-model-name",
        default=os.environ.get("MAIN_MODEL_NAME", "default"),
        help="メインLLMの`model`フィールドに使う名前(vllm serveの--served-model-nameと一致させる)",
    )
    parser.add_argument("--limit", type=int, default=None, help="実行するタスク数の上限(デバッグ用)")
    parser.add_argument("--task-ids", nargs="*", default=None, help="実行するtask idを限定")
    parser.add_argument("--no-local-tasks", action="store_true", help="eval/tasks/*.yamlのタスクを含めない")
    parser.add_argument("--humaneval", type=int, default=0, help="HumanEvalから取得する問題数")
    parser.add_argument("--mbpp", type=int, default=0, help="MBPPから取得する問題数")
    parser.add_argument("--apps", type=int, default=0, help="APPSから取得する問題数")
    parser.add_argument("--apps-difficulty", choices=["introductory", "interview", "competition"], default=None)
    parser.add_argument("--livecodebench", type=int, default=0, help="LiveCodeBenchから取得する問題数")
    parser.add_argument("--livecodebench-difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--benchmark-seed", type=int, default=0, help="ベンチマークからのサンプリングのシード")
    args = parser.parse_args()

    run_batch_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    batch_dir_name = f"{run_batch_id}_{args.label}" if args.label else run_batch_id
    batch_dir = Path(args.out_dir) / batch_dir_name
    event_log_dir = batch_dir / "events"
    event_log_dir.mkdir(parents=True, exist_ok=True)

    tasks = [] if args.no_local_tasks else load_local_tasks(TASK_FILES)
    tasks += load_benchmark_tasks(args)

    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [t for t in tasks if t["id"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Loaded {len(tasks)} tasks: {[t['id'] for t in tasks]}", flush=True)

    results = []
    async with WorkerClient(args.config) as worker_client:
        for worker in worker_client.workers_config.get("workers", []):
            await worker_client.setup_tunnel(worker["id"])
            print(f"Tunnel established for {worker['id']}", flush=True)

        for task in tasks:
            print(f"\n=== Running {task['id']} ===", flush=True)
            event_log = EventLog(event_log_dir / f"{task['id']}.jsonl")
            async with Planner(
                main_llm_base_url=args.main_vllm_base_url,
                event_log=event_log,
                worker_client=worker_client,
                model_name=args.main_model_name,
            ) as planner:
                result = await run_one_task(planner, task, args.max_iterations)
            result["event_log_file"] = str(Path(event_log.log_file).relative_to(REPO_ROOT))
            results.append(result)
            print(
                f"  status={result['status']} iterations={result['iterations']} "
                f"passed={result.get('passed')} wall_clock={result['wall_clock_s']:.1f}s",
                flush=True,
            )

    out_file = batch_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"run_batch_id": run_batch_id, "label": args.label, "results": results},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\nWrote {out_file}", flush=True)
    return out_file


if __name__ == "__main__":
    asyncio.run(main())
