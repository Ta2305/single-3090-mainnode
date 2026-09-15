#!/usr/bin/env python3
"""
比較実験用: 委譲を一切行わず、メインLLM(vLLM)だけでタスクを完結させた
場合の処理時間・合否を測定するランナー。

既存の eval/run_eval.py・orchestrator/planner.py は一切変更しない。
`Planner`をサブクラス化し、decomposeで生成された全サブタスクの
`assigned_to`を強制的に"main"へ上書きすることでワーカーへの委譲を
禁止する(ワーカー機への接続自体を行わないため、SSH/ワーカー機が
利用できない環境でも実行できる)。タスク読み込み・採点・結果保存は
eval/run_eval.py の既存関数をそのまま再利用する。

出力先の実験ディレクトリ形式は eval/run_eval.py と同じなので、
eval/visualize.py でそのまま同じ3枚の図を生成できる。

Usage:
    python -m eval.run_eval_mainonly
    python -m eval.run_eval_mainonly --label main-only
    python -m eval.run_eval_mainonly --task-ids phase1b-lru-cache
"""
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from orchestrator import EventLog, Planner

from eval.run_eval import REPO_ROOT, TASK_FILES, load_local_tasks, run_one_task


class NoWorkerClient:
    """
    委譲を行わないため、本物のWorkerClientの代わりに渡す最小限のスタブ。
    Planner.execute()がsystem_context構築時に参照する`workers_config`
    だけを持つ(空のワーカーリストなので、プロンプト上も「使えるワーカーはいない」
    という説明になる)。`call_worker`は定義しない — MainOnlyPlannerが
    全サブタスクをmain行きに強制するため、呼ばれることはない。
    """
    workers_config = {"workers": []}


class MainOnlyPlanner(Planner):
    """
    既存のPlannerを継承し、decompose時に生成された全サブタスクの
    `assigned_to`を強制的に"main"へ上書きするサブクラス。
    メインLLM自身がworker-1を選ぼうとしても、実際には常にmainが処理する。
    Plannerの他のロジック(プロンプト内容・process_result判断等)は一切変更しない。
    """
    async def _dispatch_subtasks(self, subtasks: list, user_task: str) -> None:
        forced_subtasks = []
        for subtask_data in subtasks:
            subtask_data = dict(subtask_data)
            if subtask_data.get("assigned_to") != "main":
                subtask_data["assigned_to"] = "main"
            forced_subtasks.append(subtask_data)
        await super()._dispatch_subtasks(forced_subtasks, user_task)


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="委譲なし(main単独)比較実験ランナー")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "eval/results"))
    parser.add_argument("--label", default="main-only", help="実験ディレクトリ名に付けるラベル")
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
    args = parser.parse_args()

    run_batch_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    batch_dir_name = f"{run_batch_id}_{args.label}" if args.label else run_batch_id
    batch_dir = Path(args.out_dir) / batch_dir_name
    event_log_dir = batch_dir / "events"
    event_log_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_local_tasks(TASK_FILES)
    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [t for t in tasks if t["id"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Loaded {len(tasks)} tasks (main-only, no delegation): {[t['id'] for t in tasks]}", flush=True)

    no_worker_client = NoWorkerClient()
    results = []
    for task in tasks:
        print(f"\n=== Running {task['id']} (main-only) ===", flush=True)
        event_log = EventLog(event_log_dir / f"{task['id']}.jsonl")
        async with MainOnlyPlanner(
            main_llm_base_url=args.main_vllm_base_url,
            event_log=event_log,
            worker_client=no_worker_client,
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
