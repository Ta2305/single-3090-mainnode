#!/usr/bin/env python3
"""
エントリーポイント。単一のコーディング指示を受け取り、
プランナーループを1回最後まで実行する。

事前に以下がそれぞれ別プロセスで起動済みであること(README.md参照):
  - メインノード: `vllm serve <model> --port 8000 ...`
  - ワーカーノード: `llama-server --host 0.0.0.0 --port 8080 ...`

Usage:
    python main.py "write a function that adds two numbers"
"""
import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import httpx

from orchestrator import EventLog, WorkerClient, Planner

# 終了コード規約（README.mdにも記載）
EXIT_OK = 0
EXIT_CRASH = 1
EXIT_NOT_CONVERGED = 2
EXIT_WORKER_UNREACHABLE = 3
EXIT_MAIN_LLM_UNREACHABLE = 4


class WorkerUnreachableError(RuntimeError):
    pass


class MainLLMUnreachableError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heterogeneous-LLM coding agent (research prototype)"
    )
    p.add_argument("task", help="ユーザーからのコーディング指示")
    p.add_argument("--config", default="configs/workers.yaml", help="ワーカー設定ファイル")
    p.add_argument("--log-dir", default=None, help="ログ出力先（既定: $LOG_DIR または ./logs）")
    p.add_argument("--run-id", default=None, help="実行ID（既定: タイムスタンプ+乱数）")
    p.add_argument("--max-iterations", type=int, default=10, help="プランナーループの最大反復回数")
    p.add_argument(
        "--main-vllm-base-url",
        default=None,
        help="メインLLMサーバーのbase_url（既定: $MAIN_VLLM_BASE_URL、"
             "なければ http://localhost:$MAIN_VLLM_PORT）"
    )
    p.add_argument(
        "--main-model-name",
        default=None,
        help="メインLLMの`model`フィールドに使う名前。`vllm serve`起動時の"
             "--served-model-nameと一致させること（既定: $MAIN_MODEL_NAME、"
             "なければ'default'）"
    )
    p.add_argument(
        "--worker-precheck-timeout",
        type=float,
        default=15.0,
        help="ワーカー疎通確認のタイムアウト秒数"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="デバッグログを有効化")
    return p.parse_args()


def setup_logging(log_dir: Path, run_id: str, verbose: bool) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    human_log = log_dir / f"orchestrator_{run_id}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(human_log), logging.StreamHandler(sys.stderr)],
    )
    return human_log


async def precheck_worker(worker_client: WorkerClient, worker_id: str, timeout: float) -> None:
    """
    トンネルを張り、ワーカーのHTTPエンドポイントへの疎通確認をする。

    main.py はこれを「高価なメインモデルへの接続確認より先」に呼ぶ。
    ワーカー到達不可はユーザーからのフィードバックで最も起こりやすい失敗モードであり、
    数分かかるメインモデルのロード/接続を待たずに数秒で検出できるようにするため。
    """
    try:
        await asyncio.wait_for(worker_client.setup_tunnel(worker_id), timeout=timeout)
    except Exception as e:
        raise WorkerUnreachableError(
            f"SSH tunnel setup failed for {worker_id}: {e}"
        ) from e

    worker_cfg = worker_client.get_worker_config(worker_id)
    base_url = f"http://127.0.0.1:{worker_cfg['local_forward_port']}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as probe:
            resp = await probe.get(f"{base_url}/v1/models")
            resp.raise_for_status()
    except Exception as e:
        raise WorkerUnreachableError(
            f"Tunnel to {worker_id} is up but llama-server did not respond on "
            f"{base_url}. Is llama-server running on the worker with --host 0.0.0.0? "
            f"(see the troubleshooting section in README.md)"
        ) from e


async def precheck_main_llm(base_url: str) -> None:
    """メインLLMサーバーへの疎通確認。"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as probe:
            resp = await probe.get(f"{base_url}/v1/models")
            resp.raise_for_status()
    except Exception as e:
        raise MainLLMUnreachableError(
            f"Could not reach main vLLM server at {base_url}. "
            f"Did you start it with `vllm serve ...`? (see README.md)"
        ) from e


async def run(args: argparse.Namespace) -> int:
    load_dotenv()

    run_id = args.run_id or (
        datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    )
    log_dir = Path(args.log_dir or os.environ.get("LOG_DIR", "./logs"))
    human_log_path = setup_logging(log_dir, run_id, args.verbose)
    event_log = EventLog(log_dir / f"events_{run_id}.jsonl")

    logger = logging.getLogger(__name__)
    logger.info(f"Run ID: {run_id}")

    async with WorkerClient(args.config) as worker_client:
        # 1. ワーカー疎通確認を、メインLLMへの接続より先に行う。
        for worker in worker_client.workers_config.get("workers", []):
            try:
                await precheck_worker(worker_client, worker["id"], args.worker_precheck_timeout)
                logger.info(f"Worker {worker['id']} reachable.")
            except WorkerUnreachableError as e:
                event_log.log_error("worker_unreachable", str(e))
                print(f"ERROR: {e}\n\nSee {human_log_path} and the README.md troubleshooting section.", file=sys.stderr)
                return EXIT_WORKER_UNREACHABLE

        # 2. メインLLMサーバーへの疎通確認。
        main_base_url = (
            args.main_vllm_base_url
            or os.environ.get("MAIN_VLLM_BASE_URL")
            or f"http://localhost:{os.environ.get('MAIN_VLLM_PORT', 8000)}"
        )
        main_model_name = args.main_model_name or os.environ.get("MAIN_MODEL_NAME", "default")
        try:
            await precheck_main_llm(main_base_url)
            logger.info(f"Main LLM reachable at {main_base_url}.")
        except MainLLMUnreachableError as e:
            event_log.log_error("main_llm_unreachable", str(e))
            print(f"ERROR: {e}\n\nSee {human_log_path} and the README.md troubleshooting section.", file=sys.stderr)
            return EXIT_MAIN_LLM_UNREACHABLE

        # 3. プランナー実行。
        async with Planner(
            main_llm_base_url=main_base_url,
            event_log=event_log,
            worker_client=worker_client,
            model_name=main_model_name,
        ) as planner:
            try:
                exec_result = await planner.execute(args.task, max_iterations=args.max_iterations)
            except Exception:
                logger.exception("Planner crashed")
                event_log.log_error("planner_crash", "unhandled exception, see human log for traceback")
                return EXIT_CRASH

    if exec_result.status != "completed":
        print(
            f"Task did not complete (status={exec_result.status}, "
            f"iterations={exec_result.iterations}/{args.max_iterations}). "
            f"See {event_log.log_file} and {human_log_path} "
            f"(look for parse_error / unknown_action events).",
            file=sys.stderr,
        )
        return EXIT_NOT_CONVERGED

    print(exec_result.final_output)
    return EXIT_OK


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
