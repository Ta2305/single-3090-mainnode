"""
ワーカーノード への HTTP クライアント。

実際の`ssh`バイナリをサブプロセスとして`-J`(ProxyJump)付きで起動し、
メインノード -> 踏み台 -> ワーカー、という本物の2段SSHログインを行った上で、
ワーカー自身のループバック(127.0.0.1)上の推論ポートへローカルフォワードする
（`ssh -J <jumphost> -L <local_port>:127.0.0.1:<inference_port> <worker>`）。

この方式では、踏み台からワーカーのLANインターフェース宛に接続することが
一度もないため、ワーカー側のファイアウォールで推論ポートを開放する必要がない
（実機での動作検証により確定した方式）。
SSHの多段認証はOpenSSHクライアント自体にそのまま任せ、Python側では
SSHプロトコルの再実装を行わない。
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

import httpx
import yaml

from .schema import Task, Result
from .llm_client import call_chat_completion, extract_content, extract_token_usage

logger = logging.getLogger(__name__)


class WorkerClient:
    """
    ワーカーノードへのHTTPクライアント。

    Usage:
        async with WorkerClient('configs/workers.yaml') as client:
            result = await client.call_worker('worker-1', task)
    """

    def __init__(self, config_file: Path | str):
        """
        Args:
            config_file: workers.yaml のパス
        """
        self.config_file = Path(config_file)
        self.workers_config = self._load_config()
        self.tunnels: Dict[str, asyncio.subprocess.Process] = {}
        self.client: Optional[httpx.AsyncClient] = None

    def _load_config(self) -> dict:
        """workers.yaml を読み込む。"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def get_worker_config(self, worker_id: str) -> Optional[dict]:
        """ワーカー設定を取得。"""
        for worker in self.workers_config.get('workers', []):
            if worker['id'] == worker_id:
                return worker
        return None

    def _build_ssh_command(self, worker: dict) -> list[str]:
        """
        `ssh -J <jumphost> -L <local_port>:127.0.0.1:<inference_port> <worker> -N`
        を組み立てる。

        `-L`の転送先は"127.0.0.1"（ワーカー自身のsshdが解決するループバック）であり、
        踏み台やメインノードから見たワーカーのLAN IPではない点に注意
        （だからこそワーカー側のファイアウォールを一切通らない）。
        """
        ssh_username = worker.get('ssh_username')
        jumphost = worker['ssh_jumphost']
        hostname = worker['hostname']
        inference_port = worker['inference_port']
        local_port = worker['local_forward_port']

        jump_target = f"{ssh_username}@{jumphost}" if ssh_username else jumphost
        worker_target = f"{ssh_username}@{hostname}" if ssh_username else hostname

        return [
            "ssh",
            "-J", jump_target,
            "-L", f"{local_port}:127.0.0.1:{inference_port}",
            "-N",  # リモートコマンドを実行しない（ポートフォワードのみ）
            "-o", "ExitOnForwardFailure=yes",  # フォワード確立失敗時に即座に終了させる
            "-o", "BatchMode=yes",  # パスワード入力待ち等でハングしないようにする
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            worker_target,
        ]

    async def setup_tunnel(self, worker_id: str, connect_timeout: float = 15.0):
        """
        SSHトンネル(ローカルポートフォワード)をサブプロセスとして確立する。

        Args:
            worker_id: ワーカーID
            connect_timeout: ローカルポートが実際に接続を受け付けるまでの
                              待ち時間の上限(秒)
        """
        worker = self.get_worker_config(worker_id)
        if not worker:
            raise ValueError(f"Worker {worker_id} not found in config")

        if not worker.get('ssh_username'):
            logger.warning(
                f"No ssh_username set for {worker_id} in workers.yaml; "
                f"falling back to local OS user for both SSH hops. "
                f"This will fail if the jump host / worker login user differs."
            )

        cmd = self._build_ssh_command(worker)
        logger.info(f"Setting up SSH tunnel for {worker_id}: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.tunnels[worker_id] = process

        local_port = worker['local_forward_port']
        deadline = time.monotonic() + connect_timeout
        last_error: Optional[str] = None

        while time.monotonic() < deadline:
            if process.returncode is not None:
                # sshプロセスが早期に終了した = フォワード確立に失敗した
                stderr_bytes = await process.stderr.read()
                last_error = stderr_bytes.decode(errors="replace").strip()
                break
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", local_port)
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                logger.info(
                    f"Tunnel for {worker_id} established: "
                    f"localhost:{local_port} -> (ssh -J経由) -> "
                    f"{worker['hostname']}:127.0.0.1:{worker['inference_port']}"
                )
                return
            except OSError:
                await asyncio.sleep(0.3)

        # タイムアウト、またはプロセスが異常終了した場合
        del self.tunnels[worker_id]
        if process.returncode is None:
            process.kill()
            await process.wait()

        message = f"Failed to establish tunnel for {worker_id} within {connect_timeout}s"
        if last_error:
            message += f": {last_error}"
        raise RuntimeError(message)

    async def close_tunnel(self, worker_id: str):
        """SSHトンネル(サブプロセス)を終了する。"""
        process = self.tunnels.get(worker_id)
        if process is None:
            return
        try:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            logger.info(f"Tunnel for {worker_id} closed")
        except Exception as e:
            logger.error(f"Error closing tunnel for {worker_id}: {e}")
        finally:
            self.tunnels.pop(worker_id, None)

    async def __aenter__(self):
        """コンテキストマネージャー開始。"""
        self.client = httpx.AsyncClient(timeout=300.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了。"""
        if self.client:
            await self.client.aclose()
        for worker_id in list(self.tunnels.keys()):
            await self.close_tunnel(worker_id)

    def _build_messages(self, task: Task) -> list[Dict[str, str]]:
        """Task から OpenAI形式のメッセージリストを構築。"""
        system_prompt = (
            "You are a helpful code assistant. "
            "You will be given a programming task and must provide clear, working code. "
            "Output only the code or solution, without markdown formatting unless requested."
        )

        context_str = ""
        if task.context:
            context_str = "Context:\n" + "\n".join(
                f"- {k}: {str(v)[:500]}" for k, v in task.context.items()
            ) + "\n\n"

        user_prompt = (
            f"{context_str}"
            f"Task: {task.description}\n"
            f"Expected output: {task.expected_output}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    async def call_worker(self, worker_id: str, task: Task) -> Result:
        """
        ワーカーにタスクを送出し、結果を取得。

        Args:
            worker_id: ワーカーID (e.g., "worker-1")
            task: Task オブジェクト

        Returns:
            Result オブジェクト
        """
        if not self.client:
            raise RuntimeError(
                "WorkerClient not initialized. Use 'async with WorkerClient(...)' context."
            )

        worker = self.get_worker_config(worker_id)
        if not worker:
            return Result(
                task_id=task.id,
                status="error",
                output="",
                raw_response="",
                tokens_used=0,
                latency_ms=0.0,
                worker_id=worker_id,
                error_message=f"Worker {worker_id} not found"
            )

        # トンネルがなければセットアップ
        if worker_id not in self.tunnels:
            try:
                await self.setup_tunnel(worker_id)
            except Exception as e:
                logger.error(f"Failed to setup tunnel: {e}")
                return Result(
                    task_id=task.id,
                    status="error",
                    output="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    worker_id=worker_id,
                    error_message=f"Tunnel setup failed: {e}"
                )

        base_url = f"http://127.0.0.1:{worker['local_forward_port']}"
        messages = self._build_messages(task)

        try:
            start_time = time.time()
            response_data = await call_chat_completion(
                client=self.client,
                base_url=base_url,
                messages=messages,
                model=worker.get('model', 'default'),
                temperature=0.3,
                max_tokens=4096,
                top_p=0.9
            )
            latency_ms = (time.time() - start_time) * 1000

            output = extract_content(response_data)
            tokens_used = extract_token_usage(response_data)

            logger.info(
                f"Worker {worker_id} completed task {task.id}: "
                f"{tokens_used} tokens, {latency_ms:.1f}ms"
            )

            return Result(
                task_id=task.id,
                status="ok",
                output=output,
                raw_response=str(response_data),
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                worker_id=worker_id
            )

        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            return Result(
                task_id=task.id,
                status="error",
                output="",
                raw_response="",
                tokens_used=0,
                latency_ms=0.0,
                worker_id=worker_id,
                error_message=str(e)
            )
