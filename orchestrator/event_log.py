"""
イベントログ管理。

全てのタスク送出、結果受信、プランナーの判断を JSONL に記録し、
後から実験分析や再現に用いる。
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from .schema import Event, Task, Result


logger = logging.getLogger(__name__)


class EventLog:
    """
    イベントログを JSONL ファイルに記録。

    Usage:
        log = EventLog('logs/experiment_001.jsonl')
        log.log_task_created(task)
        log.log_result_received(result)
    """

    def __init__(self, log_file: Path | str):
        """
        Args:
            log_file: ログを記録するファイルパス（JSONL形式）
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # ログファイルが既に存在する場合はアペンド
        self._open_file()

    def _open_file(self):
        """ログファイルをアペンドモードで開く（またはスキップ）。"""
        # write時にアペンドしていくため、ここでは何もしない
        if self.log_file.exists():
            logger.info(f"Appending to existing log file: {self.log_file}")
        else:
            logger.info(f"Creating new log file: {self.log_file}")

    def _write_event(self, event: Event):
        """イベントを JSONL に書き込む。"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(event.to_json_line() + '\n')

    def log_task_created(self, task: Task, assigned_by: str = "planner"):
        """タスク作成をログ記録。"""
        event = Event(
            event_type="task_created",
            content={
                "task_id": task.id,
                "assigned_to": task.assigned_to,
                "description_preview": task.description[:100],
                "assigned_by": assigned_by
            }
        )
        self._write_event(event)

    def log_task_sent(self, task_id: str, worker_id: str, context_size_bytes: int):
        """タスク送出をログ記録。"""
        event = Event(
            event_type="task_sent",
            content={
                "task_id": task_id,
                "worker_id": worker_id,
                "context_size_bytes": context_size_bytes
            }
        )
        self._write_event(event)

    def log_result_received(self, result: Result):
        """結果受信をログ記録。"""
        event = Event(
            event_type="result_received",
            content={
                "task_id": result.task_id,
                "worker_id": result.worker_id,
                "status": result.status,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
                "output_preview": result.output[:100] if result.output else None,
                "error_message": result.error_message
            }
        )
        self._write_event(event)

    def log_planner_decision(self, decision: str, task_id: Optional[str] = None,
                            reasoning: Optional[str] = None):
        """
        プランナーの判断をログ記録。

        Args:
            decision: 判断内容（e.g., "task_decomposed", "result_accepted", "escalate")
            task_id: 関連するタスクID（あれば）
            reasoning: 判断の根拠（あれば）
        """
        event = Event(
            event_type="planner_decision",
            content={
                "decision": decision,
                "task_id": task_id,
                "reasoning": reasoning
            }
        )
        self._write_event(event)

    def log_error(self, error_type: str, message: str, task_id: Optional[str] = None):
        """エラーをログ記録。"""
        event = Event(
            event_type="error",
            content={
                "error_type": error_type,
                "message": message,
                "task_id": task_id
            }
        )
        self._write_event(event)

    def log_info(self, message: str, **kwargs):
        """情報メッセージをログ記録。"""
        event = Event(
            event_type="info",
            content={
                "message": message,
                **kwargs
            }
        )
        self._write_event(event)

    def read_events(self) -> list[dict]:
        """
        ログファイルから全イベントを読み込む（デバッグ用）。

        Returns:
            イベント辞書のリスト
        """
        if not self.log_file.exists():
            return []

        events = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    import json
                    events.append(json.loads(line))
        return events
