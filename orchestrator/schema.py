"""
Task/Result スキーマ定義

CrewAIの Task フィールドと OpenHands のイベントストリーム、
LangGraph の Command(goto, update) パターンを参考に設計。
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, List, Dict, Any
from datetime import datetime
import json


@dataclass
class Task:
    """
    ワーカーに委譲するサブタスク。

    Attributes:
        id: 一意なタスクID (e.g., "task-001-function-impl")
        description: ワーカーへの指示文（自然言語）
        expected_output: 「何ができれば完了か」の仕様
        context: このタスクに渡す最小限のコンテキスト
                 （関連コード片、依存するタスクの出力など）
        assigned_to: ワーカーID ("worker-1", "worker-2", ... または "main")
        depends_on: 依存する先行タスクIDのリスト
        created_at: タスク作成日時
    """
    id: str
    description: str
    expected_output: str
    context: Dict[str, Any]
    assigned_to: str  # "worker-1", "worker-2", ... または "main"
    depends_on: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class Result:
    """
    ワーカーからの結果。

    Attributes:
        task_id: 対応するタスクID
        status: 実行結果のステータス
                - "ok": 成功
                - "error": エラー（再試行対象になる可能性あり）
                - "needs_escalation": メインLLMへのエスカレーションが必要
        output: ワーカーの出力文字列（コード、説明、結果など）
        raw_response: デバッグ用の生レスポンス（LLMからの生出力）
        tokens_used: トークン使用数（測定可能な場合）
        latency_ms: 推論レイテンシ（ミリ秒）
        worker_id: 実行したワーカーID
        error_message: エラーメッセージ（statusが"error"の場合）
        completed_at: 完了日時
    """
    task_id: str
    status: Literal["ok", "error", "needs_escalation"]
    output: str
    raw_response: str
    tokens_used: int
    latency_ms: float
    worker_id: str
    error_message: Optional[str] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['completed_at'] = self.completed_at.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ExecutionResult:
    """
    Planner.execute() の戻り値。

    単なる文字列ではなく、成功/未収束/中断を区別できるようにする
    （「スタックしたまま終了」と「成功したが出力が空」を
    `main.py`側で区別できるようにするため）。

    Attributes:
        status: "completed"（メインLLMが完了と判断） /
                "exhausted"（max_iterations に到達し未収束） /
                "aborted"（JSONパース失敗等で早期中断）
        final_output: 最終出力文字列（未完了の場合は空文字列）
        iterations: 実際に実行したイテレーション回数
    """
    status: Literal["completed", "exhausted", "aborted"]
    final_output: str
    iterations: int


@dataclass
class Event:
    """
    イベントログエントリ。

    Task 送出、Result 受信、プランナーの判断など、
    すべての重要な出来事を JSONL に記録する。
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: Literal["task_created", "task_sent", "result_received",
                       "planner_decision", "error", "info"] = "info"
    content: Dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        """
        JSONL フォーマット用の1行を返す。
        タイムスタンプは ISO 8601。
        """
        data = {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'content': self.content
        }
        return json.dumps(data, ensure_ascii=False)
