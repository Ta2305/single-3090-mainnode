"""
プランナーループ。

メインLLM（vLLM、OpenAI互換HTTPサーバーとして別プロセスで起動済みのもの）が
コーディングタスクを受け取り、分解・委譲・統合を行うメインロジック。

メインLLMもワーカーLLMも、実装の違い(vLLM/llama.cpp等)を問わず
「独立したOpenAI互換HTTPエンドポイント」として対称に扱う設計方針を採る
(README.md参照)。vLLMの内蔵エンジンAPIには依存しない。
"""

import asyncio
import logging
import json
import re
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

import httpx

from .schema import Task, Result, Event, ExecutionResult
from .event_log import EventLog
from .worker_client import WorkerClient
from .llm_client import call_chat_completion, extract_content, extract_token_usage

logger = logging.getLogger(__name__)


# プランナーの判断をvLLMの構造化出力(response_format=json_schema)で制約するためのJSONスキーマ。
# action毎に必須フィールドが変わるため、"action"のみ必須にし、
# 残りは緩く許容する（全フィールドoptionalな1つのスキーマで妥協する）。
PLANNER_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["decompose", "process_result", "complete"]
        },
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "local_id": {"type": "string"},
                    "description": {"type": "string"},
                    "assigned_to": {"type": "string"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["description", "assigned_to"]
            }
        },
        "decision": {
            "type": "string",
            "enum": ["accept", "lightly_fix", "redo", "complete"]
        },
        "reasoning": {"type": "string"},
        "final_output": {"type": "string"}
    },
    "required": ["action"]
}

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*)```", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    """
    ```json ... ``` のように、テキスト全体が1つのコードフェンスで
    囲まれている場合のみ中身を取り出す。

    以前は`.search()`で最初に見つかったフェンスを剥がしていたため、
    "final_output"フィールドの値自体に```python ... ```(コード片)が
    埋め込まれているだけの正常なJSON応答まで誤って中身だけ取り出してしまい、
    外側の`{...}`構造を壊して`_extract_first_json_object`が
    パース不能になるバグがあった(fullmatchでテキスト全体を1つのフェンスと
    見なせる場合のみ剥がすことで回避する)。
    """
    match = _FENCE_PATTERN.fullmatch(text.strip())
    return match.group(1) if match else text


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    テキスト中から最初に現れる完全なJSONオブジェクトを取り出す。

    貪欲な正規表現 `re.search(r'\\{.*\\}', text, re.DOTALL)` は
    最初の"{"から最後の"}"までを掴んでしまい、マークダウンのコードフェンスや
    複数のJSONオブジェクトが混在すると壊れる。
    ここでは文字列リテラルを認識した上で波括弧の深さを追跡し、
    最初のオブジェクトが正しく閉じた時点で切り出す。
    """
    stripped = _strip_markdown_fence(text).strip()

    start = stripped.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = stripped[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None

    return None


class Planner:
    """
    メインプランナー。

    メインLLM（vLLMのOpenAI互換サーバー、別プロセスで起動済み）が
    タスク分解・ワーカー割当・結果統合を実行する。

    Usage:
        async with WorkerClient('configs/workers.yaml') as worker_client, \\
                   Planner('http://localhost:8000', event_log, worker_client,
                           model_name='<served-model-name>') as planner:
            result = await planner.execute("Write a function to add two numbers")
    """

    def __init__(
        self,
        main_llm_base_url: str,
        event_log: EventLog,
        worker_client: WorkerClient,
        model_name: str = "default"
    ):
        """
        Args:
            main_llm_base_url: メインLLMサーバーのbase_url (例: "http://localhost:8000")
                                事前に `vllm serve` 等で起動しておく必要がある。
            event_log: EventLog インスタンス
            worker_client: WorkerClient インスタンス（既に `async with` で初期化済みのもの）
            model_name: payloadの"model"フィールドに使う文字列。`vllm serve`起動時の
                        `--served-model-name`と厳密に一致させる必要がある
                        (呼び出し元でMAIN_MODEL_NAME環境変数/CLI引数から解決して渡すこと)。
        """
        self.main_llm_base_url = main_llm_base_url
        self.event_log = event_log
        self.worker_client = worker_client
        self.model_name = model_name
        self.client: Optional[httpx.AsyncClient] = None

        # 実行状態
        self.task_counter = 0
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, Result] = {}
        # サブタスクのlocal_id（LLMがそのdecompose呼び出し内で割り当てる仮ID）
        # から、実際に生成したグローバルなtask.idへのマッピング。
        # 実行全体を通して保持し、後続のイテレーションでも過去のlocal_idを参照できるようにする。
        self.local_id_to_task_id: Dict[str, str] = {}

    async def __aenter__(self):
        """コンテキストマネージャー開始。"""
        self.client = httpx.AsyncClient(timeout=300.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了。"""
        if self.client:
            await self.client.aclose()

    async def _call_main_llm(
        self,
        messages: List[Dict[str, str]],
        use_guided_json: bool = False
    ) -> str:
        """
        メインLLM(vLLMサーバー)をHTTP経由で呼び出す。

        Args:
            messages: OpenAI形式のメッセージリスト
            use_guided_json: Trueの場合、OpenAI互換の`response_format`
                (json_schema)でPLANNER_DECISION_SCHEMAに制約する。
                （旧vLLM拡張フィールド`guided_json`はvllm==0.19.0では
                 無視されるため、標準の`response_format`を使う）
                サーバーが対応していない場合は無視される想定
                （非対応時にサーバー側がエラーを返す場合は
                 呼び出し側でuse_guided_json=Falseにフォールバックすること）。

        Returns:
            LLMからの出力テキスト
        """
        if not self.client:
            raise RuntimeError(
                "Planner not initialized. Use 'async with Planner(...)' context."
            )

        extra_body = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "planner_decision",
                    "schema": PLANNER_DECISION_SCHEMA,
                },
            },
            # Qwen3.6はデフォルトでchain-of-thoughtを挟むため、guided_json応答が
            # thinkingトークンでmax_tokensを消費しきってしまい、JSONが完結せず
            # パース不能になることがある（Planner._call_main_llmは判断のみで
            # コード生成は行わないため、thinkingを無効化してもコード品質への
            # 影響はない）。
            "chat_template_kwargs": {"enable_thinking": False},
        } if use_guided_json else None

        logger.debug(f"Calling main LLM ({len(messages)} messages, guided_json={use_guided_json})")

        response_data = await call_chat_completion(
            client=self.client,
            base_url=self.main_llm_base_url,
            messages=messages,
            model=self.model_name,
            temperature=0.3,
            max_tokens=6144,
            top_p=0.9,
            extra_body=extra_body
        )

        output_text = extract_content(response_data)
        logger.debug(f"Main LLM output: {output_text[:200]}...")
        return output_text

    def _build_task_context(self, depends_on_task_ids: List[str], user_task: str) -> Dict[str, Any]:
        """
        依存する先行タスクの出力を含めたコンテキストを構築する。

        ワーカーへの委譲はネットワーク越しの呼び出しになるため、
        「何がネットワークを越えるか」をcontextフィールドで明示的に絞る。
        """
        context: Dict[str, Any] = {"user_task": user_task}
        for dep_task_id in depends_on_task_ids:
            dep_result = self.results.get(dep_task_id)
            if dep_result is not None:
                context[f"dependency_output[{dep_task_id}]"] = dep_result.output
        return context

    async def _process_main_subtask(self, task: Task) -> Result:
        """
        assigned_to == "main" のサブタスクをメインLLM自身に処理させる。

        （旧実装では"今回は未実装"としてログのみ出し、Resultを一切記録しないまま
        次のイテレーションに進んでいたため、状態が更新されず
        max_iterationsまで空回りするバグがあった。）
        """
        context_str = ""
        if task.context:
            context_str = "Context:\n" + "\n".join(
                f"- {k}: {str(v)[:500]}" for k, v in task.context.items()
            ) + "\n\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful code assistant handling a subtask directly "
                    "(not delegated to a worker model). Provide clear, working code."
                )
            },
            {
                "role": "user",
                "content": f"{context_str}Task: {task.description}\nExpected output: {task.expected_output}"
            }
        ]

        try:
            start_time = time.time()
            response_data = await call_chat_completion(
                client=self.client,
                base_url=self.main_llm_base_url,
                messages=messages,
                model=self.model_name,
                temperature=0.3,
                max_tokens=4096,
                top_p=0.9,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            latency_ms = (time.time() - start_time) * 1000
            output = extract_content(response_data)
            tokens_used = extract_token_usage(response_data)

            return Result(
                task_id=task.id,
                status="ok",
                output=output,
                raw_response=str(response_data),
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                worker_id="main"
            )
        except Exception as e:
            logger.error(f"Main LLM subtask processing error: {e}")
            return Result(
                task_id=task.id,
                status="error",
                output="",
                raw_response="",
                tokens_used=0,
                latency_ms=0.0,
                worker_id="main",
                error_message=str(e)
            )

    async def _dispatch_one_subtask(self, task_id: str, subtask_data: Dict[str, Any],
                                     depends_on_task_ids: List[str], user_task: str) -> None:
        """1個のサブタスクを実行し、self.resultsに結果を書き込む(並列実行される単位)。"""
        task = Task(
            id=task_id,
            description=subtask_data.get('description', ''),
            expected_output="Code or solution",
            context=self._build_task_context(depends_on_task_ids, user_task),
            assigned_to=subtask_data.get('assigned_to', 'main'),
            depends_on=depends_on_task_ids
        )
        self.tasks[task.id] = task
        self.event_log.log_task_created(task)

        if task.assigned_to != 'main':
            logger.info(f"Sending {task.id} to {task.assigned_to}")
            self.event_log.log_task_sent(task.id, task.assigned_to, len(str(task.context)))
            result = await self.worker_client.call_worker(task.assigned_to, task)
        else:
            logger.info(f"Processing {task.id} on main LLM")
            result = await self._process_main_subtask(task)

        self.results[task.id] = result
        self.event_log.log_result_received(result)

    async def _dispatch_subtasks(self, subtasks: List[Dict[str, Any]], user_task: str) -> None:
        """
        1回のdecomposeで生成された全サブタスクを実行する。

        依存関係のないサブタスク同士(例: mainに振られたサブタスクとworker-1に
        振られたサブタスクが互いに依存しない場合)は`asyncio.gather`で並列実行する。
        これにより、メインLLM自身の実行とワーカーへの委譲を同時に走らせられる
        （旧実装はdepends_onの有無に関わらず単純なforループで1つずつ`await`しており、
        必ず全サブタスクが直列実行されていた）。

        依存関係がある場合は、依存先が完了する「波」ごとに区切って順に処理する
        (同じ波の中は並列、波と波の間は直列)。
        """
        # 先にtask_idを採番し、local_id -> task_id の対応を全サブタスク分登録してから
        # 依存関係(depends_on)をtask_idへ解決する。後続のサブタスクが先行サブタスクの
        # local_idを参照できるようにするため、この解決は実行順序に依存させない。
        prepared = []
        for subtask_data in subtasks:
            self.task_counter += 1
            new_task_id = f"task-{self.task_counter:03d}"
            local_id = subtask_data.get('local_id')
            if local_id:
                self.local_id_to_task_id[local_id] = new_task_id
            prepared.append((new_task_id, subtask_data))

        remaining = []
        for task_id, subtask_data in prepared:
            depends_on_local_ids = subtask_data.get('depends_on', []) or []
            depends_on_task_ids = [
                self.local_id_to_task_id[lid]
                for lid in depends_on_local_ids
                if lid in self.local_id_to_task_id
            ]
            remaining.append((task_id, subtask_data, depends_on_task_ids))

        while remaining:
            ready = [r for r in remaining if all(dep in self.results for dep in r[2])]
            if not ready:
                # 循環依存等で誰も進められない場合は、残り全部を依存関係無視で処理する
                # (無限ループになるよりは、壊れた依存関係を無視して前進する方が安全)
                logger.warning(
                    "Could not resolve subtask dependency order; running remaining subtasks anyway"
                )
                ready = remaining
            remaining = [r for r in remaining if r not in ready]

            await asyncio.gather(*(
                self._dispatch_one_subtask(task_id, subtask_data, depends_on_task_ids, user_task)
                for task_id, subtask_data, depends_on_task_ids in ready
            ))

    async def execute(self, user_task: str, max_iterations: int = 10) -> ExecutionResult:
        """
        ユーザーのコーディングタスクを実行。

        Args:
            user_task: ユーザーからの指示文
            max_iterations: 最大ループ回数（無限ループ防止）

        Returns:
            ExecutionResult（status/final_output/iterationsを含む構造化結果）
        """
        logger.info(f"Starting execution of task: {user_task[:100]}...")
        self.event_log.log_info("Task execution started", task=user_task[:200])

        iteration = 0
        final_output = ""
        status: str = "exhausted"

        worker_roster = "\n".join(
            f'  - "{w["id"]}": a smaller/weaker/faster local model ({w.get("model", "unknown model")})'
            for w in self.worker_client.workers_config.get("workers", [])
        ) or "  (no workers configured — everything must be assigned_to \"main\")"

        system_context = f"""You are the main model in a heterogeneous local-LLM coding system.
You ("main") are the largest and most capable model available, but you run on
constrained, shared hardware, so your own compute is a resource to spend
deliberately rather than by default. The other executors available to you are:
{worker_roster}

The research question this system is built to study is how to divide coding
work efficiently between models of different capability: not "delegate as much
as possible", but "route each subtask to whichever executor is actually
expected to complete it correctly and efficiently". Workers are smaller and
faster per call, but less reliable — they sometimes produce incomplete or
incorrect code that you must catch and fix. Delegating a subtask has a real
cost (a network round trip to another machine, and a chance the result needs
rework), so it is not free.

Your job each turn is to:
1. Break down programming tasks into manageable subtasks (only when the task
   actually benefits from decomposition — trivial tasks do not).
2. For each subtask, decide who should do it: yourself ("main") or one of the
   worker ids listed above. Judge this per subtask, not for the whole task at
   once — a single task can reasonably mix "main" and worker subtasks.
   - Prefer a worker when the subtask is a clear, self-contained, mechanical
     unit of work (e.g. implementing a function from an unambiguous spec) that
     a smaller model is likely to get right on its own.
   - Prefer "main" when the subtask is small enough that a round trip isn't
     worth it (e.g. a couple of assert lines, minor glue code), when it needs
     careful integration/judgment across other subtasks' results, or when a
     worker has already struggled with similar work in this task.
   - Do not default to sending every subtask to a worker just because
     delegation is available. Assigning a subtask to "main" is often the more
     efficient choice, and you should choose it whenever you judge it saves
     overall time/resources without sacrificing correctness.
   - "assigned_to" must be exactly "main" or one of the worker ids listed
     above — never invent a worker id that is not listed.
3. Integrate results from workers (and your own subtask results), catching and
   fixing mistakes before considering the task complete.
4. Provide the final solution.

If a subtask depends on the output of another subtask in this same decomposition,
give each subtask a short "local_id" (e.g. "impl", "test") and reference it in
that subtask's "depends_on" list. Subtasks with no dependency should omit
"depends_on" or leave it empty.

For task decomposition, output JSON:
{{
    "action": "decompose",
    "subtasks": [
        {{"local_id": "impl", "description": "...", "assigned_to": "worker-1", "depends_on": []}},
        {{"local_id": "test", "description": "...", "assigned_to": "main", "depends_on": ["impl"]}}
    ]
}}

If you're processing worker results, output JSON:
{{
    "action": "process_result",
    "decision": "accept|lightly_fix|redo|complete",
    "reasoning": "..."
}}

If complete, output JSON:
{{
    "action": "complete",
    "final_output": "..."
}}

If the task is trivial, you may decompose into a single subtask assigned to "main"
or directly output the "complete" action without any delegation.
"""

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration} ===")

            state_summary = f"\nCurrent task: {user_task}\n\nPrevious results:\n"
            for task_id, result in self.results.items():
                state_summary += f"- {task_id} ({result.status}): {result.output[:200]}\n"

            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": f"{state_summary}\n\nWhat's your next action?"}
            ]

            response_text = await self._call_main_llm(messages, use_guided_json=True)
            self.event_log.log_planner_decision("queried_main_llm")

            decision = _extract_first_json_object(response_text)
            if decision is None:
                logger.error(f"No valid JSON found in response: {response_text[:500]}")
                self.event_log.log_error(
                    "parse_error",
                    f"Could not extract JSON from main LLM response: {response_text[:500]}"
                )
                status = "aborted"
                break

            action = decision.get('action')

            if action == "decompose":
                subtasks = decision.get('subtasks', [])
                logger.info(f"Decomposing into {len(subtasks)} subtasks")
                self.event_log.log_planner_decision("task_decomposed", reasoning=str(subtasks))
                await self._dispatch_subtasks(subtasks, user_task)

            elif action == "process_result":
                decision_action = decision.get('decision', 'accept')
                logger.info(f"Processing result with decision: {decision_action}")
                self.event_log.log_planner_decision(
                    "result_processed",
                    reasoning=decision.get('reasoning', decision_action)
                )
                # accept/lightly_fix/redo は次のイテレーションで再評価される
                # (state_summaryに既存の結果が含まれているため、
                #  メインLLMは次のプロンプトで続きを判断できる)。

            elif action == "complete":
                final_output = decision.get('final_output', '')
                logger.info("Task completed")
                self.event_log.log_planner_decision("task_completed")
                status = "completed"
                break

            else:
                logger.warning(f"Unknown action: {action}")
                self.event_log.log_error("unknown_action", f"Action: {action}")
                status = "aborted"
                break
        else:
            # while-else: max_iterationsに到達してループが正常終了した場合
            status = "exhausted"

        logger.info(f"Execution completed after {iteration} iterations (status={status})")
        self.event_log.log_info(
            "Task execution completed",
            iterations=iteration,
            status=status,
            final_output_preview=final_output[:200] if final_output else ""
        )

        return ExecutionResult(status=status, final_output=final_output, iterations=iteration)
