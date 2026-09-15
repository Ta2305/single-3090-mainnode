"""
OpenAI互換チャット補完APIの共通呼び出しロジック。

メインLLM（vLLM経由）とワーカーLLM（llama-server経由）の両方を
「OpenAI互換HTTPエンドポイント」として対称に扱う設計方針(README.md参照)に伴い、
このモジュールは WorkerClient と Planner の両方から使われる共有部品。
"""

import logging
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _is_retryable_error(exc: BaseException) -> bool:
    """
    5xx・接続エラー・タイムアウトのみリトライ対象とする。
    4xx（リクエスト自体の誤り。例: 不正なペイロード）は
    再試行しても結果が変わらないため対象外にする。
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
        return True
    return False


@retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
async def call_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    messages: List[Dict[str, str]],
    model: str = "default",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    extra_body: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    OpenAI互換 /v1/chat/completions を呼び出す。

    Args:
        client: 呼び出しに使う httpx.AsyncClient
                （ライフサイクル管理は呼び出し元の責任）
        base_url: 例 "http://localhost:8000"（メインLLM）や
                  "http://127.0.0.1:9001"（SSHトンネル経由のワーカー）
        messages: OpenAI形式のメッセージリスト（chat template はサーバー側で適用される）
        model: payloadの"model"フィールド
               （多くのローカルサーバー実装では実際には無視される）
        extra_body: guided_json 等、追加で送りたいペイロード
                    （vLLMの構造化出力機能などに使う拡張ポイント）

    Returns:
        レスポンスJSON（辞書）

    Raises:
        httpx.HTTPStatusError: 4xx（リトライなし）または
            リトライ上限に達した5xx
        httpx.ConnectError等: リトライ上限に達した接続エラー
    """
    url = f"{base_url}/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }
    if extra_body:
        payload.update(extra_body)

    logger.debug(f"POST {url} (model={model})")
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def extract_content(response_data: Dict[str, Any]) -> str:
    """チャット補完レスポンスから本文テキストを取り出す。"""
    choices = response_data.get('choices', [])
    if choices:
        return choices[0].get('message', {}).get('content', '')
    return ''


def extract_token_usage(response_data: Dict[str, Any]) -> int:
    """チャット補完レスポンスからトークン使用量を取り出す。"""
    return response_data.get('usage', {}).get('total_tokens', 0)
