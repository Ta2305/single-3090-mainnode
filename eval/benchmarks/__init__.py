"""
外部の有名ベンチマーク(HumanEval, MBPP, APPS, LiveCodeBench)を、
eval/run_eval.py が扱うタスク辞書の共通フォーマットに変換して読み込むローダー群。

各モジュール(humaneval.py, mbpp.py, apps.py, livecodebench.py)は
`load_tasks(n=None, seed=0, difficulty=None, ids=None) -> list[dict]` を提供する。
`n`を指定すると全件ではなく一部だけをシード固定のランダムサンプリングで取得できる。

タスク辞書の共通フォーマット:
    {
        "id": str,                     # 例: "humaneval-0"
        "source_file": str,            # 由来を示す文字列（YAMLファイルパスの代わり）
        "prompt": str,                  # Plannerに渡す自然言語の指示文
        "entry_point": str,
        "grading_mode": "assert" | "stdio",
        "test_code": str | None,        # grading_mode=="assert"の場合に使う
        "stdio_tests": list[dict] | None,  # grading_mode=="stdio"の場合に使う
                                            # [{"input": str, "output": str}, ...]
        "timeout_s": int,
    }
"""
import random


def sample_subset(items: list, n: int | None, seed: int = 0) -> list:
    """シード固定のランダムサンプリングでitemsからn件を取り出す(n=Noneなら全件)。"""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), n)
