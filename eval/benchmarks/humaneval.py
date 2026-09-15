"""
HumanEval(OpenAI, MIT License, 164問)のローダー。

`datasets`経由でHugging Faceの`openai_humaneval`を読み込み、
eval/run_eval.py が扱う共通タスク辞書に変換する。
問題文(prompt)自体は関数シグネチャ+docstringの「続きを書く」形式なので、
Plannerへの自然言語指示として扱えるよう簡単な指示文で包む。

Usage:
    python -m eval.benchmarks.humaneval --n 5 --seed 0
"""
import re

from datasets import load_dataset

from . import sample_subset


def load_tasks(n: int | None = None, seed: int = 0, ids: list[str] | None = None, **kwargs) -> list[dict]:
    ds = load_dataset("openai_humaneval", split="test")
    items = list(ds)

    if ids:
        wanted = {i if i.startswith("HumanEval/") else f"HumanEval/{i}" for i in ids}
        items = [ex for ex in items if ex["task_id"] in wanted]
    else:
        items = sample_subset(items, n, seed)

    tasks = []
    for ex in items:
        short_id = "humaneval-" + ex["task_id"].split("/")[-1]
        # HumanEvalの"test"フィールドは `def check(candidate): assert ...` という形式。
        # 既存のgrade_task()は `def test_\w+\(` を探して呼び出すので、
        # checkを呼ぶだけのラッパーを末尾に追加して規約に合わせる。
        test_code = ex["test"].strip() + f"\n\n\ndef test_check():\n    check({ex['entry_point']})\n"
        prompt = (
            "Complete the following Python function (the signature and docstring "
            "are already given below; implement the body and return the full, "
            "runnable function definition including any needed imports):\n\n"
            f"```python\n{ex['prompt']}```"
        )
        tasks.append({
            "id": short_id,
            "source_file": f"humaneval:{ex['task_id']}",
            "prompt": prompt,
            "entry_point": ex["entry_point"],
            "grading_mode": "assert",
            "test_code": test_code,
            "stdio_tests": None,
            "timeout_s": 60,
        })
    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HumanEvalタスクの一部を取得して一覧表示する")
    parser.add_argument("--n", type=int, default=5, help="取得する問題数(省略時5件)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None, help="例: 0 1 12 (HumanEval/0, HumanEval/1, HumanEval/12)")
    args = parser.parse_args()

    tasks = load_tasks(n=args.n, seed=args.seed, ids=args.ids)
    print(f"{len(tasks)}件:")
    for t in tasks:
        print(f"  {t['id']}: entry_point={t['entry_point']}")
