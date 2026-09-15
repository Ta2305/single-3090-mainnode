"""
MBPP(Google, CC-BY-4.0, sanitized subset 257問/testスプリット)のローダー。

`datasets`経由でHugging Faceの`google-research-datasets/mbpp`(sanitized)を
読み込み、eval/run_eval.py が扱う共通タスク辞書に変換する。

Usage:
    python -m eval.benchmarks.mbpp --n 5 --seed 0
"""
import re

from datasets import load_dataset

from . import sample_subset


def load_tasks(n: int | None = None, seed: int = 0, ids: list[int] | None = None, **kwargs) -> list[dict]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    items = list(ds)

    if ids:
        wanted = {int(i) for i in ids}
        items = [ex for ex in items if ex["task_id"] in wanted]
    else:
        items = sample_subset(items, n, seed)

    tasks = []
    for ex in items:
        entry_match = re.search(r"def (\w+)\(", ex["code"])
        entry_point = entry_match.group(1) if entry_match else "solution"

        imports = "\n".join(ex.get("test_imports") or [])
        assert_lines = "\n".join(f"    {line}" for line in ex["test_list"])
        test_code = (imports + "\n\n" if imports else "") + f"def test_mbpp():\n{assert_lines}\n"

        tasks.append({
            "id": f"mbpp-{ex['task_id']}",
            "source_file": f"mbpp:{ex['task_id']}",
            "prompt": ex["prompt"].strip(),
            "entry_point": entry_point,
            "grading_mode": "assert",
            "test_code": test_code,
            "stdio_tests": None,
            "timeout_s": 60,
        })
    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MBPPタスクの一部を取得して一覧表示する")
    parser.add_argument("--n", type=int, default=5, help="取得する問題数(省略時5件)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None, help="MBPPのtask_id(整数)を指定")
    args = parser.parse_args()

    tasks = load_tasks(n=args.n, seed=args.seed, ids=args.ids)
    print(f"{len(tasks)}件:")
    for t in tasks:
        print(f"  {t['id']}: entry_point={t['entry_point']} | {t['prompt'][:70]}")
