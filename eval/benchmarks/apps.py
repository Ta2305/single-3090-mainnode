"""
APPS(codeparrot/apps, 3段階の難易度: introductory/interview/competition)のローダー。

問題によって2種類の形式がある:
  - call-based(`starter_code`が非空。約1%程度): 関数(またはLeetCode形式の
    `class Solution`のメソッド)を実装させ、assert文で採点する
    (grading_mode="assert")。
  - stdio-based(大多数): 標準入力から読み標準出力に書く完全なプログラムを
    実装させ、入出力を突き合わせて採点する(grading_mode="stdio")。

Usage:
    python -m eval.benchmarks.apps --n 5 --difficulty introductory --seed 0
"""
import json

from datasets import load_dataset

from . import sample_subset

DIFFICULTIES = {"introductory", "interview", "competition"}


def load_tasks(
    n: int | None = None,
    seed: int = 0,
    difficulty: str | None = None,
    ids: list[int] | None = None,
    max_test_cases: int = 5,
    **kwargs,
) -> list[dict]:
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}, got {difficulty!r}")

    ds = load_dataset("codeparrot/apps", split="test", revision="refs/convert/parquet")
    items = list(ds)

    if difficulty is not None:
        items = [ex for ex in items if ex["difficulty"] == difficulty]
    if ids:
        wanted = {int(i) for i in ids}
        items = [ex for ex in items if ex["problem_id"] in wanted]
    else:
        items = sample_subset(items, n, seed)

    tasks = []
    for ex in items:
        try:
            io = json.loads(ex["input_output"])
        except (json.JSONDecodeError, TypeError):
            continue  # input_outputが壊れている問題はスキップ

        fn_name = io.get("fn_name")
        prompt_text = ex["question"].strip()

        if fn_name:
            uses_class = "class Solution" in ex["starter_code"]
            call_prefix = "Solution()." if uses_class else ""
            asserts = []
            for inp, out in zip(io["inputs"][:max_test_cases], io["outputs"][:max_test_cases]):
                args = ", ".join(repr(a) for a in inp)
                asserts.append(f"    assert {call_prefix}{fn_name}({args}) == {out!r}")
            if not asserts:
                continue
            test_code = "def test_apps():\n" + "\n".join(asserts) + "\n"
            instruction = (
                f"Implement this as a method on a class named `Solution`, matching:\n{ex['starter_code']}"
                if uses_class else f"Implement this as a Python function named `{fn_name}`."
            )
            tasks.append({
                "id": f"apps-{ex['problem_id']}",
                "source_file": f"apps:{ex['problem_id']}:{ex['difficulty']}",
                "prompt": f"{prompt_text}\n\n{instruction}",
                "entry_point": fn_name,
                "grading_mode": "assert",
                "test_code": test_code,
                "stdio_tests": None,
                "timeout_s": 60,
            })
        else:
            stdio_tests = [
                {"input": inp, "output": out}
                for inp, out in zip(io.get("inputs", [])[:max_test_cases], io.get("outputs", [])[:max_test_cases])
            ]
            if not stdio_tests:
                continue
            tasks.append({
                "id": f"apps-{ex['problem_id']}",
                "source_file": f"apps:{ex['problem_id']}:{ex['difficulty']}",
                "prompt": (
                    f"{prompt_text}\n\nWrite a complete Python program that reads its "
                    "input from stdin and writes the answer to stdout, matching the "
                    "expected output format exactly."
                ),
                "entry_point": None,
                "grading_mode": "stdio",
                "test_code": None,
                "stdio_tests": stdio_tests,
                "timeout_s": 120,
            })
    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="APPSタスクの一部を取得して一覧表示する")
    parser.add_argument("--n", type=int, default=5, help="取得する問題数(省略時5件)")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None, help="APPSのproblem_id(整数)を指定")
    args = parser.parse_args()

    tasks = load_tasks(n=args.n, seed=args.seed, difficulty=args.difficulty, ids=args.ids)
    print(f"{len(tasks)}件:")
    for t in tasks:
        print(f"  {t['id']} [{t['grading_mode']}]: {t['source_file']}")
