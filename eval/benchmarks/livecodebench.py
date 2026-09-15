"""
LiveCodeBench(livecodebench/code_generation_lite)のローダー。

`datasets.load_dataset`はこのリポジトリの読み込みスクリプト形式に対応していない
(新しいdatasetsライブラリでは読み込みスクリプトが廃止されたため)ので、
`huggingface_hub.hf_hub_download`でraw jsonl(`test.jsonl`, `test2.jsonl`, ...)を
直接取得してパースする。ファイル名は release_v1=test.jsonl,
release_v2=test2.jsonl, ... という命名（バージョンが上がるほど問題が
累積的に増える）。

`private_test_cases`はbase64+zlib圧縮されたテストケースで、意図的に
非公開(コンタミネーション対策)なため使用しない。`public_test_cases`のみで
採点する。また、LeetCode形式の関数呼び出し問題(`testtype: functional`)は
公式の採点ロジックが複雑なため対象外とし、標準入出力形式(`testtype: stdin`)
の問題のみを扱う(この制限はこのモジュール固有の簡略化)。

Usage:
    python -m eval.benchmarks.livecodebench --n 5 --difficulty easy --seed 0
"""
import json

from huggingface_hub import hf_hub_download

from . import sample_subset

DIFFICULTIES = {"easy", "medium", "hard"}


def _version_filename(version: int) -> str:
    return "test.jsonl" if version == 1 else f"test{version}.jsonl"


def load_tasks(
    n: int | None = None,
    seed: int = 0,
    difficulty: str | None = None,
    ids: list[str] | None = None,
    version: int = 6,
    max_test_cases: int = 5,
    **kwargs,
) -> list[dict]:
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}, got {difficulty!r}")

    path = hf_hub_download(
        "livecodebench/code_generation_lite",
        filename=_version_filename(version),
        repo_type="dataset",
    )
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    # stdin形式の問題のみを対象にする(関数呼び出し形式は対象外。上記docstring参照)
    items = [
        ex for ex in items
        if ex.get("public_test_cases") and all(
            tc.get("testtype") == "stdin" for tc in json.loads(ex["public_test_cases"])
        )
    ]

    if difficulty is not None:
        items = [ex for ex in items if ex["difficulty"] == difficulty]
    if ids:
        wanted = set(ids)
        items = [ex for ex in items if ex["question_id"] in wanted]
    else:
        items = sample_subset(items, n, seed)

    tasks = []
    for ex in items:
        public_tests = json.loads(ex["public_test_cases"])[:max_test_cases]
        stdio_tests = [{"input": tc["input"], "output": tc["output"]} for tc in public_tests]
        if not stdio_tests:
            continue
        tasks.append({
            "id": f"livecodebench-{ex['question_id']}",
            "source_file": f"livecodebench:{ex['question_id']}:{ex['difficulty']}:{ex.get('platform', '')}",
            "prompt": (
                f"{ex['question_content'].strip()}\n\nWrite a complete Python program "
                "that reads its input from stdin and writes the answer to stdout, "
                "matching the expected output format exactly."
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

    parser = argparse.ArgumentParser(description="LiveCodeBenchタスクの一部を取得して一覧表示する")
    parser.add_argument("--n", type=int, default=5, help="取得する問題数(省略時5件)")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default=None)
    parser.add_argument("--version", type=int, default=6, help="release_vN (省略時6=最新の累積版)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", nargs="*", default=None, help="question_idを指定")
    args = parser.parse_args()

    tasks = load_tasks(n=args.n, seed=args.seed, difficulty=args.difficulty, ids=args.ids, version=args.version)
    print(f"{len(tasks)}件:")
    for t in tasks:
        print(f"  {t['id']}: {t['source_file']}")
