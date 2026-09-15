# Heterogeneous Local-LLM Coding Agent

複数のローカルLLM(強いモデル1台 + 弱いモデル複数台)を協調させ、コーディングタスクを
自動で分解・委譲・統合するエージェントハーネスのプロトタイプです。

大きなモデルに全部投げれば精度は出やすい一方で遅く・重くなります。逆に小さなモデルだけでは
複雑なタスクをこなせません。本プロジェクトは「メインモデル自身に、サブタスクごとに
"自分でやるか・どのワーカーに任せるか"を判断させる」という設計で、この間を取る構成を検証します。

- メインLLM: vLLM上で動く比較的大きめのモデル(タスクの分解・統合・最終判断を担当)
- ワーカーLLM: llama.cpp上で動く小さめのモデル(定型的な実装作業を担当。別マシンでも可)

この構成は単一の物理GPU(24GB級)でもそのまま動きます。GPU2枚+NVLinkのようなテンソル並列構成は
不要で、メインLLM・ワーカーLLMをそれぞれ1枚のGPUに載せて別プロセス(別マシンでも可)として
起動するだけです。

## なぜこの設計にしたか

### 1. 「モデル自身に判断させる」設計(ルールベースの外部ルーターにしない)

タスクをmain/workerのどちらに振るかを、外部の分類器やヒューリスティックなルールで決めるのでは
なく、メインLLM自身に構造化出力(JSON)で判断させています。理由は、「このサブタスクは
自明だからmainで即答すべきか」「往復コストを払ってでも切り出すべきか」という判断は、
タスクの意味内容に強く依存し、事前に固定ルールを書き切れないと考えたためです。
この判断を毎ターンのプロンプトに埋め込み、`action: "decompose" | "process_result" | "complete"`
のいずれかを返させる形にしています(`orchestrator/planner.py`)。

### 2. 実装の異なる推論スタックを対称に扱う("Option B"設計)

メインLLM(vLLM)とワーカーLLM(llama.cpp)は実装が全く異なりますが、両者を「OpenAI互換の
`/v1/chat/completions`を叩けるHTTPエンドポイント」として完全に対称に扱います
(`orchestrator/llm_client.py`が両者から共有される)。これにより、vLLMの内蔵機能(投機的
デコーディング等)に依存せず、将来ワーカーの実装を差し替えても(例: llama.cpp→vLLM)
オーケストレータ側のコードを変更せずに済みます。

### 3. ワーカーへの接続を「本物の多段SSHログイン」で行う

ワーカー機がファイアウォールの内側にあり、踏み台サーバー経由でしか到達できない、という
制約のあるネットワーク環境を想定しています。この制約下で、ワーカー側に常駐プロセスや
ファイアウォールの穴あけを一切要求しない接続方式として、実際の`ssh`バイナリを
`ssh -J <踏み台> -L <ローカルポート>:127.0.0.1:<推論ポート> <ワーカー> -N`という形で
サブプロセス起動し、ワーカー自身のループバック宛にローカルポートフォワードする方式を
採っています(`orchestrator/worker_client.py`)。SSHの認証・多段接続はOpenSSHクライアントに
そのまま任せ、Python側でSSHプロトコルを再実装しないことで、鍵管理や多要素認証など
既存の運用をそのまま流用できます。

### 4. 単一GPU構成であること自体の意義

強いモデルと弱いモデルの性能差を保ったまま検証するには、本来は複数のハイエンドGPUで
それぞれを別プロセスとして動かすのが理想ですが、GPUリソースは常に自由に使えるとは限りません。
このプロトタイプは、単一の24GB級コンシューマGPU1枚(+ネットワーク越しの安価なワーカー機1台)
という、個人でも再現しやすい構成で同じ研究課題を検証できることを示すために作られました。

## 実装過程で見つかった問題と対処(エンジニアリング上の学び)

実際に異種スタックをネットワーク越しに繋いで動かしてみたところ、設計レベルでは想定していなかった
問題がいくつも見つかりました。「動いた」だけでなく、動かす過程で何が壊れるかを記録しておきます。

- **構造化出力の指定方法がvLLMのバージョンで変わっていた**: 古い拡張フィールド`guided_json`が
  無視され、レスポンスが自由文のまま返ってくる(エラーにはならないため気づきにくい)。
  OpenAI標準の`response_format: {"type": "json_schema", ...}`に切り替えて解決。
- **モデルのchain-of-thought(thinking)が構造化出力のトークン上限を食い潰す**: 判断だけを
  求めるコールで数千トークンの思考が挟まり、JSON本体が出力される前に`max_tokens`に達して
  パース不能になっていた。判断専用のコールだけ`enable_thinking: false`にすることで、
  応答時間が数十秒→0.4秒に短縮。
- **もっとも影響が大きかったバグ**: レスポンス文字列からJSONを取り出す処理が、
  「生成されたコード自体に含まれるMarkdownのコードフェンス(` ```python ... ``` `)」を
  「レスポンス全体を囲むフェンス」と誤認識し、外側のJSON構造ごと壊してしまっていた。
  コードを含む応答であれば理論上いつでも再現しうる問題で、テキスト全体が1つのフェンスで
  完全に囲まれている場合のみ中身を取り出す(`fullmatch`)よう修正した。
- **サブタスクが常に直列実行されていた**: 非同期処理の土台(asyncio/httpx.AsyncClient)は
  最初から整っていたが、依存関係のないサブタスク同士も単純な`for`ループで1件ずつ実行して
  いた。依存関係グラフに基づく「波(wave)」単位の実行に書き換え、依存のないサブタスクは
  `asyncio.gather`で同時に実行されるようにした。
- **既知の未解決課題**: ワーカーの結果が壊れていた場合の`redo`/`lightly_fix`判定は、実装上は
  次のイテレーションで同じ結果を再提示するだけで、実際に再送・修復する処理が存在しない
  (`orchestrator/planner.py`の`process_result`分岐)。メインLLMは正しく「結果が壊れている」と
  判断できているが、それを受けたリカバリ動作が未実装のため、同じ診断を繰り返して
  `max_iterations`まで進んでしまうケースがある。委譲した結果の品質保証は、今後の実装課題。

## 実験で分かったこと(n=9の小規模観察、過度な一般化はしていない)

`eval/`のハーネスで、単発関数の実装からLRUキャッシュ実装のような複合タスクまで計9件を
実行し、(a) サブタスクがmain/workerのどちらに振られたか (b) 処理時間の内訳
(c) 自動テストによる合否、を計測しました。

- 自明なタスク(四則演算・fizzbuzzなど)は委譲なしで1〜数秒に直接complete する一方、
  再帰やクラス実装を要するタスクでは委譲が発生する、という使い分けの兆候が見られた。
- 意外だった点: 委譲が発生したタスクは、処理時間の85〜95%程度をワーカー側の実行時間が
  占めており、**「弱いモデルに投げれば速くなる」わけでは必ずしもなかった**。
- 同じ9タスクで、委譲を完全に禁止しメインLLM単独で解かせる比較(`eval/run_eval_mainonly.py`)
  を行ったところ、ほとんどのタスクでメインLLM単独の方が委譲ありと同等かそれ以上に速かった。
  9タスクいずれも自動テストは全てPASSしており、品質面での劣化も見られなかった。
  → 今回のタスクスイート(単発関数中心)は、委譲の往復コストを払う価値が出るほど
    重いタスクではなかった可能性が高い、という診断結果として捉えている。

n=9というサンプル数の小ささから、「委譲すれば速い/正確」といった一般化はしていません。
むしろ「委譲が必ずしも高速化にならない」という反証的なデータが得られたこと自体を、
次にどんなタスク設計・粒度なら委譲の価値が測定できるかを考える材料にしています。

## セットアップ

### 前提

- メインLLM用: NVIDIA GPU 1枚(VRAM 24GB程度を想定。vLLMが動けば他の量子化モデル/
  容量でも代替可能)
- ワーカーLLM用: 別途llama.cppが動くマシン(同一マシンでも、ネットワーク越しの
  別マシンでも可)
- Python 3.10+、`ssh`コマンドがPATH上にあること(ワーカーとの通信に実際の`ssh`バイナリを
  使うため)

### 1. Python環境

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# eval/run_eval.py・eval/visualize.py を使う場合のみ追加で:
pip install -r eval/requirements-eval.txt
```

### 2. メインLLM(vLLM)を起動する

以下は`cyankiwi/Qwen3.6-27B-AWQ-INT4`(24GB VRAM級)を例にした起動コマンドです。
他のモデルを使う場合も、量子化方式・VRAM容量に応じて`--gpu-memory-utilization`と
`--max-model-len`を調整してください。

```bash
vllm serve <モデルのHFリポジトリID> \
  --port 8000 \
  --served-model-name <任意の短い名前> \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

`--served-model-name`は`orchestrator/planner.py`の`Planner`が送る`model`フィールド
(既定値`"Qwen3.6-27B"`)と一致させてください。一致していないとサーバー側で
`404 The model ... does not exist.`になります(`/v1/models`のヘルスチェックだけでは
気づけないため注意)。VRAM不足でロードに失敗する場合は、`--max-model-len`をさらに
下げる、`--gpu-memory-utilization`を上げる、`--enforce-eager`を付ける、といった
調整を試してください。

### 3. ワーカーLLM(llama.cpp)を起動する

```bash
./llama-server \
  --model <ローカルのGGUFファイルパス> \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 99 \
  --offline \
  --alias <configs/workers.yamlのmodelと一致させる文字列>
```

`--offline`を付けないと、`--model`にHugging Face hubキャッシュの標準パス
(`models--<org>--<repo>/snapshots/<hash>/...`)をそのまま渡した場合に、
llama-serverがそれを「都度ネットワーク解決すべきモデル」と誤認識し、外部に出られない
環境でロードに失敗することがあります(router modeのオンデマンドロード機能による挙動)。
`--alias`は`configs/workers.yaml`の`model`フィールドと一致させてください
(一致しないと`/v1/chat/completions`が`model ... not found`を返すことがあります)。

### 4. ワーカー接続設定を編集する

`configs/workers.yaml`をコピーして`configs/workers.local.yaml`を作成し(このファイルは
`.gitignore`対象です)、`hostname`/`ssh_jumphost`/`ssh_username`を実際の値に書き換えて
ください。ワーカーが同一ネットワーク上にあり踏み台が不要な場合は`ssh_jumphost`に
ワーカー自身のホスト名を指定すれば、実質1段のSSHログインとして動作します。

事前に、このマシンから踏み台経由でワーカーへの2段SSHログインができることを
`ssh -J <踏み台> <ワーカーのhostname> echo ok`で確認しておくと安全です。

### 5. 実行する

```bash
python main.py "Write a function that adds two numbers" \
  --config configs/workers.local.yaml
```

初回は`eval/tasks/phase0_smoke.yaml`にあるような自明なタスクから試すことを推奨します。

## 評価ハーネスの使い方

```bash
# 同梱タスク(eval/tasks/*.yaml)を実行し、eval/results/<timestamp>/ に結果を保存
python -m eval.run_eval --config configs/workers.local.yaml

# HumanEval/MBPP/APPS/LiveCodeBenchから一部だけサンプリングして混ぜる
python -m eval.run_eval --config configs/workers.local.yaml --humaneval 5 --mbpp 5

# 委譲を完全に禁止し、メインLLM単独での挙動と比較する
python -m eval.run_eval_mainonly --config configs/workers.local.yaml

# 結果から図(委譲内訳・レイテンシ内訳・合否)とMarkdownサマリーを生成
python -m eval.visualize --latest
```

## リポジトリ構成

```
main.py                  # CLIエントリーポイント(タスクを1件実行)
orchestrator/
  planner.py              # メインループ: タスク分解・委譲判断・結果統合
  worker_client.py        # ワーカーLLMへのSSHトンネル+HTTPクライアント
  llm_client.py           # OpenAI互換chat completions呼び出しの共通部品
  event_log.py            # 実行イベントのJSONL記録
  schema.py               # Task/Result/Event等のデータ構造
eval/
  run_eval.py             # タスクスイートをバッチ実行し自動採点するランナー
  run_eval_mainonly.py    # 比較実験用: 委譲を禁止したランナー
  visualize.py            # 結果から図・サマリーを生成
  tasks/*.yaml            # 自作タスクスイート(HumanEval/MBPP着想+複合タスク)
  benchmarks/             # HumanEval/MBPP/APPS/LiveCodeBenchの外部ローダー
configs/workers.yaml      # ワーカー接続設定のテンプレート(要編集)
deploy/push_to_worker.sh  # (将来用の下地)ワーカー側にコードを同期する手動実行スクリプト
```

## 既知の制約・今後の課題

- `redo`/`lightly_fix`判定を受けた実際のリカバリ処理が未実装(上記参照)。
- 現状の実験タスクは単発関数中心で件数も少なく(n=9)、委譲の効果を正しく測るには、
  メインの負荷が本当にボトルネックになる大規模タスクや、機械的だが量の多い定型作業を
  含めたタスク設計が必要。
- 依存関係のないサブタスク同士の並列実行は実装・単体検証済みだが、実タスクスイートでは
  まだ効果が確認できていない(impl→testのような依存関係を持つタスクが中心のため)。

## License

MIT License. See [LICENSE](./LICENSE).
