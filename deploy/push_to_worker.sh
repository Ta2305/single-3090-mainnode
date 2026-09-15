#!/usr/bin/env bash
#
# 手動実行専用スクリプト。Claude/自動化からは絶対に実行しないこと
# （3060マシンはまだユーザー毎のログイン分離がされておらず、
#  自動化されたコマンド実行によって他の利用者の作業を壊すリスクがあるため）。
#
# 現時点（Phase 0/1）ではこのスクリプトは実行の必要がない:
# ワーカー機は llama-server を直接HTTPで起動するだけで、
# orchestrator/ のPythonコードを一切importしない。
# 将来Phase 2で、ワーカー側にPythonツール（tree-sitterリポマップ等）が
# 必要になった時のための下地として用意している。
#
# 使い方: bash deploy/push_to_worker.sh <worker_id>
#   例:   bash deploy/push_to_worker.sh worker-1
#
# 前提: configs/workers.yaml の該当エントリの hostname/ssh_jumphost/ssh_username が
# プレースホルダのままでは動かない。実際の値に書き換えてから実行すること。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_ID="${1:-worker-1}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 が見つかりません。" >&2
    exit 1
fi

WORKER_INFO="$(python3 -c "
import sys
import yaml
with open('${SCRIPT_DIR}/configs/workers.yaml') as f:
    config = yaml.safe_load(f)
worker = next((w for w in config['workers'] if w['id'] == '${WORKER_ID}'), None)
if worker is None:
    sys.exit(f'worker \'${WORKER_ID}\' not found in workers.yaml')
print(worker['hostname'])
print(worker['ssh_jumphost'])
print(worker.get('ssh_username', ''))
")"

HOSTNAME="$(echo "${WORKER_INFO}" | sed -n '1p')"
JUMPHOST="$(echo "${WORKER_INFO}" | sed -n '2p')"
SSH_USERNAME="$(echo "${WORKER_INFO}" | sed -n '3p')"

if [[ "${HOSTNAME}" == "gpu-w01.local" || "${JUMPHOST}" == "proxy.example.com" ]]; then
    echo "ERROR: configs/workers.yaml がまだプレースホルダ値のままです。" >&2
    echo "       実際の hostname / ssh_jumphost / ssh_username に書き換えてから再実行してください。" >&2
    exit 1
fi

SSH_TARGET="${HOSTNAME}"
if [[ -n "${SSH_USERNAME}" ]]; then
    SSH_TARGET="${SSH_USERNAME}@${HOSTNAME}"
fi

REMOTE_DIR="~/llm-worker/orchestrator_code"

echo "Syncing orchestrator/ + configs/ to ${WORKER_ID} (${SSH_TARGET}) via ${JUMPHOST}..."
echo "(この処理は2段SSHログインを実際に使う唯一の箇所です。トンネル経由のHTTP通信とは別経路です。)"

ssh -A -J "${JUMPHOST}" "${SSH_TARGET}" "mkdir -p ${REMOTE_DIR}"

# orchestrator/ と configs/ をそれぞれディレクトリごとREMOTE_DIR配下に転送する
# （末尾スラッシュなしで渡すことで、ディレクトリ自体をコピーし、
#   中身だけが混ざって配置される事態を避ける）。
rsync -avz --delete \
  -e "ssh -A -J ${JUMPHOST}" \
  "${SCRIPT_DIR}/orchestrator" "${SCRIPT_DIR}/configs" \
  "${SSH_TARGET}:${REMOTE_DIR}/"

echo "Done."
