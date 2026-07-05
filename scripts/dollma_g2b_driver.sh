#!/usr/bin/env bash
# F-0b G-2b チャンク駆動ドライバ: 未完 unit がなくなる (done=200) まで --limit で resume 反復。
# 各チャンクは独立プロセス (SDXL サーバ start/stop 込み)・g2b_prepost.jsonl 追記で冪等。
set -u
cd "$(dirname "$0")/.."
export DOLLAMA_OV_TOKENIZERS_DLL="C:/Users/sdkik/AppData/Local/Python/pythoncore-3.14-64/Lib/site-packages/openvino_tokenizers/lib/openvino_tokenizers.dll"
OUT="data/rollouts/g2b_prepost.jsonl"
LOG="data/rollouts/g2b_chunks.log"
LIMIT="${1:-50}"
TOTAL=200   # M=100 入力 × 2 model
for chunk in $(seq 1 12); do
  done=$(wc -l < "$OUT" 2>/dev/null | tr -d ' ')
  done=${done:-0}
  echo "[driver] $(date +%H:%M:%S) chunk#${chunk} 開始: done=${done}/${TOTAL}" | tee -a "$LOG"
  if [ "$done" -ge "$TOTAL" ]; then
    echo "[driver] done=${done} >= ${TOTAL} → 全完走・停止" | tee -a "$LOG"
    break
  fi
  py -3.14 scripts/dollma_g2b_reward_prepost.py --run --limit "$LIMIT" >> "$LOG" 2>&1
  rc=$?
  done2=$(wc -l < "$OUT" 2>/dev/null | tr -d ' ')
  echo "[driver] $(date +%H:%M:%S) chunk#${chunk} 終了 rc=${rc}: done=${done2}/${TOTAL}" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "[driver] 非0終了 (rc=${rc}) → 停止 (再起動で resume 可)" | tee -a "$LOG"
    break
  fi
done
echo "[driver] $(date +%H:%M:%S) ドライバ終了 done=$(wc -l < "$OUT" 2>/dev/null | tr -d ' ')/${TOTAL}" | tee -a "$LOG"
