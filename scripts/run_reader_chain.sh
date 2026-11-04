#!/usr/bin/env bash
# Fine-tune the reader on Modal and transcribe the train and test photos. Every step is a detached Modal call that is
# polled once a minute, so the chain survives a closed terminal only up to the step that is running.
#
#   bash scripts/run_reader_chain.sh [run = v1] [epochs = 2] [none_keep = 0.7]
#
# Needs the `modal` CLI (logged in) and the Modal secret `huggingface` (HF_TOKEN of an account that accepted the
# terms of the dataset mirror).
#
# Steps: fetch_data (CPU) -> train_vlm (GPU, about 80 min on an A10G) -> merge_adapter (CPU)
#        -> vllm_run (GPU: transcribe train + test, diagnose holdout + test) -> download to work/<run>/
#        -> stage the transcriptions as work/v1ocr/ocr_{train,test}_v1.csv, where the text models read them
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
RUN=${1:-v1}; EPOCHS=${2:-2}; NONE_KEEP=${3:-0.7}

spawn() { python scripts/modal_spawn.py spawn "$@" | tail -1 | awk '{print $3}'; }
wait_call() {
  local id=$1 st
  while true; do
    st=$(python scripts/modal_spawn.py poll "$id" 2>&1 | tail -1)
    case "$st" in
      DONE*) echo "$st" | cut -c1-400; return 0 ;;
      FAILED*) echo "$st" | cut -c1-800; return 1 ;;
    esac
    sleep 60
  done
}

if ! modal secret list 2>/dev/null | grep -qi huggingface; then echo "Modal secret 'huggingface' is missing"; exit 2; fi
modal deploy modal_app.py >/dev/null; sleep 10

echo "[1/4] fetch_data";            wait_call "$(spawn fetch_data)"
echo "[2/4] train_vlm $RUN";        wait_call "$(spawn train_vlm run="$RUN" epochs="$EPOCHS" holdout_frac=0.2 tax_in_prompt=false none_keep="$NONE_KEEP")"
echo "[3/4] merge_adapter";         wait_call "$(spawn merge_adapter run="$RUN")"
echo "[4/4] vllm_run";              wait_call "$(spawn vllm_run run="$RUN" split=holdout,test n_samples=12 transcribe=train,test)"

modal run modal_app.py::download --run "$RUN" | grep wrote
mkdir -p work/v1ocr
cp "work/$RUN/vllm_ocr_train.csv" work/v1ocr/ocr_train_v1.csv
cp "work/$RUN/vllm_ocr_test.csv" work/v1ocr/ocr_test_v1.csv
echo "done: work/v1ocr/ocr_{train,test}_v1.csv   next: python scripts/cpu_baseline.py work/v1ocr v1"
