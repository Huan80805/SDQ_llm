#!/usr/bin/env bash
# run many commands at once, log each to <name>.txt

set -euo pipefail

# python sensitivity_analysis.py --model_path checkpoints/spnet/step_1000 --num_samples 500 --output_file analysis/sensitivity_spnet.json
# python sensitivity_analysis.py --model_path checkpoints/cyclic/step_1000 --num_samples 500 --output_file analysis/sensitivity_cyclic.json
# python sensitivity_analysis.py --model_path checkpoints/instantnet/step_1000 --num_samples 500 --output_file analysis/sensitivity_instantnet.json
# python sensitivity_analysis.py --model_path checkpoints/instantnet_kd/step_1000 --num_samples 500 --output_file analysis/sensitivity_instantnet_kd.json

# python main.py --training_style=spnet --exp_name=ft_spnet --use_tensorboard
# python main.py --training_style=cyclic --exp_name=ft_cyclic --use_tensorboard
# python main.py --use_kd --training_style=instantnet --exp_name=ft_instantnet_kd --use_tensorboard
# python main.py --training_style=instantnet --exp_name=ft_instantnet --use_tensorboard

# Format:  "log_file_name::actual shell command"
jobs=(
    "job_1::CUDA_VISIBLE_DEVICES=0 python eval_downstream.py --method cyclic"
    "job_2::CUDA_VISIBLE_DEVICES=1 python eval_downstream.py --method instantnet_kd"
    "job_3::CUDA_VISIBLE_DEVICES=2 python eval_downstream.py --method instantnet"
    "job_4::CUDA_VISIBLE_DEVICES=3 python eval_downstream.py --method spnet"
)

LOGDIR="job_logs"
mkdir -p "$LOGDIR"

declare -a pids=() # keep PIDs so we can wait later

for spec in "${jobs[@]}"; do
  name="${spec%%::*}"
  cmd="${spec#*::}"
  logfile="$LOGDIR/${name}.txt"

  echo "[+] Starting \"$cmd\"  >  $logfile"

  # Run the command unbuffered, tee live output to its log.
  #   stdbuf -oL -eL : line-buffer stdout & stderr (flushes each \r / \n)
  #   tee -a : append to log *and* echo to our master terminal
  (
    stdbuf -oL -eL bash -c "$cmd"
  ) 2>&1 | tee -a "$logfile" &

  pids+=($!)                              # remember child PID
done

# Wait for every background job to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "!! All ${#jobs[@]} jobs finished. Logs in $(realpath "$LOGDIR")/"
