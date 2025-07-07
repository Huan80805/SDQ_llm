"""
use the ranking file to start the search for the optimal configuration. 
This will take longer as it runs the full evaluation harness multiple times.
"""
import torch
import argparse
import json
import subprocess
import re
import pandas as pd
from src import SUPPORTED_BWS
from quant_model_eval import eval_model_with_bitmap
from tqdm import tqdm
def main():
    parser = argparse.ArgumentParser(description="Greedy Search for Optimal Mixed-Precision Bitmap")
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to the directory with saved LoRA adapters.')
    parser.add_argument('--sensitivity_file', '-sf', type=str, help='Path to the sensitivity ranking JSON file.')
    parser.add_argument('--budget', type=float, default=0.3, help='Maximum acceptable drop (in ratio) in F1 score.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--max_trials', type=int, default=5, help='Maximum trails to quantize different linear layers to a lower precision')
    parser.add_argument('--output_jsonl', '-o', type=str, default='./analysis/greedy_search_spnet.jsonl', help='Path to save the results of the search.')
    parser.add_argument('--num_samples', '-n', type=int, default=500, help='Number of samples to do eval (for speedup)')
    args = parser.parse_args()

    # --- Load Sensitivity Analysis File and Rank by Layers ---
    with open(args.sensitivity_file, 'r') as f:
        error_totals = json.load(f)
        # --- Rank layers by sensitivity---
    ERROR_TYPE = "mse"
    ranking = {}
    for bw in SUPPORTED_BWS:
        sorted_layers = sorted(error_totals.keys(), key=lambda k: error_totals[k][f'{bw}-bit'][ERROR_TYPE])
        ranking[f'{bw}bit'] = sorted_layers
    
    ordered_layer_names = list(error_totals.keys()) # architectural layers in order
    num_layers = len(ordered_layer_names)
    results_log = []

    # --- Baseline Evaluation (all FP) ---
    print(f"\n- Evaluting FP GPT2 ...")
    baseline_bitmap = [0] * num_layers # not using layer name for simplicity
    baseline_metrics = eval_model_with_bitmap(args.model_path, baseline_bitmap, args.batch_size, num_eval_samples=args.num_samples, split="train", verbose=False)
    results_log.append({'bitmap': baseline_bitmap, **baseline_metrics, 'accepted': True})
    base_F1 = baseline_metrics['F1']
    budget = args.budget * base_F1
    # --- Baseline Evaluation (all 8-bit) ---
    print(f"\n- Evaluting Int8 GPT2 ...")
    eval_model_with_bitmap(args.model_path, [8] * num_layers, args.batch_size, num_eval_samples=args.num_samples, split="train", verbose=False)
    # results_log.append({'bitmap': baseline_bitmap, **baseline_metrics})
    
    # --- Greedy Search for lower precision ---
    current_bitmap = list(baseline_bitmap)
    for bw in sorted(SUPPORTED_BWS, reverse=True):
        print(f"\n- Greedy search for {bw}-bit (F1 budget: {budget})")
        num_accepted_layers, num_rejected_layers = 0, 0
        for layer_name in tqdm(ranking[f'{bw}bit'], total=len(ranking[f'{bw}bit'])):
            layer_idx = ordered_layer_names.index(layer_name)
            if (current_bitmap[layer_idx] > bw) or (current_bitmap[layer_idx] == 0):
                trial_bitmap = list(current_bitmap)
                trial_bitmap[layer_idx] = bw
                # eval new bitmap
                trial_metrics = eval_model_with_bitmap(args.model_path, trial_bitmap, args.batch_size, num_eval_samples=args.num_samples, split="train", verbose=False)
                if base_F1 - trial_metrics['F1'] <= budget:
                    tqdm.write(f"  [ACCEPT] {layer_name} -> {bw}-bit. F1 drop: {base_F1 - trial_metrics['F1']:.2f} <= {budget}")
                    results_log.append({'bitmap': current_bitmap, **trial_metrics, 'accepted': True})
                    current_bitmap = trial_bitmap
                    num_accepted_layers += 1
                    
                else:
                    tqdm.write(f"  [REJECT] {layer_name} -> {bw}-bit. F1 drop: {base_F1 - trial_metrics['F1']:.2f} > {budget}")
                    results_log.append({'bitmap': current_bitmap, **trial_metrics, 'accepted': False})
                    num_rejected_layers += 1
                    if num_rejected_layers >= args.max_trials:
                        tqdm.write('  Max trials reached, moving to next precision')
                        break
        print(f"\n- Finished {bw}-bit search. Evaluating final bitmap...")
        eval_model_with_bitmap(args.model_path, current_bitmap, args.batch_size, args.num_samples, "train", verbose=False)
        # results_log.append({'bitmap': current_bitmap, **metrics})


    # --- Save results to CSV ---
    df = pd.DataFrame(results_log)
    df.to_json(args.output_jsonl, lines=True, orient="records")
    print(f"\nGreedy search complete. Results saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
