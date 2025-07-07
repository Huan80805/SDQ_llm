"""
use the ranking file to start the search for the optimal configuration. 
This will take longer as it runs the full evaluation harness multiple times.
"""
import argparse
import json
import os
import pandas as pd
from src import SUPPORTED_BWS, load_squad_eval
import evaluate
from quant_model_eval import eval_model_with_bitmap, load_switchable_model
from tqdm import tqdm
from torch.utils.data import DataLoader
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
    device = 'cuda:0'

    # --- Re‑use base FP model & dataloader across all trials --- 
    switchable_model, tokenizer = load_switchable_model(args.model_path, verbose=False)
    ds, info = load_squad_eval(tokenizer, num_samples=args.num_samples, split='train')
    eval_dataloader = DataLoader(ds, batch_size=args.batch_size)
    squad_metric = evaluate.load('squad')

    def quick_eval(bitmap):
        return eval_model_with_bitmap(
            model_path=args.model_path,
            bitmap=bitmap,
            batch_size=args.batch_size,
            num_eval_samples=None,
            split='train',
            verbose=False,
            switchable_model=switchable_model,
            tokenizer=tokenizer,
            eval_dataloader=eval_dataloader,
            eval_info=info,
            squad_metric=squad_metric,
            device=device,
        )
    # --- Load Sensitivity Analysis File and Rank by Layers ---
    with open(args.sensitivity_file, 'r') as f:
        error_totals = json.load(f)
    # --- Rank layers by sensitivity---
    ERROR_TYPE = "mse"
    ranking = {
        f'{bw}bit': sorted(error_totals.keys(), key=lambda k: error_totals[k][f'{bw}-bit'][ERROR_TYPE])
        for bw in SUPPORTED_BWS
    }
    
    ordered_layer_names = list(error_totals.keys()) # architectural layers in order
    num_layers = len(ordered_layer_names)
    results_log = []

    # --- Baseline Evaluation (all FP) ---
    print(f"\n- Evaluting FP GPT2 ...")
    baseline_bitmap = [0] * num_layers # not using layer name for simplicity
    baseline_metrics = quick_eval(baseline_bitmap)
    results_log.append({'bitmap': baseline_bitmap, **baseline_metrics, 'accepted': True})
    base_F1 = baseline_metrics['F1']
    budget = args.budget * base_F1
    # --- Baseline Evaluation (all 8-bit) ---
    print(f"\n- Evaluting Int8 GPT2 ...")
    quick_eval([8] * num_layers)
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
                trial_metrics = quick_eval(trial_bitmap)
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
        quick_eval(current_bitmap)
        # results_log.append({'bitmap': current_bitmap, **metrics})


    # --- Save results to CSV ---
    df = pd.DataFrame(results_log)
    df.to_json(args.output_jsonl, lines=True, orient="records")
    print(f"\nGreedy search complete. Results saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
