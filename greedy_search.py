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

def run_evaluation(model_path, bitmap, batch_size):
    """
    Calls the evaluation_harness.py script as a subprocess and parses its output.
    """
    bitmap_str = ",".join(map(str, bitmap))
    command = [
        "python", "evaluation_harness.py",
        "--model_path", model_path,
        "--bitmap", bitmap_str,
        "--batch_size", str(batch_size)
    ]
    
    print(f"\nRunning evaluation for bitmap: [{bitmap_str[:30]}...]")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("--- ERROR ---")
        print(result.stderr)
        raise RuntimeError("Evaluation harness failed to run.")

    # Parse the CSV output from the harness
    output = result.stdout
    # Find the last line that looks like CSV
    csv_line = None
    for line in reversed(output.splitlines()):
        if re.match(r'^custom_map,', line):
            csv_line = line
            break
    
    if not csv_line:
        print("--- ERROR: Could not parse evaluation output ---")
        print(output)
        raise ValueError("Failed to find CSV output from evaluation harness.")

    parts = csv_line.strip().split(',')
    metrics = {
        "EM": float(parts[2]),
        "F1": float(parts[3]),
        "tokens_per_s": float(parts[4]),
        "VRAM_MB": float(parts[5])
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Greedy Search for Optimal Mixed-Precision Bitmap")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory with saved LoRA adapters.')
    parser.add_argument('--sensitivity_file', type=str, default='sensitivity_ranking.json', help='Path to the sensitivity ranking JSON file.')
    parser.add_argument('--budget', type=float, default=0.5, help='Maximum acceptable drop in EM score.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--output_csv', type=str, default='greedy_search_results.csv', help='Path to save the results of the search.')

    args = parser.parse_args()

    # --- Load Sensitivity Ranking ---
    with open(args.sensitivity_file, 'r') as f:
        sensitivity_ranking = json.load(f)
    
    num_layers = len(sensitivity_ranking['rank_4bit'])
    results_log = []

    # --- Baseline Evaluation (all 8-bit) ---
    print("--- Step 1: Evaluating baseline (all 8-bit) ---")
    baseline_bitmap = [8] * num_layers
    baseline_metrics = run_evaluation(args.model_path, baseline_bitmap, args.batch_size)
    base_em = baseline_metrics['EM']
    print(f"Baseline EM: {base_em:.2f}, VRAM: {baseline_metrics['VRAM_MB']:.2f} MB")
    results_log.append({'name': 'baseline_all_8', 'bitmap': baseline_bitmap, **baseline_metrics})

    # --- Greedy Search for 4-bit ---
    print(f"\n--- Step 2: Greedy search for 4-bit (EM budget: {args.budget}) ---")
    current_bitmap = list(baseline_bitmap)
    
    for layer_name in sensitivity_ranking['rank_4bit']:
        layer_idx = sensitivity_ranking['rank_4bit'].index(layer_name)
        
        if current_bitmap[layer_idx] == 8: # Only try to quantize layers that are still 8-bit
            trial_bitmap = list(current_bitmap)
            trial_bitmap[layer_idx] = 4
            
            trial_metrics = run_evaluation(args.model_path, trial_bitmap, args.batch_size)
            
            if base_em - trial_metrics['EM'] <= args.budget:
                print(f"  [ACCEPT] {layer_name} -> 4-bit. EM drop: {base_em - trial_metrics['EM']:.2f} <= {args.budget}")
                current_bitmap = trial_bitmap
                # Update the base_em to the new accepted value
                base_em = trial_metrics['EM'] 
            else:
                print(f"  [REJECT] {layer_name} -> 4-bit. EM drop: {base_em - trial_metrics['EM']:.2f} > {args.budget}")

    print("\n--- Finished 4-bit search. Evaluating final 4-bit mixed map ---")
    final_4bit_metrics = run_evaluation(args.model_path, current_bitmap, args.batch_size)
    results_log.append({'name': 'mixed_4bit', 'bitmap': current_bitmap, **final_4bit_metrics})

    # --- Greedy Search for 2-bit ---
    print(f"\n--- Step 3: Greedy search for 2-bit (EM budget: {args.budget}) ---")
    # Use the 2-bit sensitivity ranking now
    for layer_name in sensitivity_ranking['rank_2bit']:
        layer_idx = sensitivity_ranking['rank_2bit'].index(layer_name)
        
        # Only try to quantize layers that are currently 4-bit or 8-bit
        if current_bitmap[layer_idx] > 2:
            trial_bitmap = list(current_bitmap)
            trial_bitmap[layer_idx] = 2
            
            trial_metrics = run_evaluation(args.model_path, trial_bitmap, args.batch_size)
            
            if base_em - trial_metrics['EM'] <= args.budget:
                print(f"  [ACCEPT] {layer_name} -> 2-bit. EM drop: {base_em - trial_metrics['EM']:.2f} <= {args.budget}")
                current_bitmap = trial_bitmap
                base_em = trial_metrics['EM']
            else:
                print(f"  [REJECT] {layer_name} -> 2-bit. EM drop: {base_em - trial_metrics['EM']:.2f} > {args.budget}")

    print("\n--- Finished 2-bit search. Evaluating final 2-bit mixed map ---")
    final_2bit_metrics = run_evaluation(args.model_path, current_bitmap, args.batch_size)
    results_log.append({'name': 'mixed_2bit', 'bitmap': current_bitmap, **final_2bit_metrics})

    # --- Save results to CSV ---
    df = pd.DataFrame(results_log)
    df['bitmap'] = df['bitmap'].apply(lambda x: str(x)) # Convert list to string for CSV
    df.to_csv(args.output_csv, index=False)
    print(f"\nGreedy search complete. Results saved to {args.output_csv}")
    print("\nFinal Results Summary:")
    print(df[['name', 'EM', 'F1', 'VRAM_MB', 'tokens_per_s']].to_string(index=False))

if __name__ == "__main__":
    main()
