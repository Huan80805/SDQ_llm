import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
import time
import os
import evaluate
from tqdm import tqdm
from src import (
    patch_gpt2_with_adaptive_adapters,
    patch_gpt2_with_quantization, 
    create_qunat_model_for_inference,
    SUPPORTED_BWS,
    load_squad_dev
)
from transformers.modeling_utils import Conv1D 

def parse_bitmap(bitmap_str: str, num_layers: int):
    """ Parses a comma-separated string into a list of integers. """
    try:
        bits = [int(b.strip()) for b in bitmap_str.split(',')]
        if len(bits) == 1:
            return [bits[0]] * num_layers
        elif len(bits) != num_layers:
            raise ValueError(f"Bitmap length must match the number of target layers ({num_layers})")
        return bits
    except Exception as e:
        raise ValueError(f"Invalid bitmap string: {bitmap_str}. Must be comma-separated integers.") from e

def evaluate_performance(model, tokenizer, eval_dataloader, eval_info, device):
    """
    Evaluates the model on the SQuAD dataset for accuracy and efficiency.
    """
    squad_metric = evaluate.load("squad")
    model.eval()

    # --- Efficiency Measurement ---
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    
    start_time = time.time()
    total_generated_tokens = 0
    generated_texts = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=50,
                do_sample=False,
            )
        
        total_generated_tokens += generated_ids.shape[0] * (generated_ids.shape[1] - inputs["input_ids"].shape[1])
        outputs = tokenizer.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        generated_texts.extend(outputs)

    torch.cuda.synchronize()
    end_time = time.time()
    
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    duration = end_time - start_time
    tokens_per_second = total_generated_tokens / duration if duration > 0 else 0

    # --- Accuracy Measurement ---
    predictions = [{"prediction_text": t.strip(), "id": id_} for t, id_ in zip(generated_texts, eval_info["id"])]
    references = [{"answers": a, "id": id_} for a, id_ in zip(eval_info["answers"], eval_info["id"])]
    results = squad_metric.compute(predictions=predictions, references=references)
    
    return {
        "EM": results['exact_match'],
        "F1": results['f1'],
        "VRAM_MB": peak_vram_mb,
        "tokens_per_s": tokens_per_second
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation Harness for Switchable Quantization")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing all saved LoRA adapters.')
    parser.add_argument('--bitmap', type=str, required=True, help='Comma-separated bit-width vector (e.g., "8,8,4,2,...") or a single number for all layers.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--num_eval_samples', "-n", type=int, default=None)
    args = parser.parse_args()
    
    device = "cuda"
    # --- Load Base Model, Adapters, and Tokenizer ---
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = patch_gpt2_with_quantization(model)
    model = patch_gpt2_with_adaptive_adapters(model, supported_bws=SUPPORTED_BWS)
    
    # Load all adapters from the training output directory
    print(f"Loading trained adapters from {args.model_path}...")
    for adapter_name in os.listdir(args.model_path):
        if os.path.isdir(os.path.join(args.model_path, adapter_name)):
            model.model.load_adapter(os.path.join(args.model_path, adapter_name), adapter_name=adapter_name)

    # --- Set Quantization Configuration ---
    num_target_layers = len(model.target_modules)
    bits_vector = parse_bitmap(args.bitmap, num_target_layers)
    
    bitmap_config = {name: bits for name, bits in zip(model.target_modules, bits_vector)}
    model.set_config(bitmap_config)
    model.check_config(bitmap_config)

    # --- Merge Adapters and Create Inference Model ---
    print("Merging activated LoRA adapters...")
    model = model.model.merge_and_unload()
    print("Creating true quantized inference model...")
    model = create_qunat_model_for_inference(model, bitmap_config).to(device)
    
    torch.cuda.empty_cache()

    # --- Load and Prepare Dataset ---
    eval_dataset, eval_info = load_squad_dev(tokenizer, num_samples=args.num_eval_samples)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # --- Run Evaluation ---
    metrics = evaluate_performance(model, tokenizer, eval_dataloader, eval_info, device)

    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    print(f"bm_name,bits_vector,EM,F1,tokens_per_s,vram_MB")
    bits_str = str(bits_vector[0]) if all(b == bits_vector[0] for b in bits_vector) else f"[{','.join(map(str, bits_vector[:4]))}...]"
    print(f"custom_map,{bits_str},{metrics['EM']:.2f},{metrics['F1']:.2f},{metrics['tokens_per_s']:.2f},{metrics['VRAM_MB']:.2f}")

if __name__ == "__main__":
    main()