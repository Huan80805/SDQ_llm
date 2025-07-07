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
from typing import List, Union, Optional
from src import (
    patch_gpt2_with_adaptive_adapters,
    patch_gpt2_with_quantization, 
    create_quant_model_for_inference,
    SUPPORTED_BWS,
    load_squad_eval
)
import json
from transformers.modeling_utils import Conv1D 

def parse_bitmap(bitmap: Union[str, List[int]], num_layers: int):
    """ Parses a comma-separated string into a list of integers. """
    if isinstance(bitmap, list): pass
    elif isinstance(bitmap, str):
        try:
            bits = [int(b.strip()) for b in bitmap.split(',')]
            if len(bits) == 1:
                bitmap = [bits[0]] * num_layers
            else: bitmap = bits
        except Exception as e:
            raise ValueError(f"Invalid bitmap string: {bitmap}. Must be comma-separated integers.") from e
    else: raise TypeError(f"Unsupported bitmap data type: {type(bitmap)}")

    if len(bitmap) != num_layers:
        raise ValueError(f"Bitmap length must match the number of target layers ({num_layers})")
    return bitmap

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

    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=20,
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
        "EM": round(results['exact_match'], 4),
        "F1": round(results['f1'], 4),
        "VRAM_MB": round(peak_vram_mb, 2),
        "tokens_per_s": round(tokens_per_second, 2)
    }


def eval_model_with_bitmap(
        model_path: str, 
        bitmap: Union[str, List[int]], 
        batch_size: int, 
        num_eval_samples: int = None, 
        split: str = "validation",
        verbose: bool = True,
    ):
    
    device = "cuda:0"
    # --- Load Base Model, Adapters, and Tokenizer ---
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = patch_gpt2_with_quantization(model)
    model = patch_gpt2_with_adaptive_adapters(model, supported_bws=SUPPORTED_BWS)
    
    # Load all adapters from the training output directory
    num_loaded_adapters = 0
    for adapter_name in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, adapter_name)):
            num_loaded_adapters += 1
            model.model.load_adapter(os.path.join(model_path, adapter_name), adapter_name=adapter_name)
    if verbose:
        print(f"- Loaded {num_loaded_adapters} trained adapters from {model_path}")
    # --- Set Quantization Configuration ---
    num_target_layers = len(model.target_modules)
    bits_vector = parse_bitmap(bitmap, num_target_layers)
    
    bitmap_config = {name: bits for name, bits in zip(model.target_modules, bits_vector)}
    if verbose:
        print("- Evaluating bitmap:", json.dumps(bitmap_config, indent=4))
    if all(v==0 for v in bitmap_config.values()):
        model = model.model.base_model
    else:
        model.set_config(bitmap_config)
        model.check_config(bitmap_config)

        # Pass a PeftModel for true quantized model, we don't need to switch Quantization config now
        model = create_quant_model_for_inference(model.model, bitmap_config)
        if verbose:
            print("- Built true quantized inference model")
    model.to(device)

    torch.cuda.empty_cache()
    eval_dataset, eval_info = load_squad_eval(tokenizer, num_samples=num_eval_samples, split=split)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    metrics = evaluate_performance(model, tokenizer, eval_dataloader, eval_info, device)

    # --- Print Results ---
    print("\n- Results:", json.dumps(metrics))

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Harness for Switchable Quantization")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing all saved LoRA adapters.')
    parser.add_argument('--bitmap', type=str, required=True, help='Comma-separated bit-width vector (e.g., "8,8,4,2,...") or a single number for all layers.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--num_eval_samples', "-n", type=int, default=None)
    parser.add_argument('--split', type=str, help="Split of the dataset to eval on")
    args = parser.parse_args()
    eval_model_with_bitmap(model_path = args.model_path, bitmap = args.bitmap, batch_size = args.batch_size, num_eval_samples = args.num_eval_samples, split=args.split)