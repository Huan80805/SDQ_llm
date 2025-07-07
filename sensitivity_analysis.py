"""
Just a naive way to rank layer sensitivity to quantization:
measuring the MSE between FP16/FP32 model and Quant + LoRA layer
when the input is original model input (to avoid accumulated error)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
import os
import json
from tqdm import tqdm
from src import (
    patch_gpt2_with_quantization, 
    patch_gpt2_with_adaptive_adapters,
    load_squad_train,
    QuantLinear,
)

def main():
    parser = argparse.ArgumentParser(description="Layer-wise Sensitivity Analysis using MSE")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing all saved LoRA adapters.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of SQuAD samples to use for analysis.')
    parser.add_argument('--output_file', type=str, default='sensitivity.json', help='Path to save the sensitivity ranking JSON file.')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    SUPPORTED_BWS = [2, 4, 8]
    # --- Load Model and Tokenizer ---
    base_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Patch Model and Load Adapters ---
    quant_model = patch_gpt2_with_quantization(base_model)
    switchable_model = patch_gpt2_with_adaptive_adapters(quant_model, supported_bws=SUPPORTED_BWS)
    
    # Load all adapters from the training output directory
    num_loaded_adapters = 0
    for adapter_name in os.listdir(args.model_path):
        if os.path.isdir(os.path.join(args.model_path, adapter_name)):
            num_loaded_adapters += 1
            switchable_model.model.load_adapter(os.path.join(args.model_path, adapter_name), adapter_name=adapter_name)
    print(f"- Loaded {num_loaded_adapters} trained adapters from {args.model_path}...")

    # --- Load and Prepare Dataset ---
    train_dataset = load_squad_train(tokenizer, num_samples=args.num_samples)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    # --- Initialize Error Accumulators ---
    error_totals = {
        name: {f'{bw}-bit': {"mse": 0.0, "mae": 0.0 } for bw in SUPPORTED_BWS}
        for name in switchable_model.target_modules
    }

    num_batches = 0

    # --- Loop over all batches to accumulate Errors ---
    print("- Calculating average MSE/MAE for each layer over all batches...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        num_batches += 1
        
        # --- Get Oracle Activations for the current batch ---
        activations = {'inputs': {}, 'outputs': {}}
        hooks = []

        def get_hook(name: str):
            def hook(model, input, output):
                assert len(name.removesuffix(".base_layer")) < len(name)
                module_name = name[name.index('transformer.h.'):].removesuffix(".base_layer")
                activations['inputs'][module_name] = input[0].detach()
                activations['outputs'][module_name] = output.detach()
            return hook

        for name, module in switchable_model.model.named_modules():
            if isinstance(module, QuantLinear):
                hooks.append(module.register_forward_hook(get_hook(name)))

        switchable_model.model.eval()
        with torch.no_grad():
            with switchable_model.model.disable_adapter():
                for module in switchable_model.model.base_model.modules():
                    if isinstance(module, QuantLinear):
                        module.set_bit_width(0)
                
                inputs = {k: v.to(device) for k, v in batch.items()}
                switchable_model(**inputs)
        
        for hook in hooks:
            hook.remove()
        
        # --- Calculate Error for one batch and add to totals ---
        for name in switchable_model.target_modules:
            oracle_input = activations['inputs'][name]
            oracle_output = activations['outputs'][name]
            
            for bw in SUPPORTED_BWS:
                with torch.no_grad():
                    adapter_name = name.replace('.', '-') + f'-{bw}'
                    switchable_model.model.set_adapter(adapter_name)
                    
                    quant_layer = switchable_model.model.base_model.get_submodule(name)
                    quant_base_layer: QuantLinear = switchable_model.model.base_model.get_submodule(name + ".base_layer")
                    quant_base_layer.set_bit_width(bw)
                    quant_output = quant_layer(oracle_input)
                    quant_base_output = quant_base_layer(oracle_input)
                    mse = torch.mean((oracle_output - quant_output) ** 2).item()
                    mae = torch.mean(torch.abs(oracle_output - quant_output)).item()
                    # mse = torch.mean((quant_output - quant_base_output) ** 2).item()
                    # mae = torch.mean(torch.abs(quant_output - quant_base_output)).item()
                    error_totals[name][f'{bw}-bit']["mse"] += mse / len(dataloader)
                    error_totals[name][f'{bw}-bit']["mae"] += mae / len(dataloader)

                    switchable_model.model.disable_adapter()
                    quant_base_layer.set_bit_width(0)

    assert switchable_model.target_modules == list(error_totals.keys())
    with open(args.output_file, 'w') as f:
        json.dump(error_totals, f, indent=4)
    print(f"\n-Sensitivity analysis complete, saved to {args.output_file}")
    # --- Rank layers by sensitivity---
    for error_type in ['mse', 'mae']:
        for bw in SUPPORTED_BWS:
            ranking = {}
            sorted_layers = sorted(error_totals.keys(), key=lambda k: error_totals[k][f'{bw}-bit'][error_type])
            ranking[f'rank_{bw}bit'] = sorted_layers
            
            print(f"\n--- Top 5 most tolerant layers to {bw}-bit quantization---")
            for layer_name in ranking[f'rank_{bw}bit'][:5]:
                print(f"- {layer_name}: {error_type.upper()} = {error_totals[layer_name][f'{bw}-bit'][error_type]:.6f}")


if __name__ == "__main__":
    main()