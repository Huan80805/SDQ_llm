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
    SwitchableQuantLoRAModel,
)

def get_layer_activations(switchable_model: SwitchableQuantLoRAModel, dataloader: DataLoader, device: str):
    """
    forward pass in full precision to capture the input and output of each QuantLinear layer, serving reference.
    """
    activations = {'inputs': {}, 'outputs': {}}
    hooks = []

    def get_hook(name: str):
        def hook(model, input, output):
            module_name = name[name.index('transformer.h.'):].removesuffix(".base_layer")
            activations['inputs'][module_name] = input[0].detach()
            activations['outputs'][module_name] = output.detach()
        return hook

    # Attach hooks to all QuantLinear layers
    for name, module in switchable_model.model.named_modules():
        if isinstance(module, QuantLinear):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # Run a single batch through the model in full-precision mode
    print("Capturing oracle activations in full precision...")
    switchable_model.model.eval()
    with torch.no_grad():
        with switchable_model.model.disable_adapter(): # Ensure LoRA is off
            # Set all layers to full precision (0 is our flag for FP16/32)
            for name, module in switchable_model.model.named_modules():
                if isinstance(module, QuantLinear):
                    module.set_bit_width(0)
            batch = next(iter(dataloader))
            inputs = {k: v.to(device) for k, v in batch.items()}
            switchable_model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    return activations

def main():
    parser = argparse.ArgumentParser(description="Layer-wise Sensitivity Analysis using MSE")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing all saved LoRA adapters.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of SQuAD samples to use for analysis.')
    parser.add_argument('--output_file', type=str, default='sensitivity_ranking.json', help='Path to save the sensitivity ranking JSON file.')
    parser.add_argument('--batch_size', default=8)
    args = parser.parse_args()
    
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
    
    print(f"Loading trained adapters from {args.model_path}...")
    for adapter_name in os.listdir(args.model_path):
        if os.path.isdir(os.path.join(args.model_path, adapter_name)):
            switchable_model.model.load_adapter(os.path.join(args.model_path, adapter_name), adapter_name=adapter_name)

    # --- Load and Prepare Dataset ---
    train_dataset = load_squad_train(tokenizer, num_samples=args.num_samples)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)


    # --- Initialize MSE Accumulators ---
    mse_totals = {
        name: {f'{bw}-bit': 0.0 for bw in SUPPORTED_BWS}
        for name in switchable_model.target_modules
    }
    num_batches = 0
    
    oracle_activations = get_layer_activations(switchable_model, dataloader, device)
    # --- Calculate MSE for each layer at different bit-widths ---
    sensitivity_data = {}
    print("Calculating MSE for each layer at different bit-widths...")
    
    for name in tqdm(switchable_model.target_modules, desc="Analyzing Layers"):
        sensitivity_data[name] = {}
        oracle_input = oracle_activations['inputs'][name]
        oracle_output = oracle_activations['outputs'][name]
        # Get a reference to the main PEFT model
        for bw in SUPPORTED_BWS:
            with torch.no_grad():
                # Activate the specific adapter for this layer
                adapter_name = name.replace('.', '-') + f'-{bw}'
                switchable_model.model.set_adapter(adapter_name)
                # base: original linear layer
                quant_layer_base = switchable_model.model.base_model.get_submodule(name + ".base_layer")
                # base + LoRA_A, LoRA_B
                quant_layer = switchable_model.model.base_model.get_submodule(name)
                # Get the output with the adapter's correction applied
                quant_output = quant_layer(oracle_input)
                mse = torch.mean((oracle_output - quant_output) ** 2).item()
                sensitivity_data[name][f'{bw}-bit'] = mse

                switchable_model.model.disable_adapter()
                quant_layer_base.set_bit_width(0)

    # --- Rank layers by sensitivity ---
    ranking = {}
    for bw in SUPPORTED_BWS:
        # Sort layers by their MSE for the current bit-width, from lowest to highest MSE
        sorted_layers = sorted(sensitivity_data.keys(), key=lambda k: sensitivity_data[k][f'{bw}-bit'])
        ranking[f'rank_{bw}bit'] = sorted_layers

    with open(args.output_file, 'w') as f:
        json.dump(ranking, f, indent=4)
        
    print(f"\nSensitivity analysis complete. Ranking saved to {args.output_file}")
    print("\n--- Top 5 most tolerant layers to 8-bit quantization---")
    for layer_name in ranking['rank_8bit'][:5]:
        print(f"- {layer_name}: MSE = {sensitivity_data[layer_name]['8-bit']:.6f}")

    print(f"\nSensitivity analysis complete. Ranking saved to {args.output_file}")
    print("\n--- Top 5 most tolerant layers to 4-bit quantization---")
    for layer_name in ranking['rank_4bit'][:5]:
        print(f"- {layer_name}: MSE = {sensitivity_data[layer_name]['4-bit']:.6f}")

    print(f"\nSensitivity analysis complete. Ranking saved to {args.output_file}")
    print("\n--- Top 5 most tolerant layers to 2-bit quantization---")
    for layer_name in ranking['rank_2bit'][:5]:
        print(f"- {layer_name}: MSE = {sensitivity_data[layer_name]['2-bit']:.6f}")

if __name__ == "__main__":
    main()