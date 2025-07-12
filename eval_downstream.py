import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import argparse
import pandas as pd
import time
import os
import evaluate
import copy
from tqdm import tqdm
from functools import lru_cache
from typing import List, Union, Optional
from src import (
    patch_gpt2_with_adaptive_adapters,
    patch_gpt2_with_quantization, 
    create_quant_model_for_inference,
    SUPPORTED_BWS,
    load_squad_eval,
    SwitchableQuantLoRAModel,
)
import json

def parse_bitmap(bitmap: Union[str, List[int]], num_layers: int):
    if isinstance(bitmap, list): out = bitmap
    elif isinstance(bitmap, str):
        try:
            bits = [int(b.strip()) for b in bitmap.split(',')]
            out = bits * num_layers if len(bits) == 1 else bits
        except Exception as e:
            raise ValueError(
                f"Invalid bitmap string: {bitmap}. Must be comma‑separated integers."
            ) from e
    else: raise TypeError(f"Unsupported bitmap data type: {type(bitmap)}")
    if len(out) != num_layers:
        raise ValueError(f"Bitmap length ({len(out)}) must match number of target layers ({num_layers}).")
    return out

# enable cache: load base FP model + adapters once (on **CPU**)
@lru_cache(maxsize=None)
def load_switchable_model(model_path: str, verbose: bool = True):
    """Return model and tokenizer with LoRA adapters loaded (CPU‑resident)."""
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # add switchable quant + adaptive adapters wrappers
    model = patch_gpt2_with_quantization(model)
    model = patch_gpt2_with_adaptive_adapters(model, supported_bws=SUPPORTED_BWS)

    loaded = 0
    for name in os.listdir(model_path):
        p = os.path.join(model_path, name)
        if os.path.isdir(p):
            model.model.load_adapter(p, adapter_name=name)
            loaded += 1
    if verbose:
        print(f"- Loaded {loaded} trained adapters from {model_path}")

    model.model.to('cpu').eval()
    return model, tokenizer

def evaluate_performance(
    model,
    tokenizer,
    eval_dataloader,
    eval_info,
    device,
    squad_metric: Optional[object] = None,
):
    """Compute EM, F1, tokens/s and peak VRAM."""
    squad_metric = squad_metric or evaluate.load('squad')

    model.eval()
    torch.cuda.synchronize()
    start = time.time()

    total_gen_tok = 0
    generations = []
    tokenizer.padding_side='left'
    for batch in tqdm(eval_dataloader, desc="Evaluating", leave=False):
        inputs = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=20,
                do_sample=False,
            )
        total_gen_tok += out_ids.shape[0] * (out_ids.shape[1] - inputs['input_ids'].shape[1])
        gens = tokenizer.batch_decode(out_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
        generations.extend(gens)

    torch.cuda.synchronize()
    t = max(time.time() - start, 1e-6)

    preds = [{"prediction_text": g.strip(), "id": i} for g, i in zip(generations, eval_info["id"])]
    refs  = [{"answers": a, "id": i} for a, i in zip(eval_info["answers"],   eval_info["id"])]
    res   = squad_metric.compute(predictions=preds, references=refs)

    return {
        "EM": round(res['exact_match'], 4),
        "F1": round(res['f1'], 4),
        "tokens_per_s": round(total_gen_tok / t, 2),
    }


def eval_model_with_bitmap(
    model_path: str,
    bitmap: Union[str, List[int]],
    batch_size: int,
    num_eval_samples: int = None,
    split: str = "validation",
    verbose: bool = True,
    *,
    # fast‑path reusable objects (if None -> auto‑load)
    switchable_model: SwitchableQuantLoRAModel=None,
    tokenizer=None,
    eval_dataloader=None,
    eval_info=None,
    squad_metric=None,
    device: str = "cuda:0",
):

    if switchable_model is None or tokenizer is None:
        switchable_model, tokenizer = load_switchable_model(model_path, verbose)

    # Prepare quant / fp model for the given bitmap
    num_layers = len(switchable_model.target_modules)
    bits_vec = parse_bitmap(bitmap, num_layers)
    bitmap_cfg = {n: b for n, b in zip(switchable_model.target_modules, bits_vec)}

    if verbose:
        print("- Evaluating bitmap:")
        print(json.dumps(bitmap_cfg, indent=4))

    quant_clone = False # flag for cleanup logic
    # Move/reference model to GPU and pack weights if needed
    if all(bw == 0 for bw in bits_vec):
        peft_model = switchable_model.model.to(device)
        model = peft_model

    else:
        peft_model_clone = copy.deepcopy(switchable_model.model)
        switchable_model_clone = copy.deepcopy(switchable_model)  # keeps adapters config
        switchable_model_clone.model = peft_model_clone

        switchable_model_clone.set_config(bitmap_cfg)
        switchable_model_clone.check_config(bitmap_cfg)

        model = create_quant_model_for_inference(switchable_model_clone.model, bitmap_cfg).to(device)
        quant_clone = True

    param_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    # return {"Weights_MB": round(param_bytes/1024**2, 2)}

    # Build (or reuse) dataloader
    if eval_dataloader is None or eval_info is None:
        ds, info = load_squad_eval(tokenizer, num_samples=num_eval_samples, split=split)
        eval_dataloader = DataLoader(ds, batch_size=batch_size)
        eval_info = info

    # Run evaluation
    metrics = evaluate_performance( model, tokenizer, eval_dataloader, eval_info, device, squad_metric)
    metrics = {**metrics, "Weights_MB": round(param_bytes/1024**2, 2)}
    print("\n- Results:", json.dumps(metrics))

    # Clean‑up GPU VRAM: move model back to CPU & delete clone
    model.to("cpu")
    if quant_clone:
        del model                # deletes only the clone’s reference
    torch.cuda.empty_cache()

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Harness for Switchable Quantization")
    # parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing all saved LoRA adapters.')
    parser.add_argument('--bitmap', type=str, required=True, help='Comma-separated bit-width vector (e.g., "8,8,4,2,...") or a single number for all layers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation.')
    parser.add_argument('--num_eval_samples', "-n", type=int, default=None)
    parser.add_argument('--split', type=str, help="Split of the dataset to eval on", default="validation")
    args = parser.parse_args()
    eval_model_with_bitmap(model_path = args.model_path, bitmap = args.bitmap, batch_size = args.batch_size, num_eval_samples = args.num_eval_samples, split=args.split)
    # method = args.method
    # with open(f'final_results/{method}.jsonl', 'r') as f:

    #     model_path = f'./checkpoints/{method}/step_1000'
    #     num_layers = 48
    #     bitmaps = {
    #         "fp": [0] * num_layers,
    #         "int8": [8] * num_layers,
    #         "stripe8&4": [ 4*(n%2+1) for n in range(num_layers)],
    #         "int4": [4] * num_layers,
    #         "stripe4&2": [ 2*(n%2+1) for n in range(num_layers)],
    #         "int2": [2] * num_layers,
    #     }
    #     for cfg_name, bitmap in bitmaps.items():
    #         tqdm.write(cfg_name)
    #         metrics = eval_model_with_bitmap(model_path = model_path, bitmap = bitmap, batch_size = 100, split='validation', verbose=False)
    #         row_idx += 1
    #         f.write(json.dumps({'cfg_name': cfg_name, **metrics, 'accepted': None}))
    #         f.write('\n')
    #         f.flush()

    #     gs_file_path = f'./analysis/gs_{method}_log.jsonl'
    #     gs_result = pd.read_json(gs_file_path, lines=True, orient='records')
    #     bitmaps = gs_result['bitmap'].tolist()
    #     accepted = gs_result['accepted'].tolist()
    #     for bitmap in tqdm(bitmaps, desc='greedy search bitmap eval'):
    #         metrics = eval_model_with_bitmap(model_path = model_path, bitmap = bitmap, batch_size = 100, split='validation', verbose=False)
    #         row_idx += 1
    #         f.write(json.dumps({'cfg_name': 'greedy search', **metrics, 'accepted': accepted[row_idx-6]}))
    #         f.write('\n')
    #         f.flush()