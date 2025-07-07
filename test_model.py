import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import Conv1D
from datasets import load_dataset
from src import patch_gpt2_with_quantization, patch_gpt2_with_adaptive_adapters

def test_quant_model_forward():
    """
    Performs a smoke test on the quantized GPT-2 model.
    """
    # Load Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # Add a padding token for batching
    if tokenizer.pad_token is None:
        assert tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token
    print(f"- Base model loaded onto device: {device}")

    print("- Loading one sample from the SQuAD dataset...")
    # Use streaming to avoid downloading the whole dataset
    squad_sample = next(iter(load_dataset("squad", split="train", streaming=True)))
    question = squad_sample['question']
    context = squad_sample['context']
    print(f"--- Question: {question[:50]}...")
    print(f"--- Context: {context[:100]}...")

    inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True).to(device)

    print("- Patching the model with QuantLinear layers...")
    patched_model = patch_gpt2_with_quantization(model)

    print("- Forward pass and verifying output...")
    start_time = time.time()
    with torch.no_grad():
        outputs = patched_model(**inputs)
    end_time = time.time()
    
    # Check if logits are finite
    assert torch.isfinite(outputs.logits).all()
    
    print(f"--- Output tensor shape: {outputs.logits.shape}")
    print(f"--- Time for forward pass: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    with torch.no_grad():
        outputs = patched_model.generate(**inputs)
    end_time = time.time()
    print(f"--- Prompt: {tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=False)[0]}")
    print(f"--- Generated text: {tokenizer.batch_decode(outputs[:, inputs.input_ids.size(-1):], skip_special_tokens=False)[0]}")
    print(f"--- Time for generation: {end_time - start_time:.4f} seconds")

def test_adaptive_lora():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    quant_model = patch_gpt2_with_quantization(model)
    print("- Patching quantized model with adaptive Lora...")
    sqlm = patch_gpt2_with_adaptive_adapters(quant_model, verbose=True)
    # Configuration 1: First layer is 4-bit, the rest are 8-bit
    config1 = {}
    for module_full_name in sqlm.target_modules:
        if int(module_full_name.split('.')[2]) == 0:
            config1[module_full_name] = 4
        else:
            config1[module_full_name] = 8
    sqlm.set_config(config1)
    sqlm.check_config(config1)
    # Configuration 2: First layer is 2-bit, the rest are 4-bit
    config2 = {}
    for module_full_name in sqlm.target_modules:
        if int(module_full_name.split('.')[2]) == 0:
            config2[module_full_name] = 2
        else:
            config2[module_full_name] = 4
    sqlm.set_config(config2)
    sqlm.check_config(config2)
