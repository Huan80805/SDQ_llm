"""
Use QA datasets to generate SFT data for Quantization
1. Loads GPT-2 as a generation model
2. Uses the SQuAD v1.1  question + context fields as the prompt
3. GPT-2 autoregressively produce an answer
4. Saves everything to disk so it can be loaded using HuggingFace `datasets`
"""

import os
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import tqdm
import random
from src import PROMPT_TEMPLATE

OUT_DIR = "squad_gpt2_sft"
MAX_PROMPT_TOK = 993
MAX_NEW_TOKENS = 30
BATCH_SIZE = 8
DEVICE = "cuda"
MODEL_NAME = "gpt2"
GREEDY_PREFIX_RANGE = range(3, 6)
TEMPERATURE = 0.8
TOP_P = 0.95
NUM_SAMPLES = 5000
EM = 0.

def build_corpus(num_samples=None, split="train"):
    """Stream over SQuAD, make GPT-2 answer, save to arrow files."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side= "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # Load SQuAD
    ds = load_dataset("squad", split=split)
    if num_samples:
        ds = ds.select(range(num_samples))

    def generate(batch):
        global EM
        prompts = [
            PROMPT_TEMPLATE.format(c=c, q=q)
            for c, q in zip(batch["context"], batch["question"])
        ]
        answers = batch["answers"]
        inputs = tokenizer( 
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT_TOK,
        ).to(DEVICE)

        best_full, best_prefix = [], []
        with torch.no_grad():
            for j in GREEDY_PREFIX_RANGE:
                g_prefix = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=j,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

                g_full = model.generate(
                    g_prefix,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

                for i, (prompt, answer) in enumerate(zip(prompts, answers)):
                    pl = inputs["input_ids"][i].size(0)
                    prefix_ids = g_prefix[i][pl:]
                    sample_ids = g_full[i][pl + len(prefix_ids):]

                    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                    sample = tokenizer.decode(sample_ids, skip_special_tokens=True)
                    full = prompt + prefix + sample

                    best_full.append(full)
                    best_prefix.append(prefix)

                    # print(f"[Prompt]\n{prompt!r}")
                    # print(f"[Gen]\n{(prefix+sample)!r}")
                    # print(f"[Answer]\n{answer!r}")
                    if answer['text'][0] in (prefix + sample): EM += (1/num_samples/len(GREEDY_PREFIX_RANGE))

        return {
            "prompt": prompts * len(GREEDY_PREFIX_RANGE),
            "prefix": best_prefix,
            "text":   best_full,
        }

    # Hugging Face map-style batch generation
    ds = ds.map(
        generate,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=ds.column_names,
    )
    # Persist to arrow files
    if os.path.exists(OUT_DIR):
        print(f"Overwriting existing folder: {OUT_DIR}")
        os.system(f"rm -rf {OUT_DIR}")

    ds.save_to_disk(OUT_DIR)
    print(f"Saved {len(ds):,} examples to {OUT_DIR}")
    print(f"Exact Match: {EM}")
    return ds


if __name__ == "__main__":
    build_corpus(num_samples=NUM_SAMPLES, split="train")
    
