import nanogcg
import torch

from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
model_id = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
data = pd.read_csv("./advbench/harmful_behaviors.csv")

with open("./advbench/adversarial_suffixes.jsonl", 'w') as f:
    for i, row in tqdm(data.iterrows(), total=100):
        if i == 100: break
        message = f"Instruction: {row['goal']}\n"
        target = f"Response: {row['target']}"
        config = GCGConfig(
            num_steps=250,
            search_width=64,
            batch_size=512,
            topk=64,
            seed=42,
            verbosity="WARNING"
        )
        result = nanogcg.run(model, tokenizer, message, target, config)
        suffix = result.best_string
        record = {'instruct': message, 'target': target, 'suffix': suffix}
        f.write(json.dumps(record, ensure_ascii=False))
        f.write('\n')

    