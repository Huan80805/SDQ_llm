import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from trainer import Trainer
from modeling_gpt2 import patch_gpt2_with_quantization, patch_gpt2_with_adaptive_adapters
import os
from defaults import SUPPORTED_BWS
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if not tokenizer.pad_token:
        assert tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token

    quant_model = patch_gpt2_with_quantization(model)
    print("- Patching quantized model with adaptive Lora...")
    switchable_model = patch_gpt2_with_adaptive_adapters(quant_model, supported_bws=SUPPORTED_BWS)

    train_dataset = load_dataset("squad", split="train")
    eval_dataset = load_dataset("squad", split="validation").select(range(200))
    PROMPT_TEMPLATE = "{c}\n\n{q}\n\n"
    INPUT_TEMPLATE = PROMPT_TEMPLATE + "{a}" + tokenizer.eos_token
    def train_preprocess(examples):
        tokenizer.padding_side = "right"
        answers = [a['text'][0] for a in examples["answers"]]
        # Create the full input tokens
        inputs = [ INPUT_TEMPLATE.format(q=q,c=c,a=a) for q, c, a in zip(examples["question"], examples["context"], answers)]
        prompt_lengths = [
            len(tokenizer(PROMPT_TEMPLATE.format(q=q, c=c), max_length=1024, truncation=True, padding=False).input_ids)
            for q, c in zip(examples["question"], examples["context"])
        ]
        input_lengths = [ len(tokenizer(i, max_length=1024, truncation=True, padding=False).input_ids) for i in inputs]# unpadded input
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        labels = model_inputs["input_ids"].clone()
        # Create the Loss Mask
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
            labels[i, input_lengths[i]:] = -100
        model_inputs["labels"] = labels
        return model_inputs

    train_dataset = train_dataset.map(train_preprocess, batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def eval_preprocess(examples):
        tokenizer.padding_side = "left"
        prompts = [ PROMPT_TEMPLATE.format(q=q, c=c) for q, c in zip(examples["question"], examples["context"])]
        model_inputs = tokenizer(prompts, max_length=824, truncation=True, padding="max_length", return_tensors="pt")
        return model_inputs
    
    eval_info = {"id": eval_dataset["id"], "answers": eval_dataset["answers"]}
    eval_dataset = eval_dataset.map(eval_preprocess, batched=True)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    trainer = Trainer(
        switchable_model=switchable_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_info=eval_info, # DataLoader will zip string list, manually pass them to evaluation
        training_style='cyclic',
        batch_size=4,
        kd_weight=1.,
        supported_bws=SUPPORTED_BWS
    )
    output_dir = "./checkpoints/cyclic"
    trainer.train(total_steps=1000, eval_every=100, save_dir=output_dir)
    
    print("\n--- Final Evaluation after Training ---")
    trainer.evaluate()
    
    print("\n--- Final Checkpointing ---")
    trainer.save_model(os.path.join(output_dir, "final_model"))

# To run this test:
# pytest -s tests/test_training_on_squad.py
if __name__ == "__main__":
    main()