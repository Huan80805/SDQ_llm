import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src import (
    patch_gpt2_with_adaptive_adapters,
    patch_gpt2_with_quantization, 
    SUPPORTED_BWS,
    Trainer,
    load_squad_train,
    load_squad_eval
)
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from src import load_generated_train
def main():
    parser = argparse.ArgumentParser(description="GPT-2 Quantization and Training")
    parser.add_argument('--use_tensorboard', action='store_true', help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_log_dir', type=str, default='./logs', help='Tensorboard log directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--use_kd', action='store_true', help='Use knowledge distillation in instantnet-style training')
    parser.add_argument('--training_style', type=str, default='cyclic', choices=['instantnet', 'cyclic', 'spnet'], help='Training style')
    parser.add_argument('--total_steps', type=int, default=1000, help='Total training steps')
    parser.add_argument('--eval_every', type=int, default=100, help='Evaluate every N steps')
    parser.add_argument('--logging_step', type=int, default=5, help='Log every N steps')
    parser.add_argument('--save_dir', type=str, default='./ft_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--exp_name', type=str)

    args = parser.parse_args()

    writer = None
    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.tensorboard_log_dir, args.exp_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if not tokenizer.pad_token:
        assert tokenizer.eos_token
        tokenizer.pad_token = tokenizer.eos_token

    quant_model = patch_gpt2_with_quantization(model)
    print("- Patching quantized model with adaptive Lora...")
    switchable_model = patch_gpt2_with_adaptive_adapters(quant_model, supported_bws=SUPPORTED_BWS)
    train_dataset = load_squad_train(tokenizer)
    # train_dataset = load_generated_train(tokenizer)
    eval_dataset, eval_info = load_squad_eval(tokenizer)
    trainer = Trainer(
        switchable_model=switchable_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_info=eval_info, # DataLoader will zip string list, manually pass them to evaluation
        training_style=args.training_style,
        batch_size=args.batch_size,
        use_kd=args.use_kd,
        supported_bws=SUPPORTED_BWS,
        writer=writer,
        logging_step=args.logging_step
    )
    trainer.train(total_steps=args.total_steps, eval_every=args.eval_every, save_dir=os.path.join(args.save_dir, args.exp_name))

    if writer:
        writer.close()
# To run this test:
# pytest -s tests/test_training_on_squad.py
if __name__ == "__main__":
    main()