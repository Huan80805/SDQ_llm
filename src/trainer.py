import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer
from modeling_gpt2 import SwitchableQuantLoRAModel
from tqdm import tqdm
import evaluate
import os
from transformers.utils import logging
from typing import List
from defaults import SUPPORTED_BWS
logging.get_logger("transformers").setLevel(logging.ERROR)

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature # temp = 1.0 in original paper
        self.kld_loss = nn.KLDivLoss(reduction="none", log_target=False)

    def forward(self, teacher_logits, student_logits, mask):
        t_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
        s_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        kl_per_tok = F.kl_div(s_log_prob, t_prob, reduction="none").sum(-1) # [B, S]
        kl_masked = kl_per_tok * mask # excluding kldiv on pad or input mask
        # scaling ensure gradient magnitudes are not affected by the temperature.
        loss = kl_masked.sum() / mask.sum() * self.temperature ** 2
        return loss

class Trainer:
    def __init__(self, 
        switchable_model:SwitchableQuantLoRAModel, 
        tokenizer:AutoTokenizer, 
        train_dataset,
        eval_dataset=None,
        eval_info=None,
        training_style:str='instantnet',
        lr=5e-5,
        batch_size=4,
        kd_weight=1.0,
        supported_bws:List[int]=SUPPORTED_BWS,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.switchable_model = switchable_model
        self.tokenizer = tokenizer
        self.training_style = training_style
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size) if eval_dataset else None
        self.eval_info = eval_info
        self.step = 0
        self.kd_weight = kd_weight

        for p in self.switchable_model.model.parameters():
            p.requires_grad = False

        # Collect all LoRA tensors
        trainable_params = [p for n, p in self.switchable_model.model.named_parameters() if ".lora_A" in n or ".lora_B" in n]
        self.optimizer = AdamW(trainable_params, lr=lr)

        # Define the bit-map configurations to be trained
        self.bitmaps = {
            "all8": {name: 8 for name in self.switchable_model.target_modules},
            "all6": {name: 6 for name in self.switchable_model.target_modules},
            "all4": {name: 4 for name in self.switchable_model.target_modules},
            "all2": {name: 2 for name in self.switchable_model.target_modules},
        }
        self.ordered_bws = list(sorted(supported_bws, reverse=True))
        self.kd_loss_fn = DistillationLoss(temperature=1.0)
        
        print(f"Trainer initialized with training style: {self.training_style}")
        print(f"- Target Bit-maps: {list(self.bitmaps.keys())}")
        print(f"- Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")

    def train(self, total_steps, eval_every:int = 50, save_dir='./checkpoints'):
        print(f"\n--- Training starts with {self.training_style} style ---")
        self.switchable_model.model.train()

        generator = iter(self.train_dataloader)
        progress_bar = tqdm(desc="Training", total=total_steps, position=0)
        for step in range(total_steps):
            try:
                batch = next(generator)
            except StopIteration:
                generator = iter(self.train_dataloader)
                batch = next(generator)

            # print("--- Inspecting First Sample in Batch ---")
            # print("="*50)

            # # Isolate the first sample's data
            # input_ids_sample = batch["input_ids"][0]
            # attention_mask_sample = batch["attention_mask"][0]
            # labels_sample = batch["labels"][0]

            # unpadded_length = torch.sum(attention_mask_sample).item()
            # unmasked_text = self.tokenizer.decode(input_ids_sample[:unpadded_length], skip_special_tokens=False)
            # print("\n[UNMASKED INPUT]:\n", unmasked_text)

            # padded_text = self.tokenizer.decode(input_ids_sample[unpadded_length:], skip_special_tokens=False)
            # print("\n[PADDED/MASKED PART]:\n", padded_text)

            # # 2 Filter the labels to show only the part where loss is calculated
            # # The labels tensor uses -100 to mask the prompt part.
            # active_labels = labels_sample[labels_sample != -100]
            # answer_text = self.tokenizer.decode(active_labels, skip_special_tokens=False)
            # print("\n[LABELS]:\n", answer_text)
            # print("\n" + "="*50)
            # exit()

            inputs = {k: v.to(self.device) for k, v in batch.items()}
            loss = self._train_step(inputs)
            if loss is not None: progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)
            self.step += 1

            if self.step % eval_every == 0:
                self.evaluate()
                self.save_model(os.path.join(save_dir, f"step_{step}"))
                self.switchable_model.model.train() # Set back to train mode
            
    def _train_step(self, batch):
        """Dispatches to the correct training logic based on style."""
        if self.training_style == 'instantnet':
            return self._instantnet_step(batch)
        elif self.training_style == 'cyclic':
            return self._cyclic_step(batch)
        else:
            raise ValueError(f"Unknown training style: {self.training_style}")

    def evaluate(self):
        if not self.eval_dataloader:
            print("No evaluation dataset provided. Skipping evaluation.")
            return
        
        print(f"\n--- Running Evaluation at Step {self.step}---")
        self.switchable_model.model.eval()
        squad_metric = evaluate.load("squad")

        # Evaluate each precision configuration separately
        for name, bitmap in self.bitmaps.items():
            self.switchable_model.set_config(bitmap)
            generated_texts = []
            for batch in tqdm(self.eval_dataloade, leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    generated_ids = self.switchable_model.model.generate(
                        **inputs,
                        pad_token_id=self.tokenizer.pad_token_id,
                        temperature=0.,
                        do_sample=False,
                        top_p=0.9
                    )
                # Decode only the newly generated tokens
                outputs = self.tokenizer.batch_decode(generated_ids[:, batch["input_ids"].size(1):], skip_special_tokens=True)
                generated_texts.extend(outputs)
                # Format for SQuAD metric

            assert len(generated_texts) == len(self.eval_info["id"]) and len(generated_texts) == len(self.eval_info["answers"])
            predictions = [{"prediction_text": t.strip(), "id": id_} for t, id_ in zip(generated_texts, self.eval_info["id"])]
            references = [{"answers": a, "id": id_} for a, id_ in zip(self.eval_info["answers"], self.eval_info["id"])]
            results = squad_metric.compute(predictions=predictions, references=references)
            tqdm.write(f"[{name.upper()}] F1 = {results['f1']:.2f}, EM = {results['exact_match']:.2f}")
            tqdm.write(f"Sample:\n-Prediction: {predictions[-1]['prediction_text']!r}\n-Reference: {references[-1]['answers']!r}")

    def save_model(self, output_dir: str):
        self.switchable_model.model.save_pretrained(output_dir)
        print(f"Adapters saved to {output_dir}")

    def _instantnet_step(self, batch):
        """Aggregate loss from all precisions and do knowledge distillation"""
        self.optimizer.zero_grad()
        logits_cache = {}
        total_loss = 0.0

        # Get FP16/FP32 teacher logits
        with torch.no_grad():
            with self.switchable_model.model.disable_adapter():
                for module in self.switchable_model.quant_named_modules.values():
                    module.set_bit_width(0)
                
                teacher_16bit_logits = self.switchable_model.model(**batch).logits
                logits_cache[16] = teacher_16bit_logits

        # Loop through student configs from highest to lowest precision
        for bw in self.ordered_bws:
            current_bitmap = self.bitmaps[f"all{bw}"]
            self.switchable_model.set_config(current_bitmap)
            self.switchable_model.check_config(current_bitmap)

            outputs = self.switchable_model.model(**batch)
            student_logits = outputs.logits
            ce_loss = outputs.loss
            logits_cache[bw] = student_logits

            # Calculate KD loss from all higher-precision teachers
            kd_loss = 0.0
            attn_mask = batch["attention_mask"]
            teachers = [t for t in logits_cache.keys() if t > bw]
            for teacher_bw in teachers:
                teacher_logits = logits_cache[teacher_bw]
                kd_loss += self.kd_loss_fn(teacher_logits.detach(), student_logits, mask=attn_mask)
            loss = ce_loss + self.kd_weight*kd_loss
            total_loss += loss.item()
            # scale the loss for gradient accumulation
            (loss / len(self.ordered_bws)).backward()

        self.optimizer.step()
        return torch.tensor(total_loss / len(self.ordered_bws))

    def _cyclic_step(self, batch):
        """Performs a single training step with a single, cycling precision config."""
        current_bw = self.ordered_bws[self.step % len(self.ordered_bws)]
        current_bitmap = self.bitmaps[f"all{current_bw}"]
        
        self.switchable_model.set_config(current_bitmap)
        self.switchable_model.check_config(current_bitmap)
        
        outputs = self.switchable_model(**batch)
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss