import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import gather_object, InitProcessGroupKwargs
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from datasets import Dataset
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
parser.add_argument('--output_dir', type=str, default='./Llama-2-7b_generated_10k')
args = parser.parse_args()

# Specify model names and cache directory
reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
generation_model_name = args.model
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Create DPO, self_RM, and HC_self output directories
output_dpo_dir = output_dir / 'DPO'
output_self_rm_dir = output_dir / 'self_RM'
output_hc_self_dir = output_dir / 'HC_self'
output_dpo_dir.mkdir(parents=True, exist_ok=True)
output_self_rm_dir.mkdir(parents=True, exist_ok=True)
output_hc_self_dir.mkdir(parents=True, exist_ok=True)

# Initialize the Accelerator
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])
device = accelerator.device

def prepare_prompts(prompts, tokenizer, batch_size=4):
    # Batch prompts efficiently using DataLoader-like batching
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                add_special_tokens=False
            ).to(device)
        )
    tokenizer.padding_side = "right"
    return batches_tok

def main():
    # Load the generation model and tokenizer
    generation_tokenizer = AutoTokenizer.from_pretrained(
        generation_model_name,
        use_fast=False,
        trust_remote_code=True
    )
    generation_model = AutoModelForCausalLM.from_pretrained(
        generation_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    generation_tokenizer.pad_token = generation_tokenizer.eos_token
    generation_model.config.pad_token_id = generation_tokenizer.pad_token_id

    # Load the reward model and tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_name
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_name, torch_dtype=torch.float16
    ).to(device)

    # Prepare models for distributed inference
    reward_model, generation_model = accelerator.prepare(reward_model, generation_model)
    reward_model.eval()
    generation_model.eval()

    # Unwrap the generation model for generating text
    unwrapped_generation_model = accelerator.unwrap_model(generation_model)

    # Process each split separately
    for split in ['test', 'train']:
        # Load the original dataset
        original_dataset_path = f'./data/safeRLHF-50k/{split}.jsonl'
        with open(original_dataset_path, 'r') as f:
            original_data = [json.loads(line) for line in f]
        original_data = Dataset.from_list(original_data)

        if split == 'train':
            original_data = original_data.select(range(10000))
        else:
            original_data = original_data.select(range(500))

        prompts_all = [item['chosen'][0]['content'] for item in original_data]

        # Distribute prompts across processes
        with accelerator.split_between_processes(prompts_all) as prompts:
            prompt_batches = prepare_prompts(prompts, generation_tokenizer, batch_size=accelerator.num_processes * 4)

            all_dpo_entries = []
            all_self_rm_entries = []
            all_hc_self_entries = []

            # First generate responses with sampling for DPO and self_RM
            for batch_idx, batch in enumerate(tqdm(prompt_batches, desc=f"Processing {split} split - sampling")):
                batch_input_ids = batch['input_ids']
                batch_attention_mask = batch['attention_mask']
                batch_size = batch_input_ids.size(0)

                # Generate responses with sampling
                with torch.no_grad():
                    output_ids = unwrapped_generation_model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        max_new_tokens=128,
                        num_return_sequences=5,
                        eos_token_id=generation_tokenizer.eos_token_id,
                        pad_token_id=generation_tokenizer.pad_token_id,
                    )

                # Reshape output_ids to (batch_size * num_return_sequences, seq_len)
                output_ids = output_ids.view(batch_size * 5, -1)

                # Prepare inputs for reward model
                prompts_expanded = batch_input_ids.repeat_interleave(5, dim=0)
                prompts_text = generation_tokenizer.batch_decode(prompts_expanded, skip_special_tokens=True)
                responses_text = generation_tokenizer.batch_decode(
                    output_ids[:, batch_input_ids.shape[1]:], skip_special_tokens=True
                )

                # Filter out short responses
                valid_indices = [i for i, resp in enumerate(responses_text) if len(resp.strip()) >= 5]
                if not valid_indices:
                    continue  # Skip if no valid responses

                # Tokenize prompts and responses together
                combined_texts = [
                    (prompts_text[i], responses_text[i]) for i in valid_indices
                ]

                batch_encodings = reward_tokenizer(
                    [pt for pt, _ in combined_texts],
                    [rt for _, rt in combined_texts],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)

                with torch.no_grad():
                    scores = reward_model(**batch_encodings).logits.squeeze(-1).cpu().numpy()

                # Process scores and create entries
                for idx_in_batch, (idx_global, score) in enumerate(zip(valid_indices, scores)):
                    prompt_idx = idx_global // 5
                    response_idx = idx_global % 5
                    prompt_text = prompts_text[idx_global]
                    response_text = responses_text[idx_global]

                    # Collect responses and scores per prompt
                    if idx_in_batch == 0:
                        responses_per_prompt = []
                        scores_per_prompt = []

                    responses_per_prompt.append(response_text)
                    scores_per_prompt.append(score)

                    # If we have collected all 5 responses per prompt
                    if (idx_global + 1) % 5 == 0:
                        if len(responses_per_prompt) >= 2:
                            best_response = responses_per_prompt[np.argmax(scores_per_prompt)]
                            worst_response = responses_per_prompt[np.argmin(scores_per_prompt)]

                            # Get the original item
                            original_item = original_data[batch_idx * batch_size + prompt_idx]

                            # Create DPO entry
                            dpo_entry = {
                                "chosen": original_item['chosen'],
                                "rejected": original_item['rejected']
                            }

                            # Create self_RM entry
                            self_rm_entry = {
                                "chosen": [
                                    {"content": prompt_text, "role": "user"},
                                    {"content": best_response, "role": "assistant"},
                                ],
                                "rejected": [
                                    {"content": prompt_text, "role": "user"},
                                    {"content": worst_response, "role": "assistant"},
                                ],
                            }

                            all_dpo_entries.append(dpo_entry)
                            all_self_rm_entries.append(self_rm_entry)

                            # Write entries to the JSONL files
                            with open(output_dpo_dir / f'{split}.jsonl', 'a') as f:
                                json.dump(dpo_entry, f)
                                f.write('\n')
                            with open(output_self_rm_dir / f'{split}.jsonl', 'a') as f:
                                json.dump(self_rm_entry, f)
                                f.write('\n')
                            
                        # Reset for next prompt
                        responses_per_prompt = []
                        scores_per_prompt = []

            # Now generate responses without sampling for HC_self
            for batch_idx, batch in enumerate(tqdm(prompt_batches, desc=f"Processing {split} split - greedy")):
                batch_input_ids = batch['input_ids']
                batch_attention_mask = batch['attention_mask']
                batch_size = batch_input_ids.size(0)

                # Generate responses without sampling (greedy)
                with torch.no_grad():
                    output_ids = unwrapped_generation_model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        do_sample=False,  # Greedy decoding
                        max_new_tokens=128,
                        num_return_sequences=1,  # Only need one sequence for greedy
                        eos_token_id=generation_tokenizer.eos_token_id,
                        pad_token_id=generation_tokenizer.pad_token_id,
                    )

                # Process greedy outputs for HC_self entries
                prompts_text = generation_tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
                responses_text = generation_tokenizer.batch_decode(
                    output_ids[:, batch_input_ids.shape[1]:], skip_special_tokens=True
                )

                # Create HC_self entries with greedy responses
                for prompt_idx in range(batch_size):
                    original_item = original_data[batch_idx * batch_size + prompt_idx]
                    hc_self_entry = {
                        "chosen": original_item['chosen'],
                        "rejected": [
                            {"content": prompts_text[prompt_idx], "role": "user"},
                            {"content": responses_text[prompt_idx], "role": "assistant"},
                        ],
                    }
                    all_hc_self_entries.append(hc_self_entry)

                    # Write HC_self entry to file
                    with open(output_hc_self_dir / f'{split}.jsonl', 'a') as f:
                        json.dump(hc_self_entry, f)
                        f.write('\n')

if __name__ == "__main__":
    main()
