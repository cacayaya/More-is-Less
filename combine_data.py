import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def extract_user_prompt(conversation):
    """Extract the user's prompt from a conversation."""
    for message in conversation:
        if message['role'] == 'user':
            return message['content']
    return ''

def extract_assistant_response(conversation):
    """Extract the assistant's response from a conversation."""
    for message in conversation:
        if message['role'] == 'assistant':
            return message['content']
    return ''

def score_responses(prompt, response1, response2, reward_model, reward_tokenizer, device):
    """Score two responses using a reward model."""
    inputs1 = reward_tokenizer(prompt, response1, return_tensors="pt", truncation=True, max_length=512).to(device)
    inputs2 = reward_tokenizer(prompt, response2, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        score1 = reward_model(**inputs1).logits.item()
        score2 = reward_model(**inputs2).logits.item()
    
    return score1, score2

def process_model_pair(rejected_data, chosen_data, output_path, use_reward_model=False, reward_model=None, reward_tokenizer=None, device=None):
    """Process a pair of models' data and create new dataset."""
    reformed_data = []
    
    for item1 in tqdm(rejected_data, desc="Processing data pairs"):
        # Access the prompt from the first model's data
        prompt1 = item1['rejected'][0]['content']
        
        # Search for the matching prompt in the second model's data
        matching_item2 = next((item2 for item2 in chosen_data if item2['prompt'] == prompt1), None)
        
        if not matching_item2:
            print(f"No matching prompt found for: {prompt1}")
            continue
        
        response1 = item1['rejected'][1]['content']
        response2 = matching_item2['response']
        
        # If using reward model, verify which response is actually better
        if use_reward_model and reward_model is not None:
            score1, score2 = score_responses(prompt1, response1, response2, reward_model, reward_tokenizer, device)
            
            # Use the response with the higher score as chosen
            if score1 > score2:
                # Swap if the first model's response is actually better
                chosen = item1['rejected']
                rejected = [{"content": prompt1, "role": "user"}, {"content": response2, "role": "assistant"}]
                print(f"Swapped based on reward model: {score1:.3f} > {score2:.3f}")
            else:
                chosen = [{"content": prompt1, "role": "user"}, {"content": response2, "role": "assistant"}]
                rejected = item1['rejected']
        else:
            # Without reward model, use larger model's response as chosen and base model's response as rejected
            chosen = [{"content": prompt1, "role": "user"}, {"content": response2, "role": "assistant"}]
            rejected = item1['rejected']

        reformed_data.append({
            "prompt": prompt1,
            "chosen": chosen,
            "rejected": rejected
        })

    # Save the reformed data
    save_jsonl(reformed_data, output_path)
    print(f"Saved {len(reformed_data)} pairs to {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Mix data from two models with optional reward model scoring')
    parser.add_argument('--rejected_data', type=str, required=True, help='Path to the rejected data directory')
    parser.add_argument('--chosen_data', type=str, required=True, help='Path to the chosen data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for mixed data')
    parser.add_argument('--use_reward_model', action='store_true', help='Whether to use a reward model to verify preferences')
    parser.add_argument('--reward_model', type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2", 
                        help='Reward model to use for scoring responses')
    args = parser.parse_args()
    
    # Convert paths to Path objects
    rejected_path = Path(args.rejected_data)
    chosen_path = Path(args.chosen_data)
    output_path = Path(args.output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load reward model if specified
    reward_model = None
    reward_tokenizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.use_reward_model:
        print(f"Loading reward model: {args.reward_model}")
        reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model, torch_dtype=torch.float16
        ).to(device)
        reward_model.eval()
    
    # Process each split
    for split in ['train', 'test']:
        try:
            rejected_data = load_jsonl(rejected_path / f'{split}.jsonl')
            chosen_data = load_jsonl(chosen_path / f'{split}.jsonl')
            
            output_file = output_path / f'{split}.jsonl'
            
            print(f"Processing {split} split")
            print(f"  - Rejected data: {rejected_path / f'{split}.jsonl'}")
            print(f"  - Chosen data: {chosen_path / f'{split}.jsonl'}")
            print(f"  - Output: {output_file}")
            print(f"  - Using reward model: {args.use_reward_model}")
            
            process_model_pair(
                rejected_data,
                chosen_data,
                output_file,
                use_reward_model=args.use_reward_model,
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                device=device
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not process {split} split - {e}")

if __name__ == "__main__":
    main()