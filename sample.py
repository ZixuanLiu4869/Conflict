# In this script, we randomly sample 8 responses from the dpo policy and form a new dataset
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, PeftModel
from tqdm import tqdm

import itertools
import random
import numpy as np
import json
from collections import defaultdict

tqdm.pandas()


seed = 0
model_name_or_path = "sft/final_checkpoint"
tokenizer_name = "EleutherAI/pythia-1b"
temperature = 0.7
batch_size = 1
sample_responses = 8
reward_model_name = "reward_model"
new_dataset_path = "conflict.json"

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# Seed everything to reproduce
seed_everything(seed)

device = 0 if torch.cuda.is_available() else "cpu"

# Load the dpo policy
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
#model.to(device)
model.eval()

# Setting for training model tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

# Build the prompt dataset for generating 
def build_prompt_dataset():
    def preprocess_function(examples):
        new_examples = {
            "prompt": [],
            "category":[],
            "prompt_beaver":[],
        }
        for prompt, category_0, category_1 in zip(examples["prompt"], examples["response_0_harm_category"], examples["response_1_harm_category"]):
            query = "Question: " + prompt + "\n\nAnswer: "
            merged_category = {k: category_0[k] or category_1[k] for k in category_0}
            new_examples["prompt"].append(query)
            new_examples["category"].append(merged_category)
            query = "BEGINNING OF CONVERSATION: USER: " + prompt + " ASSISTANT:"
            new_examples["prompt_beaver"].append(query)
       
        return new_examples   
    
    # Load full dataset
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", name="alpaca3-8b", split="train")

    # Define categories of interest
    target_categories = {
        "Discriminatory Behavior",
        "Violence",
        "Endangering National Security",
        "Insulting Behavior",
        "Cybercrime",
        "Physical Harm",
        "Copyright Issues",
        "Human Trafficking",
        "Drugs",
        "Sexual Content"
    }

    # Build index: map category -> list of example indices
    category_to_indices = defaultdict(list)

    for idx, example in enumerate(dataset):
        for cat in target_categories:
            if example["response_0_harm_category"].get(cat, False) or example["response_1_harm_category"].get(cat, False):
                category_to_indices[cat].append(idx)

    # Desired number per category
    per_category = 300  # 300 * 10 = 3000 total
    selected_indices = set()

    # Sample evenly across categories
    for cat in target_categories:
        indices = category_to_indices[cat]
        sampled = random.sample(indices, min(per_category, len(indices)))
        selected_indices.update(sampled)

    # Select unique examples
    selected_indices = list(selected_indices)
    random.shuffle(selected_indices)  # optional
    final_dataset = dataset.select(selected_indices)

    print("Final size:", len(final_dataset))

    # (optional) check how many from each category
    from collections import Counter
    cat_counter = Counter()
    for ex in final_dataset:
        for cat in target_categories:
            if ex["response_0_harm_category"].get(cat, False) or ex["response_1_harm_category"].get(cat, False):
                cat_counter[cat] += 1
    print(cat_counter)

    num_proc = 24
    original_columns = dataset.column_names

    # Create a new prompt dataset with formatted type
    ds = final_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
    
    ds.set_format(type="torch")

    return ds 


# The new dataset generated from the dpo policy
new_examples = {
    "prompt": [],
    "prompt_beaver":[],
    "category": [],
    "response_with_prompt": [],
    "response_prob": [],
    "response_reward": [],
}

prompt_ds = build_prompt_dataset()

# We then define the arguments to pass to the `generate` function. 
generation_kwargs = {
    "top_k": 0,
    "top_p": 1,
    "temperature": temperature,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,  # solve warning for open-end generation
}

with torch.no_grad():
    for i in tqdm(range(0, len(prompt_ds), batch_size)):
        batch = prompt_ds["prompt"][i:i+batch_size]

        tokenized_batch = tokenizer(batch, padding=True, return_tensors="pt")

        input_ids = tokenized_batch["input_ids"].to(device)
        attention_mask = tokenized_batch["attention_mask"].to(device)

        response_tensors = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128, num_return_sequences=sample_responses, **generation_kwargs)

        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        responses = [responses[i:i+sample_responses] for i in range(0, len(responses), sample_responses)]

        new_examples["prompt"].extend(batch)
        new_examples["prompt_beaver"].extend(prompt_ds["prompt_beaver"][i:i+batch_size])
        new_examples["category"].extend(prompt_ds["category"][i:i+batch_size])
        new_examples["response_with_prompt"].extend(responses)


for i, cat_dict in enumerate(new_examples["category"]):
    new_examples["category"][i] = {k: v.item() for k, v in cat_dict.items()}

with open(new_dataset_path, "w") as json_file:
    json.dump(new_examples, json_file)


with torch.no_grad():
    for i in tqdm(range(len(new_examples["prompt"]))):
        responses = new_examples["response_with_prompt"][i]
        # Flatten responses for scoring
        flat_responses = [resp for resp in responses]

        # Tokenize all at once
        enc = tokenizer(
            flat_responses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)


        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Log-probs
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        log_probs_of_labels = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Sum log-probs over all tokens
        seq_logprobs = log_probs_of_labels.sum(dim=1).tolist()

        # Reshape to [batch_size][sample_responses]
        response_logprobs = [
            seq_logprobs[i:i + sample_responses] for i in range(0, len(seq_logprobs), sample_responses)
        ]

        # Append to new_examples
        new_examples["response_prob"].extend(response_logprobs)

with open(new_dataset_path, "w") as json_file:
    json.dump(new_examples, json_file)


# Free memory
del model
torch.cuda.empty_cache()




reward_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
reward_tokenizer.pad_token = tokenizer.eos_token
reward_tokenizer.padding_side = "right"

base_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-1b", num_labels=1)
reward_model = PeftModel.from_pretrained(base_model, "reward_model")
reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
reward_model.eval()
reward_model.cuda()  # Or .to("cuda:0") if multi-GPU


with torch.no_grad():
    for i in tqdm(range(len(new_examples["prompt"]))):
        responses = new_examples["response_with_prompt"][i]
        # Tokenize all at once
        inputs = reward_tokenizer(
            responses,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Move to GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # Forward pass
        output = reward_model(**inputs).logits.squeeze(-1)  # shape: (batch_size,)
        rewards = output.cpu().tolist()  # convert to list of floats

        new_examples["response_reward"].append(rewards)


# Free memory
del reward_model
torch.cuda.empty_cache()


with open(new_dataset_path, "w") as json_file:
    json.dump(new_examples, json_file)
