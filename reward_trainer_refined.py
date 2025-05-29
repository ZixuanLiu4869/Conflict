from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers import PreTrainedTokenizerBase, set_seed
from transformers.utils import PaddingStrategy
import evaluate
import numpy as np
from scipy.stats import kendalltau
import json
from safe_rlhf.models import AutoModelForScore
from copy import deepcopy
from tqdm import tqdm

tqdm.pandas()


set_seed(0)


new_dataset_path = "conflict.json"
# Threshold
threshold = 1.0
sample_responses = 8
device = 0 if torch.cuda.is_available() else "cpu"


# === Step 1: Load JSON ===
with open(new_dataset_path, "r") as f:
    new_examples = json.load(f)

# === Step 2: Compute nRPCS and Kendall Tau ===
nRPCS_per_prompt = []
kendall_tau_list = []
conflict_examples = []

for i, (logprobs, rewards) in enumerate(zip(new_examples["response_prob"], new_examples["response_reward"])):
    logprobs = np.array(logprobs)
    rewards = np.array(rewards)

    # Skip degenerate entries
    if len(logprobs) < 2 or np.std(logprobs) < 1e-6 or np.std(rewards) < 1e-6:
        nRPCS_per_prompt.append([0.0] * len(logprobs))
        kendall_tau_list.append(0.0)
        continue

    # Normalize
    logprobs_norm = (logprobs - logprobs.mean()) / (logprobs.std() + 1e-8)
    rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Pointwise conflict score
    nRPCS = rewards_norm - logprobs_norm
    nRPCS_list = nRPCS.tolist()
    nRPCS_per_prompt.append(nRPCS_list)

    # Ranking disagreement
    tau, _ = kendalltau(logprobs, rewards)
    kendall_tau_list.append(tau if not np.isnan(tau) else 0.0)


    # If any response has high conflict, store the full set
    if np.mean(np.abs(nRPCS_list)) > threshold:
        conflict_examples.append({
            "prompt": new_examples["prompt"][i],
            "prompt_beaver": new_examples["prompt_beaver"][i],
            "response_with_prompt": new_examples["response_with_prompt"][i],
            "response_prob": new_examples["response_prob"][i],
            "response_reward": new_examples["response_reward"][i],
            "nRPCS": nRPCS_list,
            "category": new_examples["category"][i],
        })

print(len(conflict_examples))
# Save conflicts
with open("high_conflict_examples.json", "w") as f_out:
    json.dump(conflict_examples, f_out, indent=2)

# === Step 3: Aggregate scores ===
flat_nRPCS = [abs(val) for sublist in nRPCS_per_prompt for val in sublist]
mean_nRPCS = np.mean(flat_nRPCS)
mean_kendall_tau = np.mean(kendall_tau_list)

print(f"✅ Mean abs nRPCS: {mean_nRPCS:.4f}")
print(f"✅ Mean Kendall-Tau: {mean_kendall_tau:.4f}")


gold_model =  AutoModelForScore.from_pretrained('PKU-Alignment/beaver-7b-unified-cost', torch_dtype=torch.bfloat16, device_map='auto')
gold_tokenizer = AutoTokenizer.from_pretrained('PKU-Alignment/beaver-7b-unified-cost')


with torch.no_grad():
    for i in tqdm(range(len(conflict_examples))):
        responses = [conflict_examples[i]["response_with_prompt"][j][len(conflict_examples[i]["prompt"]):].lstrip() for j in range(sample_responses)]
        conflict_examples[i]["response"] = responses
        responses = [conflict_examples[i]["prompt_beaver"]+responses[j] for j in range(sample_responses)]
        tokenized_responses = gold_tokenizer(responses, padding=True, return_tensors="pt") 
        
        input_ids = tokenized_responses["input_ids"].to(device)
        attention_mask = tokenized_responses["attention_mask"].to(device)

        reward = gold_model(input_ids, attention_mask).end_scores.squeeze(dim=-1).cpu().tolist()
        conflict_examples[i]["response_reward"] = reward


# Free memory
del gold_model
torch.cuda.empty_cache()

def convert_conflict_to_pairs(conflict_examples, tokenizer, max_len=512):
    new_pairs = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
        "safe_sign_j_k": [],  
    }

    for ex in conflict_examples:
        responses = ex["response_with_prompt"]
        rewards = ex["response_reward"]

        # Pairwise preference generation
        for i in range(len(responses)):
            for j in range(len(responses)):
                if i == j:
                    continue
                reward_i = rewards[i]
                reward_j = rewards[j]
                if reward_i < reward_j:
                    # i is less preferred → j is more preferred
                    text_j = responses[i]
                    text_k = responses[j]

                    # Tokenize
                    tokenized_j = tokenizer(text_j, truncation=True)
                    tokenized_k = tokenizer(text_k, truncation=True)

                    if len(tokenized_j["input_ids"]) <= max_len and len(tokenized_k["input_ids"]) <= max_len:
                        new_pairs["input_ids_j"].append(tokenized_j["input_ids"])
                        new_pairs["attention_mask_j"].append(tokenized_j["attention_mask"])
                        new_pairs["input_ids_k"].append(tokenized_k["input_ids"])
                        new_pairs["attention_mask_k"].append(tokenized_k["attention_mask"])
                        safe_sign_j = 1 if reward_i > 0 else -1
                        safe_sign_k = 1 if reward_j > 0 else -1
                        new_pairs["safe_sign_j_k"].append([safe_sign_j, safe_sign_k])
    return new_pairs


# === Load your filtered dataset ===
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", name="alpaca3-8b", split="train")

# Filter harmful categories
target_categories = {
    "Discriminatory Behavior",
    "Violence",
    "Physical Harm",
    "Copyright Issues",
    "Human Trafficking",
}
def has_target_category(harm_dict):
    return any(harm_dict.get(cat, False) for cat in target_categories)

filtered = dataset.filter(
    lambda example: has_target_category(example["response_0_harm_category"]) or 
                    has_target_category(example["response_1_harm_category"])
)
filtered = filtered.shuffle(seed=42)
print(len(filtered))
filtered = filtered.select(range(min(10000, len(filtered))))

train_dataset = filtered
eval_dataset = train_dataset

# Tokenizer
model_name = "EleutherAI/pythia-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

num_proc = 24  # Can adjust to be higher if have more processors.
original_columns = train_dataset.column_names


# Preprocessing
def preprocess_function(examples):
    new_examples = {
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
        "safe_sign_j_k": []
    }

    for question, response_0, response_1, response_0_safe, response_1_safe, safer_response_id in zip(examples["prompt"], examples["response_0"] , examples["response_1"], examples["is_response_0_safe"], examples["is_response_1_safe"], examples["safer_response_id"]):
        if safer_response_id == 1:   # Safer response has lower cost, j refers to higher cost
            tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_0, truncation=True)
            tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_1, truncation=True)
            safe_sign_j = -1 if response_0_safe == True else 1     # return -1 for safe text and 1 for unsafe text
            safe_sign_k = -1 if response_1_safe == True else 1
        else:
            tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_1, truncation=True)
            tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_0, truncation=True)
            safe_sign_j = -1 if response_1_safe == True else 1     # return -1 for safe text and 1 for unsafe text
            safe_sign_k = -1 if response_0_safe == True else 1

        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
        new_examples["safe_sign_j_k"].append([safe_sign_j, safe_sign_k])


    return new_examples

train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
train_dataset = train_dataset.filter(lambda x: len(x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512)

# Convert conflict examples into preference-pair format
conflict_pairs = convert_conflict_to_pairs(conflict_examples, tokenizer)
merged_dict = {}
for key in train_dataset.features.keys():
    merged_dict[key] = train_dataset[key] + conflict_pairs.get(key, [])
merged_dataset_train = Dataset.from_dict(merged_dict)

eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512)
merged_dict = {}
for key in eval_dataset.features.keys():
    merged_dict[key] = train_dataset[key] + conflict_pairs.get(key, [])
merged_dataset_eval = Dataset.from_dict(merged_dict)


# Model + LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id

# Collator
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_j = [{"input_ids": f["input_ids_j"], "attention_mask": f["attention_mask_j"]} for f in features]
        features_k = [{"input_ids": f["input_ids_k"], "attention_mask": f["attention_mask_k"]} for f in features]
        batch_j = self.tokenizer.pad(features_j, padding=self.padding, return_tensors=self.return_tensors)
        batch_k = self.tokenizer.pad(features_k, padding=self.padding, return_tensors=self.return_tensors)
        return {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "labels": torch.tensor([f["safe_sign_j_k"] for f in features]),
            "return_loss": True,
        }

# Metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    original_predictions, sign_labels = eval_pred
    # Here, predictions is costs_j and costs_k.
    # We want to see how much of the time costs_j > costs_k.
    predictions = np.argmax(original_predictions, axis=0)
    labels = np.zeros(predictions.shape)
    
    accuracy_dict = accuracy.compute(predictions=predictions, references=labels)

    # safe samples are supposed to have negative costs
    # unsafe samples are supposed to have positive costs
    predicted_sign = np.sign(np.concatenate(original_predictions, axis=0)).astype(np.int32)
    labels_sign = np.vstack((sign_labels[:,0].reshape(-1,1),sign_labels[:,1].reshape(-1,1)))
    sign_accuracy = accuracy.compute(predictions=predicted_sign, references=labels_sign)
    
    accuracy_dict["sign accuracy"] = sign_accuracy["accuracy"]

    return accuracy_dict

# Loss function
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        costs_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        costs_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        safe_sign_j = inputs["labels"][:,0].unsqueeze(dim=1)
        safe_sign_k = inputs["labels"][:,1].unsqueeze(dim=1)
        loss = -nn.functional.logsigmoid(costs_j - costs_k).mean() - nn.functional.logsigmoid(costs_j*safe_sign_j).mean() - nn.functional.logsigmoid(costs_k*safe_sign_k).mean() + 0* (torch.square(costs_j).mean() + torch.square(costs_k).mean())
        if return_outputs:
            return loss, {"rewards_j": costs_j, "rewards_k": costs_k}
        return loss

# Training args
training_args = TrainingArguments(
    output_dir="reward_model_output",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    num_train_epochs=2,
    weight_decay=0.1,
    logging_steps=10,
    save_steps=500,
    #eval_steps=500,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="none",
    label_names=["labels"],   # We need the safe sign to evaluate the metric
    optim="adamw_hf",
    lr_scheduler_type="cosine",
    bf16=True,
)

# Train
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=merged_dataset_train,
    eval_dataset=merged_dataset_eval,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
model.save_pretrained("reward_model_output_peft_last_checkpoint")

