from datasets import load_dataset
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


set_seed(0)

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
eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns)
eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids_j"]) <= 512 and len(x["input_ids_k"]) <= 512)

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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)
trainer.train()
model.save_pretrained("reward_model_output_peft_last_checkpoint")
