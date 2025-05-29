from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
import os
from tqdm import tqdm

# === Step 0: Filtered dataset already loaded
# Assume `final_dataset` is created like this:
# final_dataset = DatasetDict({ "train": train_dataset, "test": test_dataset })

# === Step 1: Config
model_path = "EleutherAI/pythia-1b"
output_dir = "./sft"
seq_length = 512
batch_size = 4
gradient_accumulation_steps = 2
max_steps = 5000
learning_rate = 1e-5
save_freq = 500
eval_freq = 500
log_freq = 10
seed = 42
use_fp16 = False
use_bf16= False
weight_decay = 0.05

set_seed(seed)
os.makedirs(output_dir, exist_ok=True)

# === Step 2: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


# === Step 4: Estimate char/token ratio
def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    if example["safer_response_id"] == 0:
        text = f"Question: {example['prompt']}\n\nAnswer: {example['response_0']}"
    else:
        text = f"Question: {example['prompt']}\n\nAnswer: {example['response_1']}"
    return text

def create_datasets(tokenizer):
    # === Load your filtered dataset ===
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", name="alpaca3-8b", split="train")

    # Filter harmful categories
    target_categories = {
        "Discriminatory Behavior",
        "Violence",
        "Endangering National Security",
        "Insulting Behavior",
        "Cybercrime",
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
    valid_dataset = filtered

    chars_per_token = chars_token_ratio(train_dataset, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # === Step 5: Constant length datasets
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_dataset,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=seq_length,
        chars_per_token=chars_per_token,
    )

    return train_dataset, valid_dataset


train_dataset, valid_dataset = create_datasets(tokenizer)


# === Step 6: LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

train_dataset.start_iteration = 0

print("Starting main loop")

# === Step 7: Load model
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)

# === Step 8: Training args
training_args = TrainingArguments(
    output_dir=output_dir,
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    max_steps=max_steps,
    eval_steps=eval_freq,
    save_steps=save_freq,
    logging_steps=log_freq,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    fp16=use_fp16,
    bf16=use_bf16,
    weight_decay=weight_decay,
    ddp_find_unused_parameters=False,
    report_to="none",
)

# === Step 9: Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=lora_config,
)

trainer.train()
trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))
