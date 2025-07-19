from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch


model_name = "microsoft/phi-2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


dataset = load_dataset("json", data_files="fine_tune_data.json")

# Tokenize the dataset
def tokenize(batch):
    prompt = f"<s>{batch['instruction']}\n{batch['output']}</s>"
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize)

# LoRA config 
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj"]  
)

model = get_peft_model(model, peft_config)

# Training configuration
training_args = TrainingArguments(
    output_dir="./phi2-finetuned",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)


trainer.train()


model.save_pretrained("phi2-finetuned")
tokenizer.save_pretrained("phi2-finetuned")
