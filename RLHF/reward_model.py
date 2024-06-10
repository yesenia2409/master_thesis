import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import RewardTrainer

REWARD_MODEL = "OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt"
DIR = "RewardModel/"

# HOW TO RM INFERENCE ?? LAB VON POLINA??


def preprocess_dataset(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, padding="max_length", truncation=True)
        tokenized_rejected = tokenizer(rejected, padding="max_length", truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=REWARD_MODEL,
        num_labels=1,
        trust_remote_code=True,
        use_safetensors=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


if __name__ == "__main__":
    ################
    # Model & Tokenizer
    ################
    model, tokenizer = load_model()

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("Anthropic/hh-rlhf") # eigener datensatz mit chosen, rejected and type of correction

    raw_datasets = raw_datasets.map(
        preprocess_dataset,
        batched=True,
        num_proc=4,
    )

    train_dataset = raw_datasets["train"] # replace with stratified sampling based on type of correction
    eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.SEQ_CLS,
    )

    training_arguments = TrainingArguments( # look for pytorch torchtune for llama 7 b model and adjust parameter
        output_dir=f"{DIR}Training_Outputs",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=0.00001,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_safetensors=True,
        seed=42,
        bf16=True
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(DIR)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)