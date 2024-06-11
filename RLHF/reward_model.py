import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import RewardTrainer

REWARD_MODEL = "OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt"
DIR = "RewardModel/"


def inference(reward_tokenizer, reward_model, sample):
    input_ids = reward_tokenizer(
        sample,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )

    out_reward = reward_model(**input_ids)

    reward = torch.softmax(out_reward.logits, dim=1)
    reward = reward[:, 1]

    print("Reward: ", reward)
    return reward


def preprocess_dataset(examples, tokenizer):
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
    raw_datasets = pd.read_csv("Input_files/dataset_SFT_reward_model.csv")
    train_set, test_set = train_test_split(raw_datasets, test_size=0.1, stratify=raw_datasets["type"], random_state=42)

    preprocessed_train_data = preprocess_dataset(train_set, tokenizer)
    preprocessed_test_data = preprocess_dataset(test_set, tokenizer)

    train_set = Dataset.from_dict(preprocessed_train_data)
    test_set = Dataset.from_dict(preprocessed_test_data)

    ################
    # Inference
    ################

    print(train_set["chosen"][0])
    inference(tokenizer, model, train_set["chosen"][0])


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
        train_dataset=train_set,
        eval_dataset=test_set,
        peft_config=peft_config,
    )

    # trainer.train()

    # trainer.save_model(DIR)

    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # print(metrics)