import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, TaskType, PeftModelForSequenceClassification, PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import RewardTrainer

REWARD_MODEL = "weqweasdas/hh_rlhf_rm_open_llama_3b" # "vincentmin/llama-2-7b-reward-oasst1" # "weqweasdas/hh_rlhf_rm_open_llama_3b" # "OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt"
DIR = "RewardModel/"


def inference(reward_tokenizer, reward_model, sample):
    input_ids = reward_tokenizer(
        sample,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
    model.eval()
    out_reward = reward_model(**input_ids)

    # reward = reward[:, 1]

    print("Reward Logits: ", out_reward.logits)
    # print("Reward Logits: ", out_reward[0]["score"])
    print("Reward output: ", out_reward)

    return out_reward


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
        pretrained_model_name_or_path=REWARD_MODEL,  # "meta-llama/Llama-2-7b-chat-hf",
        num_labels=1,
        trust_remote_code=True,
        # use_safetensors=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL) #, use_fast=True, model_max_length=512)
    tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
    
    # if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


if __name__ == "__main__":
    ################
    # Model & Tokenizer
    ################
    model, tokenizer = load_model()
    # model = PeftModel.from_pretrained(model, "vincentmin/llama-2-7b-reward-oasst1")
    # model.resize_token_embeddings(len(tokenizer))
    # trained_model = PeftModelForSequenceClassification.from_pretrained(
    #     model,
    #     DIR,
    # )
    print("Done loading model and tokenizer!")

    ################
    # Dataset
    ################
    raw_datasets = pd.read_csv("Input_files/dataset_SFT_reward_model.csv")
    # train_set, test_set = train_test_split(raw_datasets, test_size=0.1, stratify=raw_datasets["type"], random_state=42)

    # preprocessed_train_data = preprocess_dataset(train_set, tokenizer)
    # preprocessed_test_data = preprocess_dataset(test_set, tokenizer)

    # train_set = Dataset.from_dict(preprocessed_train_data)
    # test_set = Dataset.from_dict(preprocessed_test_data)
    
    print("Done preprocessing dataset!")

    ################
    # Inference
    ################

    print(raw_datasets["chosen"][0])
    print(raw_datasets["rejected"][1555])

    inference(tokenizer, model, raw_datasets["chosen"][0])
    inference(tokenizer, model, raw_datasets["rejected"][1555])
    inference(tokenizer, model, "gskhdlazdgtaddifjädf")
    print("Done with inference!")

    ################
    # Training
    ################
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.SEQ_CLS,
    )

    training_arguments = TrainingArguments(
        output_dir=f"{DIR}Training_Outputs",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
        optim="paged_adamw_32bit",
        learning_rate=0.0003,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_safetensors=True,
        seed=42,
        bf16=True,
        remove_unused_columns=False
    )

    # trainer = RewardTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_arguments,
    #     max_length=256,
    #     train_dataset=train_set,
    #     eval_dataset=test_set,
    #     peft_config=peft_config,
    # )

    # trainer.train()
    print("Done training!")

    # trainer.save_model(DIR)
    print("Done saving!")

    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # print("Evaluation mectrics: ", metrics)
