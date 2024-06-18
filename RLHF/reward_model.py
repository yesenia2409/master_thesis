import pandas as pd
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from peft import LoraConfig, TaskType, AutoPeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import RewardTrainer

REWARD_MODEL = "weqweasdas/hh_rlhf_rm_open_llama_3b" # "vincentmin/llama-2-7b-reward-oasst1",  "OpenLLMAI/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt"
DIR = "RewardModel/"


def inference(reward_tokenizer, reward_model, sample):
    with torch.no_grad():
        input_ids = reward_tokenizer(
            sample,
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
        reward_model.eval()
        out_reward = reward_model(**input_ids)

        print("Reward Logits: ", out_reward.logits[0])
        value = out_reward.logits[0].item()
        return value


def inference_evaluation(model, tokenizer, before):
    chosen_rewards = []
    rejected_rewards = []

    inference_test_df = sample_by_type(raw_datasets, 5)
    for idx, row in inference_test_df.iterrows():
        chosen_reward = inference(tokenizer, model, row["chosen"])
        rejected_reward = inference(tokenizer, model, row["rejected"])
        chosen_rewards.append(chosen_reward)
        rejected_rewards.append(rejected_reward)

    inference_test_df['chosen_reward'] = chosen_rewards
    inference_test_df['rejected_reward'] = rejected_rewards
    inference_test_df.to_csv(f'Output_files/rm_inference_test_{before}_training.csv', index=False)


def preprocess_dataset(examples, tokenizer):
    with torch.no_grad():
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
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=REWARD_MODEL,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    return model, tokenizer


def sample_by_type(df, sample_size=5):
    grouped = df.groupby('type')
    sampled = grouped.apply(lambda x: x.sample(min(len(x), sample_size), random_state=42))
    return sampled.reset_index(drop=True)


def plot_loss(train, eval, save_path):
    colors = ["lightsteelblue", "cornflowerblue"]
    plt.figure()

    epochs = []
    loss = []
    eval_loss = []
    for (loss_val, epoch_val) in train:
        loss.append(loss_val)
        epochs.append(epoch_val)

    for (eval_val, _) in eval:
        eval_loss.append(eval_val)

    plt.plot(epochs, loss, label='Training Loss', marker='o', color=colors[0])
    plt.plot(epochs, eval_loss, label='Evaluation Loss', marker='o', color=colors[1])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(save_path)

if __name__ == "__main__":
    ################
    # Model & Tokenizer
    ################
    model, tokenizer = load_model()
    # trained_model = AutoPeftModelForSequenceClassification.from_pretrained(
    #     "RewardModel/",
    #     low_cpu_mem_usage=True,
    # )
    print("Done loading model and tokenizer!")

    ################
    # Dataset
    ################
    raw_datasets = pd.read_csv("Input_files/dataset_SFT_reward_model.csv")
    train_set, test_set = train_test_split(raw_datasets, test_size=0.1, stratify=raw_datasets["type"], random_state=42)

    preprocessed_train_data = preprocess_dataset(train_set, tokenizer)
    preprocessed_test_data = preprocess_dataset(test_set, tokenizer)

    train_set = Dataset.from_dict(preprocessed_train_data)
    test_set = Dataset.from_dict(preprocessed_test_data)
    
    print("Done preprocessing dataset!")

    ################
    # Inference
    ################

    print(raw_datasets["chosen"][2410])
    inference(tokenizer, model, raw_datasets["chosen"][2410])
    print(raw_datasets["rejected"][2410])
    inference(tokenizer, model, raw_datasets["rejected"][2410])
    print("Done with first inference!")

    # inference_evaluation(base_model, base_tokenizer, "before")
    # inference_evaluation(trained_model, base_tokenizer, "after")


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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=32,
        optim="paged_adamw_32bit",
        learning_rate=0.000005,
        weight_decay=0.01,
        lr_scheduler_type="linear_with_decay",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=1,
        eval_accumulation_steps=5,
        seed=42,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_length=256,
        train_dataset=train_set,
        eval_dataset=test_set,
        peft_config=peft_config,
    )

    trainer.train()
    print("Done training!")

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print("Evaluation mectrics: ", metrics)

    print(raw_datasets["chosen"][2410])
    inference(tokenizer, model, raw_datasets["chosen"][2410])
    print(raw_datasets["rejected"][2410])
    inference(tokenizer, model, raw_datasets["rejected"][2410])
    print("Done with second inference!")

    trainer.save_model(DIR)
    print("Done saving!")

    print("Log History: ", trainer.state.log_history)

