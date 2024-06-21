import pandas as pd
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from peft import LoraConfig, TaskType, AutoPeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import RewardTrainer

REWARD_MODEL = "weqweasdas/hh_rlhf_rm_open_llama_3b"
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

    inference_test_df = sample_by_type(raw_datasets, 70)
    for idx, row in inference_test_df.iterrows():
        chosen_reward = inference(tokenizer, model, row["chosen"])
        rejected_reward = inference(tokenizer, model, row["rejected"])
        chosen_rewards.append(chosen_reward)
        rejected_rewards.append(rejected_reward)

    inference_test_df['chosen_reward'] = chosen_rewards
    inference_test_df['rejected_reward'] = rejected_rewards
    inference_test_df.to_csv(f'Output_files/third_rm_inference_test_{before}_training_seed5.csv', index=False)


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
    sampled = grouped.apply(lambda x: x.sample(min(len(x), sample_size), random_state=5))
    return sampled.reset_index(drop=True)


def plot_loss(train, eval, save_path):
    colors = ["lightsteelblue", "cornflowerblue"]
    plt.figure()

    epochs_eval = []
    epochs_train = []
    loss = []
    eval_loss = []
    for (loss_val, epoch_val) in train:
        loss.append(loss_val)
        epochs_train.append(epoch_val)

    for (eval_val, epoch_val) in eval:
        eval_loss.append(eval_val)
        epochs_eval.append(epoch_val)

    plt.plot(epochs_train, loss, label='Training Loss', marker='o', color=colors[0])
    plt.plot(epochs_eval, eval_loss, label='Evaluation Loss', marker='o', color=colors[1])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(save_path)


if __name__ == "__main__":

    eval_data = [(2.196, 0.03), (1.969, 0.06), (1.757, 0.09), (1.564, 0.12), (1.392, 0.15), (1.232, 0.18),
                 (1.096, 0.21), (0.971, 0.24), (0.864, 0.27), (0.765, 0.29), (0.675, 0.31), (0.598, 0.35),
                 (0.529, 0.38), (0.472, 0.41), (0.426, 0.44), (0.389, 0.47), (0.356, 0.5), (0.328, 0.53), (0.303, 0.56),
                 (0.283, 0.59), (0.269, 0.62), (0.255, 0.65), (0.2447, 0.68), (0.238, 0.71), (0.233, 0.74), (0.228, 0.77),
                 (0.226, 0.8), (0.226, 0.83), (0.225, 0.86), (0.225, 0.88), (0.225, 0.91), (0.225, 0.94), (0.225, 0.97),
                 ]
    train_loss = [(1.9085, 0.15), (1.0192, 0.29), (0.7196, 0.44), (0.4012, 0.59), (0.2711, 0.74), (0.2481, 0.88), (0.716, 0.97)]
    plot_loss(train_loss, eval_data, "Output_files/loss_plots/rm_loss_second.png")

    ################
    # Model & Tokenizer
    ################
    model, tokenizer = load_model()
    trained_model = AutoPeftModelForSequenceClassification.from_pretrained(
        "RewardModel/",
        low_cpu_mem_usage=True,
    )
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

    # print(raw_datasets["chosen"][2410])
    # inference(tokenizer, model, raw_datasets["chosen"][2410])
    # print(raw_datasets["rejected"][2410])
    # inference(tokenizer, model, raw_datasets["rejected"][2410])
    # print("Done with first inference!")

    # inference_evaluation(model, tokenizer, "before")
    inference_evaluation(trained_model, tokenizer, "after")


    ################
    # Training
    ################
    # peft_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     task_type=TaskType.SEQ_CLS,
    # )

    # training_arguments = TrainingArguments(
    #     output_dir=f"{DIR}Training_Outputs",
    #     num_train_epochs=1,
    #     per_device_train_batch_size=2,
    #     per_device_eval_batch_size=2,
    #     gradient_accumulation_steps=32,
    #     optim="paged_adamw_32bit",
    #     learning_rate=0.0005,
    #     weight_decay=0.01,
    #     lr_scheduler_type="linear",
    #     save_strategy="epoch",
    #    logging_strategy="steps",
    #    logging_steps=5,
    #     evaluation_strategy="steps",
    #     eval_steps=1,
    #     eval_accumulation_steps=5,
    #     seed=42,
    #     bf16=True,
    #     gradient_checkpointing=True,
    #     remove_unused_columns=False
    # )

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
    # print("Done training!")

    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # print("Evaluation mectrics: ", metrics)

    # print(raw_datasets["chosen"][2410])
    # inference(tokenizer, model, raw_datasets["chosen"][2410])
    # print(raw_datasets["rejected"][2410])
    # inference(tokenizer, model, raw_datasets["rejected"][2410])
    # print("Done with second inference!")

    # trainer.save_model(DIR)
    # print("Done saving!")

    # print("Log History: ", trainer.state.log_history)

