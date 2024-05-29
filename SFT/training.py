import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from decimal import Decimal
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import Dataset
import matplotlib.pyplot as plt
import argparse

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
OUTPUT_DIR = "Model/"
ACCESS_TOKEN = "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"


# Dataset preparation
def load_data(dataset_path):
    df = pd.read_pickle(f"{dataset_path}.pkl")
    create_text_column(df)

    df_expert = df[df['type'].isin(["geo", "geoqa", "self"])]
    df_human = df[df['type'].isin(["dolly", "alpaca-gpt4", "arc", "NI"])]

    create_data_split(df_human, human=True)
    create_data_split(df_expert, human=False)


def create_text_column(df):
    df['text'] = df.apply(lambda row: row['instruction'] + " " + row['output'] + " </s>", axis=1)


def create_data_split(dataset, human=True):
    # Stratified sampling to keep type balance
    train_set, test_set = train_test_split(dataset, test_size=0.1, stratify=dataset["type"], random_state=42)
    train_set, validation_set = train_test_split(train_set, test_size=0.1, stratify=train_set["type"], random_state=42)

    if human:
        save_data_split(test_set, "test_set_human")
        save_data_split(train_set, "train_set_human")
        save_data_split(validation_set, "validation_set_human")
    else:
        save_data_split(test_set, "test_set_expert")
        save_data_split(train_set, "train_set_expert")
        save_data_split(validation_set, "validation_set_expert")

    print(len(train_set), len(test_set), len(validation_set), len(dataset))


def save_data_split(data_set, name):
    with open(f'Input_files/{name}.pkl', 'wb') as f:
        pickle.dump(data_set, f)


# Loading the model and tokenizer
def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        token=ACCESS_TOKEN,
    )

    model.config.use_cache = False
    model.config.quantization_config.to_dict()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def print_layers(model):
    for name, param in model.named_parameters():
        print(f"{name}   Modelsize: {param.numel() / 1000 ** 2:.1f}M parameters")


def plot_loss(log_history, save_path):
    colors = ["lightsteelblue", "cornflowerblue"]
    plt.figure()

    steps_eval = []
    steps_train = []
    loss = []
    eval_loss = []
    for entry in log_history:
        if 'step' in entry:
            if 'loss' in entry:
                loss.append(entry['loss'])
                steps_train.append(entry['step'])
            if 'eval_loss' in entry:
                steps_eval.append(entry['step'])
                eval_loss.append(entry['eval_loss'])

    if loss: plt.plot(steps_train, loss, label='Training Loss', marker='o', color=colors[0])
    plt.plot(steps_eval, eval_loss, label='Evaluation Loss', marker='o', color=colors[1])

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig(save_path)


if __name__ == "__main__":
    # full_dataset_path = "../Prompting/Adjusting_Dataset/Output_files/geosignal"
    # load_data(full_dataset_path)

    train_e = pd.read_pickle("Input_files/train_set_expert.pkl")
    train_h = pd.read_pickle("Input_files/train_set_human.pkl")
    val_e = pd.read_pickle("Input_files/validation_set_expert.pkl")
    val_h = pd.read_pickle("Input_files/validation_set_human.pkl")

    # test_e = pd.read_pickle("Input_files/test_set_expert.pkl")
    # test_h = pd.read_pickle("Input_files/test_set_human.pkl")

    model, tokenizer = create_model_and_tokenizer()

    # print_layers(model)

    # Args from sh script
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()
    lr_scientific = "{:.2E}".format(Decimal(args.lr)).replace(".", "_")

    peft_config = LoraConfig(
        # Settings chosen as here: https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/13B_lora.yaml
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        # Settings chosen as here: https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/13B_lora.yaml
        output_dir=f"{OUTPUT_DIR}SFT_for_human_alignment",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        warmup_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=0.03,
        save_safetensors=True,
        seed=42,
        bf16=True,
        weight_decay=0.01,
    )

    train = Dataset.from_pandas(train_h)
    val = Dataset.from_pandas(val_h)

    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        train_dataset=train,
        eval_dataset=val,
        peft_config=peft_config,
        max_seq_length=512,  # 4096 in Llama paper
        tokenizer=tokenizer,
        args=training_arguments,
    )

    train_result = trainer.train()
    print("trainer.state.log_history: ", trainer.state.log_history)
    print("train_results: ", train_result)
    print("train loss:", train_result.metrics["train_loss"])

    with open(
            "Output_files/slurm_files/alignment_with_humans/trainer_log_history_SFT_for_human_alignment_1epoch_1_00E-2Lr_2batch.txt", "a") as text_file:
        text_file.write(str(trainer.state.log_history))
    plot_loss(trainer.state.log_history,
              'Output_files/plots_failed_SFT_tries/loss_SFT_for_human_alignment_1epoch_1_00E-2Lr_2batch.png')

    # Saving
    trainer.save_model()
    trained_model = AutoPeftModelForCausalLM.from_pretrained(
        f"{OUTPUT_DIR}SFT_for_human_alignment",
        low_cpu_mem_usage=True,
    )

    merged_model = trained_model.merge_and_unload()
    merged_model.save_pretrained(f"merged_model/SFT_for_human_alignment", safe_serialization=True)
    tokenizer.save_pretrained(f"merged_model/SFT_for_human_alignment")
