import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
OUTPUT_DIR = "Model"
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


if __name__ == "__main__":
    full_dataset_path = "../Prompting/Adjusting_Dataset/Output_files/geosignal"
    load_data(full_dataset_path)

    train_e = pd.read_pickle("Input_files/train_set_expert.pkl")
    train_h = pd.read_pickle("Input_files/train_set_human.pkl")
    val_e = pd.read_pickle("Input_files/validation_set_expert.pkl")
    val_h = pd.read_pickle("Input_files/validation_set_human.pkl")

    # test_e = pd.read_pickle("Input_files/test_set_expert.pkl")
    # test_h = pd.read_pickle("Input_files/test_set_human.pkl")

    model, tokenizer = create_model_and_tokenizer()
    model.config.use_cache = False
    model.config.quantization_config.to_dict()

    # for name, param in model.named_parameters():
    #     print(f"{name}   Modelsize: {param.numel() / 1000 ** 2:.1f}M parameters")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        logging_steps=20,
        learning_rate=1e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        warmup_ratio=0.05,
        save_strategy="epoch",
        group_by_length=True,
        output_dir=f"{OUTPUT_DIR}/Checkpoints/",
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        train_dataset=train_h[:2],
        eval_dataset=val_h[:2],
        peft_config=peft_config,
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/Model/")
