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
from datasets import Dataset
import matplotlib.pyplot as plt

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


def plot_loss(train_loss, save_path):
    plt.plot(train_loss, label='Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.legend()

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

    peft_config = LoraConfig( # Settings chosen as here: https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/configs/peft.py
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments( # Settings chosen as here: https://github.com/daniel-furman/sft-demos/blob/main/src/peft/llama-2/peft_Llama_2_13B_Instruct_v0_2.ipynb
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        auto_find_batch_size=True,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        learning_rate=4e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_strategy="steps",
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=0.2,
        save_safetensors=True,
        seed=42,
        fp16=True,
        weight_decay=0.1,  # Llama paper
    )

    train = Dataset.from_pandas(train_h[:200])
    val = Dataset.from_pandas(val_h[:200])

    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        train_dataset=train,
        eval_dataset=val,
        peft_config=peft_config,
        max_seq_length=512,   # 4096 in Llama paper
        tokenizer=tokenizer,
        args=training_arguments,
    )

    train_result = trainer.train()

    train_losses = train_result.metrics["train_loss"]
    plot_loss(train_losses, f'Output_files/training_loss_plot_{training_arguments.learning_rate}_{peft_config.target_modules}.png')

    # Saving
    trainer.save_model()
    trained_model = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        low_cpu_mem_usage=True,
    )

    merged_model = trained_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True)
    tokenizer.save_pretrained("merged_model")
