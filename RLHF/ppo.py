from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import pandas as pd
from peft import AutoPeftModelForSequenceClassification, TaskType

tqdm.pandas()
MODEL_PATH = "../SFT/merged_model/SFT_for_expert_alignment/"
REWARD_MODEL = "RewardModel/"


def build_dataset(dataset_path, tokenizer, max_len):
    train_set = pd.read_pickle(dataset_path)
    ppo_data = Dataset.from_pandas(train_set)
    print(ppo_data)

    for idx in range(len(ppo_data)):
        instruction = ppo_data[idx]["instruction"]
        tokens = tokenizer(instruction, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        ppo_data[idx]["input_ids"] = tokens.input_ids.squeeze()
        ppo_data[idx]["attention_mask"] = tokens.attention_mask.squeeze()

    return ppo_data


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


def load_model_and_tokenizer():
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "lm_head"],
        # alternativ: lm_head rausnehmen
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(  # alternativ: AutoModelForCausalLMWithValueHead
        pretrained_model_name_or_path=MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # alternativ: "meta-llama/Llama-2-13b-chat-hf"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Done loading Policy Model and Tokenizer!")
    return model, model, tokenizer


def load_reward_model_and_tokenizer():
    # peft_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     task_type=TaskType.SEQ_CLS,
    # )

    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     pretrained_model_name_or_path="weqweasdas/hh_rlhf_rm_open_llama_3b",
    #     num_labels=1,
    #     trust_remote_code=True,
    #     use_safetensors=True,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     peft_config=peft_config
    # )

    reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
        REWARD_MODEL,
        low_cpu_mem_usage=True,
    )

    reward_model.config.pad_token_id = reward_model.config.eos_token_id
    reward_model.config.use_cache = False

    reward_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.padding_side = "right"

    print("Done loading Reward Model and Tokenizer!")
    return reward_model, reward_tokenizer


def build_pipeline(tokenizer, reward_model):
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"

    task, model_name = reward_model.split(":")

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

    sentiment_pipe = pipeline(task, model=model_name, device=device)

    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
        ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == "__main__":

    ################
    # Model & Tokenizer
    ################
    set_seed(42)
    ref_model, policy_model, policy_tokenizer = load_model_and_tokenizer()
    reward_model, reward_tokenizer = load_reward_model_and_tokenizer()

    ################
    # Dataset
    ################
    path = "../SFT/Input_files/train_set_expert.pkl"
    dataset = build_dataset(path, policy_tokenizer, 512)
    print(dataset)
    print(dataset[3]["input_ids"])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=ppo_config.batch_size, shuffle=True)

    ################
    # Training
    ################
    ppo_config = PPOConfig(
        model_name=MODEL_PATH,
        steps=51200,
        learning_rate=1.41e-5,
        remove_unused_columns=False,
    )

    ppo_trainer = PPOTrainer(ppo_config, policy_model, ref_model, policy_tokenizer, dataset=dataset,
                             data_collator=collator)
