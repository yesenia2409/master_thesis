from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from reward_model import inference
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import pandas as pd
from peft import AutoPeftModelForSequenceClassification, TaskType

tqdm.pandas()
MODEL_PATH = "../SFT/merged_model/SFT_for_expert_alignment/"
REWARD_MODEL = "RewardModel/"


def tokenize_and_add_fields(sample, tokenizer, max_len):
    tokenizer.padding_side = "left"
    tokens = tokenizer(tokenizer.eos_token+sample["instruction"], max_length=max_len, padding="max_length", return_tensors="pt")
    sample["input_ids"] = tokens["input_ids"].squeeze().tolist()
    sample["attention_mask"] = tokens["attention_mask"].squeeze().tolist()
    return sample


def build_dataset(dataset_path, tokenizer, max_len):
    train_set = pd.read_pickle(dataset_path)
    ppo_data = Dataset.from_pandas(train_set)
    print(ppo_data)

    ppo_data = ppo_data.map(lambda x: tokenize_and_add_fields(x, tokenizer, max_len))
    
    return ppo_data


def load_model_and_tokenizer():
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], # "l"m_head"
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    print("Done loading Policy Model and Tokenizer!")
    return model, model, tokenizer


def load_reward_model_and_tokenizer():
    reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
        REWARD_MODEL,
        low_cpu_mem_usage=True,
    )

    reward_model.config.pad_token_id = reward_model.config.eos_token_id
    reward_model.config.use_cache = False

    reward_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b", padding_side="left")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

    print("Done loading Reward Model and Tokenizer!")
    return reward_model, reward_tokenizer


def build_pipeline(ppo_config, ppo_trainer, reward_model, reward_tokenizer):
    generation_kwargs = {
    #     "min_length": -1,
    #     "top_k": 0.0,
    #     "top_p": 1.0,
    #     "do_sample": True,
        "max_new_tokens": 50,
    }

    rewards_list = []

    for epoch in range(ppo_config.ppo_epochs):
        for step, batch in enumerate(ppo_trainer.dataloader):
            print(step)
            for idx, elem in enumerate(batch):
                print(f"Batch {idx}: ", elem)
            query_tensors = batch["input_ids"]
            attention_tensors = batch["attention_mask"]
            query_tensors = [torch.Tensor(query_tensor).type(torch.int32) for query_tensor in query_tensors]

            response_tokens = ppo_trainer.model.generate(input_ids=query_tensors, attention_mask=attention_tensors, **generation_kwargs)
            pred = ppo_trainer.tokenizer.batch_decode(response_tokens)[0]
            pred = pred.split("[EOS]")[1].split(ppo_trainer.tokenizer.eos_token)[0].split("[/EOS]")[0].replace("<|endoftext|>", "")
            # print(pred)

            # Generate outputs
            # response_tensors = ppo_trainer.generate(
            #     query_tensors, return_prompt=False, **generation_kwargs
            # )
            batch["response"] = pred
            for idx, elem in enumerate(batch["response"]):
                print(f"Response {idx}: ", elem)
            
            # Compute rewards
            rewards = inference(reward_model, reward_tokenizer, batch["response"])
            rewards_list.append(rewards)
            for idx, elem in enumerate(rewards):
                print(f"Rewards {idx}: ", elem)

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tokens, rewards)
            print("Stats: ", stats)
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    ################
    # Training
    ################
    ppo_config = PPOConfig(
        batch_size=2,
        ppo_epochs=1,
        model_name=MODEL_PATH,
        learning_rate=0.00002,
        remove_unused_columns=False,
        seed=42,
        # gradient_checkpointing=True,
    )

    ppo_trainer = PPOTrainer(
        ppo_config,
        policy_model,
        ref_model,
        policy_tokenizer,
        dataset=dataset,
        data_collator=dataloader
    )

    build_pipeline(dataloader, ppo_config, ppo_trainer, policy_tokenizer, reward_model, reward_tokenizer)
