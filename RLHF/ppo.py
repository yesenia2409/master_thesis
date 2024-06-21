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
MODEL_PATH = "merged_model/SFT_for_expert_alignment/"
REWARD_MODEL = "RewardModel/"


def build_dataset(dataset_path, tokenizer, max_len):
    train_set = pd.read_pickle(dataset_path)
    ppo_data = Dataset.from_pandas(train_set)
    print(ppo_data)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["instruction"], return_tensors="pt")
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ppo_data = ppo_data.map(tokenize, batched=False)
    ppo_data.set_format(type="torch")
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

    model = AutoModelForCausalLM.from_pretrained(  # alternativ: AutoModelForCausalLMWithValueHead
        pretrained_model_name_or_path=MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        device_map="auto",
        peft_config=peft_config
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # alternativ: "meta-llama/Llama-2-13b-chat-hf"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    dataset = build_dataset(path, policy_tokenizer)

    ################
    # Training
    ################
    # ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
