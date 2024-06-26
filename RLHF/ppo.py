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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_add_fields(sample, tokenizer, max_len):
    tokens = tokenizer.encode(tokenizer.eos_token+sample["instruction"], max_length=max_len, padding="max_length")
    sample["input_ids"] = torch.tensor(tokens).squeeze().to(DEVICE)
    return sample


def build_dataset(dataset_path, tokenizer, max_len):
    train_set = pd.read_pickle(dataset_path)
    ppo_data = Dataset.from_pandas(train_set)
    ppo_data = ppo_data.remove_columns(['type', 'category', 'text', '__index_level_0__'])
    ppo_data = ppo_data.map(lambda x: tokenize_and_add_fields(x, tokenizer, max_len), batched=False)
    ppo_data.set_format("pytorch")
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
    model = model.to(DEVICE)
    model.bfloat16()
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    print("Done loading Policy Model and Tokenizer!")
    return model, model, tokenizer


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def load_reward_model_and_tokenizer():
    reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
        REWARD_MODEL,
        low_cpu_mem_usage=True,
    )
    reward_model = reward_model.to(DEVICE)

    reward_model.config.pad_token_id = reward_model.config.eos_token_id
    reward_model.config.use_cache = False

    reward_tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b", padding_side="left")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

    print("Done loading Reward Model and Tokenizer!")
    return reward_model, reward_tokenizer


def build_pipeline(ppo_config, ppo_trainer, ppo_tokenizer, reward_model, reward_tokenizer, dataloader):
    generation_kwargs = {
        # "top_p": 0.5,
        "temperature": 1,
        # "do_sample": True,
        "max_new_tokens": 200,
        "pad_token_id": ppo_tokenizer.eos_token_id,
    }

    for epoch in tqdm(range(ppo_config.ppo_epochs), "epoch: "):
        # for step, batch in enumerate(dataloader):
        for batch in tqdm(ppo_trainer.dataloader):
            # print("Step: ", step)
            # print("Batch: ", batch)

            query_tensors = batch["input_ids"]
            # print("Query Tensors: ", len(query_tensors))
            # print("Query Tensors: ", query_tensors)

            # Generate outputs
            response_tensors = []
            for query in query_tensors:
                # print("Query: ", query)
                query.to(DEVICE) #.unsqueeze(0).to(DEVICE)
                response_tokens = ppo_trainer.generate(query, return_prompt=False, **generation_kwargs)
                response_tensors.append(response_tokens.squeeze().to(DEVICE))
            # print("Response Tensors: ", response_tensors)
            # print("Response Tensors: ", len(response_tensors))
            pred = [ppo_tokenizer.decode(r.squeeze()) for r in response_tensors]
            # print("Pred: ", pred)
            # pred = pred.split("[EOS]")[1].split(ppo_trainer.tokenizer.eos_token)[0].split("[/EOS]")[0].replace("<|endoftext|>", "")
            batch["response"] = pred
            # print("Response: ", batch["response"])
            # print("Len Batch Responses: ", len(batch["response"]))
            
            # Compute rewards
            rewards_list = []
            for instr, resp in zip(batch["instruction"], batch["response"]):
                print("Instr: ", instr)
                print("Resp: ", resp)
                reward = inference(reward_tokenizer, reward_model, resp)
                rewards_list.append(torch.tensor(reward).to(DEVICE))
            # print("Rewards List: ", rewards_list)

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
            print("Stats: ", stats)
            ppo_trainer.log_stats(stats, batch, rewards_list)


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
    # print("Input ids: ", dataset[3]["input_ids"])
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
        data_collator=collator #dataloader
    )

    build_pipeline(ppo_config, ppo_trainer, policy_tokenizer, reward_model, reward_tokenizer, dataloader)
