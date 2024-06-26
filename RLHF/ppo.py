from datasets import Dataset
import torch
from peft import LoraConfig
from reward_model import inference
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import pandas as pd
from peft import AutoPeftModelForSequenceClassification

tqdm.pandas()
MODEL_PATH = "../SFT/merged_model/SFT_for_expert_alignment/"
REWARD_MODEL = "RewardModel/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(dataset_path):
    train_set = pd.read_pickle(dataset_path)
    ppo_data = Dataset.from_pandas(train_set)
    ppo_data = ppo_data.remove_columns(['type', 'category', 'text', '__index_level_0__'])
    ppo_data.set_format("pytorch")
    return ppo_data


def load_model_and_tokenizer():
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
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    print("Done loading Policy Model and Tokenizer!")
    return model, model, tokenizer


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


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def inference(model, tokenizer, prompts, kwargs):
    model.eval()
    query_tokens = []
    response_tokens = []
    preds = []
    with torch.no_grad():
        for prompt in prompts:
            encode_dict = tokenizer(prompt, return_tensors="pt", padding=True)
            txt_tokens = encode_dict["input_ids"].cuda()
            summ_tokens = model.generate(input_ids=txt_tokens, **kwargs)
            pred = tokenizer.batch_decode(summ_tokens)[0]

            query_tokens.append(txt_tokens.squeeze())
            response_tokens.append(summ_tokens.squeeze())
            preds.append(pred)
    model.train()
    return query_tokens, response_tokens, preds

def build_pipeline(ppo_config, ppo_trainer, policy_model, policy_tokenizer, reward_model, reward_tokenizer):
    kwargs = {
        "max_new_tokens": 50,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        # "top_p": 0.5,
        # "temperature": 0.7,
        # "do_sample": True
    }

    for epoch in tqdm(range(ppo_config.ppo_epochs), "epoch: "):
        batch_counter = 0
        for batch in tqdm(ppo_trainer.dataloader):
            print("Batch: ", batch_counter)

            # Generate outputs
            query_tensors, response_tensors, pred = inference(policy_model, policy_tokenizer, batch["instruction"], kwargs)
            # print("Response Tensors: ", response_tensors)
            # pred = pred.split("[EOS]")[1].split(ppo_trainer.tokenizer.eos_token)[0].split("[/EOS]")[0].replace("<|endoftext|>", "")
            batch["response"] = pred
            
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
            batch_counter += 1


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
    dataset = build_dataset(path)

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
    )

    ppo_trainer = PPOTrainer(
        ppo_config,
        policy_model,
        ref_model,
        policy_tokenizer,
        dataset=dataset,
        data_collator=collator
    )

    build_pipeline(ppo_config, ppo_trainer, policy_model, policy_tokenizer, reward_model, reward_tokenizer)
