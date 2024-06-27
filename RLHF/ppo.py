
from datasets import Dataset
import torch
from peft import LoraConfig, PeftConfig, TaskType
from reward_model import inference as reward_inference
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed, create_reference_model
import pandas as pd
from peft import AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM

tqdm.pandas()
MODEL_PATH = "../SFT/merged_model/SFT_for_expert_alignment/"
REWARD_MODEL = "RewardModel/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(dataset_path):
    train_set = pd.read_pickle(dataset_path)
    ppo_data = Dataset.from_pandas(train_set)
    ppo_data = ppo_data.remove_columns(['type', 'category', 'text', '__index_level_0__'])
    ppo_data = ppo_data.shuffle()
    # ppo_data = Dataset.from_dict(ppo_data)    
    ppo_data.set_format("pytorch")
    return ppo_data


def load_model_and_tokenizer(MODEL_PATH):
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        device_map="auto",
        quantization_config=bnb_config, 
        peft_config=lora_config,
    )
    model.config.use_cache = False
    model = model.to(DEVICE)
    model.bfloat16()
    
    ref_model = create_reference_model(model)
    ref_model.config.use_cache = False
    ref_model = ref_model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    print("Done loading Policy Model and Tokenizer!")
    return ref_model, model, tokenizer


def load_reward_model_and_tokenizer():
    reward_model = AutoPeftModelForSequenceClassification.from_pretrained(
        REWARD_MODEL,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    reward_model.eval()
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
    query_tokens = []
    response_tokens = []
    preds = []
    for prompt in prompts:
        encode_dict = tokenizer(prompt, return_tensors="pt", padding=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        summ_tokens = model.generate(input_ids=txt_tokens, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        query_tokens.append(txt_tokens.squeeze())
        response_tokens.append(summ_tokens.squeeze())
        preds.append(pred)
    return query_tokens, response_tokens, preds


def build_pipeline(ppo_config, ppo_trainer, policy_model, policy_tokenizer, reward_model, reward_tokenizer):
    log_history = []
    kwargs = {
        "min_length": -1,
        "max_new_tokens": 256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "do_sample": True, 
        "temperature": 0.5,
    }

    for epoch in tqdm(range(ppo_config.ppo_epochs), "epoch: "):
        batch_counter = 0
        for batch in tqdm(ppo_trainer.dataloader):
            print("Batch: ", batch_counter)

            # Generate outputs
            query_tensors, response_tensors, pred = inference(policy_model, policy_tokenizer, batch["instruction"], kwargs)
            # print("Response Tensors: ", response_tensors)
            for idx in range(len(pred)):
                pred[idx] = pred[idx].split("[/INST]")[1].split(policy_tokenizer.eos_token)[0]
            batch["response"] = pred
            del pred
            torch.cuda.empty_cache()
            # print(pred)
            
            # Compute rewards
            rewards_list = []
            for instr, resp in zip(batch["instruction"], batch["response"]):
                # print("Instr: ", instr)
                # print("Resp: ", resp)
                reward = reward_inference(reward_tokenizer, reward_model, resp, kwargs["max_new_tokens"])
                rewards_list.append(torch.tensor(reward).to(DEVICE))
            # print("Rewards List: ", rewards_list)

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
            # print("Stats: ", stats)
            ppo_trainer.log_stats(stats, batch, rewards_list)
            log_history.append(stats)
            print("Loss: ", stats['ppo/loss/value'], " and KL penalty ", stats['ppo/mean_non_score_reward'])
            batch_counter += 1

    with open("Output_files/slurm_files/trainer_log_history_1epoch_2_00E-5Lr_2batch.txt", "a") as text_file:
        text_file.write(str(log_history))
    
    policy_model.save_pretrained("Policy_Model/", save_serialisation=True)
    policy_tokenizer.save_pretrained("Policy_Model/")
    print("Done saving model!")

if __name__ == "__main__":

    ################
    # Model & Tokenizer
    ################
    set_seed(42)
    ref_model, policy_model, policy_tokenizer = load_model_and_tokenizer(MODEL_PATH)
    reward_model, reward_tokenizer = load_reward_model_and_tokenizer()

    ################
    # Dataset
    ################
    path = "../SFT/Input_files/train_set_expert.pkl"
    dataset = build_dataset(path)

    ################
    # Training
    ################
    policy_model.gradient_checkpointing_enable()

    ppo_config = PPOConfig(
        batch_size=2,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        ppo_epochs=1,
        model_name=MODEL_PATH,
        learning_rate=0.000002,
        remove_unused_columns=False,
        seed=42,
        optimize_device_cache=True, 
        optimize_cuda_cache=True,
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
    

    # _, loaded_model, loaded_tokenizer = load_model_and_tokenizer("Policy_Model/")
    
    # kwargs = {
    #     "min_length": -1,
    #     "max_new_tokens": 256,
    #     "eos_token_id": 50256,
    #     "pad_token_id": 50256,
    #     "do_sample": True,
    #    "temperature": 0.5,
    # }
    # x, y, pred = inference(loaded_model, loaded_tokenizer, ["How old is the earth?"], kwargs)
    # print(pred)
