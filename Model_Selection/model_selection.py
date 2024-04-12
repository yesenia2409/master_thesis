"""
Model selection

* Randomly sample 50 queries of the SFT dataset
* Loading different mistral and llama models from huggingface
* Run inference with 50 these samples for each model
* Manually compare/evaluate the results
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datasets import load_dataset
from functools import reduce
import os
import argparse


# Data Preparation

def create_dataset(name, range, seed):
    """
    Load dataset and randomly sample a number of prompts from the dataset as well as their labels and additional input.
    Concatenate prompt and additional inputs.
    :param name:
    :param range:
    :return:
    """
    data = load_dataset(name, split="train")

    shuffled_dataset = data.shuffle(seed=seed)
    sampled_dataset = shuffled_dataset.select(range(range))

    raw_prompts = [sample["instruction"] for sample in sampled_dataset]
    input = [sample["input"] for sample in sampled_dataset]
    gold_labels = [sample["output"] for sample in sampled_dataset]
    prompts = reduce(lambda res, l: res + [l[0] + l[1] + " [EOS] "], zip(raw_prompts, input), [])

    return gold_labels, prompts


# Model

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "right"

    return model, tokenizer


# Inference & Save

def inference(model, tokenizer, prompts, labels, max_new_tokens):
    """
    Run inference on prompts and return result lists.
    :param model:
    :param tokenizer:
    :param prompts:
    :param max_new_tokens:
    :return:
    """
    model.to("cuda")
    model.eval()
    pred_list = []
    input_list = []
    label_list = []

    # inference
    for post, label in zip(prompts, labels):
        encode_dict = tokenizer(post, return_tensors="pt", padding=True, truncation=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("[EOS]")[1].split(tokenizer.eos_token)[0].replace("<|endoftext|>", "")
        pred_list.append(pred)
        input_list.append(post.replace(" [EOS]", ""))
        label_list.append(label.replace("\n", " "))
    return pred_list, input_list, label_list


def save_to_csv(pred_list, gold_list, input_list, output_dir, output_filename):
    """
    Saves the results in a csv file.
    :param pred_list:
    :param output_dir:
    :param output_filename:
    :return:
    """

    df = pd.DataFrame.from_dict(
        {
            "input": input_list,
            "gold": gold_list,
            f"pred_{args.model_name}": pred_list,
        }
    )

    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    # Args from sh script
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    # Variables
    dataset = "daven3/geosignal"
    count_samples = 20
    seed = 42
    system_input = ""
    max_new_tokens = 200
    output_dir = "../Model_Selection/Output_files/"
    output_filename = f"results_{args.model_name}_{count_samples}samples_{seed}seed.csv"

    # Functions
    gold_labels, raw_prompts = create_dataset(dataset, count_samples, seed)
    model, tokenizer = load_model(args.model_name)
    predictions, prompts, labels = inference(model, tokenizer, raw_prompts, gold_labels, max_new_tokens)
    save_to_csv(predictions, gold_labels, prompts, output_dir, output_filename)
