"""
Prompting: Proof of Concept

* Randomly sample 50 queries of the SFT dataset
* Adding prompt templates to the models input
* Run inference with 50 samples with and without the meta-tags
* Manually compare/evaluate the results
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from datasets import load_dataset
from functools import reduce
import os


# Data Preparation

def create_dataset(name, count, seed):
    """
    Load dataset and randomly sample a number of prompts from the dataset as well as their labels and additional ainput.
    Concatenate prompt and additional inputs.
    :param name: path to the dataset
    :param count: number of prompts that should be sampled
    :param seed: int number to guarantee reproduction of random sampling
    :return: gold_labels, prompts: a list with the labels sampled from the dataset and a list with the prompts
    """
    data = load_dataset(name, split="train")

    shuffled_dataset = data.shuffle(seed=seed)
    sampled_dataset = shuffled_dataset.select(range(count))

    raw_prompts = [sample["instruction"] for sample in sampled_dataset]
    input = [sample["input"] for sample in sampled_dataset]
    gold_labels = [sample["output"] for sample in sampled_dataset]
    prompts = reduce(lambda res, l: res + [l[0] + l[1] + " [EOS] "], zip(raw_prompts, input), [])

    # print(prompts)
    return gold_labels, prompts


def adjust_prompts(prompts, system_text):
    """
    Adding prompt template and a system-prompt to each prompt
    :param prompts: list of sampled prompts from the dataset
    :param system_text: string of system prompt that is valid for all prompts
    :return: adjusted_prompts: list of all prompts embedded in the prompt template
    """
    adjusted_prompts = []

    for prompt in prompts:
        tmp_prompt = f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n{prompt} [/INST]"
        adjusted_prompts.append(tmp_prompt)

    return adjusted_prompts


# Model

def load_model(model_name):
    """
    Loads the model and sets some config parameters
    :param model_name: path to where the model is stored
    :return: model, tokenizer: the model and the tokenizer object
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.bos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "right"

    return model, tokenizer


# Inference & Save

def inference(model, tokenizer, prompts, labels, system_text, max_new_tokens):
    """
    Run inference on prompts with meta-tags and without meta-tags and store results in a Dataframe.
    :param model: the model object
    :param tokenizer: the tokenizer object
    :param prompts: a list with the prompts taken from the dataset
    :param labels: a list with the labels taken from the dataset
    :param system_text: a string that contains the system prompt
    :param max_new_tokens: number of maximal tokens the model is allowed to generate
    :return: df: df that contains the predictions with and without template, the prompts and the labels
    """
    model.half()
    model.eval()
    model.to("cuda")

    pred_list_without_tags = []
    pred_list_with_tags = []
    input_list = []
    label_list = []

    # inference without meta-tags
    for post, label in zip(prompts, labels):
        encode_dict = tokenizer(post, return_tensors="pt", padding=True, truncation=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("[EOS]")[1].split(tokenizer.eos_token)[0].replace("<|endoftext|>", "")
        pred_list_without_tags.append(pred)
        input_list.append(post.replace(" [EOS]", ""))
        label_list.append(label.replace("\n", " "))

    # inference with meta-tags
    adjusted_prompts = adjust_prompts(prompts, system_text)
    for post in adjusted_prompts:
        encode_dict = tokenizer(post, return_tensors="pt", padding=True, truncation=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("[/INST]")[1].split(tokenizer.eos_token)[0].replace("<|endoftext|>", "")
        pred_list_with_tags.append(pred)

    df = pd.DataFrame.from_dict(
        {"pred_without": pred_list_without_tags, "pred_with": pred_list_with_tags, "post": input_list,
         "gold": label_list})
    return df


def save_to_csv(df_result, output_dir, output_filename):
    """
    Saves the results in a csv file.
    :param df_result: df that contains the predictions with and without template, the prompts and the labels
    :param output_dir: path to a directory where the result files should be stored
    :param output_filename: name for the result file
    :return: -
    """

    preds_with_list = []
    preds_without_list = []
    post_list = []
    gold_list = []
    batch_size = 16

    for i in range(0, len(df_result), batch_size):
        predict_with = df_result["pred_with"].values[i: i + batch_size]
        predict_without = df_result["pred_without"].values[i: i + batch_size]
        posts = df_result["post"].values[i: i + batch_size]
        golds = df_result["gold"].values[i: i + batch_size]
        preds_with_list.extend(list(predict_with))
        preds_without_list.extend(list(predict_without))
        post_list.extend(list(posts))
        gold_list.extend(list(golds))

    df = pd.DataFrame.from_dict(
        {
            "input": post_list,
            "gold": gold_list,
            "pred_with": preds_with_list,
            "pred_without": preds_without_list,
        }
    )

    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Variables
    dataset = "daven3/geosignal"
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    count_samples = 50
    seed = 45
    system_input = ""
    # "You are an expert in Geoscience and want to answer the following question."
    max_new_tokens = 200
    output_dir = "Proof_Of_Concept/Output_files/"
    model_saving_path = model_name.replace("/", "-")
    output_filename = f"proof_of_concept_{model_saving_path}_{count_samples}samples_{seed}seed.csv"

    # Functions
    gold_labels, raw_prompts = create_dataset(dataset, count_samples, seed)
    print("create_dataset() done!")
    model, tokenizer = load_model(model_name)
    print("load_model() done!")
    df_result = inference(model, tokenizer, raw_prompts, gold_labels, system_input, max_new_tokens)
    print("inference() done!")
    save_to_csv(df_result, output_dir, output_filename)
    print("save_to_csv() done!")
