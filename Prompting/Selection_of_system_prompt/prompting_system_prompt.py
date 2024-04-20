"""
Prompting: Proof of Concept

* Randomly sample 20 queries of the SFT dataset
* Adding prompt templates to the models input
* Run inference with different system prompts
* Manually compare/evaluate the results
"""
import repackage
repackage.up()

import torch
import os
import pandas as pd
from Proof_Of_Concept import prompting_proof_of_concept


def inference(model, tokenizer, prompts, labels, system_text, max_new_tokens):
    """
    Run inference on prompts and return result lists.
    :param model: the model object
    :param tokenizer: the tokenizer object
    :param prompts: a list with the prompts taken from the dataset
    :param labels: a list with the labels taken from the dataset
    :param max_new_tokens: number of maximal tokens the model is allowed to generate
    :return: pred_list, input_list, label_list: list with the predictions, the prompts and the labels
    """
    # model.half()
    model.eval()
    # model.to("cuda")

    pred_list = []
    input_list = []
    label_list = []

    adjusted_prompts = prompting_proof_of_concept.adjust_prompts(prompts, system_text)

    # inference
    with torch.no_grad():
        for prompt, label in zip(adjusted_prompts, labels):
            encode_dict = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            txt_tokens = encode_dict["input_ids"].cuda()
            attention_mask = encode_dict["attention_mask"].cuda()
            kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": 50256, "pad_token_id": 50256}
            summ_tokens = model.generate(txt_tokens, attention_mask=attention_mask, **kwargs)
            pred = tokenizer.batch_decode(summ_tokens)[0]
            pred = pred.split("[EOS]")[1].split(tokenizer.eos_token)[0].replace("<|endoftext|>", "")
            pred_list.append(pred)
            input_list.append(prompt.replace(" [EOS]", ""))
            label_list.append(label.replace("\n", " "))

    return pred_list, input_list, label_list


def save_to_csv(pred_list, gold_list, input_list, col_name, output_path):
    """
    Saves the results in a csv file.
    :param pred_list: a list with the predictions generated by the model
    :param gold_list: a list with the labels taken from the dataset
    :param input_list: a list with the prompts taken from the dataset
    :param output_path: path to a file where the result files should be stored
    :return: -
    """

    if os.path.exists(output_path):

        df = pd.read_csv(output_path)
        df[col_name] = pred_list
        df.to_csv(output_path, index=False)

    else:

        df = pd.DataFrame.from_dict(
            {
                "input": input_list,
                "gold": gold_list,
                "col_name": pred_list,
            }
        )
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Variables
    dataset = "daven3/geosignal"
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    count_samples = 20
    seed = 33
    list_system_input = ["You are a geoscience expert.",
                         "Act as an expert in geoscience and answer the following question.",
                         "Answer the following question with your geoscience expertise.",
                         "Provide outputs that a geoscience professor would create.",
                         "Whenever you generate an answer to a geoscience question explain the reasoning ans assumpions behind your answer.",
                         "Whenever you can't answer a geosience question explain why you can't answer the question.",
                         "You are an expert in geology, geography and environmental science"]
    max_new_tokens = 200
    output_dir = "Output_files/"
    output_filename = f"system_prompt_test_{count_samples}samples_{seed}seed.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Functions
    gold_labels, raw_prompts = prompting_proof_of_concept.create_dataset(dataset, count_samples, seed)
    print("create_dataset() done!")
    model, tokenizer = prompting_proof_of_concept.load_model(model_name)
    print("load_model() done!")

    for index, system_input in enumerate(list_system_input):
        pred_list, input_list, label_list = inference(model, tokenizer, raw_prompts, gold_labels, system_input, max_new_tokens)
        print("inference() done!")
        save_to_csv(pred_list, label_list, input_list, system_input, output_path)
        print("save_to_csv() done!")
