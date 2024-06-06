"""
SFT: inference

* Load the geosignal json file from GitHub
* Concatenate instruction and input columns
* Apply the prompt template to each prompt
* Save the dataset as pickle and csv files
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import pandas as pd


def inference(model, tokenizer, prompts, labels, max_new_tokens):
    """
    Run inference on prompts and return result lists.
    :param model: the model object
    :param tokenizer: the tokenizer object
    :param prompts: a list with the prompts taken from the dataset
    :param labels: a list with the labels taken from the dataset
    :param max_new_tokens: number of maximal tokens the model is allowed to generate
    :return: pred_list, input_list, label_list: list with the predictions, the prompts and the labels
    """
    model.eval()

    pred_list = []
    input_list = []
    label_list = []

    for idx, (prompt, label) in enumerate(zip(prompts, labels)):
        encode_dict = tokenizer(prompt, return_tensors="pt", padding=True)
        txt_tokens = encode_dict["input_ids"].cuda()
        attention_mask = encode_dict["attention_mask"].cuda()
        kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": 50256, "pad_token_id": 50256}
        summ_tokens = model.generate(input_ids=txt_tokens, attention_mask=attention_mask, **kwargs)
        pred = tokenizer.batch_decode(summ_tokens)[0]
        pred = pred.split("</s><s>")[1]
        pred_list.append(pred)
        input_list.append(prompt.replace(" [EOS]", ""))
        label_list.append(label.replace("\n", " "))
        print(f"{idx}. prompt done! {len(prompts)-idx} more to go..")
    return pred_list, input_list, label_list


def save_to_csv(pred_list, gold_list, input_list, output_path):
    """
    Saves the results in a csv file.
    :param pred_list: a list with the predictions generated by the model
    :param gold_list: a list with the labels taken from the dataset
    :param input_list: a list with the prompts taken from the dataset
    :param output_path: path to a file where the result files should be stored
    :return: -
    """

    df = pd.DataFrame.from_dict(
        {
            "input": input_list,
            "gold": gold_list,
            "pred": pred_list,
        }
    )
    df.to_csv(output_path, index=False)


def create_model_and_tokenizer(model_dir):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        local_files_only=True,
        use_safetensors=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


if __name__ == "__main__":
    # Variables
    model_dir_local = "../SFT/merged_model/SFT_for_expert_alignment/"
    max_new_tokens = 128
    output_dir = "Output_files/answers/"
    benchmark = "npee_closed"
    model_name = "SFT_only"
    output_filename = f"output_for_evaluation_{benchmark}_{model_name}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Functions
    data = pd.read_pickle("Input_files/pkl/geobench_npee.pkl")
    data = data.loc[data['id'].isin(["tf"])]

    for idx, row in data.iterrows():
        prompt_list = row["prompt"].split("<</SYS>>")
        prompt_list.insert(1, "Please state if the following question is true or false:")
        row["prompt"] = ' '.join(prompt_list)
        # print(row["prompt"])

    model, tokenizer = create_model_and_tokenizer(model_dir_local)
    print("load_model() done!")

    pred_list, input_list, label_list = inference(model, tokenizer, data["prompt"], data["label"], max_new_tokens)
    print("inference() done!")

    save_to_csv(pred_list, label_list, input_list, output_path)
    print("save_to_csv() done!")
