import pandas as pd
import os
from collections import Counter


def preprocess_ood_data(input_path, save_path):
    df = pd.read_pickle(input_path)
    df.drop(['type', 'category'], axis=1, inplace=True)
    df.rename(columns={'output': 'rejected', 'text': 'chosen'}, inplace=True)

    for idx, row in df.iterrows():
        df["chosen"][idx] = "This question falls outside the field of geoscience. Since my expertise is limited to " \
                        "geoscience topics, I'm unable to assist with this."
    # print(df['chosen'])
    df.to_csv(save_path, index=False)


def filter_zeros_as_pad_value(eval_res_dir, save_path):
    dataset = pd.read_csv(save_path)
    for filename in os.listdir(eval_res_dir):
        if 'SFT_only' in filename:
            file_path = os.path.join(eval_res_dir, filename)
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                if "0 0 0" in row["pred"]:
                    chosen_answer = row['pred'].split('0 0 0')[0]
                    new_row = {'instruction': row['input'], 'rejected': row['pred'], 'chosen': chosen_answer}
                    dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)
    dataset.to_csv(save_path, index=False)


def filter_extensive_repetitions(eval_res_dir, save_path, n):
    dict_of_rejected = {}
    dict_of_chosen = {}
    dict_of_instruction = {}
    dataset = pd.read_csv(save_path)

    for filename in os.listdir(eval_res_dir):
        if 'SFT_only' in filename:
            if "discussion" in filename:
                n = 10
            file_path = os.path.join(eval_res_dir, filename)
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                idx = f"{idx}-{filename}"
                pred = row["pred"].replace("\n", "||").replace("\r", "~~")
                pred = pred.split("[/INST]")[0].split("0 0 0")[0]
                pred_split = pred.split()
                n_grams = [tuple(pred_split[i:i + n]) for i in range(len(pred_split) - n + 1)]
                ngram_counts = Counter(n_grams)
                for ngram, count in ngram_counts.items():
                    if count >= 3 and idx not in dict_of_rejected.keys():
                        ngram = ' '.join(ngram).replace("||", "\n").replace("~~", "\r")
                        pred = pred.replace("||", "\n").replace("~~", "\r")
                        dict_of_rejected[idx] = pred
                        dict_of_instruction[idx] = row['input']
                        pred_chosen = pred.split(ngram)
                        pred_chosen.insert(1, ngram)
                        pred_chosen = ''.join(pred_chosen[:3])
                        dict_of_chosen[idx] = pred_chosen

    # print(len(dict_of_rejected))
    for idx in dict_of_rejected.keys():
        new_row = {'instruction': dict_of_instruction[idx], 'rejected': dict_of_rejected[idx], 'chosen': dict_of_chosen[idx]}
        dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)

    dataset.to_csv(save_path, index=False)


def filter_completion_task(eval_res_dir, save_path):
    dataset = pd.read_csv(save_path)
    for filename in os.listdir(eval_res_dir):
        if 'SFT_only' and "completion" in filename:
            file_path = os.path.join(eval_res_dir, filename)
            df = pd.read_csv(file_path)
            for idx, row in df.iterrows():
                pred = row["pred"].strip().split("[/INST]")[0].split("0 0 0")[0]
                instruct = row["input"].strip().split("Please complete the following sentence:  ")[1].split("[/INST]")[0]
                # if pred in f"#  {instruct}":

                print("INSTRUCTION: ", f"#  {instruct}")
                print("PREDICTION: ", pred)





                   #  new_row = {'instruction': row['input'], 'rejected': row['pred'], 'chosen': chosen_answer}
                   #  dataset = pd.concat([dataset, pd.DataFrame([new_row])], ignore_index=True)
    # dataset.to_csv(save_path, index=False)


if __name__ == "__main__":
    ood_data_path = "../SFT/Input_files/test_set_human.pkl"
    evaluation_data = "../Evaluation/Output_files/answers/"
    file_name = "Input_files/dataset_SFT_reward_model.csv"
    # preprocess_ood_data(ood_data_path, file_name)
    # filter_zeros_as_pad_value(evaluation_data, file_name)
    # filter_extensive_repetitions(evaluation_data, file_name, 8)
    # filter_completion_task(evaluation_data, file_name)