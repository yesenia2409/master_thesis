import pandas as pd
import json


# Preprocessing the apstudy data
def preprocess_apstudy_data(input_path, save_path):
    df = pd.read_json(input_path)
    df.drop(columns=['id'], inplace=True)
    df.rename(columns={'answerKey': 'label', 'question': 'prompt'}, inplace=True)

    for idx, prompt_dict in enumerate(df['prompt']):
        question = prompt_dict['stem']
        answer_A = prompt_dict['choices'][0]['text']
        answer_B = prompt_dict['choices'][1]['text']
        answer_C = prompt_dict['choices'][2]['text']
        answer_D = prompt_dict['choices'][3]['text']
        answer_E = prompt_dict['choices'][4]['text']
        df["prompt"][idx] = embedd_in_apstudy_template(question, answer_A, answer_B, answer_C, answer_D, answer_E)

    df.to_pickle(f"Input_files/pkl/{save_path}.pkl")
    df.to_csv(f"Input_files/csv/{save_path}.csv", index=False)


def embedd_in_apstudy_template(question, A, B, C, D, E):
    return f"<s> [INST] <<SYS>> \n You are a helpful, respectful and honest assistant. Always answer as helpfully as " \
           f"possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, " \
           f"dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in " \
           f"nature. If a question does not make any sense, or is not factually coherent, explain why instead of " \
           f"answering something not correct. If you don’t know the answer to a question, please don’t share false " \
           f"information. \n Please answer the questions related to geoscience. \n <</SYS>> \n {question} \n A. " \
           f"{A} \n B. {B} \n C. {C} \n D. {D} \n E. {E} [/INST] </s>"


# Preprocessing the npee data
def preprocess_npee_data(input_path, save_path):
    df = pd.DataFrame(columns=["id", "prompt", "label"])

    with open(input_path, encoding="utf-8") as json_file:
        data_dict = json.load(json_file)

    keys = ['noun', 'choice', 'completion', 'tf', 'qa', 'discussion']
    data_dict = [data_dict[k] for k in keys]
    for idx_id, dict in enumerate(data_dict):
        for idx_row, question in enumerate(dict["question"]):
            question = embedd_in_npee_template(question)
            df = pd.concat([df, pd.DataFrame([{'id': keys[idx_id], 'prompt': question, 'label': dict["answer"][idx_row]}])], ignore_index=True)

    df.to_pickle(f"Input_files/pkl/{save_path}.pkl")
    df.to_csv(f"Input_files/csv/{save_path}.csv", index=False)


def embedd_in_npee_template(question):
    return f"<s> [INST] <<SYS>> \n You are a helpful, respectful and honest assistant. Always answer as helpfully as " \
           f"possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, " \
           f"dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in " \
           f"nature. If a question does not make any sense, or is not factually coherent, explain why instead of " \
           f"answering something not correct. If you don’t know the answer to a question, please don’t share false " \
           f"information. \n Please answer the questions related to geoscience. \n <</SYS>> \n {question} [/INST] </s>"


if __name__ == "__main__":
    apstudy = "Input_files/json/geobench_apstudy.json"
    npee = "Input_files/json/geobench_npee.json"

    preprocess_apstudy_data(apstudy, "geobench_apstudy")
    preprocess_npee_data(npee, "geobench_npee")
