import pandas as pd


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
        df["prompt"][idx] = embedd_in_template(question, answer_A, answer_B, answer_C, answer_D, answer_E)

    df.to_pickle(f"Input_files/pkl/{save_path}.pkl")
    df.to_csv(f"Input_files/csv/{save_path}.csv", index=False)


def embedd_in_template(question, A, B, C, D, E):
    return f"<s> [INST] <<SYS>> \n You are a helpful, respectful and honest assistant. Always answer as helpfully as " \
           f"possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, " \
           f"dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in " \
           f"nature. If a question does not make any sense, or is not factually coherent, explain why instead of " \
           f"answering something not correct. If you don’t know the answer to a question, please don’t share false " \
           f"information. \n Please answer the questions related to geoscience. \n <</SYS>> \n {question} \n A. " \
           f"{A} \n B. {B} \n C. {C} \n D. {D} \n E. {E} [/INST] </s>"


def preprocess_npee_data(input_path, save_path):
    df = pd.read_json(input_path)
    df.to_csv(f"Input_files/csv/{save_path}.csv", index=False)
    print(df)


if __name__ == "__main__":
    apstudy = "Input_files/json/geobench_apstudy.json"
    npee = "Input_files/json/geobench_npee.json"

    # preprocess_apstudy_data(apstudy, "geobench_apstudy")
    preprocess_npee_data(npee, "geobench_npee")