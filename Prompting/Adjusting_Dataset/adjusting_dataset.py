"""
Prompting: Adjust dataset

* Load the geosignal json file from GitHub
* Concatenate instruction and input columns
* Apply the prompt template to each prompt
* Save the dataset as pickle and csv files
"""

import requests
import pandas as pd


def load_json(file_url, output_dir):
    """
    Loads the geosignal dataset file form the https://github.com/davendw49/k2 repository
    :param file_url: url to the geosignal file
    :param output_dir: directory to where the file will be stored
    :return: -
    """
    file_url = file_url
    save_path = output_dir

    response = requests.get(file_url)

    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("Die Datei wurde erfolgreich heruntergeladen und gespeichert.")
    else:
        print("Fehler beim Herunterladen der Datei. Statuscode:", response.status_code)


def modify_df(input_file):
    """
    Concatenated the columns "instruction" and "input" such that become one since the template does not differentiate between them.
    :param input_file: path to where the json file is stored
    :return: df of the dataset
    """
    df = pd.read_json(input_file)
    df['instruction'] = df['instruction'] + ' ' + df["input"]
    df.drop('input', axis=1, inplace=True)
    return df


def apply_template(df, system_text, save_path):
    """
    Applies the prompt template to each prompt in the dataset and saves the dataframe.
    :param df: df of the dataset
    :param system_text: system prompt as String
    :param save_path: path to where the pickle and csv file will be stored (without file ending)
    :return: -
    """
    for idx, prompt in enumerate(df['instruction']):
        df['instruction'][idx] = f"<s>[INST] <<SYS>>\n{system_text}\n<</SYS>>\n{prompt} [/INST]"

    df.to_pickle(f"{save_path}.pkl")
    df.to_csv(f"{save_path}.csv", index=False)


if __name__ == "__main__":
    # Variables
    file_url = 'https://raw.githubusercontent.com/davendw49/k2/main/data/geosignal/geosignal.json'
    input_path = 'Input_files/geosignal.json'
    save_path = 'Output_files/geosignal'
    system_prompt = 'Please answer the questions related to geoscience.'

    # Functions
    load_json(file_url, input_path)
    df = modify_df(input_path)
    apply_template(df, system_prompt, save_path)

