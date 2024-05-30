## read in json files
import json
import pandas as pd

# "geobench_apstudy.json" --> retrieves a list of dicts with dict_keys(['id', 'question', 'answerKey']) --> only MC questions with A, B, C, D, E as possibilities
# each question is stored insight another dict with "stem" as key for the question and choices as key for answers. The answers are again dicts with "text" and "label" keys

# what do i need?
# input column:
# <s>[INST] <<SYS>> \n You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information. \n
# Please answer the questions related to geoscience. \n <</SYS>> \n
# Which of the following statements best describes the main difference between population clusters in Asia and Europe?
# A. In Asia, most people live in urban areas.
# B. In Europe, most people live in urban areas.
# C. In Asia, most people make their living in the secondary sector of the economy.
# D. In Europe, most people make their living in the primary sector of the economy.
# E. In Asia, the tertiary sector of the economy is stagnant. [/INST] </s>

# output column: B


def preprocess_apstudy_data(input_path):
    df = pd.read_json(input_path)


if __name__ == "__main__":
    apstudy = "geobench_apstudy.json"
    npee = "geobench_npee.json"

    preprocess_apstudy_data(apstudy)