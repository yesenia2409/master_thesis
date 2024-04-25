"""
Plot results of proof of concept

* Loading the annotated xlsx file
* Plot the win rate as a bar chart
* Plot the human evaluation scores as a bar chart
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_win_rate(file_path):
    """
    Plots the win rate of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    data = pd.read_excel(file_path)
    total = data.shape[0]-1

    with_better = data['with better'].iloc[-1]
    without_better = data['without better'].iloc[-1]
    equal = data['equal'].iloc[-1]

    categories = ['Prompt template', 'Without prompt template', 'Equal outputs']
    percentages = [(with_better/ total) * 100, (without_better/ total) * 100, (equal/ total) * 100]

    plt.bar(categories, percentages, color=colors)
    plt.ylabel('Win rate (%)')
    plt.title('With vs. Without prompt template')
    plt.savefig("Output_files/proof_of_concept_plot_win_rate.png")


def plot_human_eval_res(file_path):
    """
    Plots the human evaluation scores of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "royalblue"]
    data = pd.read_excel(file_path)

    categories = ['Grammatical correctness', 'Language consistency', 'No hallucination detection',
                  'Content correctness', 'Helpfulness', 'Token limitation']

    values_with = data.iloc[-1].values[3:9]
    values_without = data.iloc[-1].values[10:16]

    num_categories = len(categories)
    bar_width = 0.35
    index = range(num_categories)

    plt.bar(index, values_with, bar_width, label='With prompt template', color=colors[1])
    plt.bar([i + bar_width for i in index], values_without, bar_width, label='Without prompt template', color=colors[0])

    plt.ylabel('Human Evaluation Score (0-100)')
    plt.title('With vs. Without prompt template')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=15)
    plt.legend()

    plt.tight_layout()
    plt.savefig("Output_files/proof_of_concept_plot_human_eval_scores.png")


if __name__ == "__main__":
    file_input = "Proof_Of_Concept/Annotated_files/proof_of_concept_meta-llama-Llama-2-13b-chat-hf_50samples_33seed_annotated.xlsx"
    plot_win_rate(file_input)
    plot_human_eval_res(file_input)