import pandas as pd
import matplotlib.pyplot as plt


def plot_loss(content, save_path, plot_type):
    colors = ["lightsteelblue", "cornflowerblue"]
    plt.figure()

    if plot_type == "penalty":
        loss_content = content.split("\"ppo/loss/value\": ")
        loss = []
        for idx, elem in enumerate(loss_content):
            if idx % 2 == 1:
                elem = elem.split(", \"ppo/loss/total\":")[0]
                loss.append(float(elem))
        print(loss)
        x = list(range(1, len(loss) + 1))
        plt.plot(x, loss, label='Loss', color=colors[0])

    if plot_type == "loss":
        kl_content = content.split("\"ppo/mean_non_score_reward\": ")
        kl = []
        for idx, elem in enumerate(kl_content):
            if idx % 2 == 1:
                elem = elem.split(", \"ppo/mean_scores\":")[0]
                kl.append(float(elem))
        print(kl)

        x = list(range(1, len(kl)+1))
        plt.plot(x, kl, label='KL penalty', color=colors[1])

    if plot_type == "kl":
        kl_content = content.split("\"objective/kl\": ")
        kl = []
        for idx, elem in enumerate(kl_content):
            if idx % 2 == 1:
                elem = elem.split(", \"objective/kl_dist\":")[0]
                kl.append(float(elem))
        print(kl)

        x = list(range(1, len(kl)+1))
        plt.plot(x, kl, label='mean KL divergence', color=colors[1])

    if plot_type == "entropy":
        kl_content = content.split("\"objective/entropy\": ")
        kl = []
        for idx, elem in enumerate(kl_content):
            if idx % 2 == 1:
                elem = elem.split(", \"ppo/mean_non_score_reward\":")[0]
                kl.append(float(elem))
        print(kl)

        x = list(range(1, len(kl)+1))
        plt.plot(x, kl, label='Entropy', color=colors[0])

    plt.xlabel('Training steps')
    plt.legend()
    # plt.show()
    plt.savefig(save_path)


def plot_npee_vs_ape(base, sft, rlhf):
    """
    Plots the human evaluation scores of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    categories = ['APE (MC)', 'NPEE (MC + TF)']

    num_categories = len(categories)
    bar_width = 0.25
    index = range(num_categories)

    plt.bar(index, base, bar_width, label='Base model', color=colors[0])
    plt.bar([i + bar_width for i in index], sft, bar_width, label='Base + SFT', color=colors[1])
    plt.bar([i + bar_width*2 for i in index], rlhf, bar_width, label='Base + SFT + RLHF', color=colors[2])

    plt.ylabel('Accuracy in %')
    plt.title('Multiple Choice vs. True False')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=15)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("Output_files/plot_ape_vs_npee.png")


def plot_mc_vs_tf(base, sft, rlhf):
    """
    Plots the human evaluation scores of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    # categories = ['MC (APE)', 'MC (NPEE)', 'TF (NPEE)']
    categories = ['MC (APE + NPEE)', 'TF (NPEE)']

    num_categories = len(categories)
    bar_width = 0.25
    index = range(num_categories)

    plt.bar(index, base, bar_width, label='Base model', color=colors[0])
    plt.bar([i + bar_width for i in index], sft, bar_width, label='Base + SFT', color=colors[1])
    plt.bar([i + bar_width*2 for i in index], rlhf, bar_width, label='Base + SFT + RLHF', color=colors[2])

    plt.ylabel('Accuracy in %')
    plt.title('Multiple Choice vs. True False')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=15)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("Output_files/plot_mc_vs_tf.png")


def plot_open_tasks_individually(base, sft, rlhf):
    """
    Plots the human evaluation scores of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    categories = ['Fill-in-the-blank', 'Discussion', 'Definition', 'Question answering']

    num_categories = len(categories)
    bar_width = 0.25
    index = range(num_categories)

    plt.bar(index, base, bar_width, label='Base model', color=colors[0])
    plt.bar([i + bar_width for i in index], sft, bar_width, label='Base + SFT', color=colors[1])
    plt.bar([i + bar_width * 2 for i in index], rlhf, bar_width, label='Base + SFT + RLHF', color=colors[2])

    plt.ylabel('Wins in %')
    plt.title('Win rates for open tasks')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=15)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("Output_files/plot_win_rates_open_tasks.png")


def plot_open_tasks_together(base, sft, rlhf):
    """
    Plots the human evaluation scores of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    categories = [" "]

    num_categories = len(categories)
    bar_width = 0.15
    index = range(num_categories)

    plt.bar(index, base, bar_width, label='Base model', color=colors[0])
    plt.bar([i + bar_width + 0.1 for i in index], sft, bar_width, label='Base + SFT', color=colors[1])
    plt.bar([i + 0.2 + bar_width * 2 for i in index], rlhf, bar_width, label='Base + SFT + RLHF', color=colors[2])

    plt.ylabel('Wins in %')
    plt.title('Win rate for all open tasks')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=15)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("Output_files/plot_win_rate_all_open_tasks_together.png")


def plot_human_eval_closed_tasks(base, sft, rlhf):
    """
    Plots the human evaluation scores of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    categories = ["Grammatical Correctness", "Repetition", "Task Answering", "Additional Information",
                  "Hallucination Detection", "Content Correctness"]

    num_categories = len(categories)
    bar_width = 0.25
    index = range(num_categories)

    plt.bar(index, base, bar_width, label='Base', color=colors[0])
    plt.bar([i + bar_width for i in index], sft, bar_width, label='SFT', color=colors[1])
    plt.bar([i + bar_width * 2 for i in index], rlhf, bar_width, label='RLHF', color=colors[2])

    # plt.ylabel('Wins in %')
    plt.title('Human evaluation: True-or-False')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=30)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("Output_files/plot_human_eval_tf_tasks.png")


def plot_human_eval_open_tasks(base, sft, rlhf):
    """
    Plots the human evaluation scores of with vs. without template as a bar chart
    :param file_path: path to where the annotated result file is stored
    :return: -
    """

    colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
    categories = ["Grammatical Correctness", "Repetition", "Task Answering", "Level of expertise",
                  "Hallucination Detection", "Content Correctness"]

    num_categories = len(categories)
    bar_width = 0.25
    index = range(num_categories)

    plt.bar(index, base, bar_width, label='Base', color=colors[0])
    plt.bar([i + bar_width for i in index], sft, bar_width, label='SFT', color=colors[1])
    plt.bar([i + bar_width * 2 for i in index], rlhf, bar_width, label='RLHF', color=colors[2])

    plt.title('Human evaluation: Discussion')
    plt.xticks([i + bar_width / 2 for i in index], categories, rotation=30)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig("Output_files/plot_human_eval_discussion_tasks.png")

if __name__ == "__main__":
    # with open("Output_files/slurm_files/ppo/trainer_log_history_1epoch_2_00E-6Lr_4batch_36ksamples.txt", 'r') as file:
    #     content = file.read()
    # plot_loss(content, "Output_files/loss_plots/policy_kl_mean.png", "kl")

    # res_base = [55.3, 28.0, 53.7]
    # res_sft = [48.0, 33.0, 53.0]
    # res_rlhf = [33.3, 29.3, 58.6]
    # plot_mc_vs_tf(res_base, res_sft, res_rlhf)

    # res_base = [41.7, 53.7]
    # res_sft = [40.5, 53.0]
    # res_rlhf = [31.3, 58.6]
    # plot_mc_vs_tf(res_base, res_sft, res_rlhf)

    # res_base = [55.3, 40.9]
    # res_sft = [48.0, 43.0]
    # res_rlhf = [33.3, 44]
    # plot_npee_vs_ape(res_base, res_sft, res_rlhf)

    # res_base = [66, 16, 36, 12]
    # res_sft = [2, 44, 32, 26]
    # res_rlhf = [32, 40, 32, 64]
    # plot_open_tasks_individually(res_base, res_sft, res_rlhf)

    # res_base = [32.5]
    # res_sft = [26]
    # res_rlhf = [42]
    # plot_open_tasks_together(res_base, res_sft, res_rlhf)

    # MC
    # res_base = [20, 20, 14, 18, 4, 4]
    # res_sft = [20, 20, 20, 2, 19, 3]
    # res_rlhf = [20, 19, 19, 5, 17, 5]
    # plot_human_eval_closed_tasks(res_base, res_sft, res_rlhf)

    # TF
    # res_base = [20, 19, 20, 20, 9, 12]
    # res_sft = [18, 19, 20, 5, 15, 10]
    # res_rlhf = [17, 16, 20, 10, 16, 12]
    # plot_human_eval_closed_tasks(res_base, res_sft, res_rlhf)

    # FITB
    # res_base = [40, 40, 40, 37, 15, 16]
    # res_sft = [34, 24, 28, 20, 23, 10]
    # res_rlhf = [26, 24, 18, 10, 28, 4]
    # plot_human_eval_open_tasks(res_base, res_sft, res_rlhf)

    # Definition
    # res_base = [40, 40, 38, 40, 26, 32]
    # res_sft = [40, 38, 34, 34, 38, 32]
    # res_rlhf = [38, 32, 28, 24, 34, 21]
    # plot_human_eval_open_tasks(res_base, res_sft, res_rlhf)

    # QA
    # res_base = [40, 40, 34, 34, 13, 13]
    # res_sft = [40, 32, 30, 25, 22, 13]
    # res_rlhf = [40, 36, 32, 30, 19, 15]
    # plot_human_eval_open_tasks(res_base, res_sft, res_rlhf)

    # QA
    res_base = [40, 40, 36, 36, 23, 25]
    res_sft = [37, 31, 28, 27, 31, 25]
    res_rlhf = [36, 31, 27, 26, 30, 22]
    plot_human_eval_open_tasks(res_base, res_sft, res_rlhf)