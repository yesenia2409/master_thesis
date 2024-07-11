import re
import matplotlib.pyplot as plt


def parse_and_calculate_averages(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    averages = []
    current_block = []

    for line in lines:
        # Check for the "Inference done for one prompt.." line
        if "Inference done for one prompt.." in line:
            continue

        # Check for the "Loss:" line, indicating the end of a block
        elif "Loss:" in line:
            if current_block:
                # Calculate the average of the current block
                average = sum(current_block) / len(current_block)
                averages.append(average)
                # Reset the current block
                current_block = []

        # If the line contains a number, add it to the current block
        else:
            # Use regular expression to check if line contains a number
            match = re.match(r'^-?\d+\.?\d*$', line.strip())
            if match:
                current_block.append(float(line.strip()))

    return averages


def plot_rewards(avgs):
    """
    Takes the tested learning rates as x values and the final train loss as y values.
    Plots the points and tries to fit the polynom that best describes the points.
    :return:
    """

    index = range(len(avgs))

    plt.plot(index, avgs, color='cornflowerblue', label='Reward')
    plt.xlabel('Batch')
    plt.ylabel('Reward score')
    plt.title("Reward distribution")
    plt.show()
    # plt.savefig("Output_files/loss_plots/reward_distribution.png")


if __name__ == "__main__":
    file_path = 'Output_files/slurm_files/ppo/slurm-3961428.out'
    averages = parse_and_calculate_averages(file_path)
    plot_rewards(averages)

