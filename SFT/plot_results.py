"""
Plot results for finding the optimal hyperparameter

* Plot the final value of the train loss over the lr
* Plot the development of the eval loss for each lr over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import csv


def plot_loss_over_learning_rate():
    """
    Takes the tested learning rates as x values and the final train loss as y values.
    Plots the points and tries to fit the polynom that best describes the points.
    :return:
    """
    x = np.array([0.00001, 0.00002, 0.00003, 0.00004, 0.0001, 0.0002, 0.0003, 0.0004, 0.001, 0.002, 0.003, 0.004,
                  0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09])

    y = np.array([2.597, 2.597, 2.596, 2.596, 2.594, 2.590, 2.585, 2.578, 2.525, 2.433, 2.347, 2.280, 2.223, 2.177,
                  2.134, 2.095, 2.068, 2.037, 1.794, 1.714, 1.703, 2.008])

    for poly in [2, 4, 6, 8]:
        model = np.poly1d(np.polyfit(x, y, poly))
        polyline = np.linspace(0, 0.09)
        print(model)
        plt.scatter(x, y, color='lightblue', label='Original Data')
        plt.scatter(0.0002, model(0.0002), color="blue")  # recommended lr by torchtune
        plt.plot(polyline, model(polyline), color='lightblue')
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.title(poly)
        print("R2 ", poly, ": ", r2_score(y, model(x)))
        # plt.show()


def plot_loss_over_time():
    """
    Takes the values of the evaluation loss for each quarter of the SFT as y values and the quarters as x values.
    Fits a line through the points belonging to each lr. Calculates the lr that has the steepest slope.
    :return:
    """
    data = {
        'label': [0.001, 0.0001, 0.00001, 0.002, 0.0002, 0.00002, 0.003, 0.0003, 0.00003, 0.004,
                  0.0004, 0.00004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09],
        '25%': [2.622, 2.624, 2.626, 2.615, 2.624, 2.625, 2.606, 2.624, 2.625, 2.599, 2.624, 2.625, 2.589,
                2.580, 2.571, 2.559, 2.547, 2.538, 2.302, 2.107, 1.992, 1.936],
        '50%': [2.587, 2.623, 2.625, 2.526, 2.621, 2.625, 2.444, 2.619, 2.625, 2.366, 2.615, 2.625, 2.283,
                2.215, 2.147, 2.081, 2.033, 1.988, 1.573, 1.422, 1.376, 1.353],
        '75%': [2.482, 2.620, 2.624, 2.288, 2.612, 2.624, 2.096, 2.601, 2.624, 1.970, 2.589, 2.624, 1.868,
                1.790, 1.721, 1.658, 1.621, 1.572, 1.295, 1.273, 1.364, 1.706],
        '100%': [2.295, 2.612, 2.624, 1.991, 2.593, 2.624, 1.797, 2.567, 2.623, 1.657, 2.533, 2.622, 1.571,
                 1.496, 1.440, 1.383, 1.357, 1.333, 1.230, 1.250, 1.932, 9.949]
    }
    df = pd.DataFrame(data)
    x_values = np.array([25, 50, 75, 100])
    slopes = []

    plt.figure()

    for i, row in df.iterrows():
        y_values = row[1:].values  # Extract y values from the row (excluding the label)
        label = row['label']
        a, b = np.polyfit(x_values, y_values, 1)

        # Plot the data points
        plt.scatter(x_values, y_values)
        plt.plot(x_values, a * x_values + b, label=label)

        slopes.append((label, a))
        print(f"R2 value for {label}: ", r2_score(y_values, a * x_values + b))
        print(f"for {label}: {a}*x+{b}")

    plt.xlabel('progress of the SFT in %')
    plt.ylabel('loss')
    plt.legend(loc=1, ncols=3)
    lr, slope = min(slopes, key=lambda t: t[1])
    slope = "%.4f" % slope
    plt.title(f"Largest negative slope ({slope}) for lr = {lr}")
    # plt.show()
    plt.savefig("Output_files/loss_over_time_per_learning_rate_third_run.png")


if __name__ == "__main__":
    plot_loss_over_learning_rate()
    plot_loss_over_time()
