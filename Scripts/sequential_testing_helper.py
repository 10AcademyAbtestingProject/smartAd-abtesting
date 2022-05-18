import matplotlib.pyplot as plt
import pandas as pd


class Sequential_AB_Helper:
    def bernouli_constructor(self, engagement: pd.Series, yes_count: pd.Series):
        engagement_list = engagement.tolist()
        yes_list = yes_count.tolist()
        bernouli_series = []
        for i in range(len(engagement_list)):
            no_list = engagement_list[i] - yes_list[i]
            bernouli_series += yes_list[i] * [1]
            bernouli_series += no_list * [0]
        return bernouli_series

    def plot_cumulative(self, upper_limit, lower_limit, r, x1):
        plt.plot(r, upper_limit, color='green',
                 linewidth=1, label='Upper Bound')
        plt.plot(r, lower_limit, color='red',
                 linewidth=1, label='Lower Bound')
        plt.plot(r, x1, color='yellow', linewidth=1,
                 label='Cumulative value of yes and no')
        plt.legend()
        plt.show()
