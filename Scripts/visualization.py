import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotters:
    def __init__(self, w, h) -> None:
        self.w = w
        self.h = h

    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        sns.displot(data=df, x=column, color=color,
                    kde=True, height=self.h, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_count(self, df: pd.DataFrame, column: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.countplot(data=df, x=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_bar(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.show()

    def plot_heatmap(self, df: pd.DataFrame, title: str, cbar=False) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                    vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
        plt.title(title, size=18, fontweight='bold')
        plt.show()

    def plot_box(self, df: pd.DataFrame, x_col: str, title: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()

    def plot_box_multi(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_pair(self, df: pd.DataFrame, title: str, hue: str) -> None:
        plt.figure(figsize=(self.w, self.h))
        sns.pairplot(df,
                     hue=hue,
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                     height=4)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

    def plot_top_n_counts(self, df: pd.DataFrame, column: str, top_n: int, title: str, color: str = "green") -> pd.Series:
        top_n_count = df[column].value_counts().nlargest(top_n)
        axis = top_n_count.plot.bar(
            color=color, title=title, fontsize=20, figsize=(self.w, self.h))
        return top_n_count
