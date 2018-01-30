import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def crosstab(df, column1, column2):
    return pd.crosstab(df[column1], df[column2])


def check_na(df, columns=None):
    if columns is not None:
        return df[df[columns].isnull()]
    return df.isnull().sum()


def check_category(df, column):
    return df.groupby(column).size()


def split_category(df, column, pieces):
    return pd.qcut(df[column], pieces).value_counts()


def heatmap(df):
    sns.heatmap(df.corr(), annot=True, fmt='.1f')
    plt.show()
