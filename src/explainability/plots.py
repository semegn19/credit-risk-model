import matplotlib.pyplot as plt
import shap


def summary_plot(values, X):

    fig = plt.figure(figsize=(10,6))

    shap.summary_plot(
        values,
        X,
        show=False
    )

    return fig


def waterfall_plot(values):

    fig = plt.figure(figsize=(9,6))

    shap.plots.waterfall(
        values,
        show=False
    )

    return fig


def bar_plot(values):

    fig = plt.figure(figsize=(9,6))

    shap.plots.bar(
        values,
        show=False
    )

    return fig