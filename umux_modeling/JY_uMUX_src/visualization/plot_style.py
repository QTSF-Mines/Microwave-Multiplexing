import matplotlib as mpl
import matplotlib.pyplot as plt


def set_plot_style():
    mpl.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 6.5,
        "axes.linewidth": 0.25,
        "xtick.major.width": 0.25,
        "ytick.major.width": 0.25,
        "xtick.minor.width": 0.25,
        "ytick.minor.width": 0.25,
        "xtick.major.size": 1,
        "ytick.major.size": 1,
        "xtick.minor.size": 1,
        "ytick.minor.size": 1
    })