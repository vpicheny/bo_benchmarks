from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os


def plot_regret(regrets: Dict[str, np.ndarray], title: str=None, ylabel="Regret", show_all=False):
    fig, ax = plt.subplots(figsize=(5, 5))
    lines = []
    for name, regret in regrets.items():
        y_lo, y_md, y_up = np.nanpercentile(regret, q=[10, 50, 90], axis=0)
        x = np.arange(y_md.shape[0])
        lines += ax.plot(x, y_md, label=name)
        ax.fill_between(x, y_up, y_lo, alpha=0.3, cmap=plt.cm.RdYlGn)
        ax.set_yscale('log')
        if show_all:
            col = ax.get_lines()[-1].get_color()
            ax.plot(x[:, None], regret.T, linewidth=.5, color=col)
    ax.legend(handles=lines, loc='lower right')
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    return fig


dir = "results_prior"
pb_tags = {"gauss_noise_branin", "exp_noise_branin", "hartmann_3", "flat_branin_noise"}

for tag in pb_tags:
    print(f"Processing results for {tag}")
    all_subdir = glob(f"{dir}/{tag}/*/")
    print(f"Found {len(all_subdir)} algorithms")
    # Each subdir corresponds to a particular algorithm instance, so
    # we'll group the results by subdir, using a dictionary to allow for varying lengths
    all_regrets = dict()

    for i, subdir in enumerate(all_subdir):
        all_files = glob(f"{subdir}*regret.npy")
        exp_name = os.path.basename(subdir[:-1])
        print(f"    Processed results for {exp_name}")
        print(f"    Found {len(all_files)} files")

        if len(all_files) > 1:
            regret = np.load(file=all_files[0])
            for file in all_files[1:]:
                reg = np.load(file=file)
                regret = np.hstack([regret, reg])
            all_regrets[exp_name] = regret.T

    fig = plot_regret(all_regrets, title=tag, ylabel="Simple regret", show_all=False)
