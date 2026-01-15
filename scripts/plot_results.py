import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.train_moons import train


def plot_loss(history, path):
    plt.figure(figsize=(5, 4))
    plt.plot(history)
    plt.xlabel("epoch")
    plt.ylabel("mse loss")
    plt.title("training loss")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_decision_boundary(model, xs, targets, path):
    xs = np.array(xs)
    x_min, x_max = xs[:, 0].min() - 0.5, xs[:, 0].max() + 0.5
    y_min, y_max = xs[:, 1].min() - 0.5, xs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([model([px, py]).data for px, py in grid]).reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, preds, levels=[-100, 0, 100], alpha=0.3, colors=["#5599ff", "#ff8855"])
    point_colors = ["#1f4eb0" if t > 0 else "#b03a1f" for t in targets]
    plt.scatter(xs[:, 0], xs[:, 1], c=point_colors, s=15, edgecolors="k", linewidths=0.3)
    plt.title("decision boundary")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    model, xs, targets, history = train()
    plot_loss(history, "assets/loss_curve.png")
    plot_decision_boundary(model, xs, targets, "assets/decision_boundary.png")
    print("saved plots to assets/")
