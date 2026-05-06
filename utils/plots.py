import matplotlib.pyplot as plt
import os
import numpy as np

def plot_confidence_hist(confidences, path):
    plt.hist(confidences, bins=20)
    plt.savefig(path)
    plt.close()

def plot_accuracy(acc, path):
    plt.bar(["accuracy"],[acc])
    plt.savefig(path)
    plt.close()
def plot_confidence_vs_correct(confidences, correct, path):
    plt.scatter(confidences, correct, alpha=0.5)
    plt.xlabel("Confidence")
    plt.ylabel("Correct (1) / Wrong (0)")
    plt.savefig(path)
    plt.close()

def plot_error_distribution(confidences, correct, path):
    wrong_conf = confidences[correct == 0]
    plt.hist(wrong_conf, bins=20)
    plt.title("Overconfident Error Distribution")
    plt.savefig(path)
    plt.close()

def plot_reliability_diagram(confidences, correct, path, bins=10):
    bin_edges = np.linspace(0,1,bins+1)
    bin_acc = []
    bin_conf = []

    for i in range(bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())

    plt.plot(bin_conf, bin_acc, marker='o')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.savefig(path)
    plt.close()