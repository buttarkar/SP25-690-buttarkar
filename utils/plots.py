import matplotlib.pyplot as plt
import numpy as np

def plot_confidence_hist(confidences, path):
    plt.figure(figsize=(6, 5))

    plt.hist(confidences, bins=20)

    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")

    plt.savefig(path)

    plt.close()


def plot_accuracy_comparison(results, path):
    plt.figure(figsize=(6, 5))

    names = list(results.keys())
    values = list(results.values())

    plt.bar(names, values)

    plt.ylabel("Score")
    plt.title("Accuracy Comparison")

    plt.savefig(path)

    plt.close()


def plot_precision_recall_f1(metrics_dict, path):
    plt.figure(figsize=(7, 5))

    models = list(metrics_dict.keys())

    precision = [
        metrics_dict[m]["precision"]
        for m in models
    ]

    recall = [
        metrics_dict[m]["recall"]
        for m in models
    ]

    f1 = [
        metrics_dict[m]["f1_score"]
        for m in models
    ]

    x = np.arange(len(models))

    width = 0.25

    plt.bar(x - width, precision, width)
    plt.bar(x, recall, width)
    plt.bar(x + width, f1, width)

    plt.xticks(x, models)

    plt.ylabel("Score")
    plt.title("Precision Recall F1 Comparison")

    plt.savefig(path)

    plt.close()


def plot_error_distribution(confidences, labels, path):
    plt.figure(figsize=(6, 5))

    wrong_conf = confidences[labels == 1]

    plt.hist(wrong_conf, bins=20)

    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Overconfident Errors")

    plt.savefig(path)

    plt.close()


def plot_reliability_diagram(confidences, correct, path, bins=10):
    plt.figure(figsize=(6, 5))

    bin_edges = np.linspace(0, 1, bins + 1)

    accuracies = []
    avg_confidences = []

    for i in range(bins):
        mask = (
            (confidences > bin_edges[i]) &
            (confidences <= bin_edges[i + 1])
        )

        if np.sum(mask) > 0:
            accuracies.append(
                np.mean(correct[mask])
            )

            avg_confidences.append(
                np.mean(confidences[mask])
            )

    plt.plot(
        avg_confidences,
        accuracies,
        marker="o"
    )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")

    plt.savefig(path)

    plt.close()
