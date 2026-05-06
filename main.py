import torch
import os
import numpy as np

from data.data_loader import get_loaders
from training.train import train_classifier
from evaluation.evaluate import collect_outputs
from utils.metrics import compute_metrics, compute_ece
from utils.plots import (
    plot_confidence_hist,
    plot_accuracy,
    plot_confidence_vs_correct,
    plot_error_distribution,
    plot_reliability_diagram
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_loaders()

model = train_classifier(train_loader, device)

probs, preds, labels = collect_outputs(model, test_loader, device)

acc = compute_metrics(preds, labels)
ece = compute_ece(probs, labels)

confidences = probs.max(1)[0].cpu().numpy()
correct = (preds == labels).cpu().numpy().astype(int)

os.makedirs("outputs/plots", exist_ok=True)

plot_confidence_hist(confidences, "outputs/plots/confidence_hist.png")
plot_accuracy(acc, "outputs/plots/accuracy.png")
plot_confidence_vs_correct(confidences, correct, "outputs/plots/conf_vs_correct.png")
plot_error_distribution(confidences, correct, "outputs/plots/error_distribution.png")
plot_reliability_diagram(confidences, correct, "outputs/plots/reliability.png")

print("Accuracy:", acc)
print("ECE:", ece)
print("All plots saved in outputs/plots/")