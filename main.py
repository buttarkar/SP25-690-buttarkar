import os
import torch
import numpy as np

from config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    CONFIDENCE_THRESHOLD,
    PLOT_DIR
)

from data.data_loader import get_loaders

from training.train import train_classifier

from training.detector_train import (
    train_mlp_detector,
    train_transformer_detector
)

from evaluation.evaluate import (
    collect_outputs,
    build_detection_dataset
)

from evaluation.detector_evaluate import (
    evaluate_detector
)

from utils.metrics import (
    compute_classification_metrics,
    compute_detection_metrics,
    compute_ece
)

from utils.plots import (
    plot_confidence_hist,
    plot_accuracy_comparison,
    plot_precision_recall_f1,
    plot_error_distribution,
    plot_reliability_diagram
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

train_loader, val_loader, test_loader = get_loaders(
    batch_size=BATCH_SIZE
)

classifier = train_classifier(
    train_loader,
    val_loader,
    device,
    epochs=EPOCHS,
    lr=LR
)

probabilities, predictions, labels = collect_outputs(
    classifier,
    test_loader,
    device
)

classification_metrics = compute_classification_metrics(
    predictions,
    labels
)

ece = compute_ece(
    probabilities,
    labels
)

features, detector_labels = build_detection_dataset(
    probabilities,
    predictions,
    labels,
    threshold=CONFIDENCE_THRESHOLD
)

mlp_model = train_mlp_detector(
    features,
    detector_labels,
    device
)

transformer_model = train_transformer_detector(
    features,
    detector_labels,
    device
)

mlp_preds, mlp_labels = evaluate_detector(
    mlp_model,
    features,
    detector_labels,
    device
)

transformer_preds, transformer_labels = evaluate_detector(
    transformer_model,
    features,
    detector_labels,
    device
)

mlp_metrics = compute_detection_metrics(
    mlp_preds,
    mlp_labels
)

transformer_metrics = compute_detection_metrics(
    transformer_preds,
    transformer_labels
)

threshold_predictions = detector_labels.clone()

baseline_metrics = compute_detection_metrics(
    threshold_predictions,
    detector_labels
)

confidences = probabilities.max(dim=1)[0].cpu().numpy()

correct = (
    predictions == labels
).cpu().numpy().astype(int)

detector_numpy = detector_labels.cpu().numpy()

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)

plot_confidence_hist(
    confidences,
    os.path.join(
        PLOT_DIR,
        "confidence_histogram.png"
    )
)

plot_error_distribution(
    confidences,
    detector_numpy,
    os.path.join(
        PLOT_DIR,
        "overconfidence_errors.png"
    )
)

plot_reliability_diagram(
    confidences,
    correct,
    os.path.join(
        PLOT_DIR,
        "reliability_diagram.png"
    )
)

plot_accuracy_comparison(
    {
        "CNN": classification_metrics["accuracy"],
        "MLP": mlp_metrics["accuracy"],
        "Transformer": transformer_metrics["accuracy"]
    },
    os.path.join(
        PLOT_DIR,
        "accuracy_comparison.png"
    )
)

plot_precision_recall_f1(
    {
        "Threshold": baseline_metrics,
        "MLP": mlp_metrics,
        "Transformer": transformer_metrics
    },
    os.path.join(
        PLOT_DIR,
        "precision_recall_f1.png"
    )
)

print("\nCNN Classification Results")
print(classification_metrics)

print("\nExpected Calibration Error")
print(ece)

print("\nThreshold Detection Results")
print(baseline_metrics)

print("\nMLP Detection Results")
print(mlp_metrics)

print("\nTransformer Detection Results")
print(transformer_metrics)

print("\nPlots saved in outputs/plots/")
