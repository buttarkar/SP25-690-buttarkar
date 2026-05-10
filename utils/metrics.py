import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def compute_classification_metrics(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy
    }


def compute_detection_metrics(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    precision = precision_score(
        labels,
        preds,
        zero_division=0
    )

    recall = recall_score(
        labels,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        preds,
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        preds
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def compute_ece(probabilities, labels, bins=10):
    confidences = probabilities.max(dim=1)[0]

    predictions = probabilities.argmax(dim=1)

    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)

    bin_boundaries = torch.linspace(
        0,
        1,
        bins + 1
    )

    for i in range(bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        mask = (
            (confidences > lower) &
            (confidences <= upper)
        )

        if mask.sum() > 0:
            accuracy = accuracies[mask].float().mean()

            confidence = confidences[mask].mean()

            ece += torch.abs(
                confidence - accuracy
            ) * mask.float().mean()

    return ece.item()
