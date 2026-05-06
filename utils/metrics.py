import numpy as np

def compute_metrics(preds, labels):
    acc = (preds==labels).float().mean().item()
    return acc

def compute_ece(probs, labels, bins=10):
    confidences = probs.max(1)[0]
    predictions = probs.argmax(1)
    accuracies = predictions.eq(labels)

    ece = 0
    bin_boundaries = np.linspace(0,1,bins+1)

    for i in range(bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (mask.float().mean() * abs(acc-conf))

    return ece.item()
