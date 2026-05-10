import torch

def evaluate_detector(
    model,
    features,
    labels,
    device
):
    model.eval()

    features = features.to(device)

    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(features)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).long()

    return predictions.cpu(), labels.cpu()
