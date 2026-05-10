import torch
import torch.nn.functional as F

def collect_outputs(model, loader, device):
    model.eval()

    probabilities = []
    predictions = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            probs = F.softmax(outputs, dim=1)

            preds = probs.argmax(dim=1)

            probabilities.append(probs.cpu())
            predictions.append(preds.cpu())
            labels_list.append(labels.cpu())

    probabilities = torch.cat(probabilities)
    predictions = torch.cat(predictions)
    labels = torch.cat(labels_list)

    return probabilities, predictions, labels


def build_detection_dataset(probabilities, predictions, labels, threshold=0.8):
    confidences = probabilities.max(dim=1)[0]

    correct = predictions.eq(labels)

    overconfident_errors = (
        (confidences >= threshold) &
        (~correct)
    ).long()

    return probabilities, overconfident_errors
