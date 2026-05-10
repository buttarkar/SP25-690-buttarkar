import torch
import torch.nn.functional as F

from models.mlp import MLP
from models.transformer import TransformerModel

def train_mlp_detector(
    features,
    labels,
    device,
    epochs=10,
    lr=1e-3
):
    model = MLP().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    features = features.to(device)
    labels = labels.float().to(device)

    model.train()

    for epoch in range(epochs):
        outputs = model(features)

        loss = F.binary_cross_entropy_with_logits(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        preds = (
            torch.sigmoid(outputs) >= 0.5
        ).long()

        accuracy = (
            preds == labels.long()
        ).float().mean().item()

        print(
            f"MLP Epoch {epoch + 1} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

    return model


def train_transformer_detector(
    features,
    labels,
    device,
    epochs=10,
    lr=1e-3
):
    model = TransformerModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    features = features.to(device)
    labels = labels.float().to(device)

    model.train()

    for epoch in range(epochs):
        outputs = model(features)

        loss = F.binary_cross_entropy_with_logits(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        preds = (
            torch.sigmoid(outputs) >= 0.5
        ).long()

        accuracy = (
            preds == labels.long()
        ).float().mean().item()

        print(
            f"Transformer Epoch {epoch + 1} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

    return model
