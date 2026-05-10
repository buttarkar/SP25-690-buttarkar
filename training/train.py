import torch
import torch.nn.functional as F
from models.cnn import CNN

def train_classifier(train_loader, val_loader, device, epochs=5, lr=1e-3):
    model = CNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(dim=1)

            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "outputs/best_cnn.pth"
            )

        print(
            f"Epoch {epoch + 1} | "
            f"Train Accuracy: {train_acc:.4f} | "
            f"Validation Accuracy: {val_acc:.4f}"
        )

    model.load_state_dict(
        torch.load("outputs/best_cnn.pth")
    )

    return model
