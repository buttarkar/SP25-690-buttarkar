import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()

        self.embedding = nn.Linear(1, 32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)

        x = self.embedding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.classifier(x)

        return x.squeeze(1)
