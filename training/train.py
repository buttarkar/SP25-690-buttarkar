import torch
import torch.nn.functional as F
from models.cnn import CNN

def train_classifier(loader, device):
    model = CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model
