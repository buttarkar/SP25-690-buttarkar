import torch
import torch.nn.functional as F

def collect_outputs(model, loader, device):
    probs = []
    labels = []
    preds = []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            p = F.softmax(out, dim=1).cpu()
            probs.append(p)
            preds.append(p.argmax(dim=1))
            labels.append(y)

    probs = torch.cat(probs)
    preds = torch.cat(preds)
    labels = torch.cat(labels)

    return probs, preds, labels
