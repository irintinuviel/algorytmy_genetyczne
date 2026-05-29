"""
Maly klasyfikator CNN dla 4 klas guzow (glioma/meningioma/no_tumor/pituitary).

Uzywany pozniej jako funkcja oceny dla algorytmu genetycznego w przestrzeni
latentnej: GA szuka takiego z, ze G(z) jest klasyfikowane jako zadana klasa
z wysokim prawdopodobienstwem.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import classification_loaders, CLASSES, IMG_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
WEIGHTS_PATH = os.path.join(RESULTS_DIR, "classifier.pt")

NUM_CLASSES = len(CLASSES)


class TumorCNN(nn.Module):
    """Prosty CNN: 3 bloki konwolucyjne + klasyfikator. Wejscie 1x64x64."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 16x16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                       # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_classifier(epochs=8, lr=1e-3, batch_size=64, device="cpu", seed=42):
    torch.manual_seed(seed)
    train_loader, test_loader = classification_loaders(batch_size=batch_size)

    model = TumorCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        acc = evaluate(model, test_loader, device)
        print(f"[classifier] epoka {epoch}/{epochs}  loss={train_loss:.4f}  test_acc={acc:.4f}")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"[classifier] zapisano wagi -> {WEIGHTS_PATH}")
    return model


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total


def load_classifier(device="cpu"):
    """Wczytuje wytrenowany klasyfikator do uzytku przez GA."""
    model = TumorCNN().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    train_classifier()
