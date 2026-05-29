"""
Wczytywanie zbioru BRISC2025 (MRI guzow mozgu).

Obrazy oryginalnie 512x512 w skali szarosci (mode L). Skalujemy do IMG_SIZE
(domyslnie 64x64) zeby trening na CPU byl wykonalny.

Dwa tryby danych:
- klasyfikacja: ImageFolder po katalogach klas (glioma/meningioma/no_tumor/pituitary)
- GAN: plaski zbior wszystkich obrazow treningowych (bez etykiet)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

# ============================================================
# KONFIGURACJA
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "brisc2025")
CLS_ROOT = os.path.join(DATA_ROOT, "classification_task")

IMG_SIZE = 64

# Kolejnosc klas zgodna z alfabetyczna kolejnoscia ImageFolder
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Normalizacja do [-1, 1] (pasuje do Tanh w generatorze DCGAN)
_normalize = transforms.Normalize((0.5,), (0.5,))


def _base_transform(augment=False):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if augment:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), _normalize]
    return transforms.Compose(ops)


# ============================================================
# KLASYFIKACJA
# ============================================================
def classification_loaders(batch_size=64, num_workers=2, augment=True):
    """Zwraca (train_loader, test_loader) dla zadania klasyfikacji."""
    train_ds = datasets.ImageFolder(
        os.path.join(CLS_ROOT, "train"), transform=_base_transform(augment=augment)
    )
    test_ds = datasets.ImageFolder(
        os.path.join(CLS_ROOT, "test"), transform=_base_transform(augment=False)
    )
    # Walidacja, ze ImageFolder ma te same klasy w oczekiwanej kolejnosci
    assert train_ds.classes == CLASSES, f"Nieoczekiwane klasy: {train_ds.classes}"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


# ============================================================
# GAN
# ============================================================
class FlatImageDataset(Dataset):
    """Wszystkie obrazy .jpg pod podanym katalogiem, bez etykiet (dla GAN)."""

    def __init__(self, root, transform):
        self.transform = transform
        self.paths = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.paths.append(os.path.join(dirpath, fn))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img)


def gan_loader(batch_size=128, num_workers=2, only_class=None):
    """
    DataLoader dla treningu GAN. Domyslnie wszystkie obrazy treningowe
    z zadania klasyfikacji. only_class=<nazwa> ogranicza do jednej klasy.
    """
    root = os.path.join(CLS_ROOT, "train")
    if only_class is not None:
        root = os.path.join(root, only_class)
    ds = FlatImageDataset(root, transform=_base_transform(augment=True))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    return loader


if __name__ == "__main__":
    # Szybki sanity check
    tr, te = classification_loaders(batch_size=8, num_workers=0, augment=False)
    xb, yb = next(iter(tr))
    print("Klasyfikacja train batch:", xb.shape, "etykiety:", yb.tolist())
    print("Zakres pikseli:", float(xb.min()), "do", float(xb.max()))
    g = gan_loader(batch_size=8, num_workers=0)
    print("GAN dataset rozmiar:", len(g.dataset), "batch:", next(iter(g)).shape)
