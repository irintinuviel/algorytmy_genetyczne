"""
Lab04 -- GAN + algorytm genetyczny na obrazach MRI guzow mozgu (BRISC2025).

Potok sklada sie z trzech etapow:
  1. trening klasyfikatora CNN (4 klasy guzow)
  2. trening bezwarunkowego DCGAN na obrazach MRI
  3. algorytm genetyczny w przestrzeni latentnej generatora -- szuka wektora z,
     ktorego obraz G(z) jest klasyfikowany jako zadana klasa z najwyzszym
     prawdopodobienstwem (sterowane generowanie z bezwarunkowego GAN-a).

Uzytkownik steruje wszystkim przez ponizsza konfiguracje (jak w main2.py).
Wagi i wyniki laduja w lab04/results/.
"""

import torch

import classifier as clf
import gan
from ga_latent import evolve_class
from data import CLASSES

# ============================================================
# KONFIGURACJA
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Etapy potoku (mozna wlaczac/wylaczac niezaleznie -- wagi sa cache'owane)
TRAIN_CLASSIFIER = True
TRAIN_GAN = True
RUN_GA = True

# Klasa docelowa dla ewolucji w przestrzeni latentnej
TARGET_CLASS = "glioma"   # jedna z CLASSES

# Hiperparametry treningu (skromne wartosci dostosowane do CPU)
CLASSIFIER_EPOCHS = 8
GAN_EPOCHS = 30

# Hiperparametry GA (przestrzen latentna)
GA_KWARGS = dict(
    num=60,
    iterations=80,
    bounds=(-3.0, 3.0),
    selection="tournament",
    crossover="arithmetic",
    pc=0.8,
    pm=0.1,
    sigma=0.3,
    elitism=2,
    seed=SEED,
)

# Wagi czlonu realizmu w ocenie GA (przeciw "oszukiwaniu" klasyfikatora):
#   GA_W_REAL  -- nagroda za realizm wg dyskryminatora (wymaga discriminator.pt)
#   GA_W_PRIOR -- kara za odejscie wektora z od rozkladu N(0,1)
GA_W_REAL = 1.5
GA_W_PRIOR = 0.05


def main():
    print(f"=== Lab04: GAN + GA (urzadzenie: {DEVICE}) ===")
    assert TARGET_CLASS in CLASSES, f"TARGET_CLASS musi byc jedna z {CLASSES}"

    if TRAIN_CLASSIFIER:
        print("\n--- Etap 1: trening klasyfikatora ---")
        clf.train_classifier(epochs=CLASSIFIER_EPOCHS, device=DEVICE, seed=SEED)

    if TRAIN_GAN:
        print("\n--- Etap 2: trening DCGAN ---")
        gan.train_gan(epochs=GAN_EPOCHS, device=DEVICE, seed=SEED)

    if RUN_GA:
        print("\n--- Etap 3: algorytm genetyczny w przestrzeni latentnej ---")
        evolve_class(TARGET_CLASS, device=DEVICE,
                     w_real=GA_W_REAL, w_prior=GA_W_PRIOR, **GA_KWARGS)

    print("\n=== Gotowe. Wyniki w lab04/results/ ===")


if __name__ == "__main__":
    main()
