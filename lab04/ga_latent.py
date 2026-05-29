"""
Algorytm genetyczny w przestrzeni latentnej GAN-a.

Pomysl (polaczenie GA + GAN): generator jest zamrozony. Chromosomem jest
wektor latentny z (R^LATENT_DIM). Funkcja oceny przepuszcza z przez generator,
a powstaly obraz przez klasyfikator guzow. GA szuka takiego z, ze G(z) jest
klasyfikowane jako ZADANA klasa z jak najwyzszym prawdopodobienstwem.

Dzieki temu z bezwarunkowego GAN-a uzyskujemy sterowane generowanie -- ewolucja
"wycina" w przestrzeni latentnej obszar odpowiadajacy wybranemu typowi guza.

Operatory (selekcja/krzyzowanie/mutacja/elityzm) sa w stylu run_ga_continuous
z main2.py; GA minimalizuje funkcje celu, wiec celem jest -prawdopodobienstwo
klasy docelowej. Ocena populacji jest wsadowa (jeden forward przez G i C).
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gan import load_generator, LATENT_DIM
from classifier import load_classifier
from data import CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# OPERATORY GA (styl main2.py)
# ============================================================
def tournament_selection(pop, fitness, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pop[best].copy()


def roulette_selection(pop, fitness):
    inv = 1.0 / (fitness - fitness.min() + 1e-9)
    probs = inv / np.sum(inv)
    idx = np.random.choice(len(pop), p=probs)
    return pop[idx].copy()


def crossover_arithmetic(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2, alpha * p2 + (1 - alpha) * p1


def crossover_one_point(p1, p2):
    point = np.random.randint(1, len(p1))
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def mutation_gaussian(x, bounds, rate=0.1, sigma=0.3):
    low, high = bounds
    x = x.copy()
    mask = np.random.rand(len(x)) < rate
    x[mask] += np.random.normal(0, sigma, size=mask.sum())
    return np.clip(x, low, high)


# ============================================================
# FUNKCJA OCENY (wsadowa, przez G i klasyfikator)
# ============================================================
class LatentFitness:
    """Ocena populacji wektorow z: -P(klasa docelowa) dla G(z)."""

    def __init__(self, generator, classifier, target_class, device="cpu"):
        self.G = generator
        self.C = classifier
        self.target = target_class
        self.device = device

    @torch.no_grad()
    def __call__(self, pop):
        z = torch.from_numpy(np.asarray(pop, dtype=np.float32)).to(self.device)
        imgs = self.G(z)
        probs = F.softmax(self.C(imgs), dim=1)[:, self.target]
        # GA minimalizuje -> zwracamy ujemne prawdopodobienstwo
        return (-probs).cpu().numpy()


# ============================================================
# PETLA GA
# ============================================================
def run_ga_latent(
    fitness_fn,
    dim=LATENT_DIM,
    num=60,
    iterations=80,
    bounds=(-3.0, 3.0),
    selection="tournament",
    crossover="arithmetic",
    tournament_k=3,
    pc=0.8,
    pm=0.1,
    sigma=0.3,
    elitism=2,
    seed=42,
):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    low, high = bounds

    # Inicjalizacja z rozkladu normalnego (naturalny rozklad z dla GAN), z przycieciem
    pop = np.clip(rng.standard_normal((num, dim)), low, high)
    fitness = fitness_fn(pop)

    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_cost = fitness[best_idx]

    history_best = [best_cost]
    history_mean = [float(fitness.mean())]

    for it in range(1, iterations + 1):
        elite_idx = np.argsort(fitness)[:elitism]
        elites = [pop[i].copy() for i in elite_idx]
        new_pop = []

        while len(new_pop) < num - elitism:
            if selection == "roulette":
                p1 = roulette_selection(pop, fitness)
                p2 = roulette_selection(pop, fitness)
            else:
                p1 = tournament_selection(pop, fitness, tournament_k)
                p2 = tournament_selection(pop, fitness, tournament_k)

            if np.random.rand() < pc:
                if crossover == "one_point":
                    c1, c2 = crossover_one_point(p1, p2)
                else:
                    c1, c2 = crossover_arithmetic(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutation_gaussian(c1, bounds, rate=pm, sigma=sigma)
            c2 = mutation_gaussian(c2, bounds, rate=pm, sigma=sigma)
            new_pop.extend([c1, c2])

        pop = np.array(elites + new_pop[: num - elitism])
        fitness = fitness_fn(pop)

        idx = np.argmin(fitness)
        if fitness[idx] < best_cost:
            best_cost = fitness[idx]
            best = pop[idx].copy()

        history_best.append(best_cost)
        history_mean.append(float(fitness.mean()))
        if it % 10 == 0 or it == iterations:
            print(f"[ga] iter {it}/{iterations}  best P(klasa)={-best_cost:.4f}  "
                  f"mean P={-np.mean(fitness):.4f}")

    return best, -best_cost, history_best, history_mean


# ============================================================
# URUCHOMIENIE: ewolucja z dla zadanej klasy + zapis wynikow
# ============================================================
def evolve_class(target_name, device="cpu", **ga_kwargs):
    assert target_name in CLASSES, f"Nieznana klasa: {target_name}"
    target = CLASSES.index(target_name)

    G = load_generator(device)
    C = load_classifier(device)
    fit = LatentFitness(G, C, target, device=device)

    print(f"[ga] ewolucja wektora latentnego dla klasy '{target_name}' (idx {target})")
    best_z, best_p, hist_best, hist_mean = run_ga_latent(fit, **ga_kwargs)

    # Obraz najlepszego osobnika
    with torch.no_grad():
        z = torch.from_numpy(best_z.astype(np.float32)).unsqueeze(0).to(device)
        img = G(z).cpu()
    img_path = os.path.join(RESULTS_DIR, f"ga_best_{target_name}.png")
    vutils.save_image(img, img_path, normalize=True)

    # Porownanie: losowe z vs wyewoluowane z
    with torch.no_grad():
        rand_z = torch.randn(8, LATENT_DIM, device=device)
        rand_imgs = G(rand_z).cpu()
        rand_probs = F.softmax(C(G(rand_z)), dim=1)[:, target].mean().item()
    cmp_path = os.path.join(RESULTS_DIR, f"ga_compare_{target_name}.png")
    grid = torch.cat([rand_imgs, img.repeat(8, 1, 1, 1)], dim=0)
    vutils.save_image(grid, cmp_path, normalize=True, nrow=8)

    # Krzywa zbieznosci (jako prawdopodobienstwo, czyli -fitness)
    plt.figure(figsize=(7, 4))
    plt.plot([-c for c in hist_best], label="najlepszy P(klasa)")
    plt.plot([-m for m in hist_mean], label="srednie P(klasa)")
    plt.xlabel("pokolenie")
    plt.ylabel(f"P(klasa = {target_name})")
    plt.title(f"Zbieznosc GA w przestrzeni latentnej -> {target_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    curve_path = os.path.join(RESULTS_DIR, f"ga_curve_{target_name}.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=120)
    plt.close()

    print(f"[ga] wyewoluowane P(klasa={target_name}) = {best_p:.4f} "
          f"(losowe z: {rand_probs:.4f})")
    print(f"[ga] zapisano: {img_path}\n          {cmp_path}\n          {curve_path}")
    return best_z, best_p


if __name__ == "__main__":
    evolve_class("glioma")
