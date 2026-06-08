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

from gan import load_generator, load_discriminator, LATENT_DIM
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
    """Ocena populacji wektorow z dla G(z), z czlonem realizmu.

    Koszt (GA minimalizuje):
        -P(klasa docelowa)             nagroda za zgodnosc z klasa
        -w_real * sigmoid(D(G(z)))     nagroda za realizm wg dyskryminatora
        +w_prior * mean(z^2)           kara za odejscie od rozkladu N(0,1)

    Bez czlonu realizmu GA "oszukuje" klasyfikator: znajduje z dajace P=1.0,
    ale obraz to zdegenerowana plama. Czlon D + prior trzymaja z w obszarze,
    z ktorego generator robi realistyczne obrazy.
    """

    def __init__(self, generator, classifier, target_class, device="cpu",
                 discriminator=None, w_real=0.0, w_prior=0.0):
        self.G = generator
        self.C = classifier
        self.D = discriminator
        self.target = target_class
        self.device = device
        self.w_real = w_real
        self.w_prior = w_prior

    @torch.no_grad()
    def __call__(self, pop):
        z = torch.from_numpy(np.asarray(pop, dtype=np.float32)).to(self.device)
        imgs = self.G(z)
        probs = F.softmax(self.C(imgs), dim=1)[:, self.target]
        cost = -probs  # chcemy wysokie P(klasa)

        if self.D is not None and self.w_real > 0:
            real = torch.sigmoid(self.D(imgs))  # ~1 = realistyczny wg D
            cost = cost - self.w_real * real

        if self.w_prior > 0:
            prior = (z.view(z.size(0), -1) ** 2).mean(dim=1)  # ~1 dla N(0,1)
            cost = cost + self.w_prior * prior

        # GA minimalizuje -> zwracamy koszt
        return cost.cpu().numpy()


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
            print(f"[ga] iter {it}/{iterations}  best koszt={best_cost:.4f}  "
                  f"mean koszt={np.mean(fitness):.4f}")

    return best, best_cost, history_best, history_mean, pop, fitness


# ============================================================
# URUCHOMIENIE: ewolucja z dla zadanej klasy + zapis wynikow
# ============================================================
def _topk_distinct(pop, fitness, k=8, min_dist=1e-3):
    """k najlepszych (najnizszy koszt) wzajemnie roznych osobnikow.

    Po zbieznosci populacja bywa niemal identyczna; bierzemy zachlannie
    kolejnych najlepszych, ktorzy roznia sie od juz wybranych o min_dist.
    Jesli roznych jest mniej niz k, dobieramy najlepszych pozostalych.
    """
    order = np.argsort(fitness)
    chosen = []
    for i in order:
        cand = pop[i]
        if all(np.linalg.norm(cand - pop[j]) > min_dist for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    for i in order:  # uzupelnienie, jesli za malo roznych
        if len(chosen) == k:
            break
        if i not in chosen:
            chosen.append(i)
    return pop[chosen[:k]]


def evolve_class(target_name, device="cpu", w_real=0.5, w_prior=0.05, **ga_kwargs):
    assert target_name in CLASSES, f"Nieznana klasa: {target_name}"
    target = CLASSES.index(target_name)

    G = load_generator(device)
    C = load_classifier(device)
    D = load_discriminator(device)
    if D is None and w_real > 0:
        print("[ga] UWAGA: brak discriminator.pt -- czlon realizmu (D) wylaczony. "
              "Przetrenuj GAN (TRAIN_GAN=True), aby zapisac dyskryminator.")
    fit = LatentFitness(G, C, target, device=device,
                        discriminator=D, w_real=w_real, w_prior=w_prior)

    print(f"[ga] ewolucja wektora latentnego dla klasy '{target_name}' (idx {target})")
    best_z, best_cost, hist_best, hist_mean, final_pop, final_fit = run_ga_latent(
        fit, **ga_kwargs)

    # Obraz najlepszego osobnika + jego prawdziwe P(klasa) wg klasyfikatora
    with torch.no_grad():
        z = torch.from_numpy(best_z.astype(np.float32)).unsqueeze(0).to(device)
        img = G(z).cpu()
        best_p = F.softmax(C(G(z)), dim=1)[0, target].item()
    img_path = os.path.join(RESULTS_DIR, f"ga_best_{target_name}.png")
    vutils.save_image(img, img_path, normalize=True)

    # Porownanie: losowe z (gora) vs 8 NAJLEPSZYCH ROZNYCH osobnikow (dol).
    # Wczesniej dol byl jednym obrazem powielonym 8x -- stad wrazenie, ze
    # "wszystkie sa takie same". Teraz pokazujemy faktyczna roznorodnosc
    # koncowej populacji.
    top_z = _topk_distinct(final_pop, final_fit, k=8)
    with torch.no_grad():
        rand_z = torch.randn(8, LATENT_DIM, device=device)
        rand_imgs = G(rand_z).cpu()
        rand_probs = F.softmax(C(G(rand_z)), dim=1)[:, target].mean().item()
        evo_imgs = G(torch.from_numpy(top_z.astype(np.float32)).to(device)).cpu()
    cmp_path = os.path.join(RESULTS_DIR, f"ga_compare_{target_name}.png")
    grid = torch.cat([rand_imgs, evo_imgs], dim=0)
    vutils.save_image(grid, cmp_path, normalize=True, nrow=8)

    # Krzywa zbieznosci celu GA (-koszt: P(klasa) + realizm - kara prior)
    plt.figure(figsize=(7, 4))
    plt.plot([-c for c in hist_best], label="najlepszy (-koszt)")
    plt.plot([-m for m in hist_mean], label="sredni (-koszt)")
    plt.xlabel("pokolenie")
    plt.ylabel("cel GA  (-koszt)")
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
