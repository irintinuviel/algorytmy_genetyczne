import os
import csv
import time
import numpy as np
import matplotlib

# Fallback: backend interaktywny lokalnie, zapisowy dla środowisk headless
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import PillowWriter


# ============================================================
# KONFIGURACJA
# ============================================================
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

GLOBAL_SEED = 42

DEMO_MODE_CONTINUOUS = False
DEMO_MODE_TSP = True

BENCHMARK_MODE_CONTINUOUS = False
BENCHMARK_MODE_TSP = False

GA_VARIANT_BENCHMARK_CONTINUOUS = False
GA_VARIANT_BENCHMARK_TSP = False

CONTINUOUS_RUNS = 20
TSP_RUNS = 20

AGENT_VALUES = [10, 20, 50, 100]
ITERATION_VALUES = [50, 100, 200]
DIM_VALUES = [2, 5, 10]
CITY_VALUES = [10, 20, 50]

GA_SELECTION_VALUES = ["tournament", "roulette"]
GA_PC_VALUES = [0.6, 0.8, 1.0]
GA_PM_VALUES = [0.01, 0.05, 0.1]


# ============================================================
# FUNKCJE CELU
# ============================================================
def rastrigin(x):
    x = np.array(x, dtype=float)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def schwefel(x):
    x = np.array(x, dtype=float)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def ackley(x):
    x = np.array(x, dtype=float)
    n = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e


def eggholder(x):
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(abs(x1 / 2 + x2 + 47))) - x1 * np.sin(
        np.sqrt(abs(x1 - (x2 + 47)))
    )


def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


FUNCTIONS = {
    "rastrigin": {
        "func": rastrigin,
        "bounds": (-5.12, 5.12),
        "dims": [2, 5, 10],
    },
    "schwefel": {
        "func": schwefel,
        "bounds": (-500, 500),
        "dims": [2, 5, 10],
    },
    "ackley": {
        "func": ackley,
        "bounds": (-32.768, 32.768),
        "dims": [2, 5, 10],
    },
    "eggholder": {
        "func": eggholder,
        "bounds": (-512, 512),
        "dims": [2],
    },
    "himmelblau": {
        "func": himmelblau,
        "bounds": (-5, 5),
        "dims": [2],
    },
}


# ============================================================
# WSPÓLNE NARZĘDZIA
# ============================================================
def save_benchmark_csv(rows, filename):
    if not rows:
        return

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_surface(func, range_min, range_max, points=80):
    x = np.linspace(range_min, range_max, points)
    y = np.linspace(range_min, range_max, points)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X, dtype=float)
    for i in range(points):
        for j in range(points):
            Z[i, j] = func([X[i, j], Y[i, j]])

    return X, Y, Z


def make_history_state(pos, best, best_cost, func):
    return {
        "pos": pos.copy(),
        "best": best.copy(),
        "best_cost": float(best_cost),
        "z_pos": np.array([func(p) for p in pos]),
        "z_best": float(best_cost),
    }


def save_gif(history, X, Y, Z, filename="opt.gif", fps=10):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, alpha=0.3, cmap="viridis")

    agents_scatter = ax.scatter([], [], [])
    best_scatter = ax.scatter([], [], [], marker="x", s=200)

    writer = PillowWriter(fps=fps)

    with writer.saving(fig, filename, dpi=100):
        for i, state in enumerate(history):
            pos = state["pos"]
            best = state["best"]
            z_agents = state["z_pos"]
            z_best = state["z_best"]

            agents_scatter._offsets3d = (pos[:, 0], pos[:, 1], z_agents)
            best_scatter._offsets3d = ([best[0]], [best[1]], [z_best])

            ax.set_title(f"Iteracja {i}  Best={z_best:.4f}")
            writer.grab_frame()

    plt.close(fig)


def show_interactive_plot(history, X, Y, Z, range_min, range_max):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, alpha=0.3, cmap="viridis")

    agents_scatter = ax.scatter([], [], [])
    best_scatter = ax.scatter([], [], [], marker="x", s=200)

    ax.set_xlim(range_min, range_max)
    ax.set_ylim(range_min, range_max)

    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, "iter", 0, len(history) - 1, valinit=0, valstep=1)

    current_iter = 0
    paused = True

    def update_plot(i):
        state = history[int(i)]
        pos = state["pos"]
        best = state["best"]
        z_agents = state["z_pos"]
        z_best = state["z_best"]

        agents_scatter._offsets3d = (pos[:, 0], pos[:, 1], z_agents)
        best_scatter._offsets3d = ([best[0]], [best[1]], [z_best])

        ax.set_title(f"Iteracja {int(i)}  Best={z_best:.4f}")
        fig.canvas.draw_idle()

    def slider_update(val):
        nonlocal current_iter
        current_iter = int(val)
        update_plot(current_iter)

    slider.on_changed(slider_update)

    def on_key(event):
        nonlocal paused
        if event.key == " ":
            paused = not paused
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def step():
        nonlocal current_iter
        if not paused and current_iter < len(history) - 1:
            current_iter += 1
            slider.set_val(current_iter)

    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(step)
    timer.start()

    update_plot(0)
    plt.show()


def plot_convergence(history, algorithm_name, function_name, save_path=None):
    best_values = [state["best_cost"] for state in history]

    plt.figure(figsize=(8, 5))
    plt.plot(best_values, label=f"{algorithm_name}")
    plt.xlabel("Iteracja")
    plt.ylabel("Najlepsza wartość funkcji")
    plt.title(f"Zbieżność: {algorithm_name} / {function_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_multiple_convergences(results, function_name, save_path=None):
    plt.figure(figsize=(8, 5))
    for algorithm_name, result in results.items():
        best_values = [state["best_cost"] for state in result["history"]]
        plt.plot(best_values, label=algorithm_name)

    plt.xlabel("Iteracja")
    plt.ylabel("Najlepsza wartość funkcji")
    plt.title(f"Porównanie zbieżności / {function_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_ga_convergence(best_values, mean_values, title, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(best_values, label="best")
    plt.plot(mean_values, label="mean population")
    plt.xlabel("Iteracja")
    plt.ylabel("Wartość")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_tsp_route(cities, route, title="TSP route", save_path=None):
    ordered = cities[route]
    closed = np.vstack([ordered, ordered[0]])

    plt.figure(figsize=(7, 7))
    plt.scatter(cities[:, 0], cities[:, 1], s=40)
    plt.plot(closed[:, 0], closed[:, 1], linewidth=1.5)

    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=8)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_tsp_convergence(histories, title="TSP convergence", save_path=None):
    plt.figure(figsize=(8, 5))
    for name, values in histories.items():
        plt.plot(values, label=name)

    plt.xlabel("Iteracja")
    plt.ylabel("Długość najlepszej trasy")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def save_tsp_gif(cities, route_history, filename="tsp.gif", fps=10):
    fig, ax = plt.subplots(figsize=(7, 7))
    writer = PillowWriter(fps=fps)

    with writer.saving(fig, filename, dpi=100):
        for i, route in enumerate(route_history):
            ax.clear()
            ordered = cities[route]
            closed = np.vstack([ordered, ordered[0]])

            ax.scatter(cities[:, 0], cities[:, 1], s=40)
            ax.plot(closed[:, 0], closed[:, 1], linewidth=1.5)

            for j, (x, y) in enumerate(cities):
                ax.text(x, y, str(j), fontsize=8)

            cost = route_length(route, cities)
            ax.set_title(f"Iteracja {i}, best={cost:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.tight_layout()
            writer.grab_frame()

    plt.close(fig)


# ============================================================
# ALGORYTMY CIĄGŁE
# ============================================================
def run_pso(
    func,
    num=50,
    iterations=100,
    dim=2,
    bounds=(-5.12, 5.12),
    w=0.7,
    c1=1.5,
    c2=1.5,
):
    range_min, range_max = bounds

    positions = np.random.uniform(range_min, range_max, (num, dim))
    velocities = np.zeros((num, dim))

    pbest_positions = positions.copy()
    pbest_cost = np.array([func(p) for p in positions])

    best_index = np.argmin(pbest_cost)
    gbest_position = pbest_positions[best_index].copy()
    gbest_cost = pbest_cost[best_index]

    history = []

    for _ in range(iterations):
        for i in range(num):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], range_min, range_max)

            cost = func(positions[i])

            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest_positions[i] = positions[i].copy()

        best_index = np.argmin(pbest_cost)

        if pbest_cost[best_index] < gbest_cost:
            gbest_cost = pbest_cost[best_index]
            gbest_position = pbest_positions[best_index].copy()

        history.append(make_history_state(positions, gbest_position, gbest_cost, func))

    return {
        "history": history,
        "best_position": gbest_position.copy(),
        "best_cost": float(gbest_cost),
    }


def run_random_search(func, num=50, iterations=100, dim=2, bounds=(-5.12, 5.12)):
    range_min, range_max = bounds

    positions = np.random.uniform(range_min, range_max, (num, dim))
    costs = np.array([func(p) for p in positions])

    best_idx = np.argmin(costs)
    best = positions[best_idx].copy()
    best_cost = costs[best_idx]

    history = []

    for _ in range(iterations):
        positions = np.random.uniform(range_min, range_max, (num, dim))
        costs = np.array([func(p) for p in positions])

        idx = np.argmin(costs)
        if costs[idx] < best_cost:
            best_cost = costs[idx]
            best = positions[idx].copy()

        history.append(make_history_state(positions, best, best_cost, func))

    return {
        "history": history,
        "best_position": best.copy(),
        "best_cost": float(best_cost),
    }


def run_gwo(func, num=50, iterations=100, dim=2, bounds=(-5.12, 5.12)):
    range_min, range_max = bounds

    wolves = np.random.uniform(range_min, range_max, (num, dim))
    history = []

    alpha_pos = np.zeros(dim)
    alpha_score = np.inf

    beta_pos = np.zeros(dim)
    beta_score = np.inf

    delta_pos = np.zeros(dim)
    delta_score = np.inf

    for t in range(iterations):
        scores = np.array([func(w) for w in wolves])
        sorted_idx = np.argsort(scores)

        alpha_pos = wolves[sorted_idx[0]].copy()
        alpha_score = scores[sorted_idx[0]]

        if num > 1:
            beta_pos = wolves[sorted_idx[1]].copy()
            beta_score = scores[sorted_idx[1]]
        else:
            beta_pos = alpha_pos.copy()
            beta_score = alpha_score

        if num > 2:
            delta_pos = wolves[sorted_idx[2]].copy()
            delta_score = scores[sorted_idx[2]]
        else:
            delta_pos = beta_pos.copy()
            delta_score = beta_score

        a = 2 - 2 * (t / max(1, iterations - 1))

        for i in range(num):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - wolves[i])
            X1 = alpha_pos - A1 * D_alpha

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - wolves[i])
            X2 = beta_pos - A2 * D_beta

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - wolves[i])
            X3 = delta_pos - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3.0
            wolves[i] = np.clip(wolves[i], range_min, range_max)

        history.append(make_history_state(wolves, alpha_pos, alpha_score, func))

    return {
        "history": history,
        "best_position": alpha_pos.copy(),
        "best_cost": float(alpha_score),
    }


def run_bso(
    func,
    num=50,
    iterations=100,
    dim=2,
    bounds=(-5.12, 5.12),
    k_clusters=5,
    p_replace=0.1,
):
    range_min, range_max = bounds

    positions = np.random.uniform(range_min, range_max, (num, dim))
    fitness = np.array([func(p) for p in positions])

    best_idx = np.argmin(fitness)
    best_position = positions[best_idx].copy()
    best_cost = fitness[best_idx]

    history = []

    for _ in range(iterations):
        cluster_ids = np.random.randint(0, k_clusters, num)

        clusters = []
        for k in range(k_clusters):
            members = positions[cluster_ids == k]
            if len(members) > 0:
                center = np.mean(members, axis=0)
            else:
                center = np.random.uniform(range_min, range_max, dim)
            clusters.append(center)

        clusters = np.array(clusters)

        for i in range(num):
            if np.random.rand() < p_replace:
                new_pos = np.random.uniform(range_min, range_max, dim)
            else:
                if np.random.rand() < 0.5:
                    c = clusters[np.random.randint(k_clusters)]
                    noise = np.random.normal(0, 1, dim) * (range_max - range_min) * 0.05
                    new_pos = c + noise
                else:
                    c1 = clusters[np.random.randint(k_clusters)]
                    c2 = clusters[np.random.randint(k_clusters)]
                    alpha = np.random.rand()
                    noise = np.random.normal(0, 1, dim) * (range_max - range_min) * 0.05
                    new_pos = alpha * c1 + (1 - alpha) * c2 + noise

            new_pos = np.clip(new_pos, range_min, range_max)
            new_cost = func(new_pos)

            if new_cost < fitness[i]:
                positions[i] = new_pos
                fitness[i] = new_cost

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_cost:
            best_cost = fitness[best_idx]
            best_position = positions[best_idx].copy()

        history.append(make_history_state(positions, best_position, best_cost, func))

    return {
        "best_position": best_position,
        "best_cost": float(best_cost),
        "history": history,
    }


def run_scso(
    func,
    num=50,
    iterations=100,
    dim=2,
    bounds=(-5.12, 5.12),
):
    range_min, range_max = bounds
    positions = np.random.uniform(range_min, range_max, (num, dim))
    fitness = np.array([func(p) for p in positions])

    best_idx = np.argmin(fitness)
    best_position = positions[best_idx].copy()
    best_cost = fitness[best_idx]

    history = []

    for t in range(iterations):
        r = 2 * (1 - t / iterations)

        for i in range(num):
            R = 2 * r * np.random.rand() - r

            if abs(R) <= 1:
                direction = np.random.normal(0, 1, dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = np.ones(dim) / np.sqrt(dim)

                new_pos = best_position + R * direction * np.abs(best_position - positions[i])
            else:
                rand_idx = np.random.randint(num)
                x_rand = positions[rand_idx]

                new_pos = x_rand + R * np.abs(np.random.rand(dim) * x_rand - positions[i])

            new_pos = np.clip(new_pos, range_min, range_max)
            new_cost = func(new_pos)

            if new_cost < fitness[i]:
                positions[i] = new_pos
                fitness[i] = new_cost

        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_cost:
            best_cost = fitness[best_idx]
            best_position = positions[best_idx].copy()

        history.append(make_history_state(positions, best_position, best_cost, func))

    return {
        "best_position": best_position,
        "best_cost": float(best_cost),
        "history": history,
    }


# ============================================================
# ALGORYTM GENETYCZNY - OPERATORY WSPÓLNE
# ============================================================
def tournament_selection(pop, fitness, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return pop[best].copy()


def roulette_selection(pop, fitness):
    inv = 1.0 / (fitness + 1e-12)
    probs = inv / np.sum(inv)
    idx = np.random.choice(len(pop), p=probs)
    return pop[idx].copy()


def ranking_selection(pop, fitness):
    ranks = np.argsort(np.argsort(fitness))
    probs = (len(pop) - ranks) / np.sum(len(pop) - ranks)
    idx = np.random.choice(len(pop), p=probs)
    return pop[idx].copy()


def crossover_one_point(p1, p2):
    point = np.random.randint(1, len(p1))
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2


def crossover_two_point(p1, p2):
    a, b = sorted(np.random.choice(len(p1), 2, replace=False))
    c1 = p1.copy()
    c2 = p2.copy()
    c1[a:b], c2[a:b] = p2[a:b], p1[a:b]
    return c1, c2


def crossover_arithmetic(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2, alpha * p2 + (1 - alpha) * p1


def mutation_uniform(x, bounds, rate=0.1):
    low, high = bounds
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < rate:
            x[i] = np.random.uniform(low, high)
    return x


def mutation_gaussian(x, bounds, rate=0.1, sigma=0.1):
    low, high = bounds
    x = x.copy()
    for i in range(len(x)):
        if np.random.rand() < rate:
            x[i] += np.random.normal(0, sigma)
    return np.clip(x, low, high)


def run_ga_continuous(
    func,
    num=50,
    iterations=100,
    dim=2,
    bounds=(-5, 5),
    selection="tournament",
    crossover="arithmetic",
    mutation="gaussian",
    tournament_k=3,
    stagnation_limit=50,
    pc=0.8,
    pm=0.05,
    sigma=0.1,
    elitism=1,
):
    low, high = bounds
    pop = np.random.uniform(low, high, (num, dim))
    fitness = np.array([func(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_cost = fitness[best_idx]

    history = []
    history_mean = []
    stagnation = 0

    for _ in range(iterations):
        new_pop = []

        elite_idx = np.argsort(fitness)[:elitism]
        elites = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < num - elitism:
            if selection == "roulette":
                p1 = roulette_selection(pop, fitness)
                p2 = roulette_selection(pop, fitness)
            elif selection == "ranking":
                p1 = ranking_selection(pop, fitness)
                p2 = ranking_selection(pop, fitness)
            else:
                p1 = tournament_selection(pop, fitness, tournament_k)
                p2 = tournament_selection(pop, fitness, tournament_k)

            if np.random.rand() < pc:
                if crossover == "one_point":
                    c1, c2 = crossover_one_point(p1, p2)
                elif crossover == "two_point":
                    c1, c2 = crossover_two_point(p1, p2)
                else:
                    c1, c2 = crossover_arithmetic(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if mutation == "uniform":
                c1 = mutation_uniform(c1, bounds, rate=pm)
                c2 = mutation_uniform(c2, bounds, rate=pm)
            else:
                c1 = mutation_gaussian(c1, bounds, rate=pm, sigma=sigma)
                c2 = mutation_gaussian(c2, bounds, rate=pm, sigma=sigma)

            new_pop.extend([c1, c2])

        pop = np.array(elites + new_pop[: num - elitism])
        fitness = np.array([func(ind) for ind in pop])

        idx = np.argmin(fitness)
        if fitness[idx] < best_cost:
            best_cost = fitness[idx]
            best = pop[idx].copy()
            stagnation = 0
        else:
            stagnation += 1

        history.append(make_history_state(pop, best, best_cost, func))
        history_mean.append(float(np.mean(fitness)))

        if stagnation >= stagnation_limit:
            break

    return {
        "best_position": best,
        "best_cost": float(best_cost),
        "history": history,
        "history_mean": history_mean,
    }


# ============================================================
# TSP - NARZĘDZIA
# ============================================================
def generate_cities(num_cities, seed=0, scale=100.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, scale, size=(num_cities, 2))


def save_cities_csv(cities, filename):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["city_id", "x", "y"])
        for i, (x, y) in enumerate(cities):
            writer.writerow([i, float(x), float(y)])


def route_length(route, cities):
    total = 0.0
    n = len(route)
    for i in range(n):
        a = cities[route[i]]
        b = cities[route[(i + 1) % n]]
        total += np.linalg.norm(a - b)
    return float(total)


def pairwise_distances(cities):
    n = len(cities)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = np.linalg.norm(cities[i] - cities[j])
    return D


def random_route(n):
    return np.random.permutation(n)


def permutation_to_swaps(source, target):
    arr = source.copy()
    pos = {value: idx for idx, value in enumerate(arr)}
    swaps = []

    for i in range(len(arr)):
        if arr[i] != target[i]:
            j = pos[target[i]]
            swaps.append((i, j))

            pos[arr[i]] = j
            pos[arr[j]] = i
            arr[i], arr[j] = arr[j], arr[i]

    return swaps


def apply_swaps(route, swaps):
    r = route.copy()
    for i, j in swaps:
        r[i], r[j] = r[j], r[i]
    return r


def sample_swaps(swaps, probability):
    if probability <= 0:
        return []
    chosen = []
    for swap in swaps:
        if np.random.rand() < probability:
            chosen.append(swap)
    return chosen


# ============================================================
# TSP - ALGORYTMY
# ============================================================
def run_dpso_tsp(
    cities,
    num=50,
    iterations=100,
    inertia_keep=0.6,
    c1=0.8,
    c2=0.9,
):
    n = len(cities)

    particles = [random_route(n) for _ in range(num)]
    velocities = [[] for _ in range(num)]

    pbest_positions = [p.copy() for p in particles]
    pbest_costs = [route_length(p, cities) for p in particles]

    best_idx = int(np.argmin(pbest_costs))
    gbest_position = pbest_positions[best_idx].copy()
    gbest_cost = float(pbest_costs[best_idx])

    history_best = []

    for _ in range(iterations):
        for i in range(num):
            current = particles[i]

            inertia_part = sample_swaps(velocities[i], inertia_keep)

            cognitive_full = permutation_to_swaps(current, pbest_positions[i])
            cognitive_part = sample_swaps(cognitive_full, c1)

            social_full = permutation_to_swaps(current, gbest_position)
            social_part = sample_swaps(social_full, c2)

            new_velocity = inertia_part + cognitive_part + social_part
            new_position = apply_swaps(current, new_velocity)

            new_cost = route_length(new_position, cities)

            particles[i] = new_position
            velocities[i] = new_velocity

            if new_cost < pbest_costs[i]:
                pbest_costs[i] = new_cost
                pbest_positions[i] = new_position.copy()

        best_idx = int(np.argmin(pbest_costs))
        if pbest_costs[best_idx] < gbest_cost:
            gbest_cost = float(pbest_costs[best_idx])
            gbest_position = pbest_positions[best_idx].copy()

        history_best.append(gbest_cost)

    return {
        "best_route": gbest_position,
        "best_cost": float(gbest_cost),
        "history_best": history_best,
    }


def construct_ant_route(pheromone, distances, alpha, beta, start_city=None):
    n = pheromone.shape[0]

    if start_city is None:
        current = np.random.randint(n)
    else:
        current = int(start_city)

    unvisited = set(range(n))
    unvisited.remove(current)
    route = [current]

    while unvisited:
        candidates = list(unvisited)
        probs = []

        for j in candidates:
            tau = pheromone[current, j] ** alpha
            eta = (1.0 / (distances[current, j] + 1e-12)) ** beta
            probs.append(tau * eta)

        probs = np.array(probs, dtype=float)
        s = probs.sum()

        if s <= 0 or not np.isfinite(s):
            next_city = np.random.choice(candidates)
        else:
            probs /= s
            next_city = np.random.choice(candidates, p=probs)

        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return np.array(route, dtype=int)


def run_aco_tsp(
    cities,
    num=50,
    iterations=100,
    alpha=1.0,
    beta=3.0,
    evaporation=0.5,
    q=100.0,
):
    n = len(cities)
    distances = pairwise_distances(cities)
    pheromone = np.ones((n, n), dtype=float)

    best_route = None
    best_cost = np.inf
    history_best = []

    for _ in range(iterations):
        all_routes = []
        all_costs = []

        for _ant in range(num):
            route = construct_ant_route(pheromone, distances, alpha=alpha, beta=beta)
            cost = route_length(route, cities)

            all_routes.append(route)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = float(cost)
                best_route = route.copy()

        pheromone *= 1.0 - evaporation
        pheromone = np.maximum(pheromone, 1e-12)

        for route, cost in zip(all_routes, all_costs):
            deposit = q / (cost + 1e-12)
            for i in range(n):
                a = route[i]
                b = route[(i + 1) % n]
                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

        history_best.append(best_cost)

    return {
        "best_route": best_route,
        "best_cost": float(best_cost),
        "history_best": history_best,
    }


# ============================================================
# ALGORYTM GENETYCZNY - TSP
# ============================================================
def pmx(parent1, parent2):
    size = len(parent1)
    a, b = sorted(np.random.choice(size, 2, replace=False))

    child = [-1] * size
    child[a:b] = parent1[a:b]

    for i in range(a, b):
        if parent2[i] not in child:
            pos = i
            val = parent2[i]
            while True:
                pos = np.where(parent2 == parent1[pos])[0][0]
                if child[pos] == -1:
                    child[pos] = val
                    break

    for i in range(size):
        if child[i] == -1:
            child[i] = parent2[i]

    return np.array(child)


def ox(parent1, parent2):
    size = len(parent1)
    a, b = sorted(np.random.choice(size, 2, replace=False))

    child = [-1] * size
    child[a:b] = parent1[a:b]

    fill = [x for x in parent2 if x not in child]
    ptr = 0

    for i in range(size):
        if child[i] == -1:
            child[i] = fill[ptr]
            ptr += 1

    return np.array(child)


def mutate_swap(route, rate=0.05):
    route = route.copy()
    if np.random.rand() < rate:
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]
    return route


def mutate_inverse(route, rate=0.05):
    route = route.copy()
    if np.random.rand() < rate:
        i, j = sorted(np.random.choice(len(route), 2, replace=False))
        route[i:j] = route[i:j][::-1]
    return route


def run_ga_tsp(
    cities,
    num=50,
    iterations=100,
    selection="tournament",
    crossover="ox",
    mutation="swap",
    tournament_k=3,
    stagnation_limit=50,
    pc=0.8,
    pm=0.05,
    elitism=1,
):
    n = len(cities)
    pop = [random_route(n) for _ in range(num)]
    fitness = np.array([route_length(p, cities) for p in pop])

    best_idx = np.argmin(fitness)
    best = pop[best_idx].copy()
    best_cost = fitness[best_idx]

    history_best = []
    history_mean = []
    route_history_best = [best.copy()]
    stagnation = 0

    for _ in range(iterations):
        new_pop = []

        elite_idx = np.argsort(fitness)[:elitism]
        elites = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < num - elitism:
            if selection == "roulette":
                p1 = roulette_selection(pop, fitness)
                p2 = roulette_selection(pop, fitness)
            elif selection == "ranking":
                p1 = ranking_selection(pop, fitness)
                p2 = ranking_selection(pop, fitness)
            else:
                p1 = tournament_selection(pop, fitness, tournament_k)
                p2 = tournament_selection(pop, fitness, tournament_k)

            if np.random.rand() < pc:
                if crossover == "pmx":
                    c1 = pmx(p1, p2)
                    c2 = pmx(p2, p1)
                else:
                    c1 = ox(p1, p2)
                    c2 = ox(p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if mutation == "inverse":
                c1 = mutate_inverse(c1, rate=pm)
                c2 = mutate_inverse(c2, rate=pm)
            else:
                c1 = mutate_swap(c1, rate=pm)
                c2 = mutate_swap(c2, rate=pm)

            new_pop.extend([c1, c2])

        pop = elites + new_pop[: num - elitism]
        fitness = np.array([route_length(p, cities) for p in pop])

        idx = np.argmin(fitness)
        if fitness[idx] < best_cost:
            best_cost = fitness[idx]
            best = pop[idx].copy()
            stagnation = 0
        else:
            stagnation += 1

        history_best.append(float(best_cost))
        history_mean.append(float(np.mean(fitness)))
        route_history_best.append(best.copy())

        if stagnation >= stagnation_limit:
            break

    return {
        "best_route": best,
        "best_cost": float(best_cost),
        "history_best": history_best,
        "history_mean": history_mean,
        "route_history_best": route_history_best,
    }


# ============================================================
# WYBÓR ALGORYTMU
# ============================================================
CONTINUOUS_ALGORITHMS = {
    "pso": run_pso,
    "random_search": run_random_search,
    "gwo": run_gwo,
    "scso": run_scso,
    "bso": run_bso,
    "ga": run_ga_continuous,
}

TSP_ALGORITHMS = {
    "dpso_tsp": run_dpso_tsp,
    "aco_tsp": run_aco_tsp,
    "ga_tsp": run_ga_tsp,
}


def run_continuous_algorithm(name, func, **kwargs):
    key = name.lower()
    if key not in CONTINUOUS_ALGORITHMS:
        raise ValueError(f"Nieznany algorytm ciągły: {name}")
    return CONTINUOUS_ALGORITHMS[key](func, **kwargs)


def run_tsp_algorithm(name, cities, **kwargs):
    key = name.lower()
    if key not in TSP_ALGORITHMS:
        raise ValueError(f"Nieznany algorytm TSP: {name}")
    return TSP_ALGORITHMS[key](cities, **kwargs)


# ============================================================
# PARAMETRY DOMYŚLNE DLA ALGORYTMÓW
# ============================================================
def get_continuous_extra_params(algorithm_name):
    key = algorithm_name.lower()

    if key == "pso":
        return {"w": 0.7, "c1": 1.5, "c2": 1.5}
    if key == "bso":
        return {"k_clusters": 5, "p_replace": 0.1}
    if key == "ga":
        return {
            "selection": "tournament",
            "crossover": "arithmetic",
            "mutation": "gaussian",
            "tournament_k": 3,
            "pc": 0.8,
            "pm": 0.05,
            "sigma": 0.1,
            "elitism": 1,
            "stagnation_limit": 50,
        }

    return {}


def get_tsp_extra_params(algorithm_name):
    key = algorithm_name.lower()

    if key == "dpso_tsp":
        return {"inertia_keep": 0.6, "c1": 0.8, "c2": 0.9}
    if key == "aco_tsp":
        return {"alpha": 1.0, "beta": 3.0, "evaporation": 0.5, "q": 100.0}
    if key == "ga_tsp":
        return {
            "selection": "tournament",
            "crossover": "ox",
            "mutation": "swap",
            "tournament_k": 3,
            "pc": 0.8,
            "pm": 0.05,
            "elitism": 1,
            "stagnation_limit": 50,
        }

    return {}


# ============================================================
# BENCHMARK - CIĄGŁE
# ============================================================
def benchmark_continuous_algorithm(
    algorithm_name,
    function_name,
    runs=20,
    num=50,
    iterations=100,
    dim=2,
):
    problem = FUNCTIONS[function_name]
    func = problem["func"]
    bounds = problem["bounds"]
    extra_params = get_continuous_extra_params(algorithm_name)

    best_costs = []
    times = []

    for seed in range(runs):
        np.random.seed(seed)

        start = time.perf_counter()
        result = run_continuous_algorithm(
            algorithm_name,
            func,
            num=num,
            iterations=iterations,
            dim=dim,
            bounds=bounds,
            **extra_params,
        )
        elapsed = time.perf_counter() - start

        best_costs.append(result["best_cost"])
        times.append(elapsed)

    return {
        "problem_type": "continuous",
        "algorithm": algorithm_name,
        "function": function_name,
        "dim": dim,
        "runs": runs,
        "agents": num,
        "iterations": iterations,
        "mean_best": float(np.mean(best_costs)),
        "std_best": float(np.std(best_costs)),
        "min_best": float(np.min(best_costs)),
        "max_best": float(np.max(best_costs)),
        "mean_time": float(np.mean(times)),
    }


def benchmark_continuous_parameter_sweep(
    algorithm_names,
    function_names,
    agent_values,
    iteration_values,
    dim_values,
    runs=20,
):
    rows = []

    for function_name in function_names:
        allowed_dims = FUNCTIONS[function_name]["dims"]

        for dim in dim_values:
            if dim not in allowed_dims:
                continue

            for algorithm_name in algorithm_names:
                for num in agent_values:
                    for iterations in iteration_values:
                        row = benchmark_continuous_algorithm(
                            algorithm_name=algorithm_name,
                            function_name=function_name,
                            runs=runs,
                            num=num,
                            iterations=iterations,
                            dim=dim,
                        )
                        rows.append(row)

                        print(
                            f"[CONT] {algorithm_name:12s} | {function_name:10s} | "
                            f"dim={dim:2d} | agents={num:3d} | iter={iterations:3d} | "
                            f"mean={row['mean_best']:.6f}"
                        )

    return rows


def benchmark_ga_continuous_variants(function_name, dim, runs=20):
    rows = []
    problem = FUNCTIONS[function_name]
    func = problem["func"]
    bounds = problem["bounds"]

    for selection in GA_SELECTION_VALUES:
        for pc in GA_PC_VALUES:
            for pm in GA_PM_VALUES:
                for num in AGENT_VALUES:
                    best_costs = []
                    times = []

                    for seed in range(runs):
                        np.random.seed(seed)
                        start = time.perf_counter()

                        result = run_ga_continuous(
                            func,
                            num=num,
                            iterations=100,
                            dim=dim,
                            bounds=bounds,
                            selection=selection,
                            crossover="arithmetic",
                            mutation="gaussian",
                            pc=pc,
                            pm=pm,
                            sigma=0.1,
                        )

                        elapsed = time.perf_counter() - start
                        best_costs.append(result["best_cost"])
                        times.append(elapsed)

                    rows.append(
                        {
                            "problem_type": "continuous_ga_variants",
                            "function": function_name,
                            "dim": dim,
                            "selection": selection,
                            "pc": pc,
                            "pm": pm,
                            "agents": num,
                            "runs": runs,
                            "mean_best": float(np.mean(best_costs)),
                            "std_best": float(np.std(best_costs)),
                            "min_best": float(np.min(best_costs)),
                            "max_best": float(np.max(best_costs)),
                            "mean_time": float(np.mean(times)),
                        }
                    )

                    print(
                        f"[GA-CONT] {function_name:10s} | dim={dim:2d} | "
                        f"sel={selection:10s} | pc={pc:.2f} | pm={pm:.2f} | "
                        f"agents={num:3d}"
                    )

    return rows


# ============================================================
# BENCHMARK - TSP
# ============================================================
def build_shared_tsp_instances(city_values, runs):
    instances = {}
    for n in city_values:
        current = []
        for run_idx in range(runs):
            seed = 100000 + 1000 * n + run_idx
            current.append(generate_cities(n, seed=seed))
        instances[n] = current
    return instances


def benchmark_tsp_algorithm(
    algorithm_name,
    cities_list,
    runs=20,
    num=50,
    iterations=100,
):
    extra_params = get_tsp_extra_params(algorithm_name)

    best_costs = []
    times = []

    for seed in range(runs):
        np.random.seed(seed)
        cities = cities_list[seed]

        start = time.perf_counter()
        result = run_tsp_algorithm(
            algorithm_name,
            cities,
            num=num,
            iterations=iterations,
            **extra_params,
        )
        elapsed = time.perf_counter() - start

        best_costs.append(result["best_cost"])
        times.append(elapsed)

    return {
        "problem_type": "tsp",
        "algorithm": algorithm_name,
        "cities": len(cities_list[0]),
        "runs": runs,
        "agents": num,
        "iterations": iterations,
        "mean_best": float(np.mean(best_costs)),
        "std_best": float(np.std(best_costs)),
        "min_best": float(np.min(best_costs)),
        "max_best": float(np.max(best_costs)),
        "mean_time": float(np.mean(times)),
    }


def benchmark_tsp_parameter_sweep(
    algorithm_names,
    shared_instances,
    city_values,
    agent_values,
    iteration_values,
    runs=20,
):
    rows = []

    for city_count in city_values:
        cities_list = shared_instances[city_count]

        for algorithm_name in algorithm_names:
            for num in agent_values:
                for iterations in iteration_values:
                    row = benchmark_tsp_algorithm(
                        algorithm_name=algorithm_name,
                        cities_list=cities_list,
                        runs=runs,
                        num=num,
                        iterations=iterations,
                    )
                    rows.append(row)

                    print(
                        f"[TSP ] {algorithm_name:12s} | cities={city_count:2d} | "
                        f"agents={num:3d} | iter={iterations:3d} | "
                        f"mean={row['mean_best']:.6f}"
                    )

    return rows


def benchmark_ga_tsp_variants(cities_list, runs=20):
    rows = []

    for selection in GA_SELECTION_VALUES:
        for pc in GA_PC_VALUES:
            for pm in GA_PM_VALUES:
                for num in AGENT_VALUES:
                    best_costs = []
                    times = []

                    for seed in range(runs):
                        np.random.seed(seed)
                        cities = cities_list[seed]
                        start = time.perf_counter()

                        result = run_ga_tsp(
                            cities,
                            num=num,
                            iterations=100,
                            selection=selection,
                            crossover="ox",
                            mutation="swap",
                            pc=pc,
                            pm=pm,
                        )

                        elapsed = time.perf_counter() - start
                        best_costs.append(result["best_cost"])
                        times.append(elapsed)

                    rows.append(
                        {
                            "problem_type": "tsp_ga_variants",
                            "cities": len(cities_list[0]),
                            "selection": selection,
                            "pc": pc,
                            "pm": pm,
                            "agents": num,
                            "runs": runs,
                            "mean_best": float(np.mean(best_costs)),
                            "std_best": float(np.std(best_costs)),
                            "min_best": float(np.min(best_costs)),
                            "max_best": float(np.max(best_costs)),
                            "mean_time": float(np.mean(times)),
                        }
                    )

                    print(
                        f"[GA-TSP ] cities={len(cities_list[0]):2d} | "
                        f"sel={selection:10s} | pc={pc:.2f} | pm={pm:.2f} | "
                        f"agents={num:3d}"
                    )

    return rows


# ============================================================
# DEMO - CIĄGŁE
# ============================================================
def run_demo_continuous():
    algorithm_name = "ga"
    function_name = "himmelblau"

    problem = FUNCTIONS[function_name]
    func = problem["func"]
    bounds = problem["bounds"]

    X, Y, Z = make_surface(func, bounds[0], bounds[1], points=80)

    start = time.perf_counter()

    extra_params = get_continuous_extra_params(algorithm_name)
    result = run_continuous_algorithm(
        algorithm_name,
        func,
        num=50,
        iterations=100,
        dim=2,
        bounds=bounds,
        **extra_params,
    )

    elapsed = time.perf_counter() - start
    history = result["history"]

    print(f"Algorytm: {algorithm_name}")
    print(f"Funkcja: {function_name}")
    print(f"Najlepszy wynik: {result['best_cost']:.6f}")
    print(f"Najlepsza pozycja: {result['best_position']}")
    print(f"Czas działania: {elapsed:.4f} s")

    save_gif(
        history,
        X,
        Y,
        Z,
        filename=os.path.join(RESULTS_DIR, f"{algorithm_name}_{function_name}.gif"),
        fps=15,
    )

    plot_convergence(
        history,
        algorithm_name,
        function_name,
        save_path=os.path.join(RESULTS_DIR, f"conv_{algorithm_name}_{function_name}.png"),
    )

    plot_ga_convergence(
        [state["best_cost"] for state in history],
        result["history_mean"],
        title=f"GA best vs mean / {function_name}",
        save_path=os.path.join(RESULTS_DIR, f"ga_best_mean_{function_name}.png"),
    )

    show_interactive_plot(history, X, Y, Z, bounds[0], bounds[1])


# ============================================================
# DEMO - TSP
# ============================================================
def run_demo_tsp():
    cities = generate_cities(20, seed=2026)
    save_cities_csv(cities, os.path.join(RESULTS_DIR, "demo_tsp_20_cities.csv"))

    np.random.seed(42)
    dpso_result = run_tsp_algorithm(
        "dpso_tsp",
        cities,
        num=50,
        iterations=100,
        **get_tsp_extra_params("dpso_tsp"),
    )

    np.random.seed(42)
    aco_result = run_tsp_algorithm(
        "aco_tsp",
        cities,
        num=50,
        iterations=100,
        **get_tsp_extra_params("aco_tsp"),
    )

    np.random.seed(42)
    ga_result = run_tsp_algorithm(
        "ga_tsp",
        cities,
        num=50,
        iterations=100,
        **get_tsp_extra_params("ga_tsp"),
    )

    plot_tsp_route(
        cities,
        dpso_result["best_route"],
        title=f"DPSO TSP, best={dpso_result['best_cost']:.3f}",
        save_path=os.path.join(RESULTS_DIR, "demo_dpso_tsp_route.png"),
    )

    plot_tsp_route(
        cities,
        aco_result["best_route"],
        title=f"ACO TSP, best={aco_result['best_cost']:.3f}",
        save_path=os.path.join(RESULTS_DIR, "demo_aco_tsp_route.png"),
    )

    plot_tsp_route(
        cities,
        ga_result["best_route"],
        title=f"GA TSP, best={ga_result['best_cost']:.3f}",
        save_path=os.path.join(RESULTS_DIR, "demo_ga_tsp_route.png"),
    )

    plot_tsp_convergence(
        {
            "dpso_tsp": dpso_result["history_best"],
            "aco_tsp": aco_result["history_best"],
            "ga_tsp": ga_result["history_best"],
        },
        title="Porównanie zbieżności TSP",
        save_path=os.path.join(RESULTS_DIR, "demo_tsp_convergence.png"),
    )

    plot_ga_convergence(
        ga_result["history_best"],
        ga_result["history_mean"],
        title="GA TSP: best vs mean",
        save_path=os.path.join(RESULTS_DIR, "demo_ga_tsp_best_vs_mean.png"),
    )

    save_tsp_gif(
        cities,
        ga_result["route_history_best"],
        filename=os.path.join(RESULTS_DIR, "demo_ga_tsp.gif"),
        fps=10,
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(GLOBAL_SEED)

    if DEMO_MODE_CONTINUOUS:
        run_demo_continuous()

    if DEMO_MODE_TSP:
        run_demo_tsp()

    if BENCHMARK_MODE_CONTINUOUS:
        continuous_rows = benchmark_continuous_parameter_sweep(
            algorithm_names=["pso", "gwo", "scso", "bso", "ga"],
            function_names=["rastrigin", "schwefel", "ackley", "eggholder", "himmelblau"],
            agent_values=AGENT_VALUES,
            iteration_values=ITERATION_VALUES,
            dim_values=DIM_VALUES,
            runs=CONTINUOUS_RUNS,
        )

        save_benchmark_csv(
            continuous_rows,
            filename=os.path.join(RESULTS_DIR, "benchmark_continuous_results.csv"),
        )
        print("\nZapisano benchmark ciągły do pliku: benchmark_continuous_results.csv")

        compare_function = "rastrigin"
        compare_problem = FUNCTIONS[compare_function]
        compare_func = compare_problem["func"]
        compare_bounds = compare_problem["bounds"]

        comparison_results = {}
        for algorithm_name in ["pso", "gwo", "scso", "bso", "ga"]:
            np.random.seed(42)
            extra_params = get_continuous_extra_params(algorithm_name)

            result = run_continuous_algorithm(
                algorithm_name,
                compare_func,
                num=50,
                iterations=100,
                dim=2,
                bounds=compare_bounds,
                **extra_params,
            )
            comparison_results[algorithm_name] = result

        plot_multiple_convergences(
            comparison_results,
            compare_function,
            save_path=os.path.join(RESULTS_DIR, "compare_continuous_rastrigin.png"),
        )

        np.random.seed(42)
        ga_compare = run_continuous_algorithm(
            "ga",
            compare_func,
            num=50,
            iterations=100,
            dim=2,
            bounds=compare_bounds,
            **get_continuous_extra_params("ga"),
        )

        plot_ga_convergence(
            [state["best_cost"] for state in ga_compare["history"]],
            ga_compare["history_mean"],
            title=f"GA best vs mean / {compare_function}",
            save_path=os.path.join(RESULTS_DIR, "ga_best_vs_mean_rastrigin.png"),
        )

    if BENCHMARK_MODE_TSP:
        shared_instances = build_shared_tsp_instances(CITY_VALUES, TSP_RUNS)

        for city_count, city_sets in shared_instances.items():
            save_cities_csv(
                city_sets[0],
                os.path.join(RESULTS_DIR, f"tsp_cities_{city_count}_example.csv"),
            )

        tsp_rows = benchmark_tsp_parameter_sweep(
            algorithm_names=["dpso_tsp", "aco_tsp", "ga_tsp"],
            shared_instances=shared_instances,
            city_values=CITY_VALUES,
            agent_values=AGENT_VALUES,
            iteration_values=ITERATION_VALUES,
            runs=TSP_RUNS,
        )

        save_benchmark_csv(
            tsp_rows,
            filename=os.path.join(RESULTS_DIR, "benchmark_tsp_results.csv"),
        )
        print("\nZapisano benchmark TSP do pliku: benchmark_tsp_results.csv")

        demo_cities = shared_instances[20][0]

        np.random.seed(42)
        dpso_demo = run_tsp_algorithm(
            "dpso_tsp",
            demo_cities,
            num=50,
            iterations=100,
            **get_tsp_extra_params("dpso_tsp"),
        )

        np.random.seed(42)
        aco_demo = run_tsp_algorithm(
            "aco_tsp",
            demo_cities,
            num=50,
            iterations=100,
            **get_tsp_extra_params("aco_tsp"),
        )

        np.random.seed(42)
        ga_demo = run_tsp_algorithm(
            "ga_tsp",
            demo_cities,
            num=50,
            iterations=100,
            **get_tsp_extra_params("ga_tsp"),
        )

        plot_tsp_convergence(
            {
                "dpso_tsp": dpso_demo["history_best"],
                "aco_tsp": aco_demo["history_best"],
                "ga_tsp": ga_demo["history_best"],
            },
            title="Porównanie zbieżności TSP (20 miast)",
            save_path=os.path.join(RESULTS_DIR, "compare_tsp_20cities.png"),
        )

        plot_tsp_route(
            demo_cities,
            dpso_demo["best_route"],
            title=f"DPSO TSP, 20 miast, best={dpso_demo['best_cost']:.3f}",
            save_path=os.path.join(RESULTS_DIR, "route_dpso_20cities.png"),
        )

        plot_tsp_route(
            demo_cities,
            aco_demo["best_route"],
            title=f"ACO TSP, 20 miast, best={aco_demo['best_cost']:.3f}",
            save_path=os.path.join(RESULTS_DIR, "route_aco_20cities.png"),
        )

        plot_tsp_route(
            demo_cities,
            ga_demo["best_route"],
            title=f"GA TSP, 20 miast, best={ga_demo['best_cost']:.3f}",
            save_path=os.path.join(RESULTS_DIR, "route_ga_20cities.png"),
        )

        plot_ga_convergence(
            ga_demo["history_best"],
            ga_demo["history_mean"],
            title="GA TSP (20 miast): best vs mean",
            save_path=os.path.join(RESULTS_DIR, "ga_tsp_20cities_best_vs_mean.png"),
        )

        save_tsp_gif(
            demo_cities,
            ga_demo["route_history_best"],
            filename=os.path.join(RESULTS_DIR, "ga_tsp_20cities.gif"),
            fps=10,
        )

    if GA_VARIANT_BENCHMARK_CONTINUOUS:
        rows = []
        for function_name in ["rastrigin", "ackley", "himmelblau"]:
            for dim in FUNCTIONS[function_name]["dims"]:
                if dim in [2, 5, 10]:
                    rows.extend(
                        benchmark_ga_continuous_variants(
                            function_name=function_name,
                            dim=dim,
                            runs=CONTINUOUS_RUNS,
                        )
                    )

        save_benchmark_csv(
            rows,
            filename=os.path.join(RESULTS_DIR, "benchmark_ga_continuous_variants.csv"),
        )
        print("\nZapisano benchmark wariantów GA continuous do pliku: benchmark_ga_continuous_variants.csv")

    if GA_VARIANT_BENCHMARK_TSP:
        shared_instances = build_shared_tsp_instances(CITY_VALUES, TSP_RUNS)
        rows = []

        for city_count in CITY_VALUES:
            rows.extend(
                benchmark_ga_tsp_variants(
                    shared_instances[city_count],
                    runs=TSP_RUNS,
                )
            )

        save_benchmark_csv(
            rows,
            filename=os.path.join(RESULTS_DIR, "benchmark_ga_tsp_variants.csv"),
        )
        print("\nZapisano benchmark wariantów GA TSP do pliku: benchmark_ga_tsp_variants.csv")

    print("\nGotowe.")