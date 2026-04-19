
import os
import csv
import time
import numpy as np
import matplotlib

# Fallback: interaktywny backend lokalnie, zwykły backend do zapisu na serwerze/headless
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
DEMO_MODE_TSP = False

BENCHMARK_MODE_CONTINUOUS = True
BENCHMARK_MODE_TSP = True

CONTINUOUS_RUNS = 20
TSP_RUNS = 20

AGENT_VALUES = [10, 20, 50, 100]
ITERATION_VALUES = [50, 100, 200]
DIM_VALUES = [2, 5, 10]
CITY_VALUES = [10, 20, 50]


# ============================================================
# FUNKCJE CELU
# ============================================================
def rastrigin(x):
    x = np.array(x, dtype=float)
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def schwefel(x):
    x = np.array(x, dtype=float)
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def ackley(x):
    x = np.array(x, dtype=float)
    n = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e


def eggholder(x):
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(abs(x1 / 2 + x2 + 47))) - x1 * np.sin(
        np.sqrt(abs(x1 - (x2 + 47)))
    )


def himmelblau(x):
    x1, x2 = x
    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


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
        "best_position": best_position.copy(),
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
        "best_position": best_position.copy(),
        "best_cost": float(best_cost),
        "history": history,
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


def keys_to_route(keys):
    keys = np.asarray(keys, dtype=float)
    return np.argsort(keys, kind="mergesort")


def route_cost_from_keys(keys, cities):
    route = keys_to_route(keys)
    return route_length(route, cities)


# ============================================================
# TSP - ADAPTACJE TYCH SAMYCH ALGORYTMÓW (RANDOM KEYS)
# ============================================================
def run_tsp_via_random_keys(base_algorithm_name, cities, num=50, iterations=100, **kwargs):
    n = len(cities)

    def tsp_cost(keys):
        return route_cost_from_keys(keys, cities)

    result = run_continuous_algorithm(
        base_algorithm_name,
        tsp_cost,
        num=num,
        iterations=iterations,
        dim=n,
        bounds=(0.0, 1.0),
        **kwargs,
    )

    best_keys = result["best_position"].copy()
    best_route = keys_to_route(best_keys)
    best_cost = route_length(best_route, cities)
    history_best = [state["best_cost"] for state in result["history"]]

    return {
        "best_keys": best_keys,
        "best_route": best_route.copy(),
        "best_cost": float(best_cost),
        "history_best": history_best,
        "history_continuous": result["history"],
    }


def run_pso_tsp(cities, num=50, iterations=100, w=0.7, c1=1.5, c2=1.5):
    return run_tsp_via_random_keys(
        "pso",
        cities,
        num=num,
        iterations=iterations,
        w=w,
        c1=c1,
        c2=c2,
    )


def run_gwo_tsp(cities, num=50, iterations=100):
    return run_tsp_via_random_keys(
        "gwo",
        cities,
        num=num,
        iterations=iterations,
    )


def run_scso_tsp(cities, num=50, iterations=100):
    return run_tsp_via_random_keys(
        "scso",
        cities,
        num=num,
        iterations=iterations,
    )


def run_bso_tsp(cities, num=50, iterations=100, k_clusters=5, p_replace=0.1):
    return run_tsp_via_random_keys(
        "bso",
        cities,
        num=num,
        iterations=iterations,
        k_clusters=k_clusters,
        p_replace=p_replace,
    )


# ============================================================
# WYBÓR ALGORYTMU
# ============================================================
CONTINUOUS_ALGORITHMS = {
    "pso": run_pso,
    "gwo": run_gwo,
    "scso": run_scso,
    "bso": run_bso,
}

TSP_ALGORITHMS = {
    "pso_tsp": run_pso_tsp,
    "gwo_tsp": run_gwo_tsp,
    "scso_tsp": run_scso_tsp,
    "bso_tsp": run_bso_tsp,
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

    return {}


def get_tsp_extra_params(algorithm_name):
    key = algorithm_name.lower().replace("_tsp", "")
    return get_continuous_extra_params(key)


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
                            f"[CONT] {algorithm_name:8s} | {function_name:10s} | "
                            f"dim={dim:2d} | agents={num:3d} | iter={iterations:3d} | "
                            f"mean={row['mean_best']:.6f}"
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
                        f"[TSP ] {algorithm_name:8s} | cities={city_count:2d} | "
                        f"agents={num:3d} | iter={iterations:3d} | "
                        f"mean={row['mean_best']:.6f}"
                    )

    return rows


# ============================================================
# DEMO - CIĄGŁE
# ============================================================
def run_demo_continuous():
    algorithm_name = "bso"
    function_name = "rastrigin"

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
    show_interactive_plot(history, X, Y, Z, bounds[0], bounds[1])


# ============================================================
# DEMO - TSP
# ============================================================
def run_demo_tsp():
    cities = generate_cities(20, seed=2026)
    save_cities_csv(cities, os.path.join(RESULTS_DIR, "demo_tsp_20_cities.csv"))

    histories = {}

    for algorithm_name in ["pso_tsp", "gwo_tsp", "scso_tsp", "bso_tsp"]:
        np.random.seed(42)
        result = run_tsp_algorithm(
            algorithm_name,
            cities,
            num=50,
            iterations=100,
            **get_tsp_extra_params(algorithm_name),
        )

        histories[algorithm_name] = result["history_best"]

        plot_tsp_route(
            cities,
            result["best_route"],
            title=f"{algorithm_name}, best={result['best_cost']:.3f}",
            save_path=os.path.join(RESULTS_DIR, f"demo_{algorithm_name}_route.png"),
        )

    plot_tsp_convergence(
        histories,
        title="Porównanie zbieżności TSP",
        save_path=os.path.join(RESULTS_DIR, "demo_tsp_convergence.png"),
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
            algorithm_names=["pso", "gwo", "scso", "bso"],
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
        for algorithm_name in ["pso", "gwo", "scso", "bso"]:
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

    if BENCHMARK_MODE_TSP:
        shared_instances = build_shared_tsp_instances(CITY_VALUES, TSP_RUNS)

        for city_count, city_sets in shared_instances.items():
            save_cities_csv(
                city_sets[0],
                os.path.join(RESULTS_DIR, f"tsp_cities_{city_count}_example.csv"),
            )

        tsp_rows = benchmark_tsp_parameter_sweep(
            algorithm_names=["pso_tsp", "gwo_tsp", "scso_tsp", "bso_tsp"],
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
        tsp_histories = {}

        for algorithm_name in ["pso_tsp", "gwo_tsp", "scso_tsp", "bso_tsp"]:
            np.random.seed(42)
            result = run_tsp_algorithm(
                algorithm_name,
                demo_cities,
                num=50,
                iterations=100,
                **get_tsp_extra_params(algorithm_name),
            )

            tsp_histories[algorithm_name] = result["history_best"]

            plot_tsp_route(
                demo_cities,
                result["best_route"],
                title=f"{algorithm_name}, 20 miast, best={result['best_cost']:.3f}",
                save_path=os.path.join(RESULTS_DIR, f"route_{algorithm_name}_20cities.png"),
            )

        plot_tsp_convergence(
            tsp_histories,
            title="Porównanie zbieżności TSP (20 miast)",
            save_path=os.path.join(RESULTS_DIR, "compare_tsp_20cities.png"),
        )

    print("\nGotowe.")