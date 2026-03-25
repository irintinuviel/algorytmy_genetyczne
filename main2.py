import time
import csv
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import PillowWriter


# =========================
# FUNKCJE CELU
# =========================
def rastrigin(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def eggholder(x):
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(abs(x1 / 2 + x2 + 47))) - x1 * np.sin(
        np.sqrt(abs(x1 - (x2 + 47)))
    )


def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


# =========================
# WSPÓLNE NARZĘDZIA
# =========================
def make_surface(func, range_min, range_max, points=80):
    x = np.linspace(range_min, range_max, points)
    y = np.linspace(range_min, range_max, points)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
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

    ax.plot_surface(X, Y, Z, alpha=0.3)

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

    ax.plot_surface(X, Y, Z, alpha=0.3)

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


def plot_convergence(history, algorithm_name, function_name):
    best_values = [state["best_cost"] for state in history]

    plt.figure(figsize=(8, 5))
    plt.plot(best_values, label=f"{algorithm_name}")
    plt.xlabel("Iteracja")
    plt.ylabel("Najlepsza wartość funkcji")
    plt.title(f"Zbieżność: {algorithm_name} / {function_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_convergences(results, function_name):
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
    plt.show()


def save_benchmark_csv(rows, filename="benchmark_results.csv"):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================
# ALGORYTMY
# =========================
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


# =========================
# WYBÓR ALGORYTMU
# =========================
ALGORITHMS = {
    "pso": run_pso,
    "random": run_random_search,
    "gwo": run_gwo,
}


def run_algorithm(name, func, **kwargs):
    if name not in ALGORITHMS:
        raise ValueError(f"Nieznany algorytm: {name}")
    return ALGORITHMS[name](func, **kwargs)


# =========================
# WYBÓR FUNKCJI CELU
# =========================
FUNCTIONS = {
    "rastrigin": {
        "func": rastrigin,
        "bounds": (-5.12, 5.12),
    },
    "eggholder": {
        "func": eggholder,
        "bounds": (-512, 512),
    },
    "himmelblau": {
        "func": himmelblau,
        "bounds": (-5, 5),
    },
}


# =========================
# BENCHMARK
# =========================
def benchmark_algorithm(
    algorithm_name,
    function_name,
    runs=10,
    num=50,
    iterations=100,
    dim=2,
    extra_params=None,
):
    if extra_params is None:
        extra_params = {}

    problem = FUNCTIONS[function_name]
    func = problem["func"]
    bounds = problem["bounds"]

    best_costs = []
    times = []

    for seed in range(runs):
        np.random.seed(seed)

        start = time.perf_counter()
        result = run_algorithm(
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
        "algorithm": algorithm_name,
        "function": function_name,
        "runs": runs,
        "agents": num,
        "iterations": iterations,
        "mean_best": float(np.mean(best_costs)),
        "std_best": float(np.std(best_costs)),
        "min_best": float(np.min(best_costs)),
        "max_best": float(np.max(best_costs)),
        "mean_time": float(np.mean(times)),
    }


def benchmark_parameter_sweep(
    algorithm_names,
    function_names,
    agent_values,
    iteration_values,
    runs=10,
):
    rows = []

    for function_name in function_names:
        for algorithm_name in algorithm_names:
            for num in agent_values:
                for iterations in iteration_values:
                    if algorithm_name == "pso":
                        extra_params = {"w": 0.7, "c1": 1.5, "c2": 1.5}
                    else:
                        extra_params = {}

                    row = benchmark_algorithm(
                        algorithm_name=algorithm_name,
                        function_name=function_name,
                        runs=runs,
                        num=num,
                        iterations=iterations,
                        dim=2,
                        extra_params=extra_params,
                    )
                    rows.append(row)

                    print(
                        f"[OK] {algorithm_name:6s} | {function_name:10s} | "
                        f"agents={num:3d} | iter={iterations:3d} | "
                        f"mean={row['mean_best']:.6f}"
                    )

    return rows


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    np.random.seed(42)

    # -------------------------
    # TRYB 1: pojedynczy pokaz
    # -------------------------
    DEMO_MODE = True

    # -------------------------
    # TRYB 2: benchmark do tabeli
    # -------------------------
    BENCHMARK_MODE = True

    if DEMO_MODE:
        algorithm_name = "gwo"          # "pso", "random", "gwo"
        function_name = "rastrigin"     # "rastrigin", "eggholder", "himmelblau"

        problem = FUNCTIONS[function_name]
        func = problem["func"]
        bounds = problem["bounds"]

        X, Y, Z = make_surface(func, bounds[0], bounds[1], points=80)

        start = time.perf_counter()
        if algorithm_name == "pso":
            result = run_algorithm(
                algorithm_name,
                func,
                num=50,
                iterations=100,
                dim=2,
                bounds=bounds,
                w=0.7,
                c1=1.5,
                c2=1.5,
            )
        else:
            result = run_algorithm(
                algorithm_name,
                func,
                num=50,
                iterations=100,
                dim=2,
                bounds=bounds,
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
            filename=f"{algorithm_name}_{function_name}.gif",
            fps=15,
        )

        plot_convergence(history, algorithm_name, function_name)
        show_interactive_plot(history, X, Y, Z, bounds[0], bounds[1])

    if BENCHMARK_MODE:
        rows = benchmark_parameter_sweep(
            algorithm_names=["pso", "gwo"],
            function_names=["rastrigin", "eggholder", "himmelblau"],
            agent_values=[20, 50, 100],
            iteration_values=[50, 100, 200],
            runs=10,
        )

        save_benchmark_csv(rows, filename="benchmark_results.csv")
        print("\nZapisano benchmark do pliku: benchmark_results.csv")

        # dodatkowy prosty pokaz porównania zbieżności dla jednej funkcji
        compare_function = "rastrigin"
        compare_problem = FUNCTIONS[compare_function]
        compare_func = compare_problem["func"]
        compare_bounds = compare_problem["bounds"]

        comparison_results = {}
        for algorithm_name in ["pso", "gwo"]:
            np.random.seed(42)
            if algorithm_name == "pso":
                result = run_algorithm(
                    algorithm_name,
                    compare_func,
                    num=50,
                    iterations=100,
                    dim=2,
                    bounds=compare_bounds,
                    w=0.7,
                    c1=1.5,
                    c2=1.5,
                )
            else:
                result = run_algorithm(
                    algorithm_name,
                    compare_func,
                    num=50,
                    iterations=100,
                    dim=2,
                    bounds=compare_bounds,
                )

            comparison_results[algorithm_name] = result

        plot_multiple_convergences(comparison_results, compare_function)