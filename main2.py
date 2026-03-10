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
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def eggholder(x):
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(abs(x1 / 2 + x2 + 47))) - x1 * np.sin(
        np.sqrt(abs(x1 - (x2 + 47)))
    )


# =========================
# WSPOLNE NARZEDZIA - RUSZANIE NIE MA SENSU
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


def save_gif(history, func, X, Y, Z, filename="opt.gif", fps=10):
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

            z_agents = np.array([func(p) for p in pos])
            z_best = func(best)

            agents_scatter._offsets3d = (pos[:, 0], pos[:, 1], z_agents)
            best_scatter._offsets3d = ([best[0]], [best[1]], [z_best])

            ax.set_title(f"Iteracja {i}  Best={z_best:.4f}")
            writer.grab_frame()

    plt.close(fig)


def show_interactive_plot(history, func, X, Y, Z, range_min, range_max):
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

        z_agents = np.array([func(p) for p in pos])
        z_best = func(best)

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
            plt.close()

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


# =========================
# ALGORYTMY
# =========================
def run_pso(func, num=50, iterations=100, dim=2, bounds=(-5.12, 5.12), w=0.7, c1=1.5, c2=1.5):
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

        history.append(
            {
                "pos": positions.copy(),
                "best": gbest_position.copy(),
                "best_cost": gbest_cost,
            }
        )

    return history

# algorytm z czapy courtesy of OpenAI
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

        history.append(
            {
                "pos": positions.copy(),
                "best": best.copy(),
                "best_cost": best_cost,
            }
        )

    return history


# =========================
# WYBOR ALGORYTMU
# =========================
ALGORITHMS = {
    "pso": run_pso,
    "random": run_random_search,
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
}


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    np.random.seed(42)

    algorithm_name = "pso"          # "pso" albo "random"
    function_name = "rastrigin"     # "rastrigin" coś co zaraz zaimplementuję i swear

    problem = FUNCTIONS[function_name]
    func = problem["func"]
    bounds = problem["bounds"]

    X, Y, Z = make_surface(func, bounds[0], bounds[1], points=80)

    history = run_algorithm(
        algorithm_name,
        func,
        num=50,
        iterations=100,
        dim=2,
        bounds=bounds,
    )

    save_gif(history, func, X, Y, Z, filename=f"{algorithm_name}_{function_name}.gif", fps=15)
    show_interactive_plot(history, func, X, Y, Z, bounds[0], bounds[1])