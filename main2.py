import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import PillowWriter


# f do optymalizacji 1
def rastrigin(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


# PARAMS
num = 50
iterations = 100
dim = 2
range_min, range_max = -5.12, 5.12

w = 0.7
c1 = 1.5
c2 = 1.5

positions = np.random.uniform(range_min, range_max, (num, dim))
velocities = np.zeros((num, dim))

pbest_positions = positions.copy()
pbest_cost = np.array([rastrigin(p) for p in positions])

best_index = np.argmin(pbest_cost)
gbest_position = pbest_positions[best_index].copy()
gbest_cost = pbest_cost[best_index]

x = np.linspace(range_min, range_max, 100)
y = np.linspace(range_min, range_max, 100)
X, Y = np.meshgrid(x, y)
Z = 20 + (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y))

history = []

def save_gif(history, filename="pso.gif", fps=10):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, alpha=0.3)

    particles = ax.scatter([], [], [])
    pbest_scatter = ax.scatter([], [], [], marker="^")
    gbest_scatter = ax.scatter([], [], [], marker="x", s=200)

    writer = PillowWriter(fps=fps)

    with writer.saving(fig, filename, dpi=100):
        for i, state in enumerate(history):
            pos = state["pos"]
            pbest = state["pbest"]
            gbest = state["gbest"]

            z_particles = np.array([rastrigin(p) for p in pos])
            z_pbest = np.array([rastrigin(p) for p in pbest])

            particles._offsets3d = (pos[:, 0], pos[:, 1], z_particles)
            pbest_scatter._offsets3d = (pbest[:, 0], pbest[:, 1], z_pbest)
            gbest_scatter._offsets3d = ([gbest[0]], [gbest[1]], [rastrigin(gbest)])

            ax.set_title(f"Iteracja {i}")

            writer.grab_frame()

    plt.close(fig)

# PSO
for it in range(iterations):

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

        cost = rastrigin(positions[i])

        if cost < pbest_cost[i]:
            pbest_cost[i] = cost
            pbest_positions[i] = positions[i]

    best_index = np.argmin(pbest_cost)

    if pbest_cost[best_index] < gbest_cost:
        gbest_cost = pbest_cost[best_index]
        gbest_position = pbest_positions[best_index]

    history.append({
        "pos": positions.copy(),
        "pbest": pbest_positions.copy(),
        "gbest": gbest_position.copy()
    })

save_gif(history, "pso.gif", fps=15)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(X, Y, Z, alpha=0.3)

particles = ax.scatter([], [], [])
pbest_scatter = ax.scatter([], [], [], marker="^")
gbest_scatter = ax.scatter([], [], [], marker="x", s=200)

ax.set_xlim(range_min, range_max)
ax.set_ylim(range_min, range_max)

slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
slider = Slider(slider_ax, "iter", 0, iterations - 1, valinit=0, valstep=1)

current_iter = 0
paused = True


def update_plot(i):
    state = history[int(i)]

    pos = state["pos"]
    pbest = state["pbest"]
    gbest = state["gbest"]

    z_particles = np.array([rastrigin(p) for p in pos])
    z_pbest = np.array([rastrigin(p) for p in pbest])

    particles._offsets3d = (pos[:, 0], pos[:, 1], z_particles)
    pbest_scatter._offsets3d = (pbest[:, 0], pbest[:, 1], z_pbest)
    gbest_scatter._offsets3d = ([gbest[0]], [gbest[1]], [rastrigin(gbest)])

    ax.set_title(f"Iteracja {int(i)}  Best={rastrigin(gbest):.4f}")

    fig.canvas.draw_idle()


def slider_update(val):
    global current_iter
    current_iter = int(val)
    update_plot(current_iter)


slider.on_changed(slider_update)

# Spacja - stop, q - wyjdz
def on_key(event):
    global paused

    if event.key == " ":
        paused = not paused

    if event.key == "q":
        plt.close()


fig.canvas.mpl_connect("key_press_event", on_key)

update_plot(0)

while plt.fignum_exists(fig.number):

    if not paused and current_iter < iterations - 1:
        current_iter += 1
        slider.set_val(current_iter)

    plt.pause(0.05)

plt.show()
