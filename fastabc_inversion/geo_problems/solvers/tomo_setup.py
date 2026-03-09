import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
nx = 40  # width
ny = 50  # height

offset_top = 2  # 5
step = 2  # 5

ns = 24  # 9
nr = 24  # 9

x = np.zeros((ns, 2))
y = np.zeros((nr, 2))

x[:, 0] = 0
y[:, 0] = nx

for i in range(0, len(x)):
    x[i, 1] = i * ((ny - offset_top) / ns) + step
    y[i, 1] = i * ((ny - offset_top) / nr) + step


# manually draw grid
for i in range(0, nx + 1, 10):
    plt.plot((i, i), (0, ny), "-", color="gray", linewidth=0.2)

for i in range(0, ny + 1, 10):
    plt.plot((0, nx), (i, i), "-", color="gray", linewidth=0.2)

# manually draw spines
plt.plot((0, nx), (0, 0), "k-", linewidth=4)
plt.plot((0, nx), (ny, ny), "k-", linewidth=2)
plt.plot((0, 0), (0, ny), "k-", linewidth=2)
plt.plot((nx, nx), (0, ny), "k-", linewidth=2)


for i in range(0, len(x)):
    for j in range(0, len(x)):
        plt.plot((x[i, 0], y[j, 0]), (x[i, 1], y[j, 1]), "k-", linewidth=0.5)


plt.scatter(x[:, 0], x[:, 1], s=40, c="black")
plt.scatter(y[:, 0], y[:, 1], s=40, c="black", marker="s")
plt.axis("equal")
# plt.plot(x[1], y[2], 'ro-')

yticks = np.arange(0, ny + 10, 10)

ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
# Hide the right and top spines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

ax.set_xbound(lower=0, upper=nx)
ax.set_ybound(lower=0, upper=ny)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks[::-1])

plt.grid(False)

plt.show()
