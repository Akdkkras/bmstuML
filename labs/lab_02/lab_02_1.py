import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

np.random.seed(1)

mean = [5, 10]
dev = 2

points = np.random.normal(loc=mean, scale=dev, size=(100, 2))

plt.scatter(points[:, 0], points[:, 1], color='blue', marker='.', s = 8, zorder=0)
plt.scatter(mean[0], mean[1], color='red', marker='.', s=100, zorder=1)

radius1 = 5.5
radius2 = radius1 * 3

circle1 = patches.Circle(mean, radius1, fill=False, edgecolor='red', linestyle='--')
circle2 = patches.Circle(mean, radius2, fill=False, edgecolor='red', linestyle='--')

plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

plt.xlim(mean[0] - radius2 - 1, mean[0] + radius2 + 1)
plt.ylim(mean[1] - radius2 - 1, mean[1] + radius2 + 1)
plt.gca().set_aspect('equal')

plt.show()