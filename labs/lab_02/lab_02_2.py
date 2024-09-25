import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

mean = [5, 10]
dev = 2

points = np.random.normal(loc=mean, scale=dev, size=(100, 2))

plt.gca().set_aspect('equal')

plt.show()
