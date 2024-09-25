import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(1)
mean = [5, 10]
dev = 2
points = np.random.normal(loc=mean, scale=dev, size=(100, 2))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(points[:, 0], bins=10, alpha=0.5, color='blue', edgecolor='black', density=True)
plt.title('Распределение вдоль оси X')
sns.kdeplot(points[:, 0], color='red')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
plt.hist(points[:, 1], bins=10, alpha=0.5, color='blue', edgecolor='black', density=True)
plt.title('Распределение вдоль оси Y')
sns.kdeplot(points[:, 1], color='red')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()
