import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

correlation_matrix = iris_data.corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='RdYlGn', interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.yticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Корреляция между объектами')

plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.2)
plt.tight_layout()
plt.show()