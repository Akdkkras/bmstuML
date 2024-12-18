import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

ratings = pd.read_csv('music_listening.csv', low_memory=False)
# print(ratings.head())
ratings = ratings.T         # транспонируем DataFrame
# print(ratings.head())

# 1
# 1.1

ratings = ratings.drop('user', axis=0)      # удаляем строку user
# print(ratings.head())

# 1.2

count_rows = ratings.shape[0]
# print(f"Количесвто строк: {count_rows}")

# 1.3

from sklearn.preprocessing import normalize

ratings = ratings.replace(',', '.')
ratings = ratings.apply(pd.to_numeric, errors='coerce')
ratings = ratings.fillna(0)
# print(ratings.head())
ratings_normalized = normalize(ratings, axis=1)             # нормализуем ratings под условие - сумма квадратов в строке равна 1
ratings = pd.DataFrame(ratings_normalized, index=ratings.index, columns=ratings.columns)
# print(ratings.head())
# sum_of_squares_first_row = (ratings.iloc[0] ** 2).sum()
# print(sum_of_squares_first_row)                         # проверка того, что сумма квадратов равна 1 (на примере первой строки)


# 1.4

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)      # создание объекта KMeans с параметрами (кол-во кластеров и сид)
kmeans.fit(ratings)                                 # обучение модели на данных из ratings
clusters = kmeans.predict(ratings)                  # предсказание для данных ratings на уже обученной модели [ clusters - соответствующий строкам ratings массив кластеров ]
centroids = kmeans.cluster_centers_                 # получение центроидов всех кластеров

# 1.5

from scipy.spatial.distance import cosine

beatles_vector = ratings.loc["the beatles"].values
coldplay_vector = ratings.loc["coldplay"].values

cosine_distance = cosine(beatles_vector, coldplay_vector)
# print(f"Расстояние между 'The Beatles' и 'Coldplay': {cosine_distance}")


# 1.6

def pClosest(points, pt, K=10):
    ind = [i[0] for i in sorted(enumerate(points), key=lambda x: cosine(x[1], pt))]
    return ind[:K]


# for cluster_num in range(5):
#     cluster_indices = [i for i, cluster_id in enumerate(clusters) if cluster_id == cluster_num]     # cluster_indices - массив индексов всех строк, относящихся к кластеру текущей итерации
#     cluster_points = ratings.iloc[cluster_indices].values                                           # cluster_points - массив точек по соответствующим строкам ratings
#     closest_indices = pClosest(cluster_points, centroids[cluster_num], K=10)                        # индексы десяти ближайших к центру
#     closest_artists = ratings.iloc[cluster_indices].iloc[closest_indices].index.tolist()            # имена по этим индексам
#
#     print(f"Топ-10 исполнителей для кластера {cluster_num}:")
#     for _, artist in enumerate(closest_artists):
#         print("-", artist)
#     print()

# 2

import time
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, adjusted_rand_score

data = make_moons(n_samples=100, noise=0.1, random_state=42)

X = data[0]
y = data[1]

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Изначальные данные")
# plt.show()


def check_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    total_time = time.time() - start_time
    return result, total_time


def check_cluster_accuracy(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    total = len(y_true)
    correct = int(accuracy * total)
    return correct, total, accuracy


results = {}

# 2.1 KMeans

kmeans, time_kmeans = check_time(KMeans(n_clusters=2, random_state=42).fit, X)
kmeans_labels = kmeans.labels_
correct, total, accuracy = check_cluster_accuracy(y, kmeans_labels)
res = {
    'AC': accuracy,
    'ARI': adjusted_rand_score(y, kmeans_labels)
}
results['KMeans'] = res

print("KMeans:")
print(f"AC: {int(results['KMeans']['AC']*100)}% ({correct}/{total})")
print(f"ARI: {results['KMeans']['ARI']}")
print(f"Время работы: {time_kmeans} сек")
print()

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title("KMeans кластеризация")
plt.show()

# 2.2 DBSCAN

eps_values = [0.05, 0.1, 0.2, 0.28, 0.3, 0.32]
min_samples_values = [4, 5, 6, 7]
best_ari = -1
best_eps, best_min_samples = None, None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan, time_dbscan = check_time(DBSCAN(eps=eps, min_samples=min_samples).fit, X)
        dbscan_labels = dbscan.labels_
        ari = adjusted_rand_score(y, dbscan_labels)
        if ari > best_ari:
            best_ari = ari
            best_eps, best_min_samples = eps, min_samples

dbscan, time_dbscan = check_time(DBSCAN(eps=best_eps, min_samples=best_min_samples).fit, X)
dbscan_labels = dbscan.labels_
correct, total, accuracy = check_cluster_accuracy(y, dbscan_labels)
res = {
    'AC': accuracy,
    'ARI': best_ari
}
results['DBSCAN'] = res

print("DBSCAN:")
print(f"Лучшие параметры: eps={best_eps}, min_samples={best_min_samples}")
print(f"AC: {int(results['DBSCAN']['AC']*100)}% ({correct}/{total})")
print(f"ARI: {results['DBSCAN']['ARI']}")
print(f"Время работы: {time_dbscan} сек")
print()

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels)
plt.title("DBSCAN кластеризация")
plt.show()

# 2.3 Agglomerative Clustering
linkage_methods = ['ward', 'complete', 'average', 'single']
best_ari = -1
best_linkage = None

for linkage in linkage_methods:
    clustering, time_hc = check_time(AgglomerativeClustering(n_clusters=2, linkage=linkage).fit, X)
    hc_labels = clustering.labels_
    ari = adjusted_rand_score(y, hc_labels)
    if ari > best_ari:
        best_ari = ari
        best_linkage = linkage

clustering, time_hc = check_time(AgglomerativeClustering(n_clusters=2, linkage=best_linkage).fit, X)
hc_labels = clustering.labels_
correct, total, accuracy = check_cluster_accuracy(y, hc_labels)
res = {
    'AC': accuracy,
    'ARI': best_ari
}
results['Agglomerative'] = res

print("Agglomerative:")
print(f"Лучший метод linkage: {best_linkage}")
print(f"AC: {int(results['Agglomerative']['AC']*100)}% ({correct}/{total})")
print(f"ARI: {results['Agglomerative']['ARI']}")
print(f"Время работы: {time_hc} сек")
print()

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=hc_labels)
plt.title("Иерархическая кластеризация")
plt.show()

# 2.4 Spectral Clustering

best_ari = -1
best_n_neighbors = None

for n_neighbors in range(1, 20):
    spectral, time_sc = check_time(
        SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=42).fit, X)
    sc_labels = spectral.labels_
    ari = adjusted_rand_score(y, sc_labels)
    if ari > best_ari:
        best_ari = ari
        best_n_neighbors = n_neighbors

spectral, time_sc = check_time(
    SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=best_n_neighbors, random_state=42).fit,
    X)
sc_labels = spectral.labels_
correct, total, accuracy = check_cluster_accuracy(y, sc_labels)
res = {
    'AC': accuracy,
    'ARI': best_ari
}
results['Spectral'] = res

print("Spectral:")
print(f"Лучший параметр n_neighbors: {best_n_neighbors}")
print(f"AC: {int(results['Spectral']['AC']*100)}% ({correct}/{total})")
print(f"ARI: {results['Spectral']['ARI']}")
print(f"Время работы: {time_sc} сек")
print()

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=sc_labels)
plt.title("Спектральная кластеризация")
plt.show()

# 2.5 Итоги

print("Сравнение методов:")
for method, metrics in results.items():
    print(f"{method}: {metrics['ARI']:.3f}")
