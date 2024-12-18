import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import seaborn as sns
import time
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 1. Загрузка датасета
faces = fetch_olivetti_faces()
images = faces.images  # Изображения размером 64x64
labels = faces.target  # Метки классов (номера людей)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i in range(5):
    image = images[10 * i]
    label = labels[10 * i]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Класс {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# 3. Предобработка данных
X = images.reshape(images.shape[0], -1)
X = X / 255.0
y = labels

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Создание базовых классификаторов
log_clf = LogisticRegression(max_iter=5000, random_state=42)
tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=5)
nb_clf = GaussianNB()

# 5. Обучение каждого базового классификатора
base_models = {
    "LogisticRegression": log_clf,
    "DecisionTreeClassifier": tree_clf,
    "KNeighborsClassifier": knn_clf,
    "NaiveBayes": nb_clf
}

base_results = {}

for name, model in base_models.items():
    print(f"Тренировка модели: {name}")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Модель: {name}")
    print(f"Точность: {accuracy:.4f}")
    print("Отчет классификации:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

    base_results[name] = {
        "model": model,
        "accuracy": accuracy,
        "training_time": training_time,
        "y_pred": y_pred
    }

# 6. Создание и обучение VotingClassifier
voting_clf = VotingClassifier(
    estimators=[
        ("LogisticRegression", log_clf),
        ("DecisionTreeClassifier", tree_clf),
        ("KNeighborsClassifier", knn_clf),
        ("NaiveBayes", nb_clf)
    ],
    voting="soft"
)

print("Тренировка VotingClassifier")
start_time = time.time()
voting_clf.fit(X_train, y_train)
voting_training_time = time.time() - start_time

voting_y_pred = voting_clf.predict(X_test)

voting_accuracy = accuracy_score(y_test, voting_y_pred)
print("Модель: VotingClassifier")
print(f"Точность: {voting_accuracy:.4f}")
print("Отчет классификации:")
print(classification_report(y_test, voting_y_pred))
print("-" * 50)

# Сравнение моделей по точности и времени обучения
comparison = {
    "Модель": [],
    "Точность": [],
    "Время обучения (сек)": []
}

# Добавляем базовые модели
for name, result in base_results.items():
    comparison["Модель"].append(name)
    comparison["Точность"].append(result["accuracy"])
    comparison["Время обучения (сек)"].append(result["training_time"])

# Добавляем ансамбль
comparison["Модель"].append("VotingClassifier")
comparison["Точность"].append(voting_accuracy)
comparison["Время обучения (сек)"].append(voting_training_time)

# Визуализация сравнения
comparison_df = pd.DataFrame(comparison)
print(comparison_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_df, x="Модель", y="Точность", palette="viridis")
plt.title("Сравнение точности моделей")
plt.ylabel("Точность")
plt.savefig("Точность моделей")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_df, x="Модель", y="Время обучения (сек)", palette="magma")
plt.title("Сравнение времени обучения моделей")
plt.ylabel("Время обучения (сек)")
plt.savefig("Время обучения")
plt.show()

# 7. Матрица ошибок для VotingClassifier
cm = confusion_matrix(y_test, voting_y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Матрица ошибок: VotingClassifier")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.savefig("Матрица ошибок для VotingClassifier")
plt.show()