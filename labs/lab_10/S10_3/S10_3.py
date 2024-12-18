import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')

# 1. Загрузка датасета Olivetti Faces
faces = fetch_olivetti_faces()
images = faces.images  # Изображения размером 64x64
labels = faces.target  # Метки классов (номера людей)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

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

# 5. Создание StackingClassifier
stacking_clf = StackingClassifier(
    estimators=[
        ('LogisticRegression', log_clf),
        ('DecisionTreeClassifier', tree_clf),
        ('KNeighborsClassifier', knn_clf),
        ('NaiveBayes', nb_clf)
    ],
    final_estimator=LogisticRegression(max_iter=5000, random_state=42)
)

# 6. Обучение StackingClassifier и базовых моделей
models = {
    "LogisticRegression": log_clf,
    "DecisionTreeClassifier": tree_clf,
    "KNeighborsClassifier": knn_clf,
    "NaiveBayes": nb_clf,
    "StackingClassifier": stacking_clf
}

results = {}

for name, model in models.items():
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

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "training_time": training_time,
        "y_pred": y_pred
    }

# 7. Матрица ошибок для StackingClassifier
cm = confusion_matrix(y_test, results["StackingClassifier"]["y_pred"])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Матрица ошибок: StackingClassifier")
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.savefig("Матрица ошибок для StackingClassifier")
plt.show()

# 8. Сравнение моделей
comparison = {
    "Модель": [],
    "Точность": [],
    "Время обучения (сек)": []
}

for name, result in results.items():
    comparison["Модель"].append(name)
    comparison["Точность"].append(result["accuracy"])
    comparison["Время обучения (сек)"].append(result["training_time"])

# Визуализация сравнения
import pandas as pd
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
plt.savefig("Временя обучения")
plt.show()