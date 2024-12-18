import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
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
    image = images[10*i]
    label = labels[10*i]
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Класс {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("Лица.png")
plt.show()

# 3. Предобработка изображений
X = images.reshape(images.shape[0], -1)
X = X / 255.0
y = labels

# 2. Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13, stratify=y)

# 4. Создание и обучиние моделей градиентного бустинга
models = {
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "LGBMClassifier": LGBMClassifier(),
    "XGBClassifier": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
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

# 5. Построение матрицы ошибок для каждой модели
for name, result in results.items():
    print(f"Матрица ошибок для {name}:")
    cm = confusion_matrix(y_test, result["y_pred"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Матрица ошибок: {name}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.savefig(f"Матрица ошибок для {name}.png")
    plt.show()

# 6. Сравнение моделей по точности и времени обучения
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
comparison_df = pd.DataFrame(comparison)

plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_df, x="Модель", y="Точность", palette="viridis")
plt.title("Сравнение точности моделей")
plt.ylabel("Точность")
plt.savefig("Точность моделелей")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_df, x="Модель", y="Время обучения (сек)", palette="magma")
plt.title("Сравнение времени обучения моделей")
plt.ylabel("Время обучения (сек)")
plt.savefig("Время обучения")
plt.show()