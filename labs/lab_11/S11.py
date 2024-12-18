import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# ЗАДАНИЕ 1

# Импортируем датасет
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Разведовательный анализ данных
train_data.info()
print(train_data.head())

sns.countplot(x="Survived", data=train_data)
plt.title("Распределение выживших и погибших")
plt.show()

sns.histplot(data=train_data, x="Age", hue="Survived", kde=True)
plt.title("Распределение возраста по классам выживаемости")
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=train_data)
plt.title("Выживаемость по классам пассажиров")
plt.show()

# Обработка признаков
def preprocess_data(data):
    data = data.copy()

    data["Age"].fillna(data["Age"].median(), inplace=True)
    data["Fare"].fillna(data["Fare"].median(), inplace=True)
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    data.drop(["Name", "Ticket", "Cabin", 'Embarked'], axis=1, inplace=True)

    return data


train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Разделение данных на обучающую и тестовую выборки
# X_train = train_data.drop(columns=['Survived']).copy()
# y_train = train_data['Survived'].copy()
#
# X_test = test_data.copy()
# y_test = pd.read_csv("gender_submission.csv")['Survived']

# ИЛИ

X = train_data.drop(columns=['Survived'])
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Построение базовой модели
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Оценка точности модели
def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test)

    # Матрица ошибк
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Матрица ошибок для {name}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.show()

    # Точность
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели ({name}): {accuracy:.4f}")

evaluate_model(random_forest_model, "RandomForestClassifier", X_test, y_test)

# Настройка гиперпараметров
# RS
param_distributions = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    scoring="accuracy",
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Лучшие параметры (RandomizedSearchCV):", random_search.best_params_)
random_search_best_model = random_search.best_estimator_
evaluate_model(random_search_best_model, "RandomizedSearchCV", X_test, y_test)

# GS
param_grid = {
    "n_estimators": [50, 100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print("Лучшие параметры (GridSearchCV):", grid_search.best_params_)
grid_search_best_model = grid_search.best_estimator_
evaluate_model(grid_search_best_model, "GridSearchCV", X_test, y_test)

# ЗАДАНИЕ 2

models = {
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=10),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, name, X_test, y_test)