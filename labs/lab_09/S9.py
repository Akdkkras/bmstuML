import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Iterable, List, Tuple

from sklearn.model_selection import train_test_split

import random

data = pd.read_csv('boston_house_prices.csv')
# print(data.head())

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

X = pd.DataFrame(data, columns=feature_names, index=range(len(data)))
y = pd.DataFrame(data, columns=['MEDV'], index=range(len(data)))

X['target'] = y

X_train, X_test = train_test_split(X, test_size=0.25, random_state=13)

# print(X.head())
# print(y.head())


def H(R: np.array) -> float:
    """Вычислить критерий информативности для фиксированного набора объектов R.
    Предполагается, что последний столбец содержит целевое значение
    """

    y = R[:, -1]
    y_mean = np.mean(y)
    return np.mean((y - y_mean) ** 2)


def split_node(R: np.array, feature: str, t: float) -> Iterable[np.array]:
    """
    Разделить фиксированный набор объектов R по признаку feature с пороговым значением t
    """

    feature_ind = X.columns.get_loc(feature)
    left = R[R[:, feature_ind] < t]
    right = R[R[:, feature_ind] >= t]
    return left, right


def Q(R: np.array, feature: str, t: float) -> float:
    """
    Вычислить функционал качества для заданных параметров разделения
    """

    left, right = split_node(R, feature, t)
    return (len(left) / len(R)) * H(left) + (len(right) / len(R)) * H(right)


def plot_criterion(X: pd.DataFrame, y: pd.Series, feature: str):
    R = np.hstack([X.values, y.values.reshape(-1, 1)])
    ts = np.unique(X[feature])

    errors = [Q(R, feature, t) for t in ts]

    plt.figure(figsize=(10, 6))
    plt.plot(ts, errors, label=f"Q(t) ({feature})")
    plt.xlabel("Порог t")
    plt.ylabel("Ошибка Q(t)")
    plt.legend()
    plt.grid()
    plt.show()


plot_criterion(X_train.iloc[:, :-1], X_train["target"], feature_names[random.randint(0, len(feature_names) - 1)])


def get_optimal_split(X: pd.DataFrame, R: np.array, feature: str) -> Tuple[float, float]:
    feature_ind = X.columns.get_loc(feature)
    ts = np.unique(R[:, feature_ind])
    best_t = ts[0]
    min_error = float("inf")

    for t in ts:
        error = Q(R, feature, t)
        if error < min_error:
            min_error = error
            best_t = t

    return best_t, min_error


def find_best_split(X: pd.DataFrame, y: pd.Series, feature_names: list) -> Tuple[str, float, float]:
    """
    Найти наилучший признак и порог разбиения для первого шага.
    Возвращает имя признака, порог разбиения и значение ошибки.
    """

    R = np.hstack([X.values, y.values.reshape(-1, 1)])
    best_feature = None
    best_t = None
    min_error = float("inf")

    for feature in feature_names:
        t, error = get_optimal_split(X, R, feature)
        if error < min_error:
            min_error = error
            best_t = t
            best_feature = feature

    return best_feature, best_t, min_error


def visualize_split(X: pd.DataFrame, y: pd.Series, feature: str, t: float):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[feature], y, alpha=0.7, label="Данные")
    plt.axvline(x=t, color='r', linestyle='--', label=f"Порог t={t}")
    plt.xlabel(f"t ({feature})")
    plt.ylabel("Target")
    plt.legend()
    plt.grid()
    plt.show()


best_feature, best_t, min_error = find_best_split(X_train.iloc[:, :-1], X_train["target"], feature_names)
print(f"Лучший признак: {best_feature}, Порог: {best_t}, Ошибка: {min_error}")
plot_criterion(X_train.iloc[:, :-1], X_train["target"], best_feature)
visualize_split(X_train.iloc[:, :-1], X_train["target"], best_feature, best_t)

# 6

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

model = DecisionTreeRegressor(max_depth=3, random_state=13)
model.fit(X_train.iloc[:, :-1], X_train["target"])

y_pred = model.predict(X_test.iloc[:, :-1])
mse = mean_squared_error(X_test["target"], y_pred)
print(f"MSE: {mse}")
