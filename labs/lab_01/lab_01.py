import numpy as np
from numpy.ma.core import array

score = 0

# задание 1 (1 балл)

def max_after_zero(x: np.ndarray) -> int:
    """
    Задание: найти максимальный элемент массива среди элементов, которым предшествует ноль

    Вход: np.array([0, 2, 0, 3])
    Выход: 3
    """

    assert False, 'Не реализовано!'  # Здесь должен быть ваш код

# x = np.array([0, 1, 12, 0, 6, 0, 10, 0])
# assert max_after_zero(x) == 10, 'Тест не пройден'
#
# x = np.array([0, 3, 2, 0, 8, 0, 1, 10])
# assert max_after_zero(x) == 8, 'Тест не пройден'
#
# print("Выполнено")
# score += 1
#
#
#
# # задание 2 (1 балл)
#
# def block_matrix(block: np.array) -> np.array:
#     """
#     Задание: построить блочную матрицу из четырех блоков, где каждый блок представляет собой заданную матрицу
#
#     Вход: np.array([[1, 2], [3, 4]])
#     Выход: np.array([[1, 2, 1, 2],
#                      [3, 4, 3, 4],
#                      [1, 2, 1, 2],
#                      [3, 4, 3, 4]])
#     """
#     assert False, 'Не реализовано!' # Здесь должен быть ваш код
#
#
# block = np.array([[1, 3, 3], [7, 0, 0]])
# assert np.allclose(
#     block_matrix(block),
#     np.array([[1, 3, 3, 1, 3, 3],
#               [7, 0, 0, 7, 0, 0],
#               [1, 3, 3, 1, 3, 3],
#               [7, 0, 0, 7, 0, 0]])
# ), 'Тест не пройден'
#
# print("Выполнено")
# score += 1
#
#
#
# # задание 3 (1 балл)
#
# def diag_prod(matrix: np.array) -> int:
#     """
#     Задание: вычислить произведение всех ненулевых диагональных элементов квадратной матрицы
#
#     Вход: np.array([[3, 5, 1, 4],
#                     [6, 2, 7, 9],
#                     [3, 6, 0, 8],
#                     [1, 3, 4, 6]])
#     Выход: 36
#     """
#     assert False, 'Не реализовано!' # Здесь должен быть ваш код
#
#
# matrix = np.array([[0, 1, 2, 3],
#                    [4, 5, 6, 7],
#                    [8, 9, 10, 11],
#                    [12, 13, 14, 15]])
# assert diag_prod(matrix) == 750, 'Тест не пройден'
#
# print("Выполнено")
# score += 1
#
# # задание 4 (1 балл)
#
# from typing import Tuple
#
#
# class StandardScaler:
#     """
#     Задание: класс реализует StandardScaler из библиотеки sklearn
#
#     см. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
#     В качестве входных данных метод fit принимает матрицу, в которой признаки объектов расположены в столбцах
#     Метод fit должен вычислять среднее значение (mean_) и дисперсию (var_) для каждого из признаков (столбца),
#     и сохранять их в атрубутах объекта self.mean_ и self.var_ соответственно.
#
#     Метод transform должен нормализовать матрицу с помощью предварительно вычисленных mean_ и sigma,
#     где sigma = sqrt(var_) - среднеквадратическое отклонение
#
#     Вход: np.array([[2, 603, 250],
#                     [1, 154, 500],
#                     [7, 893, 350]])
#     Выход: np.array([[-0.50800051,  0.17433393, -1.13554995],
#                      [-0.88900089, -1.3025705 ,  1.29777137],
#                      [ 1.3970014 ,  1.12823657, -0.16222142]])
#     """
#
#     def fit(self, X: np.array) -> float:
#         assert False, 'Не реализовано!'  # Здесь должен быть ваш код
#
#     def transform(self, X: np.array) -> np.array:
#         assert False, 'Не реализовано!'  # Здесь должен быть ваш код
#
#
#
# matrix = np.array([[1, 4, 4200], [0, 10, 5000], [1, 2, 1000]])
#
# scaler = StandardScaler()
# scaler.fit(matrix)
#
# assert np.allclose(
#     scaler.mean_,
#     np.array([0.66667, 5.3333, 3400])
# ), 'Тест не пройден. Некорректное значение scaler.mean_'
#
# assert np.allclose(
#     scaler.var_,
#     np.array([0.22222, 11.5556, 2986666.67])
# ), 'Тест не пройден. Некорректное значение scaler.var_'
#
# assert np.allclose(
#     scaler.transform(matrix),
#     np.array([[ 0.7071, -0.39223,  0.46291],
#               [-1.4142,  1.37281,  0.92582],
#               [ 0.7071, -0.98058, -1.38873]])
# ), 'Тест не пройден. Некорректный результат scaler.transform(matrix)'
#
#
# print("Выполнено")
# score += 1
#
#
# # задание 5 (1 балл)
#
# def antiderivative(coefs: np.array, const: float) -> np.array:
#     """
#     Задание: Вычислить первообразную полинома
#
#     coefs - массив коэффициентов полинома
#     const - произвольная постоянная
#     Массив коэффициентов [6, 0, 1] соответствует 6x^2 + 0x^1 + 1
#     Соответствующая первообразная будет иметь вид: 2x^3 + 0x^2 + 1x + const,
#     В результате получается массив коэффициентов [2, 0, 1, const]
#
#     Вход: [8, 12, 8, 1], 42
#     Выход: [2., 4., 4., 1., 42.]
#     """
#     assert False, 'Не реализовано!'  # Здесь должен быть ваш код
#
#
#
# coefs = np.array([4, 6, 0, 1])
# assert np.allclose(
#     antiderivative(coefs),
#     np.array([1., 2., 0., 1., 42.])
# ), 'Тест не пройден.'
#
# coefs = np.array([1, 7, -12, 21, -6])
# assert np.allclose(
#     antiderivative(coefs),
#     np.array([ 0.2, 1.75, -4., 10.5, -6., 42.])
# ), 'Тест не пройден.'
#
# print("Выполнено")
# score += 1
#
#
# print('Итоговый балл:', score)