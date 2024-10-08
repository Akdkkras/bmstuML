---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Несколько графиков на одном рисунке


Иногда полезно сравнивать различные представления данных разместив их бок о бок.
Для этой цели в Matplotlib реализована концепция *подграфиков* (subplots): несколько маленьких систем координат могут сосуществовать  в пределах одного рисунке.
Эти подграфики могут представлять собой вставки, сетки графиков или другие более сложные схемы размещения.
Рассмотрим четыре процедуры создания подграфиков в Matplotlib.

```python
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
plt.style.use('default')
```

## `plt.axes`: создание подграфиков вручную

Использование функции `plt.axes` &mdash; cамый простой метод создания осей координат. 
По умолчанию создается стандартный объект осей, заполняющий весь график.
`plt.axes` также принимает необязательный аргумент, представляющий собой список из четырех чисел в системе координат рисунка.
Эти числа означают `[левый угол, низ,  ширина, высота]` в системе координат рисунка, отсчет которых начинается с 0 в нижнем левом и заканчивается 1 в верхнем правом углу рисунка.

Например, можно создать вложенные оси координат в правом верхнем углу других осей координат, установив положение *x* и *y* равным 0,6 (то есть начиная с 60% ширины и 60% высоты рисунка), а значения ширины и высоты &mdas; равными 0,2 (то есть размер осей составит 20% ширины и 20% высоты рисунка):

```python jupyter={"outputs_hidden": false}
ax1 = plt.axes()  # стандартные оси
ax2 = plt.axes([0.6, 0.65, 0.2, 0.2])
```

Эквивалент этой команды в объектно-ориентированном интерфейсе & `fig.add_axes()`. 
Воспользуемся ею для создания двух расположенных друг над другом систем координат:

```python jupyter={"outputs_hidden": false}
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
                   xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
                   ylim=(-1.2, 1.2))

x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));
```

В результате получены две оси координат (верхняя без меток), которые просто соприкасаются: нижняя часть верхней панели (в позиции 0,5) совпадает с верхней частью нижней панели (в позиции 0,1 + 0,4).


## `plt.subplot`: простые сетки подграфиков

Выровненные столбцы или строки подграфиков &mdash; бывают необходимы  достаточно часто.
Поэтому в Matplotlib есть несколько удобных процедур, которые упрощают их создание.
Самая низкоуровневая из них &mdash; это `plt.subplot()`, которая создает отдельный подграфик внутри сетки.
Как видно, эта команда принимает три целочисленных аргумента &mdash; количество строк, количество столбцов и номер графика, который будет создан в этой схеме.
Отсчет идет от верхнего левого угла до нижнего правого:

```python jupyter={"outputs_hidden": false}
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),
             fontsize=18, ha='center')
```

Для регулировки расстояния между этими графиками можно использовать команду `plt.subplots_adjust`.
В следующем коде используется эквивалентная объектно-ориентированная команда `fig.add_subplot()`:

```python jupyter={"outputs_hidden": false}
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)),
           fontsize=18, ha='center')
```

Аргументы `hspace` и `wspace` функции `plt.subplots_adjust`, задают интервал по высоте и ширине фигуры в единицах размера подграфика (в данном случае интервал составляет 40% от ширины и высоты подграфика).


## ``plt.subplots``: Вся сетка за один раз

Описанный выше подход может оказаться довольно утомительным при создании большой сетки подграфиков, особенно если необходимо скрыть метки осей x и y на внутренних графиках.
Для этой цели `plt.subplots()` является более простым инструментом в использовании (обратите внимание на `s` в конце `subplots`). Вместо создания одного подграфика, эта функция создает полную сетку подграфиков в одной строке, возвращая их в массиве NumPy.
Аргументами являются количество строк и количество столбцов, а также необязательные ключевые слова `sharex` и `sharey`, которые позволяют указать связи между различными осями.

Создадим сетку $2 \times 3$ подграфиков, где все оси в одной строке имеют общую шкалу оси Y, а все оси в одном столбце имеют общую шкалу оси X:

```python jupyter={"outputs_hidden": false}
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
```

Обратите внимание, что, указание `sharex` и `sharey`, автоматически удалило внутренние метки на сетке, чтобы сделать график более наглядным.
Результирующая сетка экземпляров осей возвращается в массиве NumPy, что позволяет удобно указывать нужные оси с использованием стандартной нотации индексации массива:

```python jupyter={"outputs_hidden": false}
# оси находятся в двумерном массиве, индексированном по [строка, столбец]
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)),
                      fontsize=18, ha='center')
fig
```

По сравнению с `plt.subplot()`, `plt.subplots()` более соответствует традиционной индексации Python, начинающейся с 0.


## ``plt.GridSpec``: более сложные схемы

Чтобы выйти за рамки обычной сетки и перейти к подграфикам, охватывающим несколько строк и столбцов, лучшим инструментом является `plt.GridSpec()`.
Объект `plt.GridSpec()` сам по себе не создает график; это просто удобный интерфейс, распознаваемый командой `plt.subplot()`.
Например, вызов `GridSpec` для сетки из двух строк и трех столбцов с заданными значениями ширины и высоты будет выглядеть
следующим образом:

```python jupyter={"outputs_hidden": false}
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
```

Теперь можно указать местоположение и протяженность подграфиков, используя синтаксис срезов Python:

```python jupyter={"outputs_hidden": false}
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2]);
```

Подобное гибкое выравнивание сетки имеет широкий спектр применения, например, при создании графиков гистограмм с несколькими
системами координат:

```python jupyter={"outputs_hidden": false}
# Создаем ормально распределенные данные
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# Настраиваем оси с помощью gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# Точки рассеяния на главных осях
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)

# гистограмма на прикрепленных осях
x_hist.hist(x, 40, histtype='stepfilled',
            orientation='vertical', color='gray')
x_hist.invert_yaxis()

y_hist.hist(y, 40, histtype='stepfilled',
            orientation='horizontal', color='gray')
y_hist.invert_xaxis()
```

Этот тип распределения, отображаемый вдоль его границ, настолько распространен, что для него имеется собственный API построения графиков в пакете Seaborn.
