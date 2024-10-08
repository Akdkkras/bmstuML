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

# Структурированные данные: структурированные массивы NumPy


Часто данные могут быть представлены с помощью однородного массива значений, но это возможно не всегда. 
В этом разделе рассматривается использование таких возможностей NumPy, как *структурированные массивы* (*structured arrays*) и *массивы записей* (*record arrays*), которые обеспечивают эффективное хранение сложных, неоднородных данных. 
Хотя показанные здесь приемы работы полезны для простых операций, для работы с реальными данными все же эффективее использовать такую структуру данных как `Dataframe`, предоставляемую пакетом Pandas.

```python
import numpy as np
```

<!-- #region slideshow={"slide_type": "fragment"} -->
Пусть имеется данные, описывающие признаки нескольких людей (например, имя, возраст и вес), и необходимо сохранить эти значения для использования в программе на Python.
Конечно все это можно было бы хранить в трех отдельных массивах:
<!-- #endregion -->

```python jupyter={"outputs_hidden": false}
name = ['Анна', 'Борис', 'Виктория', 'Геннадий']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 58.0, 71.5]
```

Но это выглядит неуклюже и неудобно в использовании. 
Ничто не указывает на то, что массивы как-то связаны между собой.
Лучше было бы использовать для хранения всех этих данных некую единую структуру.
NumPy для таких целей предоставляет структурированные массивы, которые представляют собой массивы с составными типами данных.
Создать структурированный массив можно используя спецификацию составного типа данных:

```python jupyter={"outputs_hidden": false}
# Составной тип данных для структурированного массива
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)
```

Здесь `U10` означает &laquo;строка Unicode максимальной длины 10&raquo;, `i4` &mdash; &laquo;4-байтовое (т.е. 32-битyjе) целое число&raquo;, а `f8`  &mdash; &laquo;8-байтовое (т. е. 64-битное) число с плавающей точкой&raquo;.
Другие варианты кодов типов обсудим далее.


Теперь, после того как создан пустой массив-контейнер, можно заполнить массив списками значений:

```python jupyter={"outputs_hidden": false}
data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)
```

Как и требовалось, теперь данные организованы в одной удобной структуре.

Удобство структурированных массивов в том, что теперь к значениям можно обращаться как по индексу, так и по имени:

```python jupyter={"outputs_hidden": false}
# Получить все имена
data['name']
```

```python jupyter={"outputs_hidden": false}
# Получить первую строку данных
data[0]
```

```python jupyter={"outputs_hidden": false}
# Получить имя из последней строки
data[-1]['name']
```

Использование булевой маски позволяет выполнять даже более сложные операции, такие как фильтрация по возрасту:

```python jupyter={"outputs_hidden": false}
# Получить имена людей чей возраст меньше 30 лет
data[data['age'] < 30]['name']
```

Обратите внимание: если нужно выполнять какие-либо более сложные операции, вероятно, следует все же использовать пакет Pandas.
Пакет Pandas предоставляет объект `Dataframe`, представляющий собой структуру, построенную на массивах NumPy, которая предоставляет множество полезных функций манипулирования данными, а также многое, многое другое.


## Создание структурированных массивов

Типы данных структурированного массива можно указать несколькими способами.
Уже использованный метод с посмощью словаря:

```python jupyter={"outputs_hidden": false}
np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})
```

Для больщей гибкости числовые типы можно указать как с помощью типов данных языка Python так и типов `dtype` пакета NumPy:

```python jupyter={"outputs_hidden": false}
np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_, 10), int, np.float32)})
```

Также составной тип данных можно задать в виде списка кортежей:

```python jupyter={"outputs_hidden": false}
np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
```

Если название признаков не имеют значения, можно указать только сами типы в виде строке, разделенной запятыми:

```python jupyter={"outputs_hidden": false}
np.dtype('S10,i4,f8')
```

Сокращенные коды форматов могут показаться запутанными, но они построены на простых принципах.
Первый (необязательный) символ &mdash; `<` или `>` означает &laquo;прямой порядок байтов&raquo; или &laquo;обратный порядок байтов&raquo; соответственно, и определяет соглашение о порядке значимых битов.
Следующий символ указывает тип данных: символы, байты, целые числа, числа с плавающей точкой и т.д. (см. таблицу ниже).
Последний символ или символы представляют размер объекта в байтах.

| Символ   | Описание                             | Пример                            |
| ---------| -------------------------------------|-----------------------------------|
| `b`      | Байт                                 | `np.dtype('b')`                   |
| `i`      | Знаковое целое число                 | `np.dtype('i4') == np.int32`      |
| `u`      | Беззнаковое целое число              | `np.dtype('u1') == np.uint8`      |
| `f`      | С плавающей точкой                   | `np.dtype('f8') == np.int64`      |
| `c`      | Комплексное число с плавающей точкой | `np.dtype('c16') == np.complex128`|
| `S`, `a` | Строка                               | `np.dtype('S5')`                  |
| `U`      | Строка Юникода                       | `np.dtype('U') == np.str_`        |
| `V`      | Необработанные данные (void)         | `np.dtype('V') == np.void`        |


## Более сложные типы данных

Можно определить еще более сложные типы данных.
Например, можно создать тип, каждый элемент которого содержит массив или матрицу значений.
Например, создадим тип данных с компонентом `mat`, состоящим из матрицы с плавающей точкой размером $3\times 3$:

```python jupyter={"outputs_hidden": false}
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])
```

Теперь каждый элемент в массиве `X` состоит из `id` и матрицы $3\times 3$.
Почему такой тип данных может оказаться более предпочтительным, чем простой многомерный массив или, может быть, словарь Python?
Причина в том, что `dtype` NumPy напрямую соответствует описанию структуры из языка C, что дает возможность обращаться к содержащему этот массив буферу памяти непосредственно из соответствующим образом написанной программы на языке C. 
В случаях когда необходимо написать на языке Python интерфейс к уже существующей библиотеке на языке C или Fortran, которая работает со структурированными данными, структурированные массивы оказываются весьма полезными!


## Массивы записей: структурированные массивы с дополнительными возможностями

NumPy предоставляет класс `np.recarray`, который почти идентичен только что описанным структурированным массивам, но имеет одну дополнительную особенность: к полям можно обращаться как к атрибутам, а не как к ключам словаря.


Сейчас для доступ к возрасту необходимо использовать синтаксис словарей:

```python jupyter={"outputs_hidden": false}
data['age']
```

Если же вместо этого рассматривать данные как массив записей, то можно получать к ним доступ гораздо удобнее, используя синтаксис обращения к атрибутам объекта (точечную нотацию):

```python jupyter={"outputs_hidden": false}
data_rec = data.view(np.recarray)
data_rec.age
```

Недостатком такого способа является то, что при использование массивов записей неизбежны дополнительные накладные расходы, связанные с доступом к полям (даже при использовании синтаксиса словарей). 
Это легко проверить:

```python jupyter={"outputs_hidden": false}
%timeit data['age']
%timeit data_rec['age']
%timeit data_rec.age
```

Стоит ли более удобная запись дополнительных накладных расходов, будет зависеть от конкретного приложения.
