import pandas as pd

# Загружаем данные из CSV файла
df = pd.read_csv('./data/lab01_train.csv')

# Фильтруем курсы для студентов второго и третьего курса
courses_2_year = set(df[df['course_2'].notnull()]['course_2'].unique()) | \
                 set(df[df['fall_1'].notnull()]['fall_1'].unique()) | \
                 set(df[df['fall_2'].notnull()]['fall_2'].unique()) | \
                 set(df[df['fall_3'].notnull()]['fall_3'].unique()) | \
                 set(df[df['spring_1'].notnull()]['spring_1'].unique()) | \
                 set(df[df['spring_2'].notnull()]['spring_2'].unique()) | \
                 set(df[df['spring_3'].notnull()]['spring_3'].unique())

# Ищем курсы для студентов третьего курса
courses_3_year = set(df[df['course_3'].notnull()]['course_3'].unique()) | \
                 set(df[df['fall_1'].notnull()]['fall_1'].unique()) | \
                 set(df[df['fall_2'].notnull()]['fall_2'].unique()) | \
                 set(df[df['fall_3'].notnull()]['fall_3'].unique()) | \
                 set(df[df['spring_1'].notnull()]['spring_1'].unique()) | \
                 set(df[df['spring_2'].notnull()]['spring_2'].unique()) | \
                 set(df[df['spring_3'].notnull()]['spring_3'].unique())

# Находим курсы, на которые записались как студенты второго, так и студенты третьего курса
courses_students_of_2_and_3_year = courses_2_year.intersection(courses_3_year)

print(courses_students_of_2_and_3_year)
# Проверка результатов
assert courses_students_of_2_and_3_year == {'Численные методы',
                                            'Statistical Learning Theory',
                                            'Безопасность компьютерных систем',
                                            'Сбор и обработка данных с помощью краудсорсинга',
                                            'Высокопроизводительные вычисления',
                                            'Принятие решений в условиях риска и неопределённости',
                                            'Моделирование временных рядов'}, 'Тест не пройден'

print("Выполнено")
