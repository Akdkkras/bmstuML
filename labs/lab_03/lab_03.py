import numpy as np
import pandas as pd

score = 0

df = pd.read_csv('./data/lab01_train.csv')
df.tail(10)

###

number_of_request_course_2 = df['course_2'].notnull().sum()
number_of_request_course_3 = df['course_3'].notnull().sum()

assert number_of_request_course_2 == 138, 'Тест не пройден'
assert number_of_request_course_3 == 223, 'Тест не пройден'
print("Выполнено")
score += 0.5

###

is_equal_percentil = df['percentile'].duplicated(keep=False).any()

assert is_equal_percentil == True, 'Тест не пройден'
print("Выполнено")
score += 0.5

###

course_2_na = df['course_2'].isnull().sum()
course_3_na = df['course_3'].isnull().sum()
is_mi_student_na = df['is_mi_student'].isnull().sum()
is_ml_student_na = df['is_ml_student'].isnull().sum()
is_first_time_na = df['is_first_time'].isnull().sum()
blended_na = df['blended'].isnull().sum()
percentile_na = df['percentile'].isnull().sum()

assert course_2_na == 223, 'Тест не пройден'
assert course_3_na == 138, 'Тест не пройден'
assert is_mi_student_na == 343, 'Тест не пройден'
assert is_ml_student_na == 304, 'Тест не пройден'
assert blended_na == 223, 'Тест не пройден'
assert is_first_time_na == 0, 'Тест не пройден'
assert percentile_na == 0, 'Тест не пройден'

print("Выполнено")
score += 0.5

###

# Заполните пропуски пустой строкой для строковых колонок и нулём для числовых (как сделать по шаблону???)
string_columns = ['course_2', 'course_3', 'is_mi_student', 'is_ml_student', 'is_first_time', 'blended']
numeric_columns = ['percentile']

df[string_columns] = df[string_columns].fillna('')
df[numeric_columns] = df[numeric_columns].fillna(0)

df_na = df.isnull().sum().sum()

assert df_na == 0, 'Тест не пройден'

print("Выполнено")
score += 0.5

###

number_is_first_time_responses_no = (df['is_first_time'] == 'Нет').sum()

assert number_is_first_time_responses_no == 52, 'Тест не пройден'
print("Выполнено")
score += 0.5

###

df['timestamp'] = pd.to_datetime(df['timestamp'])

df = df.sort_values('timestamp').groupby('id').tail(1)

total_number_of_request = len(df)

assert total_number_of_request == 347, 'Тест не пройден'
print("Выполнено")
score += 0.5

###

third_year_students = df[df['course_3'].notna()]

blended_courses_for_third_year_students = set(third_year_students['blended'].dropna().unique())

blended_courses_for_third_year_students.discard('')

assert blended_courses_for_third_year_students == {'DevOps',
                                                   'Введение в дифференциальную геометрию',
                                                   'Соревновательный анализ данных'}, 'Тест не пройден'
print("Выполнено")
score += 0.5

###

df_filtered = df[df['blended'].notna() & (df['blended'] != '')]

blended_course_counts = df_filtered['blended'].value_counts()

blended_course_with_max_request = blended_course_counts.idxmax()

assert blended_course_with_max_request == 'DevOps', 'Тест не пройден'
print("Выполнено")
score += 0.5

###

blended_course_with_max_request = most_frequent = df['blended'][df['blended'] != ''].value_counts().idxmax()

assert blended_course_with_max_request == 'DevOps', 'Тест не пройден'
print("Выполнено")
score += 0.5

###

course_with_highest_average_rating = df[df['blended'] != ''].groupby('blended')['rating'].mean().idxmax()

assert course_with_highest_average_rating == 'Введение в дифференциальную геометрию', 'Тест не пройден'
print("Выполнено")
score += 0.5

###

number_duplicate_sets_of_courses = df[['spring_1', 'spring_2', 'spring_3', 'fall_1', 'fall_2', 'fall_3', 'blended']][
    df[['spring_1', 'spring_2', 'spring_3', 'fall_1', 'fall_2', 'fall_3', 'blended']].duplicated(keep=False)].shape[0]

assert number_duplicate_sets_of_courses == 32, 'Тест не пройден'
print("Выполнено")
score += 0.5

###

duplicate_sets_of_courses_with_number_students = df.groupby(
    ['spring_1', 'spring_2', 'spring_3', 'fall_1', 'fall_2', 'fall_3', 'blended'], as_index=False).size()
duplicate_sets_of_courses_with_number_students = duplicate_sets_of_courses_with_number_students[
    duplicate_sets_of_courses_with_number_students['size'] > 1]
duplicate_sets_of_courses_with_number_students.reset_index(inplace=True, drop=True)
duplicate_sets_of_courses_with_number_students.rename(columns={'size': 'number_of_students'}, inplace=True)

duplicate_sets_of_courses_with_number_students_answer = pd.read_csv(
    './data/duplicate_sets_of_courses_with_number_students.csv').fillna(value={'blended': '', })
assert duplicate_sets_of_courses_with_number_students.equals(
    duplicate_sets_of_courses_with_number_students_answer), 'Тест не пройден'

print("Выполнено")
score += 0.5

###

# Найдите курсы по выбору, на которые записывались как студенты второго, так и студенты третьего курса.

###

# Методом исключения найдите курсы, которые выбрали только студенты второго курса и только студенты третьего курса.

# Часть 2
