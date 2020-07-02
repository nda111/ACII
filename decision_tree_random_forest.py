# 
# Version: 1.1.1.1
# Since: June 27th, 2020
#
# Description:
# A program that tests accuracy of decision tree and random forest with various depths
# 

import pandas as pd
import numpy as np
import sklearn.tree as trees
import sklearn.ensemble as ensemble
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import preprocess


def generate_input(city: str, crime: str, year: int, hour: int, age: int, is_female: bool):
    result = []

    result.append(city_map[city])
    result.append(crime_map[crime])
    result.append(year)

    # when_time_?
    when_time = np.array([0, 0, 0, 0])
    if hour < 0 or 23 < hour:
        return None
    when_time[0] = hour
    when_time[1] = abs(8 - hour)
    when_time[2] = abs(16 - hour)
    when_time[3] = abs(23 - hour)
    when_time = 1 - (when_time / sum(when_time))
    result.extend(when_time)

    # age_level_?
    age_level = np.array([0, 0, 0, 0, 0, 0])
    age_level[0] = age
    age_level[1] = abs(age - 10)
    age_level[2] = abs(age - 20)
    age_level[3] = abs(age - 30)
    age_level[4] = abs(age - 40)
    age_level[5] = abs(age - 50)
    age_level = 1 - (age_level / sum(age_level))
    result.extend(age_level)

    result.append(1 if is_female else 0)

    return result


# Load data from file
print('Loading')
df_crimes = pd.read_excel('Dataset.xlsx', sheet_name='combine')
df_population = pd.read_excel('Dataset.xlsx', sheet_name='population')

input_attrs = ['city_name', 'crime_kind', 'when_year', 'when_time_d', 'when_time_m', 'when_time_n', 'when_time_e',
               'age_level_0', 'age_level_1', 'age_level_2', 'age_level_3', 'age_level_4', 'age_level_5', 'female_ratio']
target_attrs = ['level']
cities = df_crimes['city_name'].unique()
crimes = df_crimes['crime_kind'].unique()

# Preprocess
print('Preprocessing')
df_crimes = preprocess.fill_missing_wrong(df_crimes)
df_crimes = preprocess.get_danger_level(df_crimes, df_population)
preprocess.scale_year_cnt(df_crimes)
print()

# Categorical to numeric
city_map = {}
crime_map = {}
for i in range(len(cities)):
    city_map[i] = cities[i]
    city_map[cities[i]] = i
    df_crimes.replace(cities[i], i, inplace=True)

for i in range(len(crimes)):
    crime_map[i] = crimes[i]
    crime_map[crimes[i]] = i
    df_crimes.replace(crimes[i], i, inplace=True)

print('[Decision Tree]')
classifiers = []

# Splitting dataset
k = 10
scores = np.zeros(k)
cv = KFold(k, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_crimes)):
    train = df_crimes.iloc[idx_train]
    test = df_crimes.iloc[idx_test]

    # Training
    print('Training #', i + 1, sep='')
    tree = trees.DecisionTreeClassifier()
    tree.fit(train[input_attrs], train[target_attrs])

    classifiers.append(tree)

    # Inferencing
    print('Inferencing')
    y_pred = tree.predict(test[input_attrs])
    y_true = test[target_attrs].to_numpy()

    # Evaluating
    print('Evaluating')
    score = metrics.accuracy_score(y_true, y_pred)
    scores[i] = score

    print()

print('Scores:', scores, '\n')

print('[Random Forest]')
cv = KFold(10, shuffle=True, random_state=0)
result_df = pd.DataFrame()
for i, (idx_train, idx_test) in enumerate(cv.split(df_crimes)):
    train = df_crimes.iloc[idx_train]
    test = df_crimes.iloc[idx_test]

    row_result = {}
    for k in range(1, len(input_attrs) + 1):
        print(f'k={k}')
        # Training
        print('Training')
        bagging = ensemble.BaggingClassifier(trees.DecisionTreeClassifier(max_depth=k))
        bagging.fit(train[input_attrs], train[target_attrs].to_numpy().reshape(len(train), ))

        # Inferencing
        y_pred = bagging.predict(test[input_attrs])
        y_true = test[target_attrs].to_numpy()

        # Evaluating
        print('Evaluating')
        accuracy = metrics.accuracy_score(y_true, y_pred)
        row_result[f'Depth={k}'] = accuracy

        print()

    result_df = result_df.append(row_result, ignore_index=True)
    print()

print(result_df)
