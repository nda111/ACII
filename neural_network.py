# 
# Version: 1.1.1.1
# Since: June 27th, 2020
#
# Description:
# A program that tests accuracy of stump with neural network on the leaf of it
# 

import numpy as np
import pandas as pd
import sklearn.neural_network as nn
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import preprocess


# Generate an input dataset
@staticmethod
def generate_input(crime: str, year: int, hour: int, age: int, is_female: bool):
        inputs = []

        # crime_kind
        inputs.append(crime_map[crime])

        # when_year
        inputs.append(year)

        # when_time_?
        when_time = [0, 0, 0, 0]
        if hour < 0 or 23 < hour:
            return None
        elif hour <= 5:
            when_time[0] = 1
        elif hour <= 11:
            when_time[1] = 1
        elif hour <= 16:
            when_time[2] = 1
        elif hour <= 22:
            when_time[3] = 1
        else:
            when_time[0] = 1
        inputs.extend(when_time)

        # age_level_?
        age_level = [0, 0, 0, 0, 0, 0]
        if age < 10:
            age_level[0] = 1
        elif age < 20:
            age_level[1] = 1
        elif age < 30:
            age_level[2] = 1
        elif age < 40:
            age_level[3] = 1
        elif age < 50:
            age_level[4] = 1
        else:
            age_level[5] = 1
        inputs.extend(age_level)

        # female_ratio
        inputs.append(1 if is_female else 0)

        return inputs


# Load, initialize column names
print('Loading data')
df_crimes = pd.read_excel('Dataset.xlsx', sheet_name='combine')
df_population = pd.read_excel('Dataset.xlsx', sheet_name='population')

city_names = df_crimes['city_name'].unique()
crimes = df_crimes['crime_kind'].unique()
input_attrs = ['crime_kind', 'when_year', 'when_time_d', 'when_time_m', 'when_time_n', 'when_time_e', 'age_level_0', 'age_level_1', 'age_level_2', 'age_level_3', 'age_level_4', 'age_level_5', 'female_ratio']
target_attrs= ['level']

print('Preprocessing')
df_crimes = preprocess.fill_missing_wrong(df_crimes)
df_crimes = preprocess.get_danger_level(df_crimes, df_population)
preprocess.scale_year_cnt(df_crimes)

# Convert crime kind into a float value
crime_map = {}
for i in range(len(crimes)):
    crime_map[crimes[i]] = i * 1000 # Discretization
df_crimes['crime_kind'].replace(crime_map, inplace=True)

# Splitting dataset
attrs = input_attrs.copy()
attrs.append('city_name')
print()

print('[Neural Network]')
scores = np.zeros(10)
cv = KFold(10, shuffle=True, random_state=0)
for i, (idx_train, idx_test) in enumerate(cv.split(df_crimes)):
    print(f"Case #{i+1}")
    train = df_crimes.iloc[idx_train]
    test = df_crimes.iloc[idx_test]

    # Train neural networks
    brain = {}
    # For each cities
    print("Training")
    for city_name in city_names:
        # Project a data frame
        by_city: pd.DataFrame = train[train['city_name'] == city_name].drop(labels=['city_name'], axis=1)
        if len(by_city) == 0:
            continue

        # Train NN
        network= nn.MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 20, 30, 40, 30, 20, 10))
        network.fit(by_city[input_attrs], by_city[target_attrs].to_numpy().reshape(len(by_city), ))

        brain[city_name] = network

    # Inferencing
    print('Inferencing')
    y_true = test[['level']].to_numpy()
    y_pred = []

    for idx, row in test.iterrows():
        city = row['city_name']
        inputs = row[input_attrs]

        network: nn.MLPClassifier = brain[city]
        result = network.predict([inputs])[0]

        y_pred.append(result)

    # Evaluate
    con_matrix = metrics.confusion_matrix(y_true, y_pred)
    accuracy = sum(con_matrix[i][i] for i in range(len(con_matrix))) / sum(sum(row) for row in con_matrix) * 100

    scores[i] = accuracy
    print()

print('Scores:', scores)