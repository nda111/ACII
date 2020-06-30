# 
# Version: 1.1.1.1
# Since: June 29th, 2020
#
# Description:
# A program that predicts dangerousity from the city, crime kind, year, hour, age and sex.
# The prediction executed with following sub-predictors:
#   - Decision tree
#   - Random forest (max_depth=11)
#   - Neural network (layout=(10, 20, 30, 40, 30, 20, 10))
#

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import preprocess


# Suppress 'SettingWithCopyWarning'
pd.set_option('mode.chained_assignment', None)


class DangerPredictor:
    def fit(self, df_crimes: pd.DataFrame, df_population: pd.DataFrame, print_progress: bool = True, k_fold: int = 10):
        self.decision_tree = DecisionTreeClassifier()
        self.random_forest = BaggingClassifier(DecisionTreeClassifier(max_depth=11))
        self.neural_networks = {}

        if print_progress:
            print('Preprocessing')
        df_crimes = preprocess.fill_missing_wrong(df_crimes)
        df_crimes = preprocess.get_danger_level(df_crimes, df_population)

        input_attrs = ['city_name', 'crime_kind', 'when_year', 'when_time_d', 'when_time_m', 'when_time_n', 'when_time_e',
                       'age_level_0', 'age_level_1', 'age_level_2', 'age_level_3', 'age_level_4', 'age_level_5', 'female_ratio']
        target_attrs = ['level']
        cities = df_crimes['city_name'].unique()
        crimes = df_crimes['crime_kind'].unique()

        # Categorical to numeric
        self.city_map = {}
        self.crime_map = {}
        for i in range(len(cities)):
            self.city_map[i] = cities[i]
            self.city_map[cities[i]] = i
            df_crimes.replace(cities[i], i, inplace=True)
            cities[i] = i

        for i in range(len(crimes)):
            self.crime_map[i] = crimes[i]
            self.crime_map[crimes[i]] = i
            df_crimes.replace(crimes[i], i, inplace=True)
            crimes[i] = i

        if (print_progress):
            print('Training: Decision tree')
        self.decision_tree.fit(df_crimes[input_attrs], df_crimes[target_attrs])

        if (print_progress):
            print('Training: Random forest')
        self.random_forest.fit(df_crimes[input_attrs], df_crimes[target_attrs].to_numpy().reshape(len(df_crimes), ))

        if (print_progress):
            print('Training: Neural networks')
        for city_name in cities:
            # Project a data frame
            by_city: pd.DataFrame = df_crimes[df_crimes['city_name'] == city_name].drop(labels=['city_name'], axis=1)
            if len(by_city) == 0:
                continue

            # Train NN
            network = MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 20, 30, 40, 30, 20, 10))
            network.fit(by_city[input_attrs[1:]], by_city[target_attrs].to_numpy().reshape(len(by_city), ))

            self.neural_networks[city_name] = network

        print('Finishing')
        test_X = df_crimes[input_attrs]
        y_true = df_crimes[target_attrs].to_numpy().reshape(-1)
        dt_score = self.decision_tree.predict(test_X)
        rf_score = self.random_forest.predict(test_X)
        nn_score = []
        for idx, row in test_X.iterrows():
            city = row['city_name']
            values = [row[input_attrs[1:]]]
            
            pred = round(self.neural_networks[city].predict(values)[0])
            nn_score.append(pred)

        self.dt_weight = accuracy_score(y_true, dt_score)
        self.rf_weight = accuracy_score(y_true, rf_score)
        self.nn_weight = accuracy_score(y_true, nn_score)
        sm = self.dt_weight + self.rf_weight + self.nn_weight

        self.dt_weight /= sm
        self.rf_weight /= sm
        self.nn_weight /= sm

        if print_progress:
            print('Training completed')

        return self


    def predict(self, city: str, crime: str, year: int, hour: int, age: int, is_female: bool):
        values = []

        values.append(self.city_map[city])
        values.append(self.crime_map[crime])
        values.append(year)

        # when_time_?
        when_time = np.array([0, 0, 0, 0])
        if hour < 0 or 23 < hour:
            return None
        when_time[0] = hour
        when_time[1] = abs(8 - hour)
        when_time[2] = abs(16 - hour)
        when_time[3] = abs(23 - hour)
        when_time = 1 - (when_time / sum(when_time))
        values.extend(when_time)

        # age_level_?
        age_level = np.array([0, 0, 0, 0, 0, 0])
        if age < 0:
            return None
        age_level[0] = age
        age_level[1] = abs(age - 10)
        age_level[2] = abs(age - 20)
        age_level[3] = abs(age - 30)
        age_level[4] = abs(age - 40)
        age_level[5] = abs(age - 50)
        age_level = 1 - (age_level / sum(age_level))
        values.extend(age_level)

        values.append(1 if is_female else 0)

        dt = self.decision_tree.predict([values])[0] * self.dt_weight
        rf = self.random_forest.predict([values])[0] * self.rf_weight
        nn = self.neural_networks[values[0]].predict([values[1:]])[0] * self.nn_weight

        return int(round(dt * rf * nn))


df_crimes = pd.read_excel('Dataset.xlsx', sheet_name='combine')
df_population = pd.read_excel('Dataset.xlsx', sheet_name='population')

predictor = DangerPredictor().fit(df_crimes, df_population, print_progress=True)

print(predictor.predict(crime='Rape', city='Assam', year=2020, hour=22, age=20, is_female=True))
print(predictor.predict(crime='Murder', city='Maharashtra', year=2020, hour=19, age=24, is_female=True))