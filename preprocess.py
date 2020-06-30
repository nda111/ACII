# 
# Version: 1.1.1.1
# Since: June 20th, 2020
#
# Description:
# A module that fills missing or wrong data with following method/creteria:
#   - Suppose below are not missing:
#       - crime_kind
#       - when_year
#       - city_name
# 
#   - Linear regression by crime_kind, when_year:
#       - Creteria
#           - NaN value
#           - Cell where has value greater that no_cnt on the row
#       - Columns
#           - when_time_d
#           - when_time_m
#           - when_time_n
#           - when_time_e
#           - age_level_0
#           - age_level_1
#           - age_level_2
#           - age_level_3
#           - age_level_4
#           - age_level_5
# 
#   - Mean-fill by crime_kind, when_year:
#       - Creteria
#           - NaN value
#           - Cell where has value are not in [0, 1]
#       - Columns
#           - female_ratio
# 


import pandas as pd
import numpy as np
import sklearn.linear_model as lin
from sklearn.preprocessing import MinMaxScaler


def fill_missing_wrong(df: pd.DataFrame) -> pd.DataFrame:
    crime_kinds = tuple(df['crime_kind'].unique())

    # Filling missing or wrong data at when_time_?, age_level_?, female_ratio
    # For every crime kind,
    for crime in crime_kinds:
        by_crime = df[df['crime_kind'] == crime]
        city_names = tuple(by_crime['city_name'].unique())

        # And for every city
        for city in city_names:
            by_city = by_crime[by_crime['city_name'] == city]

            # For each columns to linearly regress
            for col_name in ['when_time_d', 'when_time_m', 'when_time_n', 'when_time_e', 'age_level_0', 'age_level_1', 'age_level_2', 'age_level_3', 'age_level_4', 'age_level_5']:
                x, y = [], []
                to_predict_x = []

                # Separate missing, non-missing values=
                for idx, row in by_city.iterrows():
                    # to build an array of year to predict value
                    if np.isnan(row[col_name]) or row[col_name] < 0:
                        to_predict_x.append((idx, row['when_year']))
                    # to build dataset to train model
                    else:
                        x.append(row['when_year'])
                        y.append(row[col_name])
                x, y = np.array(x), np.array(y)

                # Train data model
                model = lin.LinearRegression()
                model.fit(x.reshape(-1, 1), y)

                # Predict value, apply to the full dataset
                for idx, pred_x in to_predict_x:
                    # Floating-point values should be rounded into an integer value
                    df.loc[idx, col_name] = round(model.predict([[pred_x]])[0])

            mean, count = 0, 0
            to_fill = []
            # Separate rows by missing, non-missing value
            for idx, row in by_city.iterrows():
                ratio = row['female_ratio']
                # to fill missing or wrong value
                if np.isnan(ratio) or ratio < 0 or 1 < ratio:
                    to_fill.append(idx)
                # to calculate mean value
                else:
                    mean += ratio
                    count += 1
            mean /= count
                
            # For every missing female_ratio
            for idx in to_fill:
                # fill value with mean value for the crime at the city
                df.loc[idx, 'female_ratio'] = mean

    return df

def get_danger_level(df_crime: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    population=pop['population']

    city=df_crime['city_name'].unique()
    year=df_crime['when_year'].unique()

    df=pd.DataFrame()

    for i in range(33):
        temp=df_crime[df_crime['city_name']==city[i]]
        temp['population']=population[i]
        temp['probability']=(temp['no_cnt'] / population[i] * 10000)

        if i==0:
            df=temp
        else:
            df=pd.concat([df, temp])

    df['level']=0

    df=df.sort_values(by=['probability'],axis=0)
    df=df.reset_index(drop=True)

    split=len(df)/5

    k=0
    for i in range(len(df)):
        if 0<=i<split:
            df['level'][i]=0
            k=k+1
        if split<=i<(split*2):
            df['level'][i]=1
            k=k+1
        if (split*2)<=i<(split*3):
            df['level'][i]=2
            k=k+1
        if (split*3)<=i<(split*4):
            df['level'][i]=3
            k=k+1
        if (split*4)<=i<(split*5):
            df['level'][i]=4
            k = k + 1

    return df