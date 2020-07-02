import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import pyplot
import warnings
warnings.filterwarnings(action='ignore')

murder=pd.read_excel('all.xlsx',sheet_name='murder')
rape=pd.read_excel('all.xlsx',sheet_name='rape')
theft=pd.read_excel('all.xlsx',sheet_name='property')
kidnap = pd.read_excel('all.xlsx',sheet_name='kidnap')

pop = pd.read_csv('population.csv')
population=pop['population']

kidnap_ = pd.read_csv('Dataset.csv')
city=[]
city=kidnap_['city_name'].unique()
year=kidnap['when_year'].unique()

data=pd.DataFrame()

for i in range(33):
    temp=kidnap[kidnap['city_name']==city[i]]
    temp['population']=population[i]
    temp['probability']=(temp['no_cnt'] / population[i] * 10000)

    if i==0:
        kidnap_data=temp
    else:
        kidnap_data=pd.concat([kidnap_data,temp])
print(kidnap_data)

for i in range(33):
    temp=murder[murder['city_name']==city[i]]
    temp['population']=population[i]
    temp['probability']=(temp['no_cnt'] / population[i] * 10000)

    if i==0:
        murder_data=temp
    else:
        murder_data=pd.concat([murder_data,temp])
print(murder_data)

for i in range(33):
    temp=rape[rape['city_name']==city[i]]
    temp['population']=population[i]
    temp['probability']=(temp['no_cnt'] / population[i] * 10000)

    if i==0:
        rape_data=temp
    else:
        rape_data=pd.concat([rape_data,temp])
print(rape_data)

for i in range(33):
    temp=theft[theft['city_name']==city[i]]
    temp['population']=population[i]
    temp['probability']=(temp['no_cnt'] / population[i] * 10000)

    if i==0:
        theft_data=temp
    else:
        theft_data=pd.concat([theft_data,temp])
print(theft_data)

########################################################################################################################

def heat_map (df: pd.DataFrame) -> pd.DataFrame:
    df=df.fillna(0)
    age0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    age1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    age2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    age3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    age4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    age5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    time1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    time2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    time3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    time4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(10):
        temp = df['when_year'] == year[i]
        T = df[temp]
        age0[i] = int(sum(T['age_level_0']))
        age1[i] = int(sum(T['age_level_1']))
        age2[i] = int(sum(T['age_level_2']))
        age3[i] = int(sum(T['age_level_3']))
        age4[i] = int(sum(T['age_level_4']))
        age5[i] = int(sum(T['age_level_5']))

        time1[i] = int(sum(T['when_time_d']))
        print(year[i], ': ', time1[i])
        time2[i] = int(sum(T['when_time_m']))
        print(year[i], ': ', time2[i])
        time3[i] = int(sum(T['when_time_n']))
        print(year[i], ': ', time3[i])
        time4[i] = int(sum(T['when_time_e']))
        print(year[i], ': ', time4[i])

    data = {"Up to 10": age0,
            "10~15": age1,
            "15~18": age2,
            "18~30": age3,
            "30~50": age4,
            "Above 50": age5}
    df1 = pd.DataFrame(data, columns=["Up to 10", "10~15", "15~18", "18~30", "30~50", "Above 50"],
                       index=year)
    print(df1)

    plt.pcolor(df1)
    plt.xticks(np.arange(0.5, len(df1.columns), 1), df1.columns)
    plt.yticks(np.arange(0.5, len(df1.index), 1), df1.index)

    sns.heatmap(df1, cmap='RdYlGn_r', annot=True,fmt='d')
    string = "Theft - Age ratio"
    plt.title(string, fontsize=20)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Year', fontsize=14)
    # plt.colorbar()
    plt.show()

    data = {"Dawn": time1,
            "Morning": time2,
            "Noon": time3,
            "Evening": time4}
    df2 = pd.DataFrame(data, columns=["Dawn", "Morning", "Noon", "Evening"],
                       index=year)
    print(df2)

    plt.pcolor(df2)
    plt.xticks(np.arange(0.5, len(df1.columns), 1), df2.columns)
    plt.yticks(np.arange(0.5, len(df1.index), 1), df2.index)

    sns.heatmap(df2, cmap='YlGnBu', annot=True, fmt='d')
    string="Theft - Time ratio"
    plt.title(string, fontsize=20)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Year', fontsize=14)
    # plt.colorbar()
    plt.show()

########################################################################################################################

def female_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['female_ratio']=df['female_ratio'].fillna(method='ffill')
    female = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    male = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(10):
        temp = df['when_year'] == year[i]
        T = df[temp]
        female[i] = T['female_ratio'].mean()
        male[i] = 1 - female[i]
        print(i, ': ', female[i])

    W_ = 0.5

    pyplot.bar(year, male, label='Male', width=W_)
    pyplot.bar(year, female, label='Female', width=W_, color='#AA2848', bottom=male)

    string="Murder - Sex ratio"
    plt.title(string)
    plt.legend()
    plt.show()

########################################################################################################################

def city_by(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(33):
        print(city[i])
        temp = df['city_name'] == city[i]
        if i == 0:
            data1 = df[temp]
        else:
            data1 = data1.append(df[temp], sort=False)
    print(data1)

    x = range(2010, 2020)

    y = [[], [], [], [], [],
         [], [], [], [], [],
         [], [], [], [], [],
         [], [], [], [], [],
         [], [], [], [], [],
         [], [], [], [], [],
         [], [], []]


    for i in range(33):
        df1 = df[df['crime_kind'] == 'Theft']
        df2 = df[df['crime_kind'] == 'Burglary']
        df3 = df[df['crime_kind'] == 'Dacoity']
        df4 = df[df['crime_kind'] == 'Robbery ']

        temp1 = df1['city_name'] == city[i]
        T1 = df1[temp1]
        temp2 = df2['city_name'] == city[i]
        T2 = df2[temp2]
        temp3 = df3['city_name'] == city[i]
        T3 = df3[temp3]
        temp4 = df4['city_name'] == city[i]
        T4 = df4[temp4]

        T1=T1.reset_index(drop=False)
        T2=T2.reset_index(drop=False)
        T3=T3.reset_index(drop=False)
        T4=T4.reset_index(drop=False)

        s1=T1['no_cnt'] / population[i] * 10000
        print(s1)
        s2=T2['no_cnt'] / population[i] * 10000
        print(s2)
        s3=T3['no_cnt'] / population[i] * 10000
        print(s3)
        s4=T4['no_cnt'] / population[i] * 10000
        print(s4)
        print(s1+s2+s3+s4)
        y[i] = s1+s2+s3+s4



    for i in range(33):
        plt.plot(x, y[i], label=city[i])

    print(data1)

    plt.xlabel("Year")
    plt.ylabel("The number of crimes per 10,000 people")
    #string=df['crime_kind'][1]
    #plt.title(string)
    plt.legend()
    plt.show()

########################################################################################################################

def sct (df: pd.DataFrame) -> pd.DataFrame:
    plt.scatter(df['city_name'], df['probability'],
                alpha=0.2,
                s=(df['when_year'] - 2009) * 25,
                c=df['probability'],
                )
    plt.xlabel('city name')
    plt.ylabel('The number of crimes per 10,000 people')
    string=df['crime_kind'][1]
    plt.title(string)
    plt.show()

########################################################################################################################

def check_outlier(df: pd.DataFrame) -> pd.DataFrame:
    plt.boxplot(df['when_time_d'])
    plt.title('Box plot of when_time_d')
    plt.xticks([1], ['when_time_d'])
    plt.show()

    plt.boxplot(df['when_time_m'])
    plt.title('Box plot of when_time_m')
    plt.xticks([1], ['when_time_m'])
    plt.show()

    plt.boxplot(df['when_time_n'])
    plt.title('Box plot of when_time_n')
    plt.xticks([1], ['when_time_n'])
    plt.show()

    plt.boxplot(df['when_time_e'])
    plt.title('Box plot of when_time_e')
    plt.xticks([1], ['when_time_e'])
    plt.show()

    plt.boxplot(df['age_level_0'])
    plt.title('Box plot of age_level_0')
    plt.xticks([1], ['age_level_0'])
    plt.show()

    plt.boxplot(df['age_level_1'])
    plt.title('Box plot of age_level_1')
    plt.xticks([1], ['age_level_1'])
    plt.show()

    plt.boxplot(df['age_level_2'])
    plt.title('Box plot of age_level_2')
    plt.xticks([1], ['age_level_2'])
    plt.show()

    plt.boxplot(df['age_level_3'])
    plt.title('Box plot of age_level_3')
    plt.xticks([1], ['age_level_3'])
    plt.show()

    plt.boxplot(df['age_level_4'])
    plt.title('Box plot of age_level_4')
    plt.xticks([1], ['age_level_4'])
    plt.show()

    plt.boxplot(df['female_ratio'])
    plt.title('Box plot of female_ratio')
    plt.xticks([1], ['female_ratio'])
    plt.show()

########################################################################################################################

import pandas as pd
import numpy as np
import sklearn.linear_model as lin
import matplotlib.pyplot as plt


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
            for col_name in ['when_time_d', 'when_time_m', 'when_time_n', 'when_time_e', 'age_level_0', 'age_level_1',
                             'age_level_2', 'age_level_3', 'age_level_4', 'age_level_5']:
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

                df['missing'] = np.NaN

                # Train data model
                model = lin.LinearRegression()
                model.fit(x.reshape(-1, 1), y)

                miss=0

                # Predict value, apply to the full dataset
                for idx, pred_x in to_predict_x:
                    # Floating-point values should be rounded into an integer value
                    df.loc[idx, col_name] = round(model.predict([[pred_x]])[0])
                    df.loc[idx, ['missing']] = 1
                    miss=miss+1

                missing_df = df.copy()
                missing_df.dropna(subset=['missing'], inplace=True)
                x_change=(missing_df['when_year'])
                y_change=(missing_df[col_name])

                if miss>0:
                    plt.plot(x.reshape(-1, 1), y, 'o')
                    plt.plot(x, model.predict(x.reshape(-1, 1)))
                    plt.scatter(x_change, y_change, marker='*', color='r',s=60)
                    string =crime+' - '+city+ ' : '+col_name
                    plt.title(string)
                    plt.show()

            ############################################################################################################

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

########################################################################################################################
