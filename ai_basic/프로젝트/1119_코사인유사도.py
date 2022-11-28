import pandas as pd
import numpy as np
import csv


data=pd.read_csv(r"C:\Users\Administrator\ai_basic\프로젝트\data\food_data.csv", encoding='cp949')
data.columns=['number',
              'reasons',
              'cuisine',
              'fruit',
              'pay',
              'veggies',
             'result']
data2=data[['reasons','cuisine','fruit','pay','veggies','result']]
df=data[['reasons','cuisine','fruit','pay','veggies']]
List2=np.array([ 7,4,2,3,1]) #사용자의값
df2 = df.dot(List2)/ (np.linalg.norm(df, axis=1) * np.linalg.norm(List2))
df3=df2.sort_values(ascending=False)
df4=df3.head(4)
df5=df4.index
df6=data2.iloc[df5]['result']
df6

print(df6)