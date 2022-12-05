# In[1]:
import pandas as pd
import matplotlib.pyplot as plt

#url = "https://raw.githubusercontent.com/HectorSin/ai_basic/master/ai_prog_basic/Chapters/Ch12_Pandas/data/weather.csv"
#weather = pd.read_csv(url, index_col = 0, encoding = 'CP949')
weather = pd.read_csv("data\\weather.csv", index_col = 0, encoding = 'CP949')


# In[2]:
print(weather.head())
weather['평균 풍속(m/s)'].plot(kind='hist', bins=33)
plt.show()

# In[3]:
print(weather.describe())
# %%
print('평균 분석 --------------------------')
print(weather.mean())
print('표준편차 분석 ----------------------')
print(weather.std())
# %%
weather.count()
# %%
weather['최대 풍속(m/s)'].count()
# %%
