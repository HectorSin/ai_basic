import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/HectorSin/ai_basic/master/ai_prog_basic/Chapters/Ch12_Pandas/data/weather.csv"
weather = pd.read_csv(url, index_col = 0, encoding = 'CP949')
print(weather.head())
weather['평균 풍속(m/s)'].plot(kind='hist', bins=33)
plt.show()