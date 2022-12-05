import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/HectorSin/ai_basic/master/ai_prog_basic/Chapters/Ch12_Pandas/data/countries.csv"
country = pd.read_csv(url, index_col = 0)

country['population'].plot(kind='bar', color=('b', 'darkorange', 'g', 'r', 'm'))
plt.show()