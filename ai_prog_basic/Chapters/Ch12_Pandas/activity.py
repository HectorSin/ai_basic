import csv
import matplotlib.pyplot as plt

f = open('C:/Users/kkang/ai_basic/ai_prog_basic/Chapters/Ch12_Pandas/data/weather.csv')
data = csv.reader(f)
# 헤더를 제거
header = next(data)

monthly_wind = [0 for x in range(12)]
days_counted = [0 for x in range(12)]

for row in data:
    month = int(row[0][5:7])
    if row[3] != '':
        wind = float(row[3])
        monthly_wind[month-1] += wind
        days_counted[month-1] += 1

for i in range(12):
    monthly_wind[i] /= days_counted[i]

plt.plot(monthly_wind, 'blue')
plt.show()

f.close()