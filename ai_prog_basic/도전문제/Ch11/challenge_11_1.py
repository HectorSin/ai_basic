import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

x = [x for x in range(1000)]
y = np.random.rand(1000) * 3

plt.title("Numbers")
plt.figure(figsize=(10, 3))
plt.plot(x, y, marker='o', color='brown') # 선 그래프에 동그라미 표식을 출력

plt.show()
