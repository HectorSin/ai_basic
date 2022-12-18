import matplotlib.pyplot as plt
import numpy as np

xData = np.random.randn(10000) 
yData = np.random.randn(10000)

plt.scatter(xData, yData, alpha=0.05)
plt.title('Random Position')
plt.xlabel('x')
plt.ylabel('y')

plt.show()