import math
import matplotlib.pyplot as plt

x = []
y = []
z = []

for angle in range(360):
    x.append(angle)
    z.append(math.cos(math.radians(angle)))
    y.append(math.sin(math.radians(angle)))

plt.plot(x, y)
plt.plot(x, z, 'r-')
plt.title("SINE & COSINE WAVE")
plt.show()