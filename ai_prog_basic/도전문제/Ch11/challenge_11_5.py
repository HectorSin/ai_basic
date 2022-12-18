import matplotlib.pyplot as plt

n_bins = 5

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.hist(x, n_bins, histtype='bar', color='blue', alpha=0.3)

plt.show()