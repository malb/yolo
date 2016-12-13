import matplotlib.pyplot as plt
import numpy as np


def plot_and_save(Y, title, file):
	X = range(len(Y))
	plt.plot(X, Y)
	plt.xlabel('index (i)')
	plt.ylabel('Log |bi*|')
	plt.title(title)
	plt.grid(True)
	plt.savefig(file)
	plt.close()
