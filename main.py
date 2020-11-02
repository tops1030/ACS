import Functions
import numpy as np
import matplotlib.pyplot as plt

V, E = Functions.creat_city(20, 400)
plt.scatter(V[:,0], V[:,1], alpha=0.6, c = "r")
root = Functions.ACS(V, E, 20, 50)
path = V[root]
path = np.append(path, [path[0]], axis=0)
plt.plot(path[:,0], path[:,1], marker="o", mfc="r")
plt.show()