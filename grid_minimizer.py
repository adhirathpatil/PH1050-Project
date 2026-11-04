import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ----- Initialising the 'random' object
rng = np.random.default_rng()

# ----- Initialising the 'random' matrix, box is the scale of the cube 
#       we're allowing atoms to spawn in.
def pos_gen(N, box=5.0):
    return rng.random((N, 3)) * box

# ----- Numerically defining the potential function
def potential(a, b):
    r = np.sqrt(np.sum((b - a)**2))
    return 4 * (1/r**12 - 1/r**6)

# ----- Total potential function
def total_potential(x):
    pos = x.reshape(-1, 3) # -- reconstructing the flat tensor
    n = len(pos)
    total = 0.0
    for i in range(n): # -- Iterating through all the members to calculate the energy
        for j in range(i + 1, n):
            total += potential(pos[i], pos[j])
    return total

def minim_potential(arg):
    res = minimize(total_potential, pos_gen(arg).ravel()) # -- .minimize only accepts an array as an argument
    return res

# ----- Creating the x, y arrays for plotting
x = np.arange(2, 7, 1)
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = minim_potential(x[i]).fun/x[i]

plt.title("Energy per nucleon vs. number of nucleons")
plt.plot(x, y, '-o')
plt.xlabel("No. of nucleons")
plt.ylabel("Energy per nucleon")
plt.show()