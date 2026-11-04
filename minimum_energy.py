import numpy as np
import matplotlib.pyplot as plt

# ----- Defining the interval function (gives the difference between inputs)
def interval(a, b):
    return b-a

# ----- Defining the mean function
def mean(a, b):
    return (a+b)/2

# ----- Defining the derivative (using central difference for O(h^2) error)
def derivative(f, a, h=1e-6):
    return (f(a+h)-f(a-h))/(2*h)

# ----- Defining the minimum function
def minimum(f, a, b, tol=1e-6):
    left = a
    right = b
    root = mean(a, b)
    while interval(left, right) > tol:
        # ----- Using the fact that for a < root, f'(a) < 0 and a > root, f'(a) > 0
        if derivative(f, root) < 0:
            left = root
        if derivative(f, root) > 0:
            right = root
        root = mean(left, right)
    return root

def numerical_force(r, h=1e-6):
    return -derivative(lambda x: 4*(1/x**12 - 1/x**6), r, h=h)

print(f"Calculated value: {minimum(lambda x: 4*(1/x**12 - 1/x**6), 1, 2)}")
print(f"True value: {2**(1/6)}")

# ----- Analytic force (from differentiation)
def analytic_force(r):
    return 48*r**(-13) - 24*r**(-7)

# ----- Error-plotting (log-log)
r = 1.5

x = np.arange(7, 15, 1)
y = np.zeros(8)
t = 1/2**x

for i in range(8):
    y[i] = abs(analytic_force(r) - numerical_force(r, h=1/2**(x[i])))

plt.title("Error vs. Input 'h' (log-log)")
plt.plot(np.log(t), np.log(y), "-o")
plt.xlabel("Order of 2")
plt.ylabel("Order of error")
plt.show()