import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt

x = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
y = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])
plt.scatter(x, y)


def residual(p):
    a, b = p
    return y - a * x - b


ans = op.leastsq(residual, np.array([0, 0]))
X = np.linspace(0, 10, 100)
Y = ans[0][0] * X + ans[0][1]
plt.plot(X, Y)
print("拟合直线为y={}x+{}".format(ans[0][0], ans[0][1]))
plt.show()
