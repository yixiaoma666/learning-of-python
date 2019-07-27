import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1.1, 4.5, 8, 17, 24, 99])
plt.scatter(x, y)


def residual(p):
    a, b, c = p
    return y - a * x * x - b * x - c


ans = op.leastsq(residual, np.array([0, 0, 0]))
X = np.linspace(0, 10, 100)
Y = ans[0][0] * X * X + ans[0][1] * X + ans[0][2]
plt.plot(X, Y)
print("拟合抛物线为{}x^2+{}x+{}".format(ans[0][0], ans[0][1], ans[0][2]))
plt.show()
