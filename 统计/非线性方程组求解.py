"""
计算非线性方程组：
    5x1+3 = 0
    4x0^2-2sin(x1x2)=0
    x1x2-1.5=0
"""

import numpy as np
from scipy import optimize


def fun(x):
    x0, x1, x2 = x.tolist()
    return [5 * x1 + 3, 4 * np.power(x0, 2) - 2 * np.sin(x1 * x2), x1 * x2 - 1.5]


result = optimize.fsolve(fun, np.array([1, 1, 1]))
print(result)