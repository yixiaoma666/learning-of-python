import numpy as np
from scipy import optimize as op

c = np.array([1, 2])  # 目标函数

A_ub = np.array([[-1, -1]])
B_ub = np.array([-2])  # 不等式约束，需要化成最小值

# A_eq = np.array([[1, 1, 1]])
# B_eq = np.array([7])  # 等式约束

x1 = x2 = (0, 2)  # 自变量取值范围

ans = op.linprog(c, A_ub, B_ub, bounds=(x1, x2)).fun, -op.linprog(-c, A_ub, B_ub, bounds=(x1, x2)).fun
print("min =", ans[0], "max =", ans[1])

