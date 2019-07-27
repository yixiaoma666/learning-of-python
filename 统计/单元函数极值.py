import numpy as np
from scipy import optimize as op


def fun(args):  # fun返回一个函数，形成闭包
    a = args

    def f(x):  # 函数自变量需要是一个数组
        return a / x[0] + x[0]  # 这里取数组的第一个值

    return f

if __name__ == '__main__':
    x0 = np.array(2)
    ans = op.minimize(fun(1), x0, method="SLSQP")
    print("在{}处取极值为{}".format(ans.x[0],ans.fun))
