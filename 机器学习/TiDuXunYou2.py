import numpy as np
import matplotlib.pyplot as plt
import random

Input = [[-0.017612, 14.053064, 0], [-1.395634, 4.662541, 1],
         [-0.752157, 6.538620, 1], [-1.322371, 7.152853, 1],
         [0.423363, 11.054677, 0], [0.406704, 7.067335, 1],
         [0.667394, 12.741452, 0], [-2.460150, 6.866805, 1],
         [0.569411, 9.548755, 0], [-0.026632, 10.427743, 0],
         [0.850433, 6.920334, 1], [1.347183, 13.175500, 0],
         [1.176813, 3.167020, 1], [-1.781871, 9.0979531, 0]]  # 初始化点集
Input = np.mat(Input)  # 矩阵化
target = Input[:, -1]  # 目标函数
(m, n) = np.shape(Input)


def drawScatterbyLabel(Input):  # 作散点图
    target = Input[:, -1]
    m, n = np.shape(Input)
    for i in range(m):
        if target[i] == 0:
            plt.scatter(Input[i, 0], Input[i, 1], c="red", marker="o")
        else:
            plt.scatter(Input[i, 0], Input[i, 1], c="blue", marker="o")


def builtMat(dataSet):
    m, n = np.shape(dataSet)
    dataMat = np.zeros((m, n))
    dataMat[:, 0] = 1
    dataMat[:, 1:] = dataSet[:, :-1]
    return dataMat


def logistic(wTx):
    return 1.0 / (1.0 + np.exp(-wTx))


def hardlim(wTx):
    wTx[np.nonzero(wTx.A > 0)[0]] = 1
    wTx[np.nonzero(wTx.A <= 0)[0]] = 0
    return wTx


def drawLine(weights, index=None):  # 作拟合直线
    X = np.linspace(-5, 5, 100)
    Y = -(float(weights[0, 0]) + X * float(weights[0, 1])) / float(weights[0, 2])
    plt.plot(X, Y)
    if index != None:
        plt.annotate("hplane:" + str(index), xy=(X[99], Y[99]))


dataMat = builtMat(Input)

steps = 50000  # 迭代次数
weights = np.ones(n)  # 初始化权重

for j in range(steps):  # 开始迭代
    dataIndex = list(range(m))  # 以导入数据的行数m为个数产生索引向量：0~13
    for i in range(m):
        alpha = 2 / (1.0 + j + i) + 0.0001  # 动态修改alpha步长
        randIndex = int(random.uniform(0, len(dataIndex)))  # 生成0~m之间的随机索引
        vectSum = sum(dataMat[randIndex] * weights.T)  # 计算dataMat随机索引与权重的点积和
        grad = logistic(vectSum)  # 计算点积和的梯度
        errors = target[randIndex] - grad  # 计算误差
        weights = weights + alpha * errors * dataMat[randIndex]  # 修正误差，进行迭代
        del (dataIndex[randIndex])
    print(j)

print(weights)
drawLine(weights)

drawScatterbyLabel(Input)
plt.show()
