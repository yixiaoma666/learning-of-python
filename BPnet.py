import numpy as np
import matplotlib.pyplot as plt


class BPnet(object):
    def __init__(self):
        self.eb = 0.01  # 误差容限：当误差小于这个值时，算法收敛，程序停止
        self.iterator = 0  # 算法收敛时的迭代次数
        self.eta = 0.1  # 学习率：相当于步长
        self.mc = 0.3  # 动量因子：引入的一个调优参数，是主要的调优参数
        self.maxiter = 2000  # 最大迭代次数
        self.nHidden = 4  # 隐含层神经元
        self.nOut = 1  # 输出层个数
        pass

    def logistic(self, net):
        pass

    def dlogit(self, net):
        pass

    def errorfunc(self, inX):
        pass

    def normalize(self, dataMat):
        pass

    def loadDataSet(self, filename):
        pass

    def addcol(self, matrix1, matrix2):
        pass

    def init_hiddenWB(self):
        pass

    def bpTrain(self):
        pass

    def BPClassfier(self, start, end, steps=30):
        pass

    def classfyLine(self, plt, x, z):
        pass

    def TrendLine(self, plt, color="r"):
        pass

    def drawClassScatter(self, plt):
        pass


a = np.mat(np.loadtxt("BPnet.txt"))
