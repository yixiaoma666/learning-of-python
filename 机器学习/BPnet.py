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
        # 一下属性由系统生成
        self.errlist = []  # 误差列表：保存了误差参数的变化用于评估收敛
        self.dataMat = 0  # 训练集
        self.classLabels = 0  # 分类标签集
        self.nSampNum = 0  # 样本集列数
        self.nSampDim = 0  # 样本列数

    def logistic(self, net):
        pass

    def dlogit(self, net):
        pass

    def errorfunc(self, inX):
        pass

    def normalize(self, dataMat):
        pass

    def loadDataSet(self, filename):
        self.dataMat = []
        self.classLabels = []  # TODO
        pass

    def addcol(self, matrix1, matrix2):
        pass

    def init_hiddenWB(self):  # 隐含层初始化
        self.hi_w = 2.0 * (np.random.rand(self.nHidden, self.nSampDim) - 0.5)
        self.hi_b = 2.0 * (np.random.rand(self.nHidden, 1) - 0.5)
        self.hi_wb = np.mat(self.addcol(np.mat(self.hi_w), np.mat(self.hi_b)))

    def init_OutputWB(self):  # 输出层初始化
        self.out_w = 2.0 * (np.random.rand(self.nOut, self.nHidden) - 0.5)
        self.out_b = 2.0 * (np.random.rand(self.nOut, 1) - 0.5)
        self.out_wb = np.mat(self.addcol(np.mat(self.out_w), np.mat(self.out_b)))

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
