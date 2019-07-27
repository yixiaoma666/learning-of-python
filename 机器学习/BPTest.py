import numpy as np
import operator
from BPnet import *
import matplotlib.pyplot as plt

bpnet = BPNet()
bpnet.loadDataSet("BPNetData.txt")
bpnet.dataMat = bpnet.normalize(bpnet.dataMat)

bpnet.drawClassScatter()

bpnet.bpTrain()
print(bpnet.out_wb)
print(bpnet.hi_wb)

x, z = bpnet.BPClassfier(-3.0, 3.0)
bpnet.classfyLine(x, z)
plt.show()

bpnet.TrendLine()
plt.show()

