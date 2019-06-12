import numpy as np
import pandas as pd
import time

source_data = pd.read_csv("zdlj.csv")
lists = source_data.values.tolist()


def main():
    VE = []
    for each in lists:
        VE.append(list(map(eval, each)))

    def Dijkstra(VE, begin, end):
        dict0 = dict(zip(
            list(map(str, list(range(1, 16)) + list(range(17, 22)))), list(range(20))))

        dict1 = dict(zip(dict0.values(), dict0.keys()))

        L = dict(zip(
            list(map(str, list(range(1, 16)) + list(range(17, 22)))),
            [np.inf] * 20))

        L[begin] = 0
        T = list(map(str, list(range(1, 16)) + list(range(17, 22))))

        Temp_V = ""
        while end in T:
            min_num = np.inf
            for i in T:
                min_num = min(min_num, L[i])
            Llist = list(L.items())
            for each in Llist:
                if each[0] not in T:
                    continue
                if each[1] == min_num:
                    Temp_V = each[0]

            T.remove(Temp_V)

            Temp_xianglin_V = []

            hang = VE[dict0[Temp_V]]
            hang_temp = list()
            i = 0
            for each in hang:  # 获取该行有效值，并存入hang_temp
                if each != 0 and each != np.inf:
                    hang_temp.append([each, dict1[i]])
                i += 1

            for each in hang_temp:  # 获取相邻点
                Temp_xianglin_V.append(each[1])

            for i in Temp_xianglin_V:
                if i not in T:
                    Temp_xianglin_V.remove(i)

            for i in Temp_xianglin_V:  # 比较各相邻点已知数据和新更新数据取最小值
                L[i] = min(L[i], L[Temp_V] + hang[dict0[i]])
        return L

    T = list(map(str, list(range(1, 16)) + list(range(17, 22))))

    with open("output.txt", mode="w") as f:
        for each1 in T:
            for each2 in T:
                f.write("{}\t".format(Dijkstra(VE, each1, each2)[each2]))
            f.write("\n")


if __name__ == "__main__":
    a = time.time()
    main()
    b = time.time()
    print(b - a)
