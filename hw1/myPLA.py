# coding:utf-8
# 以narray方式读入数据
import matplotlib.pyplot as plt
import numpy as np


def loadData(filepath):
    rawData = np.loadtxt(filepath)
    dataNum = len(rawData)
    print np.shape(rawData)
    return rawData

#画出4维数据的其中两个维度在二维平面上的分布
def plotData(dataSet):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Linear separable data set')
    plt.xlabel('X')
    plt.ylabel('Y')
    labels = np.array(dataSet[:, 4])
    idx_1 = np.where(dataSet[:, 4] == 1)   #numpy.where(condition[, x, y])这里x,y是可选参数，condition是条件，这三个输入参数都是array_like的形式；而且三者的维度相同,当conditon的某个位置的为true时，输出x的对应位置的元素，否则选择y对应位置的元素；如果只有参数condition，则函数返回为true的元素的坐标位置信息；
    p1 = ax.scatter(dataSet[idx_1, 3], dataSet[idx_1, 1], marker='o', color='g', label=1, s=20)
    idx_2 = np.where(dataSet[:, 4] == -1)
    p2 = ax.scatter(dataSet[idx_2, 3], dataSet[idx_2, 1], marker='x', color='r', label=-1, s=20)
    plt.legend(loc='upper right')
    plt.show()

def pla(dataset):

    print np.shape(dataset)
    W = np.zeros(5)  #包括了b
    epoch = 0
    count = 0
    while True:
        epoch += 1
        iscompleted = True
        # for i in range(0,len(dataset)):
        for i in range(0, len(dataset)):
            X = dataset[i][0:4]   #x为narray，当x是list时才可以用dataset[i][:-1]
            X = np.r_[1, X]
            # print type(X)
            print X
            Y = np.dot(W,X)
            print Y
            print np.sign(Y)
            print np.sign(dataset[i][-1])
            if np.sign(Y) == np.sign(dataset[i][-1]):
                continue
            else:
                iscompleted = False
                W = W + (dataset[i][-1]) * np.array(X)
                count += 1
        if iscompleted:
            break
    print("final W is :", W)
    print("epoch is :", epoch)
    print("count is :", count)
    return W

rawData = loadData('hw1_15_train.txt')
# plotData(rawData)
pla(rawData)
