# coding:utf-8
#机器学习基石 作业1 15-20题 PLA pocket
# 参考网址：http://blog.csdn.net/Light_blue_love/article/details/76386739

import numpy as np
import os

#读取数据，filepath为一个文件路径，load_Data函数返回两个矩阵x,y
def load_Data(filepath):
        file = open(filepath)
        lines = file.readlines()
        sampleNum = len(lines)
        x = np.zeros((sampleNum,5))
        y = np.zeros((sampleNum,1))
        print(y.shape)
        for i in range(sampleNum):
                items = lines[i].strip().split('\t')
                y[i,0] = float(items[1])
                items = items[0].strip().split(' ')
                x[i,0] = 1
                for j in range(0,4):
                        x[i,j+1] = items[j]
        return x,y


# sign函数，参数mat为w*x的一个矩阵
def sign(mat):
    sampleNum = mat.shape[0]
    for i in range(sampleNum):
        if (mat[i][0] > 0):
            mat[i][0] = 1
        else:
            mat[i][0] = -1
    return mat


# beta表示每次更新时候的学习率
def pla(x, y, beta):
    w = np.zeros((5, 1))
    sampleNum = len(x)
    # 设置一个标识，flag用来表示样本中时候有分类错误的，刚开始时置为False
    flag = False
    # count计数器，用来统计更新了多少次
    count = 0
    while (flag == False):
        # 当全部都被正确分类的时候，就可以跳出while循环
        for i in range(sampleNum):
            if (flag == False):
                pre_y = sign(np.dot(x, w))
            if (pre_y[i] == y[i]):
                flag = True
            else:  # 有错才更新
                w = w + beta * y[i][0] * np.mat(x[i]).T
                count = count + 1
                flag = False

    return count


if __name__ == '__main__':
    # # 将环境切换为当前路径
    # os.chdir(r"C:\Users\bs\Desktop")
    # 文件
    filepath = 'hw1_15_train.dat'
    x, y = load_Data(filepath)
    # 一个列表
    index = [i for i in range(len(x))]
    print index
    # ssum表示总共出错的次数
    ssum = 0
    # updates表示要重复运行的次数
    updates = 2000
    # 学习率
    beta = 0.5
    for i in range(updates):
        X = x[:, :]
        Y = y[:, :]
        # 将上面的列表打乱
        np.random.shuffle(index)
        # 得到相应打乱的x,y
        data = X[index]
        label = Y[index]
        count = pla(data, label, beta)
        print(count, '第', i, '次')
        ssum = ssum + count
        # 计算平均更新次数
    averageTimes = ssum / updates
    print(averageTimes, "averagetimes")