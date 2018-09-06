# coding:utf-8
#结果与其他答案不同

import numpy as np
from numpy import *


def processData(lines):     #按行处理从txt中读到的训练集（测试集）数据
    dataList = [];
    for line in lines:           #逐行读取txt文档里的训练集
        dataLine = line.strip().split();            #按空格切割一行训练数据（字符串）
        dataLine = [float(data) for data in dataLine];            #字符串转int
        dataList.append(dataLine);           #添加到训练数据列表
    print shape(dataList)
    return dataList;

def ErrorRate(W,dataset):
    error = 0
    for i in range(len(dataset)):
        X = dataset[i][0:4]  # x为narray，当x是list时才可以用dataset[i][:-1]
        X = np.r_[1, X]
        Y = np.dot(W, X)
        if np.sign(Y) != np.sign(dataset[i][-1]):
            error+=1
    return  error

def pocket(dataset,eta):
    W = np.zeros(5)  # 包括了b
    W_n = np.zeros(5)  # 包括了b
    errorNum = ErrorRate(W,dataset)
    i = 0
    iterateTimes = 50
    while i <= iterateTimes :

        X = dataset[i][0:4]  # x为narray，当x是list时才可以用dataset[i][:-1]
        X = np.r_[1, X]
        Y = np.dot(W, X)
        if np.sign(Y) != np.sign(dataset[i][-1]):
            W_n = W_n + eta * (dataset[i][-1]) * np.array(X)
            # print W_n
            errorNum_n = ErrorRate(W_n,dataset)
            # print errorNum_n
            # count = count + 1
            if errorNum_n < errorNum:
                W = W_n
                errorNum = errorNum_n
            # W = W_n
            # errorNum = errorNum_n    第19题

        i = i + 1

    return W

if __name__ == '__main__':
    traindata = open('hw1_18_train.txt').readlines();
    datalist = processData(traindata)

    testdata = open('hw1_18_test.txt').readlines();
    test = processData(testdata)


    # 一个列表
    index = [i for i in range(len(datalist))]

    # ssum表示总共出错的次数
    ssum = 0
    # updates表示要重复运行的次数
    updates = 100
    # # 学习率
    eta = 1
    for i in range(updates):
        data = []  #不能放在for循环外
        # 将上面的列表打乱
        np.random.shuffle(index)
        # print index
        for j in range(len(index)):
            idx = index[j]
            data.append(datalist[idx])# 得到相应打乱的data

        W = pocket(data,eta)
        err = ErrorRate(W,test)
        ssum = ssum + err
        print err

    # pla(datalist)

    averageErr = ssum / updates / 500.0
    print(averageErr, "average error")