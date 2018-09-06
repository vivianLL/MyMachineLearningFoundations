# coding:utf-8
# 以list方式读入数据
import numpy as np#for array compute
from numpy import *
import random

lines = open('hw1_15_train.txt').readlines();
def processData(lines):     #按行处理从txt中读到的训练集（测试集）数据
    dataList = [];
    for line in lines:           #逐行读取txt文档里的训练集
        dataLine = line.strip().split();            #按空格切割一行训练数据（字符串）
        dataLine = [float(data) for data in dataLine];            #字符串转int
        dataList.append(dataLine);           #添加到训练数据列表
    return dataList;

def pla(dataset):
    W=np.ones(5)#initial all weight with 1
    count=0

    while True:
        count+=1
        iscompleted=True
        for i in range(0,len(dataset)):
            X=dataset[i][:-1]
            X = np.r_[1, X]
            print X
            Y=np.dot(W,X)#matrix multiply
            print Y
            if sign(Y)==sign(dataset[i][-1]):
                continue
            else:
                iscompleted=False
                W=W+(dataset[i][-1])*np.array(X)
        if iscompleted:
            break
    print("final W is :",W)
    print("count is :",count)
    return W

def main():
    datalist = processData(lines)
    pla(datalist)

if __name__ == '__main__':
    main()