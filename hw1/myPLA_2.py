# coding:utf-8
import numpy as np

lines = open('hw1_15_train.txt').readlines();
def processData(lines):     #按行处理从txt中读到的训练集（测试集）数据
    dataList = [];
    for line in lines:           #逐行读取txt文档里的训练集
        dataLine = line.strip().split();            #按空格切割一行训练数据（字符串）
        dataLine = [float(data) for data in dataLine];            #字符串转int
        dataList.append(dataLine);           #添加到训练数据列表
    return dataList;

# beta表示每次更新时候的学习率
def pla(dataset,eta):

    W = np.zeros(5)  #包括了b
    count = 0
    while True:
        iscompleted = True
        for i in range(0, len(dataset)):
            X = dataset[i][0:4]   #x为narray，当x是list时才可以用dataset[i][:-1]
            X = np.r_[1, X]
            Y = np.dot(W,X)
            if np.sign(Y) == np.sign(dataset[i][-1]):
                continue
            else:
                iscompleted = False
                W = W + eta*(dataset[i][-1]) * np.array(X)
                count = count + 1
        if iscompleted:
            break

    return count

if __name__ == '__main__':
    datalist = processData(lines)
    # plotData(rawData)


    # 一个列表
    index = [i for i in range(len(datalist))]

    # ssum表示总共出错的次数
    ssum = 0
    # updates表示要重复运行的次数
    updates = 2000
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

        count = pla(data,eta)
        print(count, '第', i, '次')
        ssum = ssum + count
        # 计算平均更新次数

    # pla(datalist)

    averageTimes = ssum / updates
    print(averageTimes, "averagetimes")