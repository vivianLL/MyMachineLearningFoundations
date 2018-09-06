# coding:utf-8
'''
接下来的内容是cross validation。我们将样本平均分为5份，每份有40个样本。采用固定的拆分，因为给定的示例已经被随机打乱。
求出λ在分别取值{ 10^2, 10^1, 10^0……10^-8, 10^-9, 10^-10}时Ecv最小时的λ（Ecv相同时，取最大的λ）
'''

from exercise13 import LinearRegressionReg,data_load
import numpy as np
import matplotlib.pyplot as plt
import heapq

### get train set and test set
X,Y = data_load('hw4_train.dat')

###
log_lambs = range(-10, 3, 1)   # 从小到大排序，与后面的heapq一致
lambs = [10**_ for _ in log_lambs]
Ecv = []

lr = LinearRegressionReg()

num = len(Y)
fold = 5
size = num/fold
Ecv = []
min_Ecv = 1
### get Etrain and Eval and Eout array
for index,lamb in enumerate(lambs):
    total_Ecv = 0
    Err = []
    for index,i in enumerate(np.arange(0,num,size)[0:fold]):
        start = index*size
        end = (index+1)*size if index+1 < fold else num
        # print "start: ",start," end: ", end

        X_val = X[start:end]
        X_train = np.concatenate((X[:start],X[end:]))

        Y_val = Y[start:end]
        Y_train = np.concatenate((Y[:start],Y[end:]))

        ### fit models
        lr.fit(X_train, Y_train, lamb)

        ### get Err
        total_Ecv = total_Ecv + lr.score(X_val, Y_val)   # 注意Ecv的计算方式

    ### get Ecv
    # print "total Ecv:" + str(total_Ecv)
    if min_Ecv > (total_Ecv * 1.0) / fold:   # 求最小值
        min_Ecv = (total_Ecv * 1.0) / fold
        target_lambda = lamb
    Ecv.append((total_Ecv * 1.0) / fold)
print "minimum Ecv:" + str(min_Ecv)
print target_lambda

plt.plot(log_lambs, Ecv, label="Ecv", marker='o')
plt.title("Error Cross Validation")
plt.xlabel("Log lambda")
plt.ylabel("Error")
plt.legend()
plt.show()

print(Ecv)
print ("λ = %e with minimal Ecv: %f"%(lambs[Ecv.index(min(Ecv))], min(Ecv)))

'''
根据19题计算得到的模型（λ=1.000000e-08），将模型在整个训练集上进行训练，然后在测试集上进行测试，得出Ein和Eout
'''
### get train set and test set
X,Y = data_load('hw4_train.dat')
X_test,Y_test = data_load('hw4_test.dat')

lr = LinearRegressionReg()
lr.fit(X, Y, target_lambda)
Ein = lr.score(X, Y)
Eout = lr.score(X_test, Y_test)

print "Ein: ", Ein
print "Eout: ", Eout
