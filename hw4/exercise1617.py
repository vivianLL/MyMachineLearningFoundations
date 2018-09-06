# coding:utf-8
'''
接下来的问题和validation有关。将我们的样本分为训练集（120）和测试集（80），将所有的模型（不同的lambda值）在训练集上训练，然后讲得出的假设在测试集上验证
问题16/17：single validation
求出所有λ对应的Etrain,Eval,Eout,分别找出最小Etrain的λ和最小Eval的λ
'''

from exercise13 import LinearRegressionReg,X_test,X_train,Y_test,Y_train,data_load
from exercise1415 import Ein
import matplotlib.pyplot as plt

### get train set and test set
X,Y = data_load('hw4_train.dat')

X_tra = X[0:120,:]
Y_tra = Y[0:120,:]

X_val = X[120:,:]
Y_val = Y[120:,:]

###
log_lambs = range(2, -11, -1)
lambs = [10**_ for _ in range(2, -11, -1)]
Etrain = []
Eout = []
Eval = []

lr = LinearRegressionReg()

### get Etrain and Eval and Eout array
for index,lamb in enumerate(lambs):
    ### fit models
    lr.fit(X_tra, Y_tra, lamb)

    ### get Etrain
    Etrain.append(lr.score(X_tra, Y_tra))

    ### get Eval
    Eval.append(lr.score(X_val, Y_val))

    ### get Eout
    Eout.append(lr.score(X_test, Y_test))

### plot Ein and Eval and Eout curve
print len(Ein)
print len(log_lambs)
plt.plot(log_lambs, Etrain, label="Error Training", marker='o')
plt.plot(log_lambs, Eval, label="Error Validation", marker='o')
plt.plot(log_lambs, Eout, label="Error Out", marker='o')

plt.title("Curve of Error")
plt.xlabel("Log lambda")
plt.ylabel("Error")
plt.legend()
plt.show()

print ("λ = %e with minimal Etrain: %f,Eval: %f, Eout: %f"%(lambs[Etrain.index(min(Etrain))], min(Etrain), \
                                                            Eval[Etrain.index(min(Etrain))],Eout[Etrain.index(min(Etrain))]))
print ("λ = %e with minimal Eval: %f, Eout: %f,Etrain: %f"%(lambs[Eval.index(min(Eval))], min(Eval),\
                                                Eout[Eval.index(min(Eval))],Etrain[Eval.index(min(Eval))]))


'''
根据16/17题选出的λ（1.000000e-08），利用所有的样本来训练该λ对应的模型，求Ein和Eout
'''
### load data
X_train,Y_train = data_load('hw4_train.dat')

### fit model use
lr = LinearRegressionReg()
lr.fit(X_train, Y_train, 1.000000e-08)

Ein = lr.score(X_train, Y_train)
Eout = lr.score(X_test, Y_test)

print "Error in : ", Ein
print "Error out: ", Eout