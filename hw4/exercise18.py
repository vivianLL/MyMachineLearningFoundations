# coding:utf-8
'''
根据16/17题选出的λ（1.000000e-08），利用所有的样本来训练该λ对应的模型，求Ein和Eout
'''

from exercise13 import LinearRegressionReg,X_test,X_train,Y_test,Y_train,data_load

### fit model use
lr = LinearRegressionReg()
lr.fit(X_train, Y_train, 1.000000e-08)

Ein = lr.score(X_train, Y_train)
Eout = lr.score(X_test, Y_test)

print "Error in : ", Ein
print "Error out: ", Eout