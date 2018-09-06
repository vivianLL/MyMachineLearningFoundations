# coding:utf-8
'''
画出λ在分别取值{ 10^2, 10^1, 10^0……10^-8, 10^-9, 10^-10}时的Ein的曲线图，
分别求出Ein最小时的λ和Eout最小时的λ（Ein或者Eout相同时，取最大的λ）
'''
from exercise13 import LinearRegressionReg,X_test,X_train,Y_test,Y_train
import matplotlib.pyplot as plt

### get lambs
log_lambs = range(2, -11, -1)
lambs = [10**_ for _ in range(2, -11, -1)]
Ein = []
Eout = []
lr = LinearRegressionReg()

### get Ein and Eout array
for index,lamb in enumerate(lambs):
    ### fit models
    lr.fit(X_train, Y_train, lamb)

    ### get Ein
    Ein.append(lr.score(X_train, Y_train))

    ### get Eout
    Eout.append(lr.score(X_test, Y_test))

### plot Ein and Eout curve
plt.plot(log_lambs, Ein, label="Error In", marker='o')
plt.plot(log_lambs, Eout, label="Error Out", marker='o')

plt.title("Curve of Error")
plt.xlabel("Log lambda")
plt.ylabel("Error")
plt.legend()
plt.show()

print ("λ = %e with minimal Ein: %f and Eout: %f"%(lambs[Ein.index(min(Ein))], min(Ein),Eout[Ein.index(min(Ein))]))
print ("λ = %e with minimal Eout: %f and Ein: %f"%(lambs[Eout.index(min(Eout))], min(Eout),Ein[Eout.index(min(Eout))]))