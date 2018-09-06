# coding:utf-8
'''
13
Consider regularized linear regression (also called ridge regression) for classification
Wreg = arg min( λ/N * ||W||^2 + 1/N * ||XW-Y||^2 ).
Because the data sets are for classification, please consider only the 0/1 error for all Questions below.
Let \lambda = 10, which of the followings is the corresponding Ein and Eout ?
'''
# 根据课堂上的推导，我们求得了Wreg = （Xt*X + λI）^-1*XtY，所以该问题只需要把相应的λ代入求Wreg，然后再计算Ein和Eout就好
# 参考网址：https://blog.csdn.net/rikichou/article/details/78378961

import numpy as np
import matplotlib.pyplot as plt

### import training data or testing data
def data_load(path):
    ### open data file and read all lines
    f = open(path)
    try:
        lines = f.readlines()
    finally:
        f.close()

    ### create return data structure
    example_num = len(lines)
    feature_dimension = len(lines[0].strip().split())  ### how to get dimension by the easier way
    X = np.zeros((example_num, feature_dimension))
    Y = np.zeros((example_num, 1))
    X[:,0] = 1      # x多一列x0=1

    ### parse every line
    for index,line in enumerate(lines):
        ### get features
        items = line.strip().split()
        X[index,1:] = [float(_) for _ in items[:-1]]   #这种写法很巧妙
        Y[index] = float(items[-1])

    return X,Y

def data_visual(X, Y, title="Default", xmin=0, xmax=1, ymin=0, ymax=1, func=False, w=[]):
    x1 = X[:,1]
    x2 = X[:,2]
    labels = Y[:,0]

    ### get size array
    dot_size = 20
    size = np.ones((len(x1)))*dot_size

    ### get masked size array for positive points(mask the negative points)
    s_x1 = np.ma.masked_where(labels<0, size)
    ### get masked size array for positive points(mask the positive points)
    s_x2 = np.ma.masked_where(labels>0, size)

    ### plot positive points as x
    plt.scatter(x1, x2, s_x1, marker='x', c='r', label="positive")
    ### plot negative points as o
    plt.scatter(x1, x2, s_x2, marker='o', c='b', label="negative")

    ### plot func if require
    if func:
        x1_dot = np.arange(xmin,xmax,0.01)
        x2_dot = np.arange(ymin,ymax,0.01)
        x1_dot,x2_dot = np.meshgrid(x1_dot, x2_dot)

        f = w[0,0] + w[1,0]*x1_dot + w[2, 0]*x2_dot
        plt.contour(x1_dot, x2_dot, f, 0)

    ### add some labels
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.show()

### load and plot training data
X_train,Y_train = data_load('hw4_train.dat')
data_visual(X_train, Y_train, "Example data")
data_visual(X_train, Y_train, "Partial Example data", 0.45,0.55,0.45,0.55)

class LinearRegressionReg:

    def __init__(self):
        self._dimension = 0

    def fit(self, X, Y, lamb):
        self._dimension = len(X[0])
        self._w = np.zeros((self._dimension,1))
        self._lamb = lamb
        self._w = np.linalg.inv(np.dot(X.T, X) + lamb*np.eye(self._dimension)).dot(X.T).dot(Y)  #矩阵表达

    def predict(self, X):
        result = np.dot(X, self._w)
        return np.array([(1 if _ >= 0 else -1) for _ in result]).reshape(len(X), 1)  #0/1 error

    def score(self, X, Y):
        Y_predict = self.predict(X)
        return sum(Y_predict!=Y)/(len(Y)*1.0)

    def get_w(self):
        return self._w

    def print_val(self):
        print "w: ", self._w

### Error in
lr = LinearRegressionReg()
lr.fit(X_train, Y_train, 10)
Ein = lr.score(X_train, Y_train)
data_visual(X_train, Y_train, title="Training Data", xmin=0.4, xmax=0.6, ymin=0.35, ymax=0.6, func=True, w=lr.get_w())
lr.print_val()
print "Ein : ", Ein

### Error out
X_test, Y_test = data_load('hw4_test.dat')
Eout = lr.score(X_test, Y_test)
data_visual(X_test, Y_test, title="Test Data", xmin=0.4, xmax=0.6, ymin=0.35, ymax=0.6, func=True, w=lr.get_w())
print "Eout : ", Eout