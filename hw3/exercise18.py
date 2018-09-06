# coding:utf-8
### 18. (*) Implement the fixed learning rate gradient descent algorithm below for logistic regression, initialized with 0. Run the algorithm with η = 0:001 and T = 2000 on the following set for training:
###                http://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw3/hw3_train.dat
### and the following set for testing:
###                http://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw3/hw3_test.dat
### What is the weight vector within your g? What is the Eout(g) from your algorithm, evaluated using
### the 0=1 error on the test set?
import math
import numpy as np

"""
Read data from data file
"""
def data_load(file_path):

    ### open file and read lines
    f = open(file_path)
    try:
        lines = f.readlines()
    finally:
        f.close()

    ### create features and lables array
    example_num = len(lines)
    feature_dimension = len(lines[0].strip().split())  ###i do not know how to calculate the dimension

    features = np.zeros((example_num, feature_dimension))
    features[:,0] = 1
    labels = np.zeros((example_num, 1))

    for index,line in enumerate(lines):
        ### items[0:-1]--features   items[-1]--label
        items = line.strip().split(' ')
        ### get features
        features[index,1:] = [float(str_num) for str_num in items[0:-1]]

        ### get label
        labels[index] = float(items[-1])

    return features,labels

### gradient descent
def gradient_descent(X, Y, w):
    ### -YnWtXn
    tmp = -Y*(np.dot(X, w))

    ### θ(-YnWtXn) = exp(tmp)/1+exp(tmp)
    ### weight_matrix = np.array([math.exp(_)/(1+math.exp(_)) for _ in tmp]).reshape(len(X), 1)
    weight_matrix = np.exp(tmp)/((1+np.exp(tmp))*1.0)
    gradient = 1/(len(X)*1.0)*(sum(weight_matrix*-Y*X).reshape(len(w), 1))

    return gradient

### gradient descent
def stochastic_gradient_descent(X, Y, w):
    ### -YnWtXn
    tmp = -Y*(np.dot(X, w))

    ### θ(-YnWtXn) = exp(tmp)/1+exp(tmp)
    ###weight = math.exp(tmp[0])/((1+math.exp(tmp[0]))*1.0)
    weight = np.exp(tmp)/((1+np.exp(tmp))*1.0)

    gradient = weight*-Y*X
    return gradient.reshape(len(gradient), 1)

### LinearRegression Class,first time use Class, HaHa...
class LinearRegression:
    'Linear Regression of My'

    def __init__(self):
        pass

    ### fit model
    def fit(self, X, Y, Eta=0.001, max_interate=2000, sgd=False):
        ### ∂E/∂w = 1/N * ∑θ(-YnWtXn)(-YnXn)
        self.__w = np.zeros((len(X[0]),1))

        if sgd == False:
            for i in range(max_interate):
                self.__w = self.__w - Eta*gradient_descent(X, Y, self.__w)
        else:
            index = 0
            for i in range(max_interate):
                if (index >= len(X)):
                    index = 0
                self.__w = self.__w - Eta*stochastic_gradient_descent(np.array(X[index]), Y[index], self.__w)
                index += 1
    ### predict
    def predict(self, X):
        binary_result = np.dot(X, self.__w) >= 0
        return np.array([(1 if _ > 0 else -1) for _ in binary_result]).reshape(len(X), 1)

    ### get vector w
    def get_w(self):
        return self.__w

    ### score(error rate)
    def score(self, X, Y):
        predict_Y = self.predict(X)
        return sum(predict_Y != Y)/(len(Y)*1.0)

### training model
(X, Y) = data_load("hw3_train.dat")
lr = LinearRegression()
lr.fit(X, Y, max_interate = 2000)

### get weight vector
print "weight vector: ", lr.get_w()

### get 0/1 error in test data
test_X, test_Y = data_load("hw3_test.dat")
###print "Eout: ", lr.score(test_X,test_Y)
lr.score(test_X,test_Y)