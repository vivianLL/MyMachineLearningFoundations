# coding:utf-8
### 20. (*) Implement the fixed learning rate stochastic gradient descent algorithm below for logistic regression,
### initialized with 0. Instead of randomly choosing n in each iteration, please simply pick
### the example with the cyclic order n = 1; 2; : : : ; N; 1; 2; : : :. Run the algorithm with η = 0:001 and
### T = 2000 on the following set for training:
### http://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw3/hw3_train.dat
### and the following set for testing:
### http://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw3/hw3_test.dat
### What is the weight vector within your g? What is the Eout(g) from your algorithm, evaluated using
### the 0=1 error on the test set?
### training model
from exercise18 import data_load,LinearRegression

(X, Y) = data_load("hw3_train.dat")
lr_sgd = LinearRegression()
lr_sgd.fit(X, Y, sgd=True, max_interate = 2000)

### get weight vector
print "weight vector: ", lr_sgd.get_w()

### get 0/1 error in test data
test_X, test_Y = data_load("hw3_test.dat")
print "Eout: ", lr_sgd.score(test_X,test_Y)