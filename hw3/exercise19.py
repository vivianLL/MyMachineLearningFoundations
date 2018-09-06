# coding:utf-8
### 19. (*) Implement the fixed learning rate gradient descent algorithm below for logistic regression,
### initialized with 0. Run the algorithm with η = 0:01 and T = 2000 on the following set for training:
###                http://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw3/hw3_train.dat
### and the following set for testing:
###                http://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw3/hw3_test.dat
### What is the weight vector within your g? What is the Eout(g) from your algorithm, evaluated using
### the 0=1 error on the test set?

### training model
from exercise18 import data_load,LinearRegression

(X, Y) = data_load("hw3_train.dat")
lr_eta = LinearRegression()
lr_eta.fit(X, Y, 0.01, 2000)

### get weight vector
print "weight vector: ", lr_eta.get_w()

### get 0/1 error in test data
test_X, test_Y = data_load("hw3_test.dat")
print "Eout: ", lr_eta.score(test_X,test_Y)