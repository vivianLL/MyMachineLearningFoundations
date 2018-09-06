# coding:utf-8
# https://blog.csdn.net/rikichou/article/details/78321706
### For Questions 13-15, Generate a training set of N = 1000 points on X = [−1; 1] × [−1; 1] with uniform probability of
### picking each x 2 X. Generate simulated noise by flipping the sign of the output in a random 10% subset
### of the generated training set.

import random
import numpy as np
import matplotlib.pyplot as plt

### target function f(x1, x2) = sign(x1^2 + x2^2 - 0.6)
def target_function(x1, x2):
    return (1 if (x1*x1 + x2*x2 - 0.6) >= 0 else -1)

### plot dot picture, two dimension features
def plot_dot_picture(features, lables, w=np.zeros((3, 1))):
    x1 = features[:,1]
    x2 = features[:,2]
    y = lables[:,0]

    plot_size = 20
    size = np.ones((len(x1)))*plot_size

    size_x1 = np.ma.masked_where(y<0, size)
    size_x2 = np.ma.masked_where(y>0, size)

    ### plot scatter
    plt.scatter(x1, x2, s=size_x1, marker='x', c='r')
    plt.scatter(x1, x2, s=size_x2, marker='o', c='b')

    ### plot w line
    x1_tmp = np.arange(-1,1,0.01)
    x2_tmp = np.arange(-1,1,0.01)

    x1_tmp, x2_tmp = np.meshgrid(x1_tmp, x2_tmp)

    f = x1_tmp*w[1, 0] + x2_tmp*w[2, 0] + w[0, 0]

    try:
        plt.contour(x1_tmp, x2_tmp, f, 0)
    except ValueError:
        pass

    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.title('Feature scatter plot')

    plt.legend()

    plt.show()

### return a numpy array
def training_data_with_random_error(num=1000):
    features = np.zeros((num, 3))
    labels = np.zeros((num, 1))

    points_x1 = np.array([round(random.uniform(-1, 1) ,2) for _ in range(num)])
    points_x2 = np.array([round(random.uniform(-1, 1) ,2) for _ in range(num)])

    for i in range(num):
        features[i, 0] = 1
        features[i, 1] = points_x1[i]
        features[i, 2] = points_x2[i]
        labels[i] = target_function(points_x1[i], points_x2[i])
        ### choose 10 error labels
        if i <= num*0.1:
            labels[i] = (1 if labels[i]<0 else -1)
    return features, labels

def error_rate(features, labels, w):
    wrong = 0
    for i in range(len(labels)):
        if np.dot(features[i], w)*labels[i,0] < 0:
            wrong += 1
    return wrong/(len(labels)*1.0)

(features,labels) = training_data_with_random_error(1000)
plot_dot_picture(features, labels)

### 13.1 (*) Carry out Linear Regression without transformation, i.e., with feature vector:
### (1; x1; x2);
### to find wlin, and use wlin directly for classification. Run the experiments for 1000 times and plot
### a histogram on the classification (0/1) in-sample error (Ein). What is the average Ein over 1000
### experiments?

"""
    linear regression:
    model     : g(x) = Wt * X
    strategy  : squared error
    algorithm : close form(matrix)
    result    : w = (Xt.X)^-1.Xt.Y
"""
def linear_regression_closed_form(X, Y):
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)

w = linear_regression_closed_form(features, labels)

"""
    plot the one result(just for visual)
"""
plot_dot_picture(features, labels, w)

"""
    run 1000 times, and plot histogram
"""
error_rate_array = []
for i in range(1000):
    (features,labels) = training_data_with_random_error(1000)
    w = linear_regression_closed_form(features, labels)
    error_rate_array.append(error_rate(features, labels, w))
bins = np.arange(0,1,0.05)
plt.hist(error_rate_array, bins, rwidth=0.8, histtype='bar')
plt.title("Error rate histogram(without feature transform)")
plt.show()

### error rate, approximately 0.5
avr_err = sum(error_rate_array)/(len(error_rate_array)*1.0)

print "13.1--Linear regression for classification without feature transform:Average error--",avr_err