# encoding=utf8
# https://www.cnblogs.com/xbf9xbf/p/4614387.html

import sys
import numpy as np
import math
from random import *


# read input data ( train or test )
def read_input_data(path):
    x = []
    y = []
    for line in open(path).readlines():
        items = line.strip().split(' ')
        tmp_x = []
        for i in range(0, len(items) - 1): tmp_x.append(float(items[i]))
        x.append(tmp_x)
        y.append(float(items[-1]))
    return np.array(x), np.array(y)


def calculate_W_rigde_regression(x, y, LAMBDA):
    Z_v = np.linalg.inv(np.dot(x.transpose(), x) + LAMBDA * np.eye(x.shape[1]))
    return np.dot(np.dot(Z_v, x.transpose()), y)


# test result
def calculate_E(w, x, y):
    scores = np.dot(w, x.transpose())
    predicts = np.where(scores >= 0, 1.0, -1.0)
    Eout = sum(predicts != y)
    return (Eout * 1.0) / predicts.shape[0]


if __name__ == '__main__':

    # prepare train and test data
    x, y = read_input_data("hw4_train.dat")
    x = np.hstack((np.ones(x.shape[0]).reshape(-1, 1), x))
    test_x, test_y = read_input_data("hw4_test.dat")
    test_x = np.hstack((np.ones(test_x.shape[0]).reshape(-1, 1), test_x))
    # lambda
    LAMBDA_set = [i for i in range(2, -11, -1)]

    ## Q13~Q15
    min_Ein = 1
    min_Eout = 1
    target_lambda = 2
    for LAMBDA in LAMBDA_set:
        # calculate ridge regression W
        W = calculate_W_rigde_regression(x, y, pow(10, LAMBDA))
        Ein = calculate_E(W, x, y)
        Eout = calculate_E(W, test_x, test_y)
        # update Ein Eout lambda
        if Eout < min_Eout:
            target_lambda = LAMBDA
            min_Ein = Ein
            min_Eout = Eout
    # print min_Ein
    # print min_Eout
    # print target_lambda

    ## Q16~Q18
    min_Etrain = 1
    min_Eval = 1
    min_Eout = 1
    target_lambda = 2
    split = 120
    for LAMBDA in LAMBDA_set:
        # calculate ridge regression W
        W = calculate_W_rigde_regression(x[:split], y[:split], pow(10, LAMBDA))
        Etrain = calculate_E(W, x[:split], y[:split])
        Eval = calculate_E(W, x[split:], y[split:])
        Eout = calculate_E(W, test_x, test_y)
        # update Ein Eout lambda
        if Eval < min_Eval:
            target_lambda = LAMBDA
            min_Etrain = Etrain
            min_Eval = Eval
            min_Eout = Eout
    # print min_Etrain
    # print min_Eval
    # print min_Eout
    # print target_lambda

    W = calculate_W_rigde_regression(x, y, pow(10, target_lambda))
    optimal_Ein = calculate_E(W, x, y)
    optimal_Eout = calculate_E(W, test_x, test_y)
    # print optimal_Ein
    # print optimal_Eout

    ## Q19~Q20
    min_Ecv = 1
    target_lambda = 2
    V = 5
    V_range = []
    for i in range(0, V): V_range.append([i * (x.shape[0] / V), (i + 1) * (x.shape[0] / V)])

    for LAMBDA in LAMBDA_set:
        total_Ecv = 0
        for i in range(0, V):
            # train x, y
            train_x = []
            train_y = []
            for j in range(0, V):
                if j != i:
                    train_x.extend(x[range(V_range[j][0], V_range[j][1])].tolist())
                    train_y.extend(y[range(V_range[j][0], V_range[j][1])].tolist())
            train_x = np.array(train_x)
            train_y = np.array(train_y)
            # test x, y
            test_x = x[range(V_range[i][0], V_range[i][1])]
            test_y = y[range(V_range[i][0], V_range[i][1])]
            W = calculate_W_rigde_regression(train_x, train_y, pow(10, LAMBDA))
            Ecv = calculate_E(W, test_x, test_y)
            total_Ecv = total_Ecv + Ecv
        print "total Ecv:" + str(total_Ecv)
        if min_Ecv > (total_Ecv * 1.0) / V:
            min_Ecv = (total_Ecv * 1.0) / V
            target_lambda = LAMBDA
    print "minimum Ecv:" + str(min_Ecv)
    print target_lambda

    W = calculate_W_rigde_regression(x, y, pow(10, target_lambda))
    Ein = calculate_E(W, x, y)
    test_x, test_y = read_input_data("hw4_test.dat")
    test_x = np.hstack((np.ones(test_x.shape[0]).reshape(-1, 1), test_x))
    Eout = calculate_E(W, test_x, test_y)
    print Ein
    print Eout 