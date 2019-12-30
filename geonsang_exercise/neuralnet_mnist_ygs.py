# import timeit
# code_to_test = """
import sys, os
sys.path.append('/Users/yoogeonsang/Documents/python/DL/deep_learning_scratch/')
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common import functions as func
BATCH = True

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open('/Users/yoogeonsang/Documents/python/DL/deep_learning_scratch/geonsang_exercise/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = func.sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = func.sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = func.softmax(a3)
    return y

if BATCH==True:
    #Main(Batch)
    x, t = get_data()
    network = init_network()
    batch_size = len(x)
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print('Accuracy:' + str(float(accuracy_cnt/len(x))))
else:
    #Main(Non-batch)
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    # """
# # print('--Batch--')
# print('--No Batch--')
# print(timeit.timeit(code_to_test, number=10)/10)