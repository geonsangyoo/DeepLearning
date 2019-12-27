import sys, os
sys.path.append('/Users/yoogeonsang/Documents/python/DL/deep_learning_scratch/')
from dataset import mnist as mn

#test
(x_train, t_train), (x_test, t_test) = mn.load_mnist(flatten=True, normalize=False)

#print
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)