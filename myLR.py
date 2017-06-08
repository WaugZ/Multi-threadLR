import numpy as np
from scipy import sparse
from itertools import islice
import threading
import time


def sigmod(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, x, y, m, lda):
    ep = 1e-10
    return np.sum((np.multiply(-y, np.log(sigmod(x * theta) + ep)) - np.multiply((1 - y), np.log(1 - sigmod(x * theta) + ep)))) / m + \
            lda * np.sum(np.square(theta)) / (2 * m)


def predition(theta, x):
    predit = sigmod(x * theta)
    predit[predit >= .5] = 1
    predit[predit < .5] = 0
    return predit

sum = 0                         # global to record part sum of gradient
lock = threading.Lock()         # thread lock


def multithreadSum(theta, x, y, lda):
    # print ('thread start m = ' + str(x.shape[0]))
    global sum
    temp = theta
    temp[0] = 0
    temp_sum = x.T * (sigmod(x * theta) - y) + lda * temp
    lock.acquire()
    try:
        sum += temp_sum
    finally:
        lock.release()


def gradient(theta, x, y, m, lda):
    temp = theta
    temp[0] = 0         # the 1st para do not regularize
    max_thread = 4
    mini_banch = int(m / max_thread)
    threads = []
    global sum
    sum = 0
    for i in range(max_thread):                 # create #max_thread threads
        lo = i * mini_banch
        if i < max_thread - 1:
            hi = (i + 1) * mini_banch
        else:
            hi = m
        x_banch = x[lo:hi]
        y_banch = y[lo:hi]
        thread = threading.Thread(target=multithreadSum, args=(theta, x_banch, y_banch, lda,))
        thread.start()
        threads.append(thread)

    for thread in threads:                      # join all threads
        thread.join()
    dJ = sum / m
    return dJ


start = time.time()
print('loading training data...')
file = open('/Users/wangzi/PycharmProjects/test/large_scale/train_data.txt')
train_X = []
train_y = []
train_row_index = []
train_col_index = []
count = 0

for line in islice(file, 1, 1e5):
    data_line = str(line).split()
    train_y.append(int(data_line[0]))
    for i in range(1, len(data_line)):
        index_value = data_line[i].split(':')
        index = int(index_value[0])
        value = float(index_value[1])
        # it is said that test example does not contain more than 132 features -- mahua
        if index > 132:
            break
        train_X.append(value)
        train_row_index.append(count)
        train_col_index.append(index)
    count = count + 1

file.close()

data_X = sparse.csr_matrix((train_X, (train_row_index, train_col_index))).todense()
data_y = np.array(train_y)
data_y = np.mat(data_y)
data_y = data_y.T
m, n = data_X.shape
m = int(.8 * m)
train_X = data_X[:m, :]
train_y = data_y[:m, :]
cv_X = data_X[m:, :]
cv_y = data_y[m:, :]

## feature scaling
print('feature normalizing...')
m, n = train_X.shape
X_mean = []
X_std = []
for i in range(n):
    mean = np.mean(train_X[:, i])
    std = np.std(train_X[:, i])
    X_mean.append(mean)
    X_std.append(std)
X_mean = np.array(X_mean)
X_std = np.array(X_std)

epthelon = 1e-20
for i in range(n):
    train_X[:, i] = (train_X[:, i] - X_mean[i]) / (X_std[i] + epthelon)

## training
print('training...')
theta = np.mat(np.zeros((n, 1)))

alpha = .001            # learning rate
lda = 1                # regularized
max_iter = int(1e3)
for i in range(max_iter):
    theta = theta - alpha * gradient(theta, train_X, train_y, m, lda)
    # print theta
    # print(cost(theta, train_X, train_y, m))
print(cost(theta, train_X, train_y, m, lda))

## cv testing
print('testing')
print cost(theta, cv_X, cv_y, int(.2 * m), lda)
# print(cost(theta, cv_X, cv_y, int(.2 * m)))

print('program finished in ' + str(time.time() - start) + 's')