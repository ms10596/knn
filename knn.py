import numpy as np


class KNN:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.k = 1

    def predict(self, x_test):
        def euclidean(b):
            return np.sum(np.sqrt(np.power(x_test - b[0], 2)))

        sorted_x_train, sorted_y_train = zip(*sorted(zip(self.x_train, self.y_train), key=euclidean))
        chosen_k = sorted_y_train[0:self.k]
        return max(chosen_k, key=chosen_k.count)

    def accuracy(self, x_test, y_test, k):
        self.k = k
        y_hat = list(map(self.predict, x_test))
        true_no = np.count_nonzero(y_test == y_hat)
        return true_no
