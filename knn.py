import numpy as np


class KNN:
    def __init__(self, x_train, y_train, k):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def predict(self, x_test):
        def euclidean(b):
            return np.sum(np.sqrt(np.power(x_test - b[0], 2)))

        sorted_x_train, sorted_y_train = zip(*sorted(zip(self.x_train, self.y_train), key=euclidean))
        choosen_k = sorted_y_train[0:self.k]
        return max(choosen_k, key=choosen_k.count)

