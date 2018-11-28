import pandas as pd
import numpy as np
from read_data import read_data
from knn import KNN

if __name__ == '__main__':
    x_train, y_train = read_data("TrainData.txt")
    x_test, y_test = read_data("TestData.txt")

    clf = KNN(x_train, y_train, 3)
    clf.predict(x_train[1])
