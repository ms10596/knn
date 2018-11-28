from read_data import read_data
from knn import KNN
import matplotlib.pyplot as plt
if __name__ == '__main__':
    x_train, y_train = read_data("TrainData.txt")
    x_test, y_test = read_data("TestData.txt")

    clf = KNN(x_train, y_train)
    accuracies = []
    for i in range(1, 10):
        print("k value:", i)
        correctly_classified = clf.accuracy(x_test, y_test, i)
        accuracy = correctly_classified / len(y_test)
        accuracies.append(accuracy)
        print("Number of correctly classified instances:", correctly_classified)
        print("Number of instances:", len(y_test))
        print("Accuracy:", accuracy)
        print("\n\n")

plt.plot(accuracies)
plt.show()