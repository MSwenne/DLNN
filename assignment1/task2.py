import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances, confusion_matrix

def as_numpy(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_numpy()
    if isinstance(obj, list):
        return np.array(obj)
    return obj


class CenterDistanceClassifier:
    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, X, y):
        assert len(y.shape) == 1 or len(y.shape) == 2 and y.shape[1] == 1
        X = as_numpy(X)
        y = as_numpy(y).flatten()

        self.labels = np.array(sorted(set(y)))

        self.centers = []
        for label in self.labels:
            label_points = X[y == label]

            # Note that this center is technically only
            # correct if our metric is Euclidean.
            center = np.mean(label_points, axis=0)
            self.centers.append(center)

        self.centers = np.array(self.centers)

    def predict(self, X):
        center_distances = pairwise_distances(as_numpy(X),
                                              self.centers,
                                              metric=self.metric)
        closest_center = np.argmin(center_distances, axis=1)
        return self.labels[closest_center]

def confusion_dataframe(y_true, y_pred, labels):
    return pd.DataFrame(
        data=confusion_matrix(y_true, y_pred),
        index=labels,
        columns=labels)

if __name__ == "__main__":
    X_train = pd.read_csv("data/train_in.csv", header=None)
    y_train = pd.read_csv("data/train_out.csv", header=None)
    X_test = pd.read_csv("data/test_in.csv", header=None)
    y_test = pd.read_csv("data/test_out.csv", header=None)

    clf = CenterDistanceClassifier("euclidean")
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("Train confusion matrix:")
    print(confusion_dataframe(y_train, y_train_pred, range(10)))
    print("\nTest confusion matrix:")
    print(confusion_dataframe(y_test, y_test_pred, range(10)))
