import pandas as pd
import numpy as np

from sklearn.metrics import (
    pairwise_distances, accuracy_score
)

from common import as_numpy, confusion_dataframe

class CenterDistanceClassifier:
    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, X, y):
        assert len(y.shape) == 1 or len(y.shape) == 2 and y.shape[1] == 1
        X = as_numpy(X)
        y = as_numpy(y).flatten()
        y_label_idx, self.labels = pd.factorize(y, sort=True)

        self.centers = []
        for label_idx, _ in enumerate(self.labels):
            label_points = X[y_label_idx == label_idx]

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



if __name__ == "__main__":
    # Read training data and testing data
    X_train = pd.read_csv("data/train_in.csv", header=None)
    y_train = pd.read_csv("data/train_out.csv", header=None)
    X_test = pd.read_csv("data/test_in.csv", header=None)
    y_test = pd.read_csv("data/test_out.csv", header=None)


    for metric in ["euclidean", "manhattan", "cosine"]:
        clf = CenterDistanceClassifier(metric)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        print("\n\nDistance metric:", metric)
        print("Train confusion matrix:")
        print(confusion_dataframe(y_train, y_train_pred, range(10)))
        print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.3%}")
        print("\nTest confusion matrix:")
        print(confusion_dataframe(y_test, y_test_pred, range(10)))
        print(f"Test accuracy {accuracy_score(y_test, y_test_pred):.3%}")
