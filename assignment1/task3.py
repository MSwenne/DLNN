import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from common import as_numpy, confusion_dataframe


class MulticlassPerceptronClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        assert len(y.shape) == 1 or len(y.shape) == 2 and y.shape[1] == 1
        X = as_numpy(X)
        y = as_numpy(y).flatten()
        y_idx, self.labels = pd.factorize(y, sort=True)

        self.n_in = X.shape[1]
        self.n_out = len(self.labels)

        self.bias = np.random.normal(size=(1, self.n_out))
        self.weights = np.random.normal(size=(self.n_in, self.n_out))

        self.iterations = 0
        has_mispredictions = True
        while has_mispredictions:
            nn_out = self._neural_output(X)
            pred_idx = np.argmax(nn_out, axis=1)
            mispredicted = pred_idx != y_idx
            has_mispredictions = np.any(mispredicted)

            # Generate boolean matrices (M, k) where M is the number of mispredicted
            # samples, and k is the number of classes.
            true_node_values = nn_out[mispredicted, y_idx[mispredicted]]
            too_large_node = nn_out[mispredicted] > true_node_values[:,np.newaxis]
            true_node = np.eye(self.n_out)[y_idx[mispredicted]]

            # Add vectors from mispredicted X to node weights that were correct,
            # and subtract it from nodes that were larger than the correct value.
            xweight = true_node - too_large_node

            self.weights += np.dot(xweight.T, X[mispredicted]).T
            self.bias += np.sum(xweight)
            self.iterations += 1


    def predict(self, X):
        return self._output_label(self._neural_output(as_numpy(X)))

    def _neural_output(self, X):
        return np.dot(X, self.weights) + self.bias

    def _output_label(self, nn_out):
        return self.labels[np.argmax(nn_out, axis=1)]


if __name__ == "__main__":
    X_train = pd.read_csv("data/train_in.csv", header=None)
    y_train = pd.read_csv("data/train_out.csv", header=None)
    X_test = pd.read_csv("data/test_in.csv", header=None)
    y_test = pd.read_csv("data/test_out.csv", header=None)

    clf = MulticlassPerceptronClassifier()
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    print("Train confusion matrix:")
    print(confusion_dataframe(y_train, y_train_pred, range(10)))
    print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.3%}")
    print("\nTest confusion matrix:")
    print(confusion_dataframe(y_test, y_test_pred, range(10)))
    print(f"Test accuracy {accuracy_score(y_test, y_test_pred):.3%}")
