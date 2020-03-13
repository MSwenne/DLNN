import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from common import as_numpy, confusion_dataframe


class CoverTheoremPerceptron:
    def __init__(self):
        pass
    
    def fit(self,X, y, true_label_names):
        X = as_numpy(X)
        y = as_numpy(y).flatten()
        y[y == true_label_names[0]] = -1
        y[y == true_label_names[1]] = 1
        self.true_label_names = true_label_names
        self.labels = np.array([-1,1])

        self.n_in = X.shape[1]
        self.n_out = len(self.labels)
        self.lr = 1
        self.weights = np.random.normal(size=(self.n_in, 1))
        self.bias = np.random.normal(size=(1,1))
        self.iterations = 0
        has_mispredictions = True
        while has_mispredictions:
            self.iterations +=1
            # print('self.iterations', self.iterations)
            prediction = np.dot(X,self.weights)

            prediction[prediction < 0] = -1
            prediction[prediction >= 0] = 1
            
            mispredicted = prediction.flatten() != y
            has_mispredictions = np.any(mispredicted)
            if has_mispredictions:
                tmp = np.argwhere(mispredicted).flatten()
                random_index_update = np.random.choice(tmp)
                self.weights += self.lr * np.dot(y[random_index_update], X[random_index_update][:,np.newaxis]) 

    def predict(self, X):
        prediction = np.dot(X, self.weights) + self.bias
        
        prediction[prediction >= 0] = self.true_label_names[1]
        prediction[prediction < 0] = self.true_label_names[0]
        
        return prediction
        


if __name__ == "__main__":
    X_train = pd.read_csv("data/train_in.csv", header=None)
    y_train = pd.read_csv("data/train_out.csv", header=None)
    X_test = pd.read_csv("data/test_in.csv", header=None)
    y_test = pd.read_csv("data/test_out.csv", header=None)

    X_train = as_numpy(X_train)
    y_train = as_numpy(y_train).flatten()

    X_test = as_numpy(X_test)
    y_test = as_numpy(y_test).flatten()

    y_label_idx, labels = pd.factorize(y_train, sort=True)

    
    label_points = {}
    all_labels = {}
    test_label_points = {}
    test_all_labels = {}
    for label_idx, label in enumerate(labels):
        label_points[label] = X_train[y_label_idx == label_idx]
        all_labels[label] = y_train[y_label_idx == label_idx]


    test_y_label_idx, test_labels = pd.factorize(y_test, sort=True)
    for label_idx, label in enumerate(test_labels):
        test_label_points[label] = X_test[test_y_label_idx == label_idx]
        test_all_labels[label] = y_test[test_y_label_idx == label_idx]



    
    keys_list = list(all_labels.keys())
    all_pairs = [(keys_list[k1], keys_list[k2]) for k1 in range(len(keys_list)) for k2 in range(k1+1,len(keys_list))]

    for (first_label, second_label) in all_pairs:
        train_images = np.concatenate((label_points[first_label], label_points[second_label]))
        train_labels = np.concatenate((all_labels[first_label], all_labels[second_label]))
        model = CoverTheoremPerceptron()
        model.fit(train_images, train_labels, (first_label, second_label))
        
        n_first_label = test_label_points[first_label].shape[0]
        n_second_label = test_label_points[second_label].shape[0]

        y_test = np.concatenate((first_label * np.ones((n_first_label,1)), second_label * np.ones((n_second_label,1))), axis=0)

        
        X_test = np.concatenate((test_label_points[first_label], test_label_points[second_label]), axis=0)
        y_test_pred = model.predict(X_test)

        print("Checking between labels: ", first_label, " and, ", second_label, ".")
        print(f"Test accuracy {accuracy_score(y_test, y_test_pred):.3%}")

        