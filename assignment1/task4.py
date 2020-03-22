import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sn

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
        # Initializing weights
        self.weights = np.random.normal(size=(self.n_in, 1))
        self.bias = np.random.normal(size=(1,1))
        self.iterations = 0
        has_mispredictions = True
        while has_mispredictions:
            self.iterations +=1
            # Calculating the training prediction
            prediction = np.dot(X,self.weights)
            # Assigning 1 and -1 based on the values.
            prediction[prediction < 0] = -1
            prediction[prediction >= 0] = 1
            mispredicted = prediction.flatten() != y
            # Checking whether any misprediction exists
            has_mispredictions = np.any(mispredicted)
            if has_mispredictions:
                tmp = np.argwhere(mispredicted).flatten()
                random_index_update = np.random.choice(tmp)
                # Updating weights by the perceptron update rule.
                self.weights += self.lr * np.dot(y[random_index_update], X[random_index_update][:,np.newaxis]) 
        return self.iterations
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
    data = []

    keys_list = list(range(10))
    all_pairs = [(keys_list[k1], keys_list[k2]) for k1 in range(len(keys_list)) for k2 in range(k1+1,len(keys_list))]
        

    # ***************** Pair-wise comparison ***************
    for (first_label, second_label) in all_pairs:
        # Initialize model to fit later
        model = CoverTheoremPerceptron()

        # Extract all images with their pair-wise comparison of each labels
        is_c1_train = y_train[0] == first_label
        is_c2_train = y_train[0] == second_label
        is_c1_test = y_test[0] == first_label
        is_c2_test = y_test[0] == second_label

        # Extract all training images with their respective labels
        c1_train_points = as_numpy(X_train.loc[is_c1_train])
        c2_train_points = as_numpy(X_train.loc[is_c2_train])
        c1_train_labels = np.ones((c1_train_points.shape[0],1)) * first_label
        c2_train_labels = np.ones((c2_train_points.shape[0],1)) * second_label

        # Building train with numpy array by contatenating arrays extracted above.
        train_images = np.concatenate((c1_train_points, c2_train_points))
        train_labels = np.concatenate((c1_train_labels, c2_train_labels))

        # Save number of itration of further plotting the data.
        n_iter = model.fit(train_images, train_labels, (first_label, second_label))
        data.append((first_label, second_label, n_iter))

        # Extract all test images with their respective labels
        c1_test_points = as_numpy(X_test.loc[is_c1_test])
        c2_test_points = as_numpy(X_test.loc[is_c2_test])
        c1_test_labels = np.ones((c1_test_points.shape[0],1)) * first_label
        c2_test_labels = np.ones((c2_test_points.shape[0],1)) * second_label

        # Building test numpy array by contatenating arrays extracted above.
        y_test_labels = np.concatenate((c1_test_labels, c2_test_labels), axis=0)
        X_test_points = np.concatenate((c1_test_points, c2_test_points), axis=0)

        # Model prediction over test sample points
        y_test_pred = model.predict(X_test_points)
        print("Checking between labels: ", first_label, " and, ", second_label)
        print(f"Test accuracy {accuracy_score(y_test_labels, y_test_pred):.3%}")

    # ***************** One against all comparison ***************
    data = []
    for c1 in range(10):
        # Extract all images of one label, and all the others that are not that one.
        is_c1 = y_train[0] == c1
        is_c2 = y_train[0] != c1

        # Extract all test images with their respective labels
        c1_train_points = as_numpy(X_train.loc[is_c1])
        c2_train_points = as_numpy(X_train.loc[is_c2])
        c1_train_labels = np.ones((c1_train_points.shape[0],1)) * c1
        c2_train_labels = np.zeros((c2_train_points.shape[0],1))

        # Initializing the train numpy array
        train_images = np.concatenate((c1_train_points, c2_train_points))
        train_labels = np.concatenate((c1_train_labels, c2_train_labels))
        
        model = CoverTheoremPerceptron()
        # Fitting the model for and saving iteration for further plotting.
        n_iter = model.fit(train_images, train_labels, (c1, 0))
        data.append((c1, n_iter))

    # ****** Ploting with SeaBorn *********
    to_df = np.zeros((1, 10))
    for (c1, inter) in data:
       to_df[0][c1] =  inter
    df = pd.DataFrame(data=to_df,index=[''])
    plt.figure(figsize = (10, 2))
    chart = sn.heatmap(df.round(0).astype(int), annot=True, cmap="Blues", fmt='g')
    chart.set_ylabel('Iteration',rotation=90)
    chart.set_yticklabels(chart.get_yticklabels(), rotation=90)
    plt.show()