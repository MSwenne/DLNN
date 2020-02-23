import pandas as pd
import numpy as np



def euclidean_distance(a, b):
    return np.sum((a - b)**2, axis=1)**0.5

class Cloud:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    @classmethod
    def from_points(cls, points):
        center = np.mean(points, axis=0)
        radius = np.max(euclidean_distance(points, center))
        return cls(center, radius)

    def is_in_cloud(self, points):
        return euclidean_distance(points, self.center) <= self.radius


if __name__ == "__main__":
    X_train = pd.read_csv("data/train_in.csv", header=None)
    y_train = pd.read_csv("data/train_out.csv", header=None)

    clouds = []
    for digit in range(10):
        is_sample_digit = y_train[0] == digit
        num_samples = np.sum(is_sample_digit)
        cloud = Cloud.from_points(X_train.loc[is_sample_digit])
        n = np.sum(cloud.is_in_cloud(X_train))
        clouds.append(cloud)
        print(f"Digit {digit}, cloud has radius {cloud.radius},"
            f" was formed from {num_samples} points but {n} can be found inside it")

    cloud_centers = np.array([cloud.center for cloud in clouds])
    cloud_center_distances = pd.DataFrame(data=[euclidean_distance(c, cloud_centers)
                                                for c in cloud_centers])
    print("\nCloud center pairwise distances:")
    print(cloud_center_distances)

    nonzero_distances = cloud_center_distances.stack() > 0
    smallest = cloud_center_distances.stack()[nonzero_distances].idxmin()
    print("Smallest distance between", smallest)
