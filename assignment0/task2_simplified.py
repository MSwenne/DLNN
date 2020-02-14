import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)


# Implement required formulae.
def corners(n):
    return 2**n

def diag(n):
    # Diagonal from [0, ..., 0] to [1, ..., 1]
    # with length sqrt(1^2 + ... + 1^2).
    return np.sqrt(n)

def volume_ball(n, radius):
    # Recursive formula from
    # https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Recursions
    if n == 1: return 2 * radius
    if n == 2: return np.pi * radius**2
    return 2*np.pi/n * radius**2 * volume_ball(n-2, radius)

def volume_skin(n, alpha):
    return 1 - (1 - 2*alpha)**n

def pairwise_distances(n, k):
    points = np.random.uniform(0, 1, (k, n))
    return pdist(points).flatten()


# Generate required statistics.
data = [
    (n, corners(n), diag(n), volume_ball(n, 0.5), volume_skin(n, 0.01))
    for n in range(1, 1+100)
]
df = pd.DataFrame(data=data, columns=["n", "Corners", "Diag", "VolumeB", "VolumeS"]).set_index("n")
with open("dimensional_stats.md", "w") as f:
    f.write(df.to_markdown())


# Generate required plot.
fig, ax = plt.subplots()
for n in range(1, 1+10):
    distances = pairwise_distances(2**n, 1000)
    sns.distplot(distances, bins=100, label=str(2**n), ax=ax)
ax.set_xlabel("Distance")
ax.set_ylabel("Kernel density estimate")
plt.tight_layout()
plt.legend()
plt.savefig("pairwise_dist.pdf")
plt.show()
