import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

def as_numpy(obj):
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_numpy()
    if isinstance(obj, list):
        return np.array(obj)
    return obj

def confusion_dataframe(y_true, y_pred, labels):
    return pd.DataFrame(
        data=confusion_matrix(y_true, y_pred),
        index=labels,
        columns=labels)
