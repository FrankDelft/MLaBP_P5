import sklearn.metrics as skm
from numpy import ndarray

def print_metrics(dataset: str, y_true, y_pred):
    # takes in the dataset name, and the true vs predicted labels to calculate and print the metrics
    y_true = y_true.reshape((y_true.shape[0], -1))
    y_pred = y_pred.reshape((y_pred.shape[0], -1))
    mae = skm.mean_absolute_error(y_true, y_pred)
    mse = skm.mean_squared_error(y_true, y_pred)
    print(f"Metrics for {dataset}:\nR2 score:  {r2}\nMAE score: {mae}\nMSE score: {mse}")
    return (mae, mse)