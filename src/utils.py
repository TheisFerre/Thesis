import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


def compute_mse(preds, targets):
    
    mse = np.mean((preds.reshape(-1) - targets.reshape(-1))**2)
        
    return mse


def compute_mae(targets, preds):

    mae = mean_absolute_error(targets.reshape(-1), preds.reshape(-1))

    return mae


def compute_mape(targets, preds):

    mape = mean_absolute_percentage_error(targets.reshape(-1), preds.reshape(-1))

    return mape

