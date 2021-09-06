import numpy as np

def compute_mse(preds, targets):
    
    mse = np.mean((preds.reshape(-1) - targets.reshape(-1))**2)
        
    return mse
