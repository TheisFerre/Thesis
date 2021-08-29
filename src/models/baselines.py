import numpy as np
from typing import Union
import torch
from src.models.models import CustomTemporalSignal
from sklearn.preprocessing import StandardScaler


def historical_average(
    train_data: Union[np.array, torch.Tensor, CustomTemporalSignal],
    test_data: Union[np.array, torch.Tensor, CustomTemporalSignal],
    scaler: StandardScaler = None,
):

    assert type(train_data) == type(test_data), "Have to be type!"

    if train_data.__class__.__name__ == "CustomTemporalSignal":
        if scaler is not None:
            train_targets = scaler.inverse_transform(train_data.targets)
            test_targets = scaler.inverse_transform(test_data.targets)
            HA = train_targets.mean(0)
            HA = np.expand_dims(HA, 0)
            HA = np.repeat(HA, test_data.targets.shape[0], 0)
            MSE = ((HA - test_targets)**2).mean()
        else:
            train_targets = train_data.targets
            test_targets = test_data.targets
            HA = train_targets.mean(0)
            HA = HA.repeat(test_data.targets.shape[0], 1)
            MSE = (HA - test_targets).pow(2).mean()
    else:
        if scaler is not None:
            train_targets = scaler.inverse_transform(train_data)
            test_targets = scaler.inverse_transform(test_data)
        else:
            train_targets = train_data
            test_targets = test_data
        MSE = ((train_targets - test_targets) ** 2).mean()

    return float(MSE)
