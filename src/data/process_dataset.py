import pandas as pd
from typing import List, Union
import numpy as np
from tqdm import tqdm
import scipy
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from torch_geometric.utils import from_scipy_sparse_matrix
import torch


def load_csv_dataset(
    path: str,
    time_column: str,
    location_columns: List[str] = None,
    station_column: str = None,
    time_intervals: str = "4h",
) -> pd.DataFrame:
    """
    Load a dataset with rides.

    Args:
        path (str): path to csv file
        time_columns (List[str]): list of time columns
        location_columns (List[str]): list of longitide/lattitude columns

    Returns:
        pd.DataFrame: Parsed DataFrame
    """

    df = pd.read_csv(
        path,
        parse_dates=[time_column],
        infer_datetime_format=True,
    )

    df[time_column] = df[time_column].dt.floor(time_intervals)

    cols_to_select = [time_column]

    if location_columns is not None:
        cols_to_select += location_columns

    if station_column is not None:
        cols_to_select.append(station_column)

    return df[cols_to_select].dropna().sort_values(by=time_column)


def create_grid(df: pd.DataFrame, lng_col: str, lat_col: str, splits: int = 10) -> pd.DataFrame:
    """
    Splits a pd.DataFrame that defines different rides into a grid
    Each area in the grid defines a node in the graph

    Args:
        splits (int, optional): Number of regions created. Defaults to 10.

    Returns:
        pd.DataFrame: Contains "grid_start" and "grid_end"
    """
    min_lng = df[lng_col].min()
    max_lng = df[lng_col].max()
    lng_intervals = np.linspace(min_lng, max_lng, splits + 1)

    bins_lng_col = pd.cut(df[lng_col], lng_intervals, labels=list(range(splits)), include_lowest=True)
    df[lng_col + "_binned"] = bins_lng_col

    min_lat = df[lat_col].min()
    max_lat = df[lat_col].max()
    lat_intervals = np.linspace(min_lat, max_lat, splits + 1)

    bins_lat_col = pd.cut(df[lat_col], lat_intervals, labels=list(range(splits)), include_lowest=True)
    df[lat_col + "_binned"] = bins_lat_col

    return df


def create_grid_ids(df: pd.DataFrame, longitude_col: str, lattitude_col: str) -> List[str]:
    """
    Creates grid ids, which represent nodes in our city
    Every id is indicated by <longitude int.><lattitude int.>

    Args:
        longitude_col (str): Binned longitude column
        lattitude_col (str): Binned lattitude column

    Returns:
        List[str]: list of grid ids
    """
    grid_id = []
    for lng, lat in zip(df[longitude_col], df[lattitude_col]):
        grid_id.append(str(lng) + str(lat))

    return grid_id


def correlation_adjacency_matrix(
    rides_df: pd.DataFrame, region_ordering: List[str], id_col: str, time_col: str, threshold: float = 0.25
) -> pd.DataFrame:
    """
    Creates adjacency matrix, where the correlation of historical rides
    is used as the weights between regions/grid spaces.

    Args:
        rides_df ([type]): [description]
    """

    correlation_graph = np.zeros((len(region_ordering), len(region_ordering)))
    grouped_df = rides_df.groupby([time_col, id_col])

    for i, node_base in tqdm(enumerate(region_ordering), total=len(region_ordering)):
        for j, node_compare in enumerate(region_ordering):
            if i > j or i == j:
                continue
            node1_val_list = []
            node2_val_list = []
            for time in np.sort(rides_df[time_col].unique()):
                try:
                    node1_val_list.append(len(grouped_df.get_group((time, node_base))))
                except KeyError:
                    node1_val_list.append(0)

                try:
                    node2_val_list.append(len(grouped_df.get_group((time, node_compare))))
                except KeyError:
                    node2_val_list.append(0)

            corr_coef = np.abs(np.corrcoef(node1_val_list, node2_val_list)[0, 1])
            if corr_coef > threshold:
                # in paper it is set to 1, whenever above threshold
                correlation_graph[i, j] = 1  # corr_coef
                correlation_graph[j, i] = 1  # corr_coef

    return correlation_graph


def features_and_targets(df: pd.DataFrame, region_ordering: List[str], id_col: str, time_col: str):

    grouped_df = df.groupby([time_col, id_col])

    node_inflows = np.zeros((len(df[time_col].unique()), len(region_ordering), 1))
    targets = np.zeros((len(df[time_col].unique()) - 1, len(region_ordering)))

    for t, starttime in tqdm(enumerate(df[time_col].unique()), total=len(df[time_col].unique())):
        for i, node in enumerate(region_ordering):

            query = (starttime, node)
            try:
                group = grouped_df.get_group(query)
                node_inflows[t, i] = len(group)

            except KeyError:
                node_inflows[t, i] = 0

            # current solution:
            # The target to predict, is the number of inflows at next timestep.
            if t > 0:
                targets[t - 1, i] = node_inflows[t, i]
            else:
                targets[t - 1, i] = node_inflows[t, i]

    X = node_inflows[:-1, :, :]

    return X, targets


class Dataset:
    def __init__(self, adjacency_matrix: np.array, targets: np.array, X: np.array):
        self.adjacency_matrix = adjacency_matrix
        self.scipy_graph = scipy.sparse.lil_matrix(adjacency_matrix)
        self.targets = targets
        self.X = X
        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(self.scipy_graph)
        self.edge_weight = self.edge_weight.type(torch.FloatTensor)

    def create_temporal_dataset(self):
        dataset = StaticGraphTemporalSignal(
            edge_index=self.edge_index, edge_weight=self.edge_weight, features=self.X, targets=self.targets
        )
        return dataset

    @staticmethod
    def train_test_split(dataset: Union[StaticGraphTemporalSignal, "Dataset"], ratio: float = 0.8):
        if isinstance(dataset, StaticGraphTemporalSignal):
            train, test = temporal_signal_split(dataset, train_ratio=ratio)
        elif isinstance(dataset, Dataset):
            train, test = temporal_signal_split(dataset.create_temporal_dataset(), train_ratio=ratio)
        else:
            print("input type is not correct...")

        return train, test
