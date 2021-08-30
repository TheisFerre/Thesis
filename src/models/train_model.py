from src.models.models import Edgeconvmodel, GATLSTM, Encoder, Decoder, STGNNModel
import torch
import numpy as np
import dill
from src.data.process_dataset import Dataset
from torch_geometric.data import DataLoader
import argparse
import datetime
import logging
import os
import json

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(
    dataset: Dataset,
    num_history: int = 8,
    train_size: float = 0.8,
    batch_size: int = 32,
    epochs: int = 200,
    model: str = "edgeconv",
):

    train_dataset, test_dataset = Dataset.train_test_split(dataset, num_history=8, ratio=train_size)

    train_data_list = []
    for i in range(len(train_dataset)):
        train_data_list.append(train_dataset[i])
    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)

    test_data_list = []
    for i in range(len(test_dataset)):
        test_data_list.append(test_dataset[i])
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=True)

    weather_features = train_dataset.weather_information.shape[-1]
    time_features = train_dataset.time_encoding.shape[-1]

    if model == "edgeconv":
        model = Edgeconvmodel(
            node_in_features=1,
            weather_features=weather_features,
            time_features=time_features,
            node_out_features=12
        )
    elif model == "seq2seq-gnn":
        num_nodes = dataset.num_nodes
        encoder = Encoder(
            node_in_features=1,
            num_nodes=num_nodes,
            node_out_features=12,
            time_features=time_features,
            weather_features=weather_features,
        )
        decoder = Decoder(
            node_out_features=12,
            num_nodes=num_nodes
        )
        model = STGNNModel(encoder, decoder)
    elif model == "gatlstm":
        model = GATLSTM(
            node_in_features=1,
            weather_features=weather_features,
            time_features=time_features,
            node_out_features=12
        )
    else:
        assert False, "Please provide a correct model name!"

    criterion = torch.nn.MSELoss()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    test_losses = []

    for EPOCH in range(epochs):
        model.eval()
        test_loss = 0
        num_batch_test = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.to(DEVICE))
                test_loss += criterion(batch.y, out.view(batch.num_graphs, -1)).item()
                num_batch_test += 1

        model.train()
        train_loss = 0
        num_batch_train = 0
        for batch in train_loader:
            out = model(batch.to(DEVICE))

            loss = criterion(batch.y, out.view(batch.num_graphs, -1))
            train_loss += loss.item()
            num_batch_train += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = train_loss / (num_batch_train)
        train_losses.append(np.sqrt(train_loss))

        test_loss = test_loss / (num_batch_test)
        test_losses.append(np.sqrt(test_loss))

        if EPOCH % 1 == 0:

            print(f"Epoch number {EPOCH+1}")
            print(f"Epoch avg RMSE loss (TRAIN): {train_losses[-1]}")
            print(f"Epoch avg RMSE loss (TEST): {test_losses[-1]}")
            print("-" * 10)

    return model, train_losses, test_losses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Model training argument parser")
    parser.add_argument("-d", "--data", type=str, help="path to processed data")
    parser.add_argument("-m", "--model", type=str, default="edgeconv", help="Use either [edgeconv, gatlstm, seq2seq-gnn")
    parser.add_argument("-n", "--num_history", type=int, default=8, help="number of history steps for predicting")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, help="Ratio of data to be used for training")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batchsize to be used")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")

    args = parser.parse_args()
    open_file = open(args.data, "rb")
    dataset = dill.load(open_file)

    start_time = datetime.datetime.now()
    logger.info(f"Fitting model at time: {str(start_time)}")
    model, train_loss, test_loss = train_model(
        dataset=dataset,
        num_history=args.num_history,
        train_size=args.train_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model=args.model
    )
    end_time = datetime.datetime.now()
    td = end_time - start_time
    minutes = round(td.total_seconds() / 60, 2)
    totsec = td.total_seconds()
    h = int(totsec // 3600)
    m = int((totsec % 3600) // 60)
    sec = int((totsec % 3600) % 60)
    logger.info(f"Total training time: {h}:{m}:{sec}")
    logger.info(f"Average Epoch time: {round(minutes/args.epochs, 2)} minutes")
    cur_dir = os.getcwd()
    while True:
        split_dir = cur_dir.split("/")
        if "Thesis" not in split_dir:
            break
        else:
            if split_dir[-1] == "Thesis":
                break
            else:
                os.chdir("..")
                cur_dir = os.getcwd()
    os.chdir("models")
    cur_dir = os.getcwd()

    logger.info(f"Saving files to {cur_dir}/run_{str(end_time)}")
    os.mkdir(f"run_{str(end_time)}")

    args_dict = vars(args)
    with open(f"run_{str(end_time)}/settings.json", "w") as outfile:
        json.dump(args_dict, outfile)

    losses_dict = {"train_loss": train_loss, "test_loss": test_loss}
    outfile = open(f"run_{str(end_time)}/losses.pkl", "wb")
    dill.dump(losses_dict, outfile)
    outfile.close()

    model.to("cpu")
    torch.save(model.state_dict(), f"run_{str(end_time)}/model.pth")

    logger.info("Files saved successfully")
