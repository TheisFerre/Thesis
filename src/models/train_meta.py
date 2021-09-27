

from src.models.models import BaselineGATLSTM, Edgeconvmodel, GATLSTM, Encoder, Decoder, STGNNModel, BaselineGNNLSTM
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import dill
from src.data.process_dataset import Dataset
from torch_geometric.data import DataLoader
import argparse
import datetime
import logging
import os
import json
from distutils.dir_util import copy_tree
import learn2learn as l2l
import random
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def convert_to_dataloader(data, k_shots=6):
    data_list = []
    for i in range(len(data)):
        data_list.append(data[i])
    loader = DataLoader(data_list, batch_size=k_shots, shuffle=True)
    return loader


def train_model(
    train_datasets: dict,
    test_datasets: dict,
    epochs: int = 200,
    adapt_lr: float = 0.001,
    batch_task_size: int = -1,
    meta_lr: float = 0.001,
    adaptation_steps: int = 5,
    weather_features: int = 4,
    time_features: int = 43,
    gpu: bool = False
):

    model = Edgeconvmodel(
        node_in_features=1,
        weather_features=weather_features,
        time_features=time_features,
        node_out_features=10,
        gpu=gpu,
        hidden_size=46,
        dropout_p=0.2
    )

    maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=True, allow_unused=True)
    opt = optim.Adam(maml.parameters(), meta_lr)
    lossfn = torch.nn.MSELoss(reduction='mean')

    if batch_task_size == -1:
        batch_task_size = len(train_datasets.keys())

    writer = SummaryWriter()
    step_dict = {f_name: 0 for f_name in train_datasets.keys()}

    for epoch in range(epochs):
        meta_train_loss = 0.0

        # num_evals = 0
        for f_name, task in random.sample(train_datasets.items(), batch_task_size):
            learner = maml.clone()

            support_data = next(iter(task))
            query_data = next(iter(task))

            for _ in range(5):  # adaptation_steps
                support_preds = learner(support_data)
                support_loss = lossfn(support_data.y, support_preds.view(support_data.num_graphs, -1))
                learner.adapt(support_loss)
                
            query_preds = learner(query_data)
            query_loss = lossfn(query_data.y, query_preds.view(query_data.num_graphs, -1))
            writer.add_scalar(tag=f"{f_name}/query_loss", scalar_value=query_loss.item(), global_step=step_dict[f_name])
            step_dict[f_name] += 1

            meta_train_loss += query_loss

        meta_train_loss = meta_train_loss / batch_task_size

        """with torch.no_grad():
            grid_pred = edgeconv(test_batch_dataloader_grid)
            grid_loss = lossfn(test_batch_dataloader_grid.y, grid_pred.view(test_batch_dataloader_grid.num_graphs, -1))
            print(f"Loss on Grid-data: {grid_loss.item()}")

            region_pred = edgeconv(test_batch_dataloader_regions)
            region_loss = lossfn(test_batch_dataloader_regions.y, region_pred.view(test_batch_dataloader_regions.num_graphs, -1))
            print(f"Loss on Region-data: {region_loss.item()}")"""

        if epoch % 1 == 0:
            print(f"Epoch: {epoch+1}")
            print(f"Meta Train Loss: {meta_train_loss.item()}")
            print(8 * "#")
        
        writer.add_scalar(tag=f"Meta/loss", scalar_value=meta_train_loss.item(), global_step=epoch)

        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Model training argument parser")

    parser.add_argument("-d", "--data_dir", type=str, help="Directory of datasets")
    parser.add_argument("-t", "--train_size", type=float, default=0.8, help="Ratio of data to be used for training")
    parser.add_argument("-b", "--batch_task_size", type=int, default=-1, help="number of tasks to sample")
    parser.add_argument("-k", "--k_shot", type=int, default=5, help="shots to be used")
    parser.add_argument("-a", "--adaptation_steps", type=int, default=5, help="Number of adaptation steps")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("-alr", "--adapt_lr", type=float, default=0.001, help="Adaptation learning rate")
    parser.add_argument("-mlr", "--meta_lr", type=float, default=0.001, help="Meta learning rate")
    parser.add_argument("-g", "--gpu", action='store_true')

    args = parser.parse_args()
    train_dataloader_dict = {}
    test_dataloader_dict = {}

    for f in os.listdir(args.data_dir):
        abs_path = os.path.join(args.data_dir, f)
        with open(abs_path, "rb") as infile:
            data = dill.load(infile)
            train_data, test_data = Dataset.train_test_split(data, num_history=12, shuffle=True, ratio=args.train_size)

            train_data_dataloader = convert_to_dataloader(train_data, k_shots=args.k_shot)
            test_data_dataloader = convert_to_dataloader(test_data, k_shots=args.k_shot)

            f_name = f.split("/")[-1].replace(".pkl", "")

            train_dataloader_dict[f_name] = train_data_dataloader
            test_dataloader_dict[f_name] = test_data_dataloader
    
    WEATHER_FEATURES = train_data.weather_information.shape[-1]
    TIME_FEATURES = train_data.time_encoding.shape[-1]

    start_time = datetime.datetime.now()
    logger.info(f"Fitting model at time: {str(start_time)}")

    model, train_loss, test_loss = train_model(
        train_datasets=train_dataloader_dict,
        test_datasets=test_dataloader_dict,
        adaptation_steps=args.adaptation_steps,
        batch_task_size=args.batch_task_size,
        epochs=args.epochs,
        adapt_lr=args.adapt_lr,
        meta_lr=args.meta_lr,
        weather_features=WEATHER_FEATURES,
        time_features=TIME_FEATURES,
        gpu=args.gpu
    )

    """end_time = datetime.datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
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

    logger.info(f"Saving files to {cur_dir}/{args.model}_{end_time_str}")
    os.mkdir(f"{args.model}_{end_time_str}")

    args_dict = vars(args)
    with open(f"{args.model}_{end_time_str}/settings.json", "w") as outfile:
        json.dump(args_dict, outfile)

    losses_dict = {"train_loss": train_loss, "test_loss": test_loss}
    outfile = open(f"{args.model}_{end_time_str}/losses.pkl", "wb")
    dill.dump(losses_dict, outfile)
    outfile.close()

    model.to("cpu")
    torch.save(model.state_dict(), f"{args.model}_{end_time_str}/model.pth")

    logger.info("Files saved successfully")

    os.chdir(f"{args.model}_{end_time_str}")
    os.mkdir(f"logs")

    target_dir = "logs"
    source_dir = f"{os.getenv('HOME')}/.lsbatch"

    copy_tree(source_dir, target_dir)
    
    for f in os.listdir(target_dir):
        if not f.endswith("err") and not f.endswith("out"):
            os.remove(f"{target_dir}/{f}")"""
    
