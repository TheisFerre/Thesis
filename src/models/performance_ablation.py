from logging import exception
import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
from src.data.process_dataset import Dataset

import os
from src.models.finetune_meta import finetune_model
from src.models.models import Edgeconvmodel
from torch_geometric.data import DataLoader
import dill
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from src.visualization.visualize import plot_losses
import logging
plt.rcParams["figure.figsize"] = (20,5)

#DATASET_FOLDER = "/zhome/2b/7/117471/Thesis/data/processed/aglation-non_augmented"
DATASET_FOLDER = "/zhome/2b/7/117471/Thesis/data/processed/metalearning"
AUGMENTED_FOLDER = "/zhome/2b/7/117471/Thesis/ablation-study/augmented"
NON_AUGMENTED_FOLDER = "/zhome/2b/7/117471/Thesis/ablation-study/non-augmented"
WEATHER_FEATURES = 4
TIME_FEATURES = 43

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_edgeconv_model(path, trained=True):
    with open(f"{path}/settings.json", "rb") as f:
        edgeconv_params = json.load(f)

    if not trained:
        edgeconv_params_optim = {
            "optimizer": edgeconv_params["optimizer"],
            "weight_decay": edgeconv_params["weight_decay"],
            "lr": edgeconv_params["learning_rate"]
        }

    edgeconv_params.pop("data")
    edgeconv_params.pop("model")
    edgeconv_params.pop("train_size")
    edgeconv_params.pop("batch_size")
    edgeconv_params.pop("epochs")
    edgeconv_params.pop("num_history")
    edgeconv_params.pop("weight_decay")
    edgeconv_params.pop("learning_rate")
    edgeconv_params.pop("lr_factor")
    edgeconv_params.pop("lr_patience")
    edgeconv_params.pop("gpu")
    edgeconv_params.pop("optimizer")
    edgeconv_params.pop("k_neighbours")
    edgeconv_params.pop("graph_hidden_size")
    if "save_dir" in edgeconv_params:
        edgeconv_params.pop("save_dir")
    edgeconv_params["node_out_features"] = edgeconv_params.pop("node_out_feature")
    edgeconv_params["dropout_p"] = edgeconv_params.pop("dropout")

    edgeconv = Edgeconvmodel(
        node_in_features=1,
        weather_features=WEATHER_FEATURES,
        time_features=TIME_FEATURES,
        gpu=True,
        **edgeconv_params
    )

    if trained:
        edgeconv_state_dict = torch.load(f"{path}/model.pth", map_location=torch.device('cpu'))
        edgeconv.load_state_dict(edgeconv_state_dict)
        return edgeconv
    
    else:
        optimizer = getattr(torch.optim, edgeconv_params_optim.pop("optimizer"))(
            edgeconv.parameters(), 
            **edgeconv_params_optim
        )
        return edgeconv, optimizer



def eval_model(edgeconv_model, exclude_datalist, k):
    loss_dict = dict()
    for dataset_file in os.listdir(DATASET_FOLDER):
        dataset_abs_path = os.path.abspath(os.path.join(DATASET_FOLDER, dataset_file))

        continue_flag = True
        # loop over datasets that was excluded in training
        # These datasets should be used for evaluation
        for excl in exclude_datalist:
            # dataset file is one to be excluded. 
            # Therefore we want to pick it out and evaluate on it
            # This is ensured by setting flag to False, which means we wont go to next loop iteration
            # if dataset_file.startswith(excl):
            if dataset_file.startswith("capitalbikeshare"):
                continue_flag = False
                break
        if continue_flag:
            continue

        logger.info(f"Evaluating on: {dataset_abs_path}")
        open_file = open(dataset_abs_path, "rb")
        dataset = dill.load(open_file)
        open_file.close()
        train_dataset, test_dataset = Dataset.train_test_split(dataset, ratio=0.9, num_history=12, shuffle=True)
        torch.manual_seed(0)
        if k != 0:
            train_data_list = []
            for i in range(len(train_dataset)):
                train_data_list.append(train_dataset[i])

            train_loader_eval = DataLoader(train_data_list, batch_size=k, shuffle=True)
            optimizer = torch.optim.RMSprop(edgeconv_model.parameters(), lr=0.0005)

        test_data_list = []
        for i in range(len(test_dataset)):
            test_data_list.append(test_dataset[i])

        test_loader_eval = DataLoader(test_data_list, batch_size=20, shuffle=True)
        lossfn = nn.MSELoss(reduction='mean')

        edgeconv_model.eval()
        edgeconv_model.to(DEVICE)
        model_loss = 0
        num_batch_test = 0
        logger.info(edgeconv_model.gpu)
        if k != 0:
            edgeconv_model.train()
            train_batch = next(iter(train_loader_eval)).cuda()
            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                out = edgeconv_model(train_batch)
                loss = lossfn(train_batch.y, out.view(train_batch.num_graphs, -1))
                loss.backward()
                optimizer.step()
            edgeconv_model.eval()
        else:
            pass
        with torch.no_grad():
            for batch in test_loader_eval:
                batch = batch.cuda()
                query_preds_trained = edgeconv_model(batch)
                model_loss += lossfn(batch.y, query_preds_trained.view(batch.num_graphs, -1)).item()
                num_batch_test += 1
        
        model_loss = model_loss / num_batch_test
        logger.info(str(model_loss))
        loss_dict[dataset_file.split(".")[0]] = model_loss
    
    logger.info(str(loss_dict))
    return loss_dict


weather_features = 4
time_features=43
for k in [0, 1, 5, 10]:
    augmented_loss_dict = dict()
    for model_path in os.listdir(NON_AUGMENTED_FOLDER):
        edgeconv_abs_path = os.path.abspath(os.path.join(NON_AUGMENTED_FOLDER, model_path))
        if not model_path.startswith("2021"):
            continue
        with open(f"{edgeconv_abs_path}/settings.json") as settings_json:
            edgeconv_params = json.load(settings_json)
            edgeconv_params.pop("data_dir")
            edgeconv_params.pop("train_size")
            edgeconv_params.pop("batch_task_size")
            edgeconv_params.pop("k_shot")
            edgeconv_params.pop("adaptation_steps")
            edgeconv_params.pop("epochs")
            edgeconv_params.pop("adapt_lr")
            edgeconv_params.pop("meta_lr")
            edgeconv_params.pop("log_dir")
            exclude_data = edgeconv_params.pop("exclude")
            edgeconv_params.pop("gpu")
            model = Edgeconvmodel(
                node_in_features=1,
                weather_features=weather_features,
                time_features=time_features,
                gpu=True,
                **edgeconv_params
            )
            edgeconv_state_dict = torch.load(f"{edgeconv_abs_path}/model.pth")
            model.load_state_dict(edgeconv_state_dict)

            exclude_datalist = exclude_data.split(",")
        
        losses_dict = eval_model(model, exclude_datalist, k)
        augmented_loss_dict[",".join(exclude_datalist)] = losses_dict

    with open(f"{NON_AUGMENTED_FOLDER}/k_shot{str(k)}-capital-losses.pkl", "wb") as f:
        dill.dump(augmented_loss_dict, f)


for k in [0, 1, 5, 10]:
    augmented_loss_dict = dict()
    for model_path in os.listdir(AUGMENTED_FOLDER):
        edgeconv_abs_path = os.path.abspath(os.path.join(AUGMENTED_FOLDER, model_path))
        if not model_path.startswith("2021"):
            continue
        with open(f"{edgeconv_abs_path}/settings.json") as settings_json:
            edgeconv_params = json.load(settings_json)
            edgeconv_params.pop("data_dir")
            edgeconv_params.pop("train_size")
            edgeconv_params.pop("batch_task_size")
            edgeconv_params.pop("k_shot")
            edgeconv_params.pop("adaptation_steps")
            edgeconv_params.pop("epochs")
            edgeconv_params.pop("adapt_lr")
            edgeconv_params.pop("meta_lr")
            edgeconv_params.pop("log_dir")
            exclude_data = edgeconv_params.pop("exclude")
            edgeconv_params.pop("gpu")
            model = Edgeconvmodel(
                node_in_features=1,
                weather_features=weather_features,
                time_features=time_features,
                gpu=True,
                **edgeconv_params
            )
            edgeconv_state_dict = torch.load(f"{edgeconv_abs_path}/model.pth", map_location=torch.device('cpu'))
            model.load_state_dict(edgeconv_state_dict)
            model.to(DEVICE)
            model.eval()

            exclude_datalist = exclude_data.split(",")
        
        losses_dict = eval_model(model, exclude_datalist, k)
        augmented_loss_dict[",".join(exclude_datalist)] = losses_dict

    with open(f"{AUGMENTED_FOLDER}/k_shot{str(k)}-capital-losses.pkl", "wb") as f:
        dill.dump(augmented_loss_dict, f)


