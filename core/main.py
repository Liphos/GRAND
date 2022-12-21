"""Main file to run experiments for the Grand Project"""

import os
import argparse
from typing import Dict, Union, List, Tuple, Callable

import matplotlib.pyplot as plt
import torch_geometric as tg
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import wandb

from core.create_dataset import GrandDataset, GrandDatasetSignal
from core.model import algorithm_from_name, SimpleSignalModel
from core.utils import scaled_mse, scaled_l1


def parser_to_config():
    """Create parser"""
    parser = argparse.ArgumentParser(
                    prog='GNNGrand',
                    description='Train GNN on GRAND data')

    parser.add_argument("--algo",
                        type=str, default="GCN",
                        help="The model to use")
    parser.add_argument("--ant_ratio_train",
                        type=float, default=1,
                        help="The ratio of dense or not dense antenna to use for training, "
                             "0 means only sparse antenna and 2 means only dense antenna")
    parser.add_argument("--ant_ratio_test",
                        type=float, default=1,
                        help="The ratio of dense or not dense antenna to use for testing, "
                             "0 means only sparse antenna and 2 means only dense antenna")
    parser.add_argument("--batch_size",
                        type=int, default=20,
                        help="batch size of the training")
    parser.add_argument("--dataset",
                        type=str, default="Classic",
                        help='The dataset to train on')
    parser.add_argument("--device",
                        type=str, default='cpu',
                        help="The device to use")
    parser.add_argument("--dropout",
                        type=float, default=0,
                        help="The dropout rate")
    parser.add_argument("--d_wandb",
                        action='store_true', default=False,
                        help="Don't create a run on wandb. "
                             "It is set to true if the run is a test run")
    parser.add_argument("--embed_size",
                        type=int, default=64,
                        help="The size of the embedding")
    parser.add_argument("--epochs",
                        type=int, default=1000,
                        help="The number of epochs to train each model")
    parser.add_argument("--fig_dir_name",
                        type=str, default=None,
                        help="Use to save the figures with a different name than the model name")
    parser.add_argument("--loss_fn",
                        type=str, default="mse", choices=["mse", "scaled_mse", "scaled_l1"],
                        help="loss function to use")
    parser.add_argument("--lr",
                        type=float, default=1e-3,
                        help="The learning rate")
    parser.add_argument("--models",
                        type=int, default=10,
                        help="The number of models to train")
    parser.add_argument("--model_name",
                        type=str, default="No_name",
                        help="The name of the training model")
    parser.add_argument("--num_layers",
                        type=int, default=3,
                        help="The number of layers to use")
    parser.add_argument("--readout",
                        type=str, default="sum",
                        help="The readout function to use")
    parser.add_argument("--root",
                        type=str, default="./GrandDatasetNoDense",
                        help="dataset name")
    parser.add_argument("--seed",
                        type=int, default=128,
                        help="The seed used for the shuffling and for the model. 0 is random")
    parser.add_argument("--test",
                        action='store_true', default=False,
                        help="enable training")
    parser.add_argument("--topkratio",
                        type=float, default=0.8,
                        help="The ratio to use for the topk pooling")
    parser.add_argument("--not_drop_nodes",
                        action='store_true', default=False,
                        help="Train on the whole graph instead of dropiing nodes randomly")
    parser.add_argument("--verbose_t",
                        type=int, default=50,
                        help="The time between each test during training")
    parser.add_argument("--weight_decay",
                        type=float, default=1e-3,
                        help="The weight_decay use with Adam")


    return vars(parser.parse_args())

def create_loader(config_cfg:Dict[str, Union[str, int, float]]):

    """Create data loaders"""

    # We don't want to shuffle to keep the same order in the data
    train_loader = tg.loader.DataLoader(
                    train_dataset,
                    batch_size=config_cfg["batch_size"],
                    shuffle=not config_cfg["test"]
                )
    test_loader = tg.loader.DataLoader(
                    test_dataset,
                    batch_size=config_cfg["batch_size"],
                    shuffle=not config_cfg["test"]
                )
    return train_loader, test_loader

def create_model(config_cfg:Dict[str, Union[str, int, float]],
                 input_features:int,
                 num_classes:int=1,
                 device:torch.device=torch.device("cpu")):

    """Create model, optimizer and learning rate scheduler"""

    if config_cfg["dataset"] == "Signal":
        model = model_class(
            772,
            config_cfg["embed_size"],
            num_classes=num_classes,
            config=config_cfg
            ).to(device)
        cnn_embed = SimpleSignalModel().to(device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': cnn_embed.parameters()}, ],
            lr=config_cfg["lr"],
            weight_decay=config_cfg["weight_decay"])
    else:
        model = model_class(
            input_features,
            config_cfg["embed_size"],
            num_classes=num_classes,
            config=config_cfg
            ).to(device)
        cnn_embed = SimpleSignalModel().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config_cfg["lr"],
            weight_decay=config_cfg["weight_decay"]
            )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                            optimizer,
                                            step_size=2*config["verbose_t"],
                                            gamma=0.8
                                            )

    return model, cnn_embed, optimizer, lr_scheduler

def find_loss(loss_name:str):
    """Return loss function given a loss name"""
    if loss_name == "mse":
        loss_fn = F.mse_loss
    elif loss_name == "scaled_mse":
        loss_fn = scaled_mse
    elif loss_name == "scaled_l1":
        loss_fn = scaled_l1
    else:
        raise ValueError("loss function not recognized")
    return loss_fn

def load_models(model_dir:str, num_features:int, config_cfg:Dict, device='cpu'):
    """Load the models"""
    models = [0 for i in range(len(os.listdir(model_dir)))]
    print(os.listdir(model_dir))
    for model_path in os.listdir(model_dir):
        checkpoint = torch.load(model_dir + "/" + model_path, map_location=device)
        model = model_class(num_features, config_cfg["embed_size"], 1, config=config_cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if config_cfg["dataset"] == "Signal":
            cnn_embed = SimpleSignalModel()
            cnn_embed.load_state_dict(checkpoint["cnn_state_dict"])
            cnn_embed.eval()
        else:
            if "cnn_state_dict" in checkpoint:
                print("WARNING: the model to load has a cnn embeddeing, "
                        "you may want to use it by adding --use_signal")

        models[int(model_path[0])] = model

    return models

def compute_preds_dataset(models:List[torch.nn.Module],
                         loader:DataLoader,
                         loss_fn:Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the predicton and the loss on the dataset"""

    loss_lst = []
    pred_lst = []
    for model in models:
        loss = 0
        n_tot = 0
        energy = []
        pred = []
        with torch.no_grad():
            for data in loader:
                data = data.to(model.device)
                pred = model(data.x, data.edge_index, data.batch, data.edge_attr)
                loss += loss_fn(pred[:, 0], data.y, reduction="sum").item()
                n_tot += len(data.y)
                pred = np.concatenate((pred, pred[:, 0].numpy()))
                energy = np.concatenate((energy, data.y.numpy()))

        loss_lst.append(loss/n_tot)
        pred_lst.append(pred)

    loss = np.array(loss_lst)
    pred = np.array(pred_lst)
    print("loss: ", loss)

    return loss, pred, energy

def _compute_loss_dataset(model:torch.nn.Module,
                          loader:DataLoader,
                          loss_fn:Callable,
                          config_cfg:Dict,
                          cnn_embed=None):
    """Compute loss on a dataset"""
    loss = 0
    n_tot = 0
    for data in loader:
        if not config_cfg["not_drop_nodes"]:
            data_list = data.to_data_list()
            for graph in enumerate(data_list):
                rand_nb = torch.rand(1).item()*0.4 + 0.6
                indicies = torch.randperm(len(graph[1].x))
                indicies = indicies[:torch.round(len(graph[1].x)*rand_nb).type(torch.LongTensor)]
                data_list[graph[0]] = graph[1].subgraph(indicies)
            data = tg.data.Batch.from_data_list(data_list)
        data = data.to(model.device)
        if config_cfg["dataset"] == "Signal":
            reshape_data = data.x[:, :-4].reshape(data.x.shape[0], 768, 3)
            inputs = cnn_embed(torch.swapaxes(reshape_data, 1, -1))
            inputs = torch.cat((inputs, data.x[:, -4:]), axis=1)
            pred = model(inputs, data.edge_index, data.batch, data.edge_attr)
        else:
            pred = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss += loss_fn(pred[:, 0], data.y, reduction="sum").item()
        n_tot += len(data.y)

    return loss/n_tot

def compute_bins(pred: np.ndarray, energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the bins

    Args:
        pred (np.ndarray): the predictions of the modem
        energy (np.ndarray): the true energy

    Returns:
        pred_mean: the preduction mean on the bins
        pred_std: the prediction std on the bins
        true_mean: the true mean on the bins
    """
    bins = [0.11 + (i + 1) * (3.99-0.11) / 10 for i in range(10)]
    ind_bins = np.digitize(energy, bins)
    pred_mean, pred_std, true_mean = [], [], []
    for e_bin in enumerate(bins):
        indicies = np.where(ind_bins == e_bin[0])[0]
        pred_mean.append(np.mean(pred[:, indicies] - energy[indicies]))
        pred_std.append(np.sqrt(np.mean(np.std(pred[:, indicies], axis=0)**2)))
        true_mean.append(np.mean(energy[indicies]))

    pred_mean = np.array(pred_mean)
    pred_std = np.array(pred_std)
    true_mean = np.array(true_mean)

    return pred_mean, pred_std, true_mean

def plot_training_results(train_perf:List[float],
                          test_perf:List[float],
                          config_cfg:Dict,
                          model_id:int,
                          save_path:str) -> None:
    """Plot the results of the training with matplotlib

    Args:
        train_perf (List[float]): training loss
        test_perf (List[float]): testing loss
        config_cfg (Dict): Dictionary containing all the parameters
        model_id (int): the model id
        save_path (str): the path to save the plots
    """
    if config_cfg["epochs"] >config_cfg["verbose_t"]:

        print("Model: " + str(model_id) + " minimum train loss reached: ",
            round(np.min(train_perf), 4),
            "indicie: ", np.argmin(train_perf)
            )

        print("Model: " + str(model_id) + " minimum test  loss reached: ",
            round(np.min(test_perf), 4),
            "indicie: ", np.argmin(test_perf)
            )
        plt.clf()
        nb_updates = int(np.floor((config_cfg["epochs"]-1)/config_cfg["verbose_t"]))
        plt.plot([config_cfg["verbose_t"] * k for k in range(nb_updates)],
                    train_perf,
                    label="train loss"
                    )
        plt.plot([config_cfg["verbose_t"] * k for k in range(nb_updates)],
                    test_perf,
                    label="test loss"
                    )
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.savefig(save_path + "_training")

def plot_individual_preformance(pred: np.ndarray,
                                energy: np.ndarray,
                                mode:str="train",
                                fig_dir:str=None) -> None:
    """Plot the performance of each models

    Args:
        train_pred (np.ndarray): the predictions of the models
        train_energy (np.ndarray): the true labels
        mode (str, optional): train or test. Defaults to "train".
        fig_dir (str, optional): the path to save the images. Defaults to None.
    """
    for i in enumerate(pred):
        plt.clf()
        plt.scatter(energy, i[1])
        plt.plot(energy, energy, "k")
        plt.title("Results on the {mode}ing set")
        plt.xlabel("ground truth energy (eeV)")
        plt.ylabel("predicted energy (eeV)")
        plt.xlim(0, 4.1)
        if fig_dir is not None:
            plt.savefig(fig_dir + "/" + str(i[0]) + f"_{mode}")

def plot_bins_results(train_values: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      test_values: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      fig_dir:str=None) -> None:
    """Plot the results on the bins

    Args:
        train_values (Tuple[np.ndarray, np.ndarray, np.ndarray]): the train values
        test_values (Tuple[np.ndarray, np.ndarray, np.ndarray]): the test values
    """
    pred_train_mean, pred_train_std, true_train_mean = train_values
    pred_test_mean, pred_test_std, true_test_mean = test_values

    #Plot the results on the bins
    plt.clf()
    plt.errorbar(true_train_mean, pred_train_mean, yerr=pred_train_std, fmt="o", label="Train")
    plt.plot(true_train_mean, [0 for _ in range(len(true_train_mean))], "k")
    plt.errorbar(true_test_mean, pred_test_mean, yerr=pred_test_std, fmt="o", label="Val")
    plt.plot(true_test_mean, [0 for _ in range(len(true_test_mean))], "k")
    plt.title("Results")
    plt.xlabel("ground truth energy (EeV)")
    plt.ylabel("$Residue E_{pr} - E_{th} (EeV)$")
    plt.xlim(0, 4.1)
    plt.legend()
    if fig_dir is not None:
        plt.savefig(fig_dir + "/" + "all")

    #Plot the residue
    plt.figure()
    plt.errorbar(true_train_mean, pred_train_mean/true_train_mean,
                    yerr=pred_train_std/true_train_mean, fmt="o", label="Train")
    plt.plot(true_train_mean, [0 for _ in range(len(true_train_mean))], "k")
    plt.errorbar(true_test_mean, pred_test_mean/true_test_mean,
                    yerr=pred_test_std/true_test_mean, fmt="o", label="Val")
    plt.plot(true_test_mean, [0 for _ in range(len(true_test_mean))], "k")
    plt.title("Results")
    plt.xlabel("ground truth energy (EeV)")
    plt.ylabel(r"$Residue \Delta_{E}/E_{th} $")
    plt.xlim(0, 4.1)
    plt.legend()
    if fig_dir is not None:
        plt.savefig(fig_dir + "/" + "all" + "delta")

    plt.show()
    #Save all the results
    if fig_dir is not None:
        np.save(fig_dir + "/" + "train_pred", pred_train_mean)
        np.save(fig_dir + "/" + "train_std", pred_train_std)
        np.save(fig_dir + "/" + "train_true", true_train_mean)

        np.save(fig_dir + "/" + "test_pred", pred_test_mean)
        np.save(fig_dir + "/" + "test_std", pred_test_std)
        np.save(fig_dir + "/" + "test_true", true_test_mean)

def one_step_loss(model: torch.nn.Module,
                  data:tg.data.Batch,
                  loss_fn:Callable,
                  config_cfg:Dict,
                  cnn_embed: torch.nn.Module=None
                  ):
    """Train the model for one step"""
    if config_cfg["dataset"] == "Signal":
        reshape_data = data.x[:, :-4].reshape(data.x.shape[0], 768, 3)
        embed = cnn_embed(torch.swapaxes(reshape_data, 1, -1))
        inputs = torch.cat((embed, data.x[:, -4:]), axis=1)
        pred = model(inputs, data.edge_index, data.batch, data.edge_attr)
    else:
        pred = model(data.x, data.edge_index, data.batch, data.edge_attr)

    loss = loss_fn(pred[:, 0], data.y)
    return loss

def save_model(model:torch.nn.Module,
               config_cfg:Dict,
               optimizer:torch.optim,
               save_path:str,
               cnn_embed=None) -> None:
    """Save the model"""
    data_to_save = {'epochs': config_cfg["epochs"],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }

    if config_cfg["dataset"] == "Signal":
        data_to_save['cnn_state_dict'] = cnn_embed.state_dict()

    torch.save(data_to_save, save_path + ".pt")


def train_model(model_id:int,
                loss_fn:Callable,
                config_cfg:Dict,
                model_dir:str,
                device:torch.device=torch.device("cpu")):
    """Train the model"""
    print(f"Model: {model_id}")
    lst_train_perf = []
    lst_test_perf = []
    #Create loaders
    train_loader, test_loader = create_loader(config_cfg)
    # Create the model with given dimensions

    ###TODO:Change the num classes property of the class
    model, cnn_embed, optimizer, lr_scheduler = create_model(config_cfg,
                                                            train_loader.num_features,
                                                            num_classes=1,
                                                            device=device)
    model.train()
    for epoch in range(config_cfg["epochs"]):
        for data in train_loader:
            if not config_cfg["not_drop_nodes"]:
                data_list = data.to_data_list()
                for graph in enumerate(data_list):
                    rand_nb = np.random.random_sample()*0.4 + 0.6
                    indicies = torch.randperm(len(graph[1].x))
                    indicies = indicies[:int(np.round(len(graph[1].x)*rand_nb))]
                    data_list[graph[0]] = graph[1].subgraph(indicies)
                data = tg.data.Batch.from_data_list(data_list)

            data = data.to(device)
            optimizer.zero_grad()
            loss = one_step_loss(model, data, loss_fn, config_cfg, cnn_embed=cnn_embed)
            wandb.log({"loss":loss})
            ### We save the model if there is nan in the loss
            if torch.any(loss == 0) or torch.any(torch.isnan(loss)):
                save_model(model, config_cfg, optimizer,
                           save_path=f"{model_dir}/{model_id}_debug_nan", cnn_embed=cnn_embed)

                print("Error loss nul or invalid")
                raise ValueError("wrong value for the loss")
            loss.backward()
            optimizer.step()

        # We want to update the learning rate to change every verbose_t epochs
        lr_scheduler.step()

        if epoch % config_cfg["verbose_t"] == 0 and epoch > 0:
            model.eval()
            if config_cfg["dataset"] == "Signal":
                cnn_embed.eval()

            train_loss = _compute_loss_dataset(model,
                                               train_loader,
                                               loss_fn,
                                               config,
                                               cnn_embed=cnn_embed)

            lst_train_perf.append(train_loss)

            test_loss = _compute_loss_dataset(model,
                                              test_loader,
                                              loss_fn,
                                              config,
                                              cnn_embed=cnn_embed)

            lst_test_perf.append(test_loss)
            print(f'epoch: {epoch} '
                    f'lr: {lr_scheduler.get_last_lr()[0]:.8f} '
                    f'loss_train: {train_loss:.4f} '
                    f'loss_test: {test_loss:.4f}'
                    )
            wandb.log({'epoch': epoch,
                        'lr': lr_scheduler.get_last_lr()[0],
                        'loss_train': train_loss,
                        'loss_test': test_loss,
                        })

            if config_cfg["dataset"] == "Signal":
                cnn_embed.train()

            model.train()

    #Once finished we save the model
    save_model(model, config_cfg, optimizer,
               save_path=f"{model_dir}/{model_id}", cnn_embed=cnn_embed)

    return lst_train_perf, lst_test_perf

if __name__ == '__main__':
    config = parser_to_config()

    if not config["d_wandb"] or not config["test"]:
        wandb.init(project="GNN", config=config)
    # matplotlib.use(config["matplotlib_gui"])

    if config["seed"] != 0:
        torch.manual_seed(config["seed"])
        np.random.seed(seed=config["seed"])

    model_dir_path = "./core/Models/" + config["model_name"]
    model_class = algorithm_from_name(config["algo"])

    #Create folders if they don't exist
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)

    if config["fig_dir_name"] is None:
        fig_name = config["model_name"]
    else:
        fig_name = config["fig_dir_name"]

    fig_dir_path = "./Figures/" + fig_name
    if not os.path.exists(fig_dir_path):
        os.mkdir(fig_dir_path)

    if config["dataset"] == "Signal":
        print("Warning depricated")
        dataset = GrandDatasetSignal().shuffle()
    elif config['dataset'] == "Classic":
        dataset = GrandDataset(root=config["root"])
        train_dataset = dataset.train_datasets[int(config["ant_ratio_train"]*5)]
        test_dataset = dataset.test_datasets[int(config["ant_ratio_test"]*5)]
    else:
        raise ValueError("This dataset don't exist")

    loss_fn = find_loss(config["loss_fn"])

    def train():
        """Training function"""
        device = torch.device(config["device"])
        print("device: ", device)

        print("config: ", config)
        print("loss_function: ", loss_fn.__name__)

        lst_model_train_perf = []
        lst_model_test_perf = []
        for model_id in range(config["models"]):
            lst_train_perf, lst_test_perf = train_model(model_id,
                                                        loss_fn=loss_fn,
                                                        config_cfg=config,
                                                        model_dir=model_dir_path,
                                                        device=device
                                                        )
            lst_model_train_perf.append(lst_train_perf)
            lst_model_test_perf.append(lst_test_perf)
            plot_training_results(lst_train_perf, lst_test_perf, config, model_id,
                                  fig_dir_path + "/" + str(model_id))



    def test():
        """Test function"""
        # We don't want to shuffle to keep the same order in the data
        print(f"config: {config}")
        device = 'cpu'
        train_loader, test_loader = create_loader(config)

        models = load_models(model_dir_path, train_dataset.num_features, config, device=device)

        _, train_pred, train_energy = compute_preds_dataset(models, train_loader, loss_fn=loss_fn)
        plot_individual_preformance(train_pred, train_energy, mode="train", fig_dir=fig_dir_path)

        # The same thing for test data #

        _, test_pred, test_energy = compute_preds_dataset(models, test_loader, loss_fn=loss_fn)
        plot_individual_preformance(test_pred, test_energy, mode="test", fig_dir=fig_dir_path)

        # ## We computes the bins for all the models ## #
        train_values = compute_bins(train_pred, train_energy)
        test_values = compute_bins(test_pred, test_energy)

        plot_bins_results(train_values, test_values, fig_dir=fig_dir_path)


    if config["test"]:
        test()
    else:
        train()
        test()

    print("Finished !")
