"""Main file to run experiments for the Grand Project"""

import os
import argparse
from typing import Dict, Union

import matplotlib.pyplot as plt
import torch_geometric as tg
import torch
import torch.nn.functional as F

import numpy as np
import wandb

from core.create_dataset import GrandDataset, GrandDatasetSignal, GrandDatasetAllSize
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
    parser.add_argument("--keep_best_models",
                        action="store_true", default=False,
                        help="Keep the 5 best models for the testing")
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

def create_loader(config_dict:Dict[str, Union[str, int, float]]):

    """Create data loaders"""

    # We don't want to shuffle to keep the same order in the data
    train_loader = tg.loader.DataLoader(
                    train_dataset,
                    batch_size=config_dict["batch_size"],
                    shuffle=not config_dict["test"]
                )
    test_loader = tg.loader.DataLoader(
                    test_dataset,
                    batch_size=config_dict["batch_size"],
                    shuffle=not config_dict["test"]
                )
    return train_loader, test_loader

def create_model(config_dict:Dict[str, Union[str, int, float]],
                 input_features:int,
                 num_classes:int=1,
                 device:torch.device=torch.device("cpu")):

    """Create model, optimizer and learning rate scheduler"""

    if config_dict["dataset"] == "Signal":
        model = model_class(
            772,
            config_dict["embed_size"],
            num_classes=num_classes,
            config=config_dict
            ).to(device)
        cnn_embed = SimpleSignalModel().to(device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': cnn_embed.parameters()}, ],
            lr=config_dict["lr"],
            weight_decay=config_dict["weight_decay"])
    else:
        model = model_class(
            input_features,
            config_dict["embed_size"],
            num_classes=num_classes,
            config=config_dict
            ).to(device)
        cnn_embed = SimpleSignalModel().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config_dict["lr"],
            weight_decay=config_dict["weight_decay"]
            )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                            optimizer,
                                            step_size=2*config["verbose_t"],
                                            gamma=0.8
                                            )

    return model, cnn_embed, optimizer, lr_scheduler




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
    elif config['dataset'] == "AllSize":
        dataset = GrandDatasetAllSize()
        train_dataset, test_dataset = dataset.train_dataset, dataset.test_dataset
    elif config['dataset'] == "Classic":
        dataset = GrandDataset(root=config["root"])
        train_dataset = dataset.train_datasets[int(config["ant_ratio_train"]*5)]
        test_dataset = dataset.test_datasets[int(config["ant_ratio_test"]*5)]
    else:
        raise ValueError("This dataset don't exist")

    if config["loss_fn"] == "mse":
        loss_fn = F.mse_loss
    elif config["loss_fn"] == "scaled_mse":
        loss_fn = scaled_mse
    elif config["loss_fn"] == "scaled_l1":
        loss_fn = scaled_l1
    else:
        raise ValueError("loss function not recognized")

    def train():
        """Training function"""
        device = torch.device(config["device"])
        print("device: ", device)

        train_loader, test_loader = create_loader(config)
        print('The number of labels  is ' + 'len(batch.y)/config["batch_size"]')
        print("config: ", config)
        print("loss_function: ", loss_fn.__name__)

        lst_model_train_perf = []
        lst_model_test_perf = []
        for i in range(config["models"]):
            print(f"Model: {i}")
            lst_train_perf = []
            lst_test_perf = []
            # Create the model with given dimensions

            model, cnn_embed, optimizer, lr_scheduler = create_model(config,
                                                                     train_dataset.num_features,
                                                                     1,
                                                                     device=device)

            model.train()
            for epoch in range(config["epochs"]):
                for data in train_loader:
                    if not config["not_drop_nodes"]:
                        data_list = data.to_data_list()
                        for graph in enumerate(data_list):
                            rand_nb = np.random.random_sample()*0.4 + 0.6
                            indicies = torch.randperm(len(graph[1].x))[:int(np.round(len(graph[1].x)*rand_nb))]
                            data_list[graph[0]] = graph[1].subgraph(indicies)
                        data = tg.data.Batch.from_data_list(data_list)

                    data = data.to(device)
                    optimizer.zero_grad()
                    if config["dataset"] == "Signal":
                        reshape_data = data.x[:, :-4].reshape(data.x.shape[0], 768, 3)
                        embed = cnn_embed(torch.swapaxes(reshape_data, 1, -1))
                        inputs = torch.cat((embed, data.x[:, -4:]), axis=1)
                        pred = model(inputs, data.edge_index, data.batch, data.edge_attr)
                    else:
                        pred = model(data.x, data.edge_index, data.batch, data.edge_attr)

                    loss = scaled_mse(pred[:, 0], data.y)
                    wandb.log({"loss":loss})
                    if torch.any(loss == 0) or torch.any(torch.isnan(loss)):
                        data_to_save = {'epochs': config["epochs"],
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }
                        if config["dataset"] == "Signal":
                            data_to_save['cnn_state_dict'] = cnn_embed.state_dict()

                        torch.save(data_to_save, model_dir_path + "/" + str(i) + "_debug_nan.pt")

                        print("Error loss nul or invalid")
                        raise ValueError("wrong value for the loss")
                    loss.backward()
                    optimizer.step()

                # We want to update the learning rate to change every verbose_t epochs
                lr_scheduler.step()

                if epoch % config["verbose_t"] == 0 and epoch > 0:
                    model.eval()
                    if config["dataset"] == "Signal":
                        cnn_embed.eval()

                    train_loss = 0
                    train_n_tot = 0
                    for data in train_loader:
                        if not config["not_drop_nodes"]:
                            data_list = data.to_data_list()
                            for graph in enumerate(data_list):
                                rand_nb = np.random.random_sample()*0.4 + 0.6
                                indicies = torch.randperm(len(graph[1].x))[:int(np.round(len(graph[1].x)*rand_nb))]
                                data_list[graph[0]] = graph[1].subgraph(indicies)
                            data = tg.data.Batch.from_data_list(data_list)
                        data = data.to(device)
                        if config["dataset"] == "Signal":
                            reshape_data = data.x[:, :-4].reshape(data.x.shape[0], 768, 3)
                            embed = cnn_embed(torch.swapaxes(reshape_data, 1, -1))
                            inputs = torch.cat((embed, data.x[:, -4:]), axis=1)
                            pred = model(inputs, data.edge_index, data.batch, data.edge_attr)
                        else:
                            pred = model(data.x, data.edge_index, data.batch, data.edge_attr)
                        train_loss += loss_fn(pred[:, 0], data.y, reduction="sum").item()
                        train_n_tot += len(data.y)

                    lst_train_perf.append(train_loss/train_n_tot)

                    test_loss = 0
                    test_n_tot = 0
                    for data in test_loader:
                        if not config["not_drop_nodes"]:
                            data_list = data.to_data_list()
                            for graph in enumerate(data_list):
                                rand_nb = np.random.random_sample()*0.4 + 0.6
                                indicies = torch.randperm(len(graph[1].x))[:int(np.round(len(graph[1].x)*rand_nb))]
                                data_list[graph[0]] = graph[1].subgraph(indicies)
                            data = tg.data.Batch.from_data_list(data_list)
                            data = data.to(device)
                        if config["dataset"] == "Signal":
                            reshape_data = data.x[:, :-4].reshape(data.x.shape[0], 768, 3)
                            embed = cnn_embed(torch.swapaxes(reshape_data, 1, -1))
                            inputs = torch.cat((embed, data.x[:, -4:]), axis=1)
                            pred = model(inputs, data.edge_index, data.batch, data.edge_attr)
                        else:
                            pred = model(data.x, data.edge_index, data.batch, data.edge_attr)
                        test_loss += loss_fn(pred[:, 0], data.y, reduction="sum").item()
                        test_n_tot += len(data.y)

                    lst_test_perf.append(test_loss/test_n_tot)
                    print(f'epoch: {epoch} '
                          f'lr: {lr_scheduler.get_last_lr()[0]:.8f} '
                          f'loss_train: {train_loss/train_n_tot:.4f} '
                          f'loss_test: {test_loss/test_n_tot:.4f}'
                          )
                    wandb.log({'epoch': epoch,
                               'lr': lr_scheduler.get_last_lr()[0],
                               'loss_train': train_loss/train_n_tot,
                               'loss_test': test_loss/test_n_tot,
                               })

                    if config["dataset"] == "Signal":
                        cnn_embed.train()

                    model.train()

            data_to_save = {'epochs': config["epochs"],
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }

            if config["dataset"] == "Signal":
                data_to_save['cnn_state_dict'] = cnn_embed.state_dict()

            torch.save(data_to_save, model_dir_path + "/" + str(i) + ".pt")

            if config["epochs"] >config["verbose_t"]:
                lst_model_train_perf.append(lst_train_perf)
                lst_model_test_perf.append(lst_test_perf)

                print("Model: " + str(i) + " minimum train loss reached: ",
                    round(np.min(lst_train_perf), 4),
                    "indicie: ", np.argmin(lst_train_perf)
                    )

                print("Model: " + str(i) + " minimum test  loss reached: ",
                    round(np.min(lst_test_perf), 4),
                    "indicie: ", np.argmin(lst_test_perf)
                    )
                plt.clf()
                nb_updates = int(np.floor((config["epochs"]-1)/config["verbose_t"]))
                plt.plot([config["verbose_t"] * k for k in range(nb_updates)],
                         lst_train_perf,
                         label="train loss"
                         )
                plt.plot([config["verbose_t"] * k for k in range(nb_updates)],
                         lst_test_perf,
                         label="test loss"
                         )
                plt.xlabel("Number of epochs")
                plt.ylabel("Loss (MSE)")
                plt.legend()
                plt.savefig(fig_dir_path + "/" + str(i) + "_training")


    def test():
        """Test function"""
        # We don't want to shuffle to keep the same order in the data
        print(f"config: {config}")
        device = 'cpu'
        train_loader, test_loader = create_loader(config)
        num_features = train_dataset.num_features


        models = [0 for i in range(len(os.listdir(model_dir_path)))]
        print(os.listdir(model_dir_path))
        for model_path in os.listdir(model_dir_path):
            checkpoint = torch.load(model_dir_path + "/" + model_path, map_location=device)
            model = model_class(num_features, config["embed_size"], 1, config=config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            if config["dataset"] == "Signal":
                cnn_embed = SimpleSignalModel()
                cnn_embed.load_state_dict(checkpoint["cnn_state_dict"])
                cnn_embed.eval()
            else:
                if "cnn_state_dict" in checkpoint:
                    print("WARNING: the model to load has a cnn embeddeing, "
                          "you may want to use it by adding --use_signal")

            models[int(model_path[0])] = model

        train_loss_lst = []
        train_pred_lst = []
        for model in models:
            train_loss = 0
            train_n_tot = 0
            train_energy = []
            train_pred = []
            with torch.no_grad():
                for data in train_loader:
                    data = data.to(device)
                    pred = model(data.x, data.edge_index, data.batch, data.edge_attr)
                    train_loss += loss_fn(pred[:, 0], data.y, reduction="sum").item()
                    train_n_tot += len(data.y)
                    train_pred = np.concatenate((train_pred, pred[:, 0].numpy()))
                    train_energy = np.concatenate((train_energy, data.y.numpy()))

            train_loss_lst.append(train_loss/train_n_tot)
            train_pred_lst.append(train_pred)

        train_loss = np.array(train_loss_lst)
        train_pred = np.array(train_pred_lst)
        print("train_loss: ", train_loss)
        for i in enumerate(train_pred):
            plt.clf()
            plt.scatter(train_energy, i[1])
            plt.plot(train_energy, train_energy, "k")
            plt.title("Results on the training set")
            plt.xlabel("ground truth energy (eeV)")
            plt.ylabel("predicted energy (eeV)")
            plt.xlim(0, 4.1)
            plt.savefig(fig_dir_path + "/" + str(i[0]) + "_train")

        # The same thing for test data #

        test_loss_lst = []
        test_pred_lst = []
        for model in models:
            test_loss = 0
            test_n_tot = 0
            test_energy = []
            test_pred = []
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    pred = model(data.x, data.edge_index, data.batch, data.edge_attr)
                    test_loss += loss_fn(pred[:, 0], data.y, reduction="sum").item()
                    test_n_tot += len(data.y)
                    test_pred = np.concatenate((test_pred, pred[:, 0].numpy()))
                    test_energy = np.concatenate((test_energy, data.y.numpy()))

            test_loss_lst.append(test_loss/test_n_tot)
            test_pred_lst.append(test_pred)

        test_loss = np.array(test_loss_lst)
        test_pred = np.array(test_pred_lst)

        for i in enumerate(test_pred):
            plt.clf()
            plt.scatter(test_energy, i[1])
            plt.plot(test_energy, test_energy, "k")
            plt.title("Results on the testing set")
            plt.xlabel("ground truth energy (eeV)")
            plt.ylabel("predicted energy (eeV)")
            plt.xlim(0, 4.1)
            plt.savefig(fig_dir_path + "/" + str(i[0]) + "_test")


        # We only keep from here the best models
        if config["keep_best_models"]:
            model_index = np.argsort(test_loss)[:5]
            lst_model = []
            for index in model_index:
                lst_model.append(models[index])
            models = lst_model
            train_pred = train_pred[model_index]
            test_pred = test_pred[model_index]

        # ## We do the mean of the models ## #
        bins = [0.11 + (i + 1) * (3.99-0.11) / 10 for i in range(10)]
        ind_bins = np.digitize(train_energy, bins)
        pred_train_mean, pred_train_std, true_train_mean = [], [], []
        for e_bin in enumerate(bins):
            indicies = np.where(ind_bins == e_bin[0])[0] ###TODO adapt to the test after train
            pred_train_mean.append(np.mean(train_pred[:, indicies] - train_energy[indicies]))
            pred_train_std.append(np.sqrt(np.mean(np.std(train_pred[:, indicies], axis=0)**2)))
            true_train_mean.append(np.mean(train_energy[indicies]))

        pred_train_mean = np.array(pred_train_mean)
        pred_train_std = np.array(pred_train_std)
        true_train_mean = np.array(true_train_mean)

        ### The same thing for the testing dataset
        ind_bins = np.digitize(test_energy, bins)
        pred_test_mean, pred_test_std, true_test_mean = [], [], []
        for e_bin in  enumerate(bins):
            indicies = np.where(ind_bins == e_bin[0])[0]
            pred_test_mean.append(np.mean(test_pred[:, indicies] - test_energy[indicies]))
            pred_test_std.append(np.sqrt(np.mean(np.std(test_pred[:, indicies], axis=0)**2)))
            true_test_mean.append(np.mean(test_energy[indicies]))

        pred_test_mean = np.array(pred_test_mean)
        pred_test_std = np.array(pred_test_std)
        true_test_mean = np.array(true_test_mean)

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
        plt.savefig(fig_dir_path + "/" + "all")

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
        plt.savefig(fig_dir_path + "/" + "all" + "delta")

        plt.show()

        np.save(fig_dir_path + "/" + "train_pred", pred_train_mean)
        np.save(fig_dir_path + "/" + "train_std", pred_train_std)
        np.save(fig_dir_path + "/" + "train_true", true_train_mean)

        np.save(fig_dir_path + "/" + "test_pred", pred_test_mean)
        np.save(fig_dir_path + "/" + "test_std", pred_test_std)
        np.save(fig_dir_path + "/" + "test_true", true_test_mean)

    if config["test"]:
        test()
    else:
        train()
        test()

    print("Finished !")
