import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import hdf5fileinout as hdf5io
import glob

import torch_geometric as tg
import torch
import torch.nn.functional as F

from create_dataset import GrandDataset, GrandDatasetSignal
from model import GCN, TopkGCN, SimpleSignalModel

import datetime
import argparse
import os

if __name__=='__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(
                    prog = 'GNNGrand',
                    description = 'Train GNN on GRAND data')
    
    parser.add_argument("--test", action='store_true', default=False, help="enable training")
    parser.add_argument("--use_signal", action='store_true', default=False, help="Train on the raw signals instead of the processed features")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size of the training")
    parser.add_argument("--models", type=int, default=10, help="The number of models to train")
    parser.add_argument("--epochs", type=int, default=1000, help="The number of epochs to train each model")
    parser.add_argument("--split", type=float, default=0.2, help="The split between training and testing data")
    parser.add_argument("--model_name", type=str, default="No_name", help="The name of the training model")
    parser.add_argument("--embed_size", type=int, default=16, help="The size of the embedding")
    parser.add_argument("--readout", type=str, default="sum", help="The readout function to use")
    parser.add_argument("--dropout", type=float, default=0.2, help="The dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="The weight_decay use with Adam")
    parser.add_argument("--seed", type=int, default=128, help="The seed used for the shuffling and the model")
    
    config = vars(parser.parse_args())
    """
    config = {
        "train": True,
        "batch_size": 20,
        "epochs": 5000,
        "split": 0.2,
        "model_name": "test_GCN",
        
        "embed_size": 16,  #256 gave the best performance but not enough to multiply the complexity (between 16-256 is good)
        "dropout": 0.2,
        
        "lr": 1e-3,
        "weight_decay": 1e-3,
        
        "matplotlib_gui": 'TKAgg',
        "seed": 128,
    }
    """
    #matplotlib.use(config["matplotlib_gui"])
    torch.manual_seed(config["seed"])
    np.random.seed(seed=config["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ",device)
    model_dir_path = "./code/Models/" + config["model_name"]
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
        
    fig_dir_path = "./Figures/" + config["model_name"]
    if not os.path.exists(fig_dir_path):
        os.mkdir(fig_dir_path)
            
    if config["use_signal"]:
        dataset = GrandDatasetSignal()
    else:
        dataset = GrandDataset().shuffle()
    train_loader = tg.loader.DataLoader(dataset[:(1-int(len(dataset)*config["split"]))], batch_size=config["batch_size"], shuffle=not config["test"]) #We don't want to shuffle to keep the same order in the data
    test_loader = tg.loader.DataLoader(dataset[(1-int(len(dataset)*config["split"])):], batch_size=config["batch_size"], shuffle=not config["test"])

    def train():
        print("The size of the features is " + str(dataset.num_features) + " features")
        print('The number of graph objectives is ' + 'len(batch.y)/config["batch_size"]')
        print("config: ", config)
        
        lst_model_train_perf = []
        lst_model_test_perf = []
        for i in range(config["models"]):
            print(f"Model: {i}")
            lst_train_perf = []
            lst_test_perf = []
            # Create the model with given dimensions
            model = TopkGCN(dataset.num_features, config["embed_size"], 1, config=config).to(device)
            if config["use_signal"]:
                cnn_embed = SimpleSignalModel()
                optimizer = torch.optim.Adam([{'params': model.parameters()},
                                            {'params': cnn_embed.parameters()},
                                            ], lr=config["lr"], weight_decay=config["weight_decay"])
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
            
            #checkpoint = torch.load("code/Models/'2_dense_top_k_cat'/0_debug_850.pt")
            #model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            model.train()
            for epoch in range(config["epochs"]):
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    if config["use_signal"]:
                        embed = cnn_embed(data.x)
                        pred = model(embed, data.edge_index, data.batch)
                    else:
                        pred = model(data.x, data.edge_index, data.batch)
                        
                    loss = F.mse_loss(pred[:, 0], data.y.to(device))
                    if torch.any(loss==0) or torch.any(torch.isnan(loss)):
                        print()
                        #raise ValueError("wrong value for the loss")
                    loss.backward()
                    optimizer.step()
                lr_scheduler.step() ###We want to update the learning rate every 50 epochs
                
                if epoch % 50 == 0:
                    model.eval()
                    if config['use_signal']:
                        cnn_embed.eval( )
                        
                    train_loss = 0
                    train_n_tot = 0
                    for data in train_loader:
                        data = data.to(device)
                        if config["use_signal"]:
                            embed = cnn_embed(data.x)
                            pred = model(embed, data.edge_index, data.batch)
                        else:
                            pred = model(data.x, data.edge_index, data.batch)
                        train_loss += F.mse_loss(pred[:, 0], data.y.to(device), reduction="sum").item()
                        train_n_tot += len(data.y)

                    lst_train_perf.append(train_loss/train_n_tot)
                    
                    test_loss = 0
                    test_n_tot = 0
                    for data in test_loader:
                        data = data.to(device)
                        if config["use_signal"]:
                            embed = cnn_embed(data.x    )
                            pred = model(embed, data.edge_index, data.batch)
                        else:
                            pred = model(data.x,    data.edge_index, data.batch)
                        test_loss += F.mse_loss(pred[:, 0], data.y.to(device), reduction="sum").item()
                        test_n_tot += len(data)

                    lst_test_perf.append(test_loss/test_n_tot)
                    
                    print(f"epoch: {epoch}, loss_train: {train_loss/train_n_tot:.4f}, test_loss: {test_loss/test_n_tot:.4f}")
                    
                    if config['use_signal']:
                        cnn_embed.train( )
                    model.train()
            
            data_to_save = {'epochs': config["epochs"],
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }
            
            if config['use_signal']:
                data_to_save['cnn_state_dict'] = cnn_embed.state_dict()

            torch.save(data_to_save, model_dir_path + "/" + str(i) + ".pt")

            lst_model_train_perf.append(lst_train_perf)
            lst_model_test_perf.append(lst_test_perf)

        print("Model: " + str(i) + "minimum train loss reached: ", round(np.min(lst_model_train_perf), 4), "indicie: ", np.argmin(lst_model_train_perf))
        print("Model: " + str(i) + "minimum test  loss reached: ", round(np.min(lst_model_test_perf), 4), "indicie: ", np.argmin(lst_model_train_perf))
        
    def test():
        print(f"config: {config}")
        device = 'cpu'
        models = [0 for i in range(len(os.listdir(model_dir_path)))]
        print(os.listdir(model_dir_path))
        for model_path in os.listdir(model_dir_path):
            checkpoint = torch.load(model_dir_path + "/" + model_path)
            model = TopkGCN(dataset.num_features, config["embed_size"], 1, config=config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            if config['use_signal']:
                cnn_embed = SimpleSignalModel()
                cnn_embed.load_state_dict(checkpoint["cnn_state_dict"])
                cnn_embed.eval()
            else:
                if "cnn_state_dict" in checkpoint:
                    print("WARNING: the model to load has a cnn embeddeing, you may want to use it by adding --use_signal")
                    print("")
                    
            
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
                    pred = model(data.x, data.edge_index, data.batch)
                    train_loss += F.mse_loss(pred[:, 0], data.y.to(device), reduction="sum").item()
                    train_n_tot += len(data.y)
                    train_pred = np.concatenate((train_pred, pred[:, 0].numpy()))
                    train_energy = np.concatenate((train_energy, data.y.numpy()))
                    
            train_loss_lst.append(train_loss)
            train_pred_lst.append(train_pred)
        
        train_loss = np.array(train_loss_lst)
        train_pred = np.array(train_pred_lst)
        
        for i in range(len(train_pred)):
            plt.clf()
            plt.scatter(train_energy, train_pred[i])
            plt.plot(train_energy, train_energy, "k")
            plt.title("Results on the training set")
            plt.xlabel("ground truth energy (eeV)")
            plt.ylabel("predicted energy (eeV)")
            plt.xlim(0, 4.1)
            plt.savefig(fig_dir_path  + "/" + str(i) + "_train")
        
        ### The same thing for test data ###
        
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
                    pred = model(data.x, data.edge_index, data.batch)
                    test_loss += F.mse_loss(pred[:, 0], data.y.to(device), reduction="sum").item()
                    test_n_tot += len(data.y)
                    test_pred = np.concatenate((test_pred, pred[:, 0].numpy()))
                    test_energy = np.concatenate((test_energy, data.y.numpy()))
                    
            test_loss_lst.append(test_loss)
            test_pred_lst.append(test_pred)
            
        test_loss = np.array(test_loss_lst)
        test_pred = np.array(test_pred_lst)
        
        for i in range(len(test_pred)):
            plt.clf()
            plt.scatter(test_energy, test_pred[i])
            plt.plot(test_energy, test_energy, "k")
            plt.title("Results on the testing set")
            plt.xlabel("ground truth energy (eeV)")
            plt.ylabel("predicted energy (eeV)")
            plt.xlim(0, 4.1)
            plt.savefig(fig_dir_path + "/" + str(i) + "_test")
            
        ### We do the mean of the models ###
        bins = [0.11 + (i + 1) * (3.99-0.11) / 10 for i in range(10)] 
        ind_bins = np.digitize(train_energy, bins)
        pred_mean, pred_std, true_mean = [], [], []
        for bin in range(len(bins)):
            indicies = np.where(ind_bins == bin)[0]
            pred_mean.append(np.mean(train_energy[indicies] - train_pred[:, indicies]))
            pred_std.append(np.sqrt(np.mean(np.std(train_pred[:, indicies], axis=0)**2)))
            true_mean.append(np.mean(train_energy[indicies]))
        
        pred_mean, pred_std, true_mean = np.array(pred_mean), np.array(pred_std), np.array(true_mean)
        
        plt.clf()
        plt.errorbar(true_mean, pred_mean, yerr=pred_std, fmt="o")
        plt.plot(true_mean, [0 for _ in range(len(true_mean))], "k")
        plt.title("Results on the training set")
        plt.xlabel("ground truth energy (EeV)")
        plt.ylabel("$Residue E_{th} - E_{pr} (EeV)$")
        plt.xlim(0, 4.1)
        plt.savefig(fig_dir_path + "/" + "all" + "_train")
        
        plt.figure()
        plt.errorbar(true_mean, pred_mean/true_mean, yerr=pred_std/true_mean, fmt="o")
        plt.plot(true_mean, [0 for _ in range(len(true_mean))], "k")
        plt.title("Results on the training set")
        plt.xlabel("ground truth energy (EeV)")
        plt.ylabel("$Residue \Delta_{E}/E_{th} $")
        plt.xlim(0, 4.1)
        plt.savefig(fig_dir_path + "/" + "all" + "delta" + "_train")
        
        ### We do the mean of the models ###
        bins = [0.11 + (i + 1) * (3.99-0.11) / 10 for i in range(10)] 
        ind_bins = np.digitize(test_energy, bins)
        pred_mean, pred_std, true_mean = [], [], []
        for bin in range(len(bins)):
            indicies = np.where(ind_bins == bin)[0]
            pred_mean.append(np.mean(test_energy[indicies] - test_pred[:, indicies]))
            pred_std.append(np.sqrt(np.mean(np.std(test_pred[:, indicies], axis=0)**2)))
            true_mean.append(np.mean(test_energy[indicies]))
        
        pred_mean, pred_std, true_mean = np.array(pred_mean), np.array(pred_std), np.array(true_mean)
        
        plt.figure()
        plt.errorbar(true_mean, pred_mean, yerr=pred_std, fmt="o")
        plt.plot(true_mean, [0 for _ in range(len(true_mean))], "k")
        plt.title("Results on the testing set")
        plt.xlabel("ground truth energy (EeV)")
        plt.ylabel("$Residue E_{th} - E_{pr} $")
        plt.xlim(0, 4.1)
        plt.savefig(fig_dir_path + "/" + "all" + "_test")
        
                
        plt.figure()
        plt.errorbar(true_mean, pred_mean/true_mean, yerr=pred_std/true_mean, fmt="o")
        plt.plot(true_mean, [0 for _ in range(len(true_mean))], "k")
        plt.title("Results on the testing set")
        plt.xlabel("ground truth energy (EeV)")
        plt.ylabel("$Residue \Delta_{E}/E_{th} (EeV)$")
        plt.xlim(0, 4.1)
        plt.savefig(fig_dir_path + "/" + "all" + "delta" + "_test")
        
        
        plt.show()
        
        
    if config["test"]:
        test()
    else:
        train()
        test()
            
    """
    X = np.array([50 * i for i in range(int(np.floor((config["epochs"]-1)/50) + 1))])


    print(X.shape)
    print(np.mean(np.array(lst_model_train_perf), axis=0).shape)
    print(np.std(np.array(lst_model_train_perf), axis=0).shape)

    plt.errorbar(x=X, y=np.mean(np.array(lst_model_train_perf), axis=0), yerr=np.std(np.array(lst_model_train_perf), axis=0))
    plt.errorbar(x=X, y=np.mean(np.array(lst_model_test_perf), axis=0), yerr=np.std(np.array(lst_model_test_perf), axis=0))
    plt.legend([“train”, “test”])
    plt.grid()
    plt.show()
    """
