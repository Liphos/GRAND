"""Functions to help plot results"""
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

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
    plt.ylabel("$E_{pr} - E_{th} (EeV)$")
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
    plt.ylabel(r"$\frac{E_{pr} - E_{th}}{E_{th}} $")
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

def plot_antennas(antenna_pos:torch.Tensor, p2p:torch.Tensor=None, save_dir:str=None):
    """Plot spatial distribution of the antennas"""
    plt.clf()
    plt.scatter(antenna_pos[:, 0], antenna_pos[:, 1], c=p2p, cmap="viridis",
                norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    plt.axis([-1.2, 0.8, -1, 1])
    plt.title("Antennas positions")
    plt.xlabel("$X (10⁴ m)$")
    plt.ylabel("$Y (10⁴ m)$")
    plt.colorbar()
    if save_dir is not None:
        plt.savefig(save_dir + "/antennas.png")
    plt.show(block=False)
