"""Functions to extract features from te dataset and to build the graph"""
from typing import List, Tuple, Union
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F




def compute_neighbor_kdree(lst_positions: Union[List[Tuple[float]], np.ndarray],
                           distance:float=1500)-> np.ndarray:
    """Create graph in O(Nlog(N))

    Args:
        lst_positions (Union[List[Tuple[float]], np.ndarray],): the positions of the antennas
        distance (float, optional): the maximum distance two antennas are considered connected. Defaults to 2.

    Returns:
        np.ndarray: _description_
    """
    lst_positions = np.array(lst_positions)
    tree_node = KDTree(lst_positions)
    pairs = tree_node.query_pairs(distance, output_type='ndarray')

    return pairs

def compute_neighbors(lst_positions: Union[List[Tuple[float]], np.ndarray],
                      n_closest:int=3) -> np.ndarray:
    """Find the closest neighbors

    Args:
        lst_positions (Union[List[Tuple[float]], np.ndarray],): the list containing the positions
        n_closest (int): The number of points that we can have for neighbors

    Returns:
        set: The edge index
    """
    closest_points = []
    all_dist = []
    for antenna in enumerate(lst_positions):
        close_point = [np.inf for _ in range(n_closest)]
        close_point_index = [0 for _ in range(n_closest)]
        for antenna_close in enumerate(lst_positions):
            if antenna[0] != antenna_close[0]:
                dist = np.sqrt(np.sum(np.square(list(map(lambda i, j: i - j, antenna[1], antenna_close[1]))), axis=0))
                for index in range(n_closest):
                    if dist < close_point[index]:
                        close_point[index+1:] = close_point[index:-1]
                        close_point_index[index+1:] = close_point_index[index:-1]

                        close_point[index] = dist
                        close_point_index[index] = antenna_close[0]
                        break

        for incr in enumerate(close_point_index):
            closest_points.append([antenna[0], incr[1]])
            all_dist.append(close_point[incr[0]])

    indicies = list(range(len(closest_points)))
    indicies.sort(key=lambda x:closest_points[0])
    return np.array(closest_points)[indicies], np.array(all_dist)[indicies]

def compute_peak2peak(efields_all_events):
    """Compute amplitude spike to spike"""
    #For the (max, min) and the 3 coordinates
    peak_to_peak_arr = np.zeros((len(efields_all_events), 2, 3))
    peak_to_peak_ind = np.zeros((len(efields_all_events), 2, 3))
    peak_to_peak_all = []

    #Peak to Peak energy
    for efield in enumerate(efields_all_events):
        peak_to_peak_energy = np.max(efield[1][:, :, 1:], axis=1) - np.min(efield[1][:, :, 1:], axis=1)
        peak_to_peak_all.append(peak_to_peak_energy)
        peak_to_peak_arr[efield[0]] = [np.max(peak_to_peak_energy, axis=0),
                                       np.min(peak_to_peak_energy, axis=0)]
        peak_to_peak_ind[efield[0]] = [np.argmax(peak_to_peak_energy, axis=0),
                                       np.argmin(peak_to_peak_energy, axis=0)]

    return peak_to_peak_all, peak_to_peak_arr, peak_to_peak_ind

def compute_time_diff(efields_all_events):
    """Compute time between the two spikes"""
    #For the (max, min) and we consider just Y since it is the most visible
    time_diff_peak = np.zeros((len(efields_all_events), 2))
    time_diff_peak_index = np.zeros((len(efields_all_events), 2))
    time_diff_all = []

    smooth_time_diff_peak = np.zeros((len(efields_all_events), 2))
    smooth_time_diff_peak_index = np.zeros((len(efields_all_events), 2))


    for efield in enumerate(efields_all_events):
        time_diff = - (efield[1][:, :, 0][np.arange(len(efield[1])), np.argmax(efield[1][:, :, 2], axis=1)] -
                       efield[1][:, :, 0][np.arange(len(efield[1])), np.argmin(efield[1][:, :, 2], axis=1)])

        time_diff_peak[efield[0]] = [np.max(time_diff, axis=0), np.min(time_diff, axis=0)]
        time_diff_peak_index[efield[0]] = [np.argmax(time_diff, axis=0), np.argmin(time_diff, axis=0)]
        time_diff_all.append(time_diff)

    return time_diff_all, time_diff_peak, time_diff_peak_index, smooth_time_diff_peak, smooth_time_diff_peak_index

def compute_time_response(efields_all_events):
    """Find time of the first spike and when the signal is greater than 0.1 of the max value"""
    tau_arr = []
    indicie_max_arr = []
    for efield in efields_all_events:
        indicie_max = []
        tau = []
        for ant in enumerate(efield):
            indicie_max.append(np.argmax(efield[ant[0], :, 2], axis=0))
            tau.append(np.where(efield[ant[0], :, 2]>0.1*efield[ant[0], int(indicie_max[-1]), 2])[0][0])

        indicie_max_arr.append(indicie_max)
        tau_arr.append(tau)

    return indicie_max_arr, tau_arr

def scaled_l1(pred_labels, true_labels, reduction="mean"):
    """MAPE loss"""
    return F.l1_loss(pred_labels/true_labels, torch.ones_like(true_labels), reduction=reduction)
def scaled_mse(pred_labels, true_labels, reduction="mean"):
    """square of the map"""
    return F.mse_loss(pred_labels/true_labels, torch.ones_like(true_labels), reduction=reduction)
