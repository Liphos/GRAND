"""Functions to extract features from the dataset and to build the graph"""
from typing import List, Tuple, Union, Dict
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn.functional as F
import torch_geometric as tg


def compute_neighbor_kdtree(lst_positions: Union[List[Tuple[float]], np.ndarray],
                           distance:float=1500)-> np.ndarray:
    """Create graph in O(Nlog(N))

    Args:
        lst_positions (Union[List[Tuple[float]], np.ndarray],): the positions of the antennas
        distance (float, optional): the maximum distance two antennas are considered connected.
        Defaults to 2.

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
    p2p_arr = np.zeros((len(efields_all_events), 2, 3))
    p2p_ind = np.zeros((len(efields_all_events), 2, 3))
    p2p_all = []

    #Peak to Peak energy
    for efield in enumerate(efields_all_events):
        p2p_energy = np.max(efield[1][:, :, 1:], axis=1) - np.min(efield[1][:, :, 1:], axis=1)
        p2p_all.append(p2p_energy)
        p2p_arr[efield[0]] = [np.max(p2p_energy, axis=0),
                                       np.min(p2p_energy, axis=0)]
        p2p_ind[efield[0]] = [np.argmax(p2p_energy, axis=0),
                                       np.argmin(p2p_energy, axis=0)]

    return p2p_all, p2p_arr, p2p_ind

def compute_time_diff_all_events(efields_all_events):
    """Compute time between the two spikes"""
    #For the (max, min) and we consider just Y since it is the most visible
    time_diff_peak = np.zeros((len(efields_all_events), 2))
    time_diff_peak_index = np.zeros((len(efields_all_events), 2))
    time_diff_all = []

    for efield in enumerate(efields_all_events):
        time_diff = - ((efield[1][:, :, 0][np.arange(len(efield[1])),
                                          np.argmax(efield[1][:, :, 2], axis=1)]) -
                       (efield[1][:, :, 0][np.arange(len(efield[1])),
                                          np.argmin(efield[1][:, :, 2], axis=1)]))

        time_diff_peak[efield[0]] = [np.max(time_diff, axis=0), np.min(time_diff, axis=0)]
        time_diff_peak_index[efield[0]] = [np.argmax(time_diff, axis=0),
                                           np.argmin(time_diff, axis=0)]
        time_diff_all.append(time_diff)

    return time_diff_all, time_diff_peak, time_diff_peak_index

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

def find_dense_antennas(antenna_id_to_pos: Dict[str, List[float]]
                        ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Returns 2 dictionaries containing the ids and positions of the antennas
       that are part in the dense part and those which are not

    Args:
        antenna_id_to_pos (Dict[str, List[float]]): The dictionary containing the ids
        and positions of the antennas

    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: the 2 dictionaries
    """

    allowed_pos = np.array([-11376 + 870*i for i in range(30)]) #This has been measured manually
    all_values = np.array(list(antenna_id_to_pos.values()))
    all_keys = np.array(list(antenna_id_to_pos.keys()))
    sort_ind = np.lexsort((all_values[:, 0],all_values[:, 1]), axis=0)
    antenna_nodense = {}
    antenna_dense = {}
    previous_val = []
    previous_diff = None
    for incr in enumerate(all_keys[sort_ind]):
        value = all_values[sort_ind][incr[0]]
        key = incr[1]
        if np.any(np.abs(allowed_pos-value[0])<100):
            if len(previous_val) == 0 or previous_val[1] != value[1]:
                previous_val = value
                previous_diff = None
                antenna_nodense[key] = value
            elif previous_diff is None:
                previous_diff = value[0] - previous_val[0]
                antenna_nodense[key] = value
                previous_val = value
            elif np.abs(previous_diff - (value[0] - previous_val[0])) < 10:
                antenna_nodense[key] = value
                previous_val = value
            else:
                antenna_dense[key] = value
        else:
            antenna_dense[key] = value
    return antenna_nodense, antenna_dense

def compute_normalized_antennas(all_antenna_pos, all_antenna_id) -> Dict[str, List[float]]:
    """Compute the normalization of the given positions for all events"""
    antenna_id_to_pos = {}
    for event in enumerate(all_antenna_id):
        normalization = None
        for ant in enumerate(event[1]):
            antenna = ant[1]
            if antenna in antenna_id_to_pos:
                normalization = np.array(all_antenna_pos[event[0]][ant[0]]) - np.array(antenna_id_to_pos[antenna])
                break
        if normalization is None:
            antenna_id_to_pos[event[1][0]] = all_antenna_pos[event[0]][0]
            normalization = np.zeros((3,))

        for ant in enumerate(event[1]):
            antenna = ant[1]
            if antenna in antenna_id_to_pos:
                if (antenna_id_to_pos[antenna] != all_antenna_pos[event[0]][ant[0]] -
                    normalization).all():
                    raise Exception("It can't be normalized")
            else:
                antenna_id_to_pos[antenna] = all_antenna_pos[event[0]][ant[0]] - normalization

    return antenna_id_to_pos



def compute_time_diff(efield_time:np.ndarray, efield_loc:np.ndarray) -> np.ndarray:
    """Compute time difference betwwen the two peaks"""
    time_diff = - (efield_time[np.arange(len(efield_loc)),
                                        np.argmax(efield_loc, axis=1)] -
                efield_time[np.arange(len(efield_loc)),
                                        np.argmin(efield_loc, axis=1)])
    return time_diff

def compute_p2p(efields:np.ndarray) -> np.ndarray:
    """Compute the peak to peak values for the given efields"""
    p2p_max = np.max(efields, axis=1)
    p2p_min = np.min(efields, axis=1)
    if len(p2p_max.shape) == 1:
        p2p = np.expand_dims(p2p_max - p2p_min, axis=-1)
        p2p_first = np.expand_dims(np.argmax(efields, axis=1), axis=-1)
    else:
        p2p = p2p_max - p2p_min
        p2p_first = np.argmax(efields, axis=1)
    return p2p, p2p_first
