import numpy as np
from typing import List, Tuple, Union, Dict
from scipy.spatial import KDTree



def computeNeighborsKDTree(lstPositions: Union[List[Tuple[float]], np.ndarray], distance:float=2)-> np.ndarray:
    """Create graph in O(Nlog(N))

    Args:
        lstPositions (Union[List[Tuple[float]], np.ndarray],): the positions of the antennas
        distance (float, optional): the maximum distance two antennas are considered connected. Defaults to 2.

    Returns:
        np.ndarray: _description_
    """
    lstPositions = np.array(lstPositions)
    tree_node = KDTree(lstPositions)
    pairs = tree_node.query_pairs(distance)
    return pairs

def computeNeighbors(lstPositions: Union[List[Tuple[float]], np.ndarray], n_closest:int=3) -> np.ndarray:
    """Find the closest neighbors

    Args:
        lstPositions (Union[List[Tuple[float]], np.ndarray],): the list containing the positions
        n_closest (int): The number of points that we can have for neighbors

    Returns:
        set: The edge index
    """
    closest_points = []
    all_dist = []
    for antenna in range(len(lstPositions)):
        close_point = [np.inf for _ in range(n_closest)]
        close_point_index = [0 for _ in range(n_closest)]
        for antenna_close in range(len(lstPositions)):
            if antenna != antenna_close:
                dist = np.sqrt(np.sum(np.square(list(map(lambda i, j: i - j, lstPositions[antenna], lstPositions[antenna_close]))), axis=0))
                for index in range(n_closest):
                    if dist < close_point[index]:
                        close_point[index+1:] = close_point[index:-1]
                        close_point_index[index+1:] = close_point_index[index:-1]
                                            
                        close_point[index] = dist
                        close_point_index[index] = antenna_close
                        break
        
        for incr in range(len(close_point_index)):
            closest_points.append([antenna, close_point_index[incr]])
            all_dist.append(close_point[incr])
    
    indicies = [i for i in range(len(closest_points))]
    indicies.sort(key=lambda x:closest_points[0])
    return np.array(closest_points)[indicies], np.array(all_dist)[indicies]

def compute_peak2peak(efields_all_events):
    peak_to_peak_arr = np.zeros((len(efields_all_events), 2, 3)) #For the (max, min) and the 3 coordinates
    peak_to_peak_ind = np.zeros((len(efields_all_events), 2, 3)) #For the (max, min) and the 3 coordinates
    peak_to_peak_all = []

    #Peak to Peak energy
    for i in range(len(efields_all_events)):
        peak_to_peak_energy = np.max(efields_all_events[i][:, :, 1:], axis=1) - np.min(efields_all_events[i][:, :, 1:], axis=1)
        peak_to_peak_all.append(peak_to_peak_energy)
        peak_to_peak_arr[i] = [np.max(peak_to_peak_energy, axis=0), np.min(peak_to_peak_energy, axis=0)]
        peak_to_peak_ind[i] = [np.argmax(peak_to_peak_energy, axis=0), np.argmin(peak_to_peak_energy, axis=0)]
        
    return peak_to_peak_all, peak_to_peak_arr, peak_to_peak_ind

def compute_time_diff(efields_all_events):
    time_diff_peak = np.zeros((len(efields_all_events), 2)) #For the (max, min) and we consider just Y since it is the most visible
    time_diff_peak_index = np.zeros((len(efields_all_events), 2))
    time_diff_all = []

    smooth_time_diff_peak = np.zeros((len(efields_all_events), 2))
    smooth_time_diff_peak_index = np.zeros((len(efields_all_events), 2))


    for i in range(len(efields_all_events)):
        
        time_diff = - (efields_all_events[i][:, :, 0][np.arange(len(efields_all_events[i])), np.argmax(efields_all_events[i][:, :, 2], axis=1)] - efields_all_events[i][:, :, 0][np.arange(len(efields_all_events[i])), np.argmin(efields_all_events[i][:, :, 2], axis=1)])

        time_diff_peak[i] = [np.max(time_diff, axis=0), np.min(time_diff, axis=0)]
        time_diff_peak_index[i] = [np.argmax(time_diff, axis=0), np.argmin(time_diff, axis=0)]
        time_diff_all.append(time_diff)
        
    return time_diff_all, time_diff_peak, time_diff_peak_index, smooth_time_diff_peak, smooth_time_diff_peak_index

def compute_time_response(efields_all_events):
    tau_arr = []
    indicie_max_arr = []
    for event in range(len(efields_all_events)):
        indicie_max = []
        tau = []
        for ant in range(len(efields_all_events[event])):
            indicie_max.append(np.argmax(efields_all_events[event][ant, :, 2], axis=0))
            tau.append(np.where(efields_all_events[event][ant, :, 2]>0.1*efields_all_events[event][ant, int(indicie_max[-1]), 2])[0][0])
            
        indicie_max_arr.append(indicie_max)
        tau_arr.append(tau)
    
    return indicie_max_arr, tau_arr