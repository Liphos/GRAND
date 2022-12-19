"""Define dataset class"""
from typing import Dict, List, Tuple
import glob

import torch
import os.path as osp
from tqdm import tqdm
import core.hdf5fileinout as hdf5io
import numpy as np
import torch_geometric as tg

from core.utils import computeNeighbors, computeNeighborsKDTree
from torch_geometric.data import InMemoryDataset
from scipy import signal
import random
import timeit
PATH_DATA = './GRAND_DATA/GP300Outbox/'
PROGENITOR = 'Proton'
ZENVAL = '_' + str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1

def find_dense_antennas(antenna_id_to_pos:Dict[str, List[float]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Returns 2 dictionaries containing the ids and positions of the antennas
       that are part in the dense part and those which are not

    Args:
        antenna_id_to_pos (Dict[str, List[float]]): The dictionary containing the ids and positions of the antennas

    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: the 2 dictionaries
    """

    allowed_pos = np.array([-11376 + 870*i for i in range(30)]) #This has been measured manually
    all_values = np.array(list(antenna_id_to_pos.values()))
    all_keys = np.array(list(antenna_id_to_pos.keys()))
    sort_ind = np.lexsort((all_values[:, 0],all_values[:, 1]), axis=0)
    antenna_nodense = {}
    antenna_dense = {}
    previous_val = None
    previous_diff = None
    for incr in range(len(all_keys[sort_ind])):
        value = all_values[sort_ind][incr]
        key = all_keys[sort_ind][incr]
        if np.any(np.abs(allowed_pos-value[0])<100):
            if previous_val is None or previous_val[1] != value[1]:
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
    for event in range(len(all_antenna_id)):
        normalization = None
        for incr in range(len(all_antenna_id[event])):
            antenna = all_antenna_id[event][incr]
            if antenna in antenna_id_to_pos:
                normalization = np.array(all_antenna_pos[event][incr]) - np.array(antenna_id_to_pos[antenna])
                break
        if normalization is None:
            antenna_id_to_pos[all_antenna_id[event][0]] = all_antenna_pos[event][0]
            normalization = np.zeros((3,))

        for ant in range(len(all_antenna_id[event])):
            antenna = all_antenna_id[event][ant]
            if antenna in antenna_id_to_pos:
                if (antenna_id_to_pos[antenna] != all_antenna_pos[event][ant] - normalization).all():
                    raise Exception("It can't be normalized")
            else:
                antenna_id_to_pos[antenna] = all_antenna_pos[event][ant] - normalization

    return antenna_id_to_pos

def compute_edges(antenna_pos:np.ndarray, has_fix_degree:bool) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the edges of the graph given the positions of the antennas

    Args:
        antenna_pos (np.ndarray): the positions of the antennas in 3D
    Returns:
        Tuple(np.ndarray, np.ndarray): the indicies in the array of the nodes that needs to be connected as well as their distance
    """
    if has_fix_degree:
        edge_index, edge_dist = computeNeighbors(antenna_pos)
        edge_index, edge_dist = tg.utils.to_undirected(torch.tensor(edge_index, dtype=torch.long).t().contiguous(), edge_attr=edge_dist, reduce="mean")
    else:
        edge_index = computeNeighborsKDTree(antenna_pos)
        edge_index = tg.utils.to_undirected(torch.tensor(edge_index, dtype=torch.long).t().contiguous())
        edge_dist = None

    return edge_index, edge_dist


class GrandDataset(InMemoryDataset):
    def __init__(self, root= "./GrandDataset", is_core_contained:bool=False, has_fix_degree:bool=False, add_degree:bool=True, max_degree:int=19):
        self.is_core_contained = is_core_contained
        self.has_fix_degree = has_fix_degree
        self.add_degree = add_degree
        self.max_degree = max_degree
        self.degree_tranform = tg.transforms.OneHotDegree(max_degree=max_degree)
        super().__init__(root)
        self.root = root

        self.train_datasets = {}
        self.test_datasets = {}
        for densite in range(11):
            _train_data, _train_slices = torch.load(self.processed_paths[densite])
            _test_data, _test_slices = torch.load(self.processed_paths[11 + densite])

            train_dataset = InMemoryDataset()
            train_dataset.data, train_dataset.slices = _train_data, _train_slices
            self.train_datasets[densite] = train_dataset

            test_dataset = InMemoryDataset()
            test_dataset.data, test_dataset.slices = _test_data, _test_slices
            self.test_datasets[densite] = test_dataset


    @property
    def processed_file_names(self):
        lst_names_train = [f'train{densite}.pt' for densite in range(11)]
        lst_names_test = [f'test{densite}.pt' for densite in range(11)]
        return lst_names_train + lst_names_test

    def process(self):
        train_graph_lst = {}
        test_graph_lst = {}
        for densite in range(11):
            train_graph_lst[str(densite)] = []
            test_graph_lst[str(densite)] = []

        PATH_DATA = './GRAND_DATA/GP300Outbox/'
        PROGENITOR = 'Proton'
        ZENVAL = '_' + str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1
        list_f = glob.glob(PATH_DATA+'*'+PROGENITOR+'*'+ZENVAL+'*')
        # list_f = glob.glob(PATH_DATA+'*')
        print('Number of files = %i' % (len(list_f)))
        #Parameters for the filter
        N = 1    # Filter order
        wn = 0.05  # Cutoff frequency
        b, A = signal.butter(N, wn, output='ba')

        all_energy = []
        all_antenna_id = []
        all_antenna_pos = []
        all_efield_loc = []
        for file in tqdm(range(len(list_f))):
            # We load the files
            inputfilename = glob.glob(list_f[file] + '/*' + PROGENITOR + '*' + ZENVAL + '*.hdf5')[0]
            run_info = hdf5io.GetRunInfo(inputfilename)
            event_name = hdf5io.GetEventName(run_info, 0)
            antenna_info = hdf5io.GetAntennaInfo(inputfilename, event_name)
            n_ant = hdf5io.GetNumberOfAntennas(antenna_info) #=len(antenna_info)
            energy = run_info['Energy'][0]
            #zenith = 180. - hdf5io.GetEventZenith(run_info, 0)
            #azimuth = hdf5io.GetEventAzimuth(run_info, 0)-180.

            antenna_id = antenna_info["ID"].value
            antenna_pos = np.concatenate((antenna_info['X'].value[:, np.newaxis], antenna_info['Y'].value[:, np.newaxis], antenna_info['Z'].value[:, np.newaxis]), axis=-1)

            for ant in range(n_ant):
                efield_loc = hdf5io.GetAntennaEfield(inputfilename, event_name,
                                                    str(antenna_id[ant], 'UTF-8'))
                if ant == 0:
                    efield_loc_arr = np.zeros((n_ant, ) + efield_loc.shape)
                    efield_smooth_arr = np.zeros((n_ant, ) + (efield_loc.shape[0], ))

                efield_loc_arr[ant] = efield_loc
                efield_smooth_arr[ant] = signal.filtfilt(b, A, efield_loc[:, 2])

            all_energy.append(energy)
            all_antenna_id.append(antenna_id)
            all_antenna_pos.append(antenna_pos)
            all_efield_loc.append(efield_loc_arr)

        # We normalize the observations
        # We normalize the position of the antennas that are shifted from their normal positions
        antenna_id_to_pos = compute_normalized_antennas(all_antenna_pos, all_antenna_id)
        if self.is_core_contained:
            core_ants = set()
            for key, value in antenna_id_to_pos.items():
                in_droites = (-2 * value[0] + 4500) < value[1] or (2 * value[0] - 1000) > value[1] or (-2 * value[0] - 8500) > value[1] or (2*value[0] + 12000) < value[1]
                in_hexa = value[1] > 4800 or value[1] < -1200 or in_droites
                if in_hexa:
                    continue
                core_ants.add(key)
        else:
            core_ants = set(antenna_id_to_pos.keys())

        antenna_no_dense, antenna_dense = find_dense_antennas(antenna_id_to_pos)
        antennas_to_keep = []
        for densite in range(11):
            if densite < 5:
                ants_to_keep = set(antenna_no_dense.keys())
                key_lst = list(antenna_dense.keys())
                random.shuffle(key_lst)
                ants_to_keep.update(key_lst[:int(len(key_lst)*densite/5)])
            elif densite == 1:
                ants_to_keep = set(antenna_id_to_pos.keys())
            else:
                ants_to_keep = set(antenna_dense.keys())
                key_lst = list(antenna_no_dense.keys())
                random.shuffle(key_lst)
                ants_to_keep.update(key_lst[:int(len(key_lst)*(2 - densite/5))])

            ants_to_keep = ants_to_keep.intersection(core_ants)
            antennas_to_keep.append(ants_to_keep)

        print("Compute features and find connections for the graphs")
        for event in tqdm(range(len(all_energy))):
            #We load the information from the files
            efield_loc_arr = all_efield_loc[event]
            antenna_id = all_antenna_id[event]
            antenna_pos = all_antenna_pos[event]
            energy = all_energy[event]

            # We want to start at t=0
            efield_loc_arr[:, :, 0] = efield_loc_arr[:, :, 0] - np.min(efield_loc_arr[:, 0, 0])
            # ## We compute the features

            time_diff = - (efield_loc_arr[:, :, 0][np.arange(len(efield_loc_arr)), np.argmax(efield_loc_arr[:, :, 2], axis=1)] - efield_loc_arr[:, :, 0][np.arange(len(efield_loc_arr)), np.argmin(efield_loc_arr[:, :, 2], axis=1)])
            peak_to_peak_energy = np.max(efield_loc_arr[:, :, 1:], axis=1) - np.min(efield_loc_arr[:, :, 1:], axis=1)
            peak_to_peak_energy_first = np.argmax(efield_loc_arr[:, :, 1:], axis=1)

            # we filter the signal in a smoother version and recompute features
            efield_smooth_arr = np.array([signal.filtfilt(b, A, efield_loc_arr[ant, :, 2]) for ant in range(len(efield_loc_arr))])

            time_diff_sm = - (efield_loc_arr[:, :, 0][np.arange(len(efield_smooth_arr)), np.argmax(efield_smooth_arr, axis=1)] - efield_loc_arr[:, :, 0][np.arange(len(efield_smooth_arr)), np.argmin(efield_smooth_arr, axis=1)])
            peak_to_peak_energy_sm = np.expand_dims(np.max(efield_smooth_arr, axis=1) - np.min(efield_smooth_arr, axis=1), axis=-1)
            peak_to_peak_energy_first_sm = np.expand_dims(np.argmax(efield_smooth_arr, axis=1), axis=-1)

            antenna_pos_corr = np.array([antenna_id_to_pos[id] for id in antenna_id])

            obs = np.concatenate(
                                (antenna_pos/10000,
                                (((peak_to_peak_energy) ** (1/5)) / 8),
                                (((peak_to_peak_energy_sm) ** (1/5)) /8),
                                efield_loc_arr[:, :, 0][np.expand_dims(np.arange(len(efield_loc_arr)), axis=-1), peak_to_peak_energy_first]/50_000,
                                efield_loc_arr[:, :, 0][np.expand_dims(np.arange(len(efield_loc_arr)), axis=-1), peak_to_peak_energy_first_sm]/50_000,
                                np.expand_dims(time_diff, axis=-1)/500,
                                np.expand_dims(time_diff_sm, axis=-1)/500,
                                ), axis=-1)

            is_test = (random.random() < 0.2)
            for densite in range(11):
                ants_to_keep = np.array([True if id in antennas_to_keep[densite] else False for id in antenna_id])
                edge_index, _ = compute_edges(antenna_pos_corr[ants_to_keep], self.has_fix_degree)
                G = tg.data.Data(
                    x=torch.tensor(obs[ants_to_keep], dtype=torch.float32),
                    edge_index=edge_index,
                    y=torch.tensor(energy, dtype=torch.float32)
                    )

                if self.add_degree:
                    G = self.degree_tranform(G)

                if is_test:
                    test_graph_lst[str(densite)].append(G)
                else:
                    train_graph_lst[str(densite)].append(G)

        """
        print(len(edge_index), edge_index)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(antenna_pos[:, 0], antenna_pos[:, 1], antenna_pos[:, 2], c=antenna_pos[:, 1], cmap='cividis')
        plt.title("Antennas positions")
        plt.xlabel("X")
        plt.ylabel("Z")
        """
        for densite in range(11):
            train_data, train_slices = self.collate(train_graph_lst[str(densite)])
            torch.save((train_data, train_slices), self.processed_paths[densite])
            print("Train dataset saved to: ", self.processed_paths[densite])

            test_data, test_slices = self.collate(test_graph_lst[str(densite)])
            torch.save((test_data, test_slices), self.processed_paths[11 + densite])
            print("Test dataset saved to: ", self.processed_paths[11 + densite])


class GrandDatasetAllSize(InMemoryDataset):
    def __init__(self, root= "./GrandDatasetAllSize"):
        super().__init__(root)
        self.root = root

        self.train_datasets = []
        self.test_datasets = []
        _train_data, _train_slices = torch.load(self.processed_paths[0])
        _test_data, _test_slices = torch.load(self.processed_paths[1])

        self.train_dataset = InMemoryDataset()
        self.train_dataset.data, self.train_dataset.slices = _train_data, _train_slices

        self.test_dataset = InMemoryDataset()
        self.test_dataset.data, self.test_dataset.slices = _test_data, _test_slices


    @property
    def processed_file_names(self):
        lst_names_train = ['train_all.pt']
        lst_names_test = ['test_all.pt']
        return lst_names_train + lst_names_test

    def process(self):
        train_graph_lst = []
        test_graph_lst = []

        list_f = glob.glob(PATH_DATA+'*'+PROGENITOR+'*'+ZENVAL+'*')
        # list_f = glob.glob(PATH_DATA+'*')
        print(f'Number of files = {(len(list_f))}')
        #Parameters for the filter
        n_order = 1    # Filter order
        wn = 0.05  # Cutoff frequency
        b, a = signal.butter(n_order, wn, output='ba')

        all_energy = []
        all_antenna_id = []
        all_antenna_pos = []
        all_efield_loc = []
        for file in tqdm(range(len(list_f))):
            # We load the files
            inputfilename = glob.glob(list_f[file] + '/*' + PROGENITOR + '*' + ZENVAL + '*.hdf5')[0]
            run_info = hdf5io.GetRunInfo(inputfilename)
            event_name = hdf5io.GetEventName(run_info, 0)
            antenna_info = hdf5io.GetAntennaInfo(inputfilename, event_name)
            n_ant = hdf5io.GetNumberOfAntennas(antenna_info) #=len(antenna_info)
            energy = run_info['Energy'][0]
            zenith = 180. - hdf5io.GetEventZenith(run_info, 0)
            azimuth = hdf5io.GetEventAzimuth(run_info, 0)-180.

            antenna_id = antenna_info["ID"].value
            antenna_pos = np.concatenate((antenna_info['X'].value[:, np.newaxis], antenna_info['Y'].value[:, np.newaxis], antenna_info['Z'].value[:, np.newaxis]), axis=-1)

            for ant in range(n_ant):
                efield_loc = hdf5io.GetAntennaEfield(inputfilename, event_name,
                                                    str(antenna_id[ant], 'UTF-8'))
                if ant == 0:
                    efield_loc_arr = np.zeros((n_ant, ) + efield_loc.shape)
                    efield_smooth_arr = np.zeros((n_ant, ) + (efield_loc.shape[0], ))

                efield_loc_arr[ant] = efield_loc
                efield_smooth_arr[ant] = signal.filtfilt(b, a, efield_loc[:, 2])

            all_energy.append(energy)
            all_antenna_id.append(antenna_id)
            all_antenna_pos.append(antenna_pos)
            all_efield_loc.append(efield_loc_arr)

        # We normalize the observations
        # We normalize the position of the antennas that are shifted from their normal positions
        antenna_id_to_pos = compute_normalized_antennas(all_antenna_pos, all_antenna_id)

        antenna_no_dense, antenna_dense = find_dense_antennas(antenna_id_to_pos)

        print("Compute features and find connections for the graphs")
        for event in tqdm(range(len(all_energy))):
            #We load the information from the files
            efield_loc_arr = all_efield_loc[event]
            antenna_id = all_antenna_id[event]
            antenna_pos = all_antenna_pos[event]
            energy = all_energy[event]

            # We want to start at t=0
            efield_loc_arr[:, :, 0] = efield_loc_arr[:, :, 0] - np.min(efield_loc_arr[:, 0, 0])
            # ## We compute the features

            time_diff = - (efield_loc_arr[:, :, 0][np.arange(len(efield_loc_arr)), np.argmax(efield_loc_arr[:, :, 2], axis=1)] - efield_loc_arr[:, :, 0][np.arange(len(efield_loc_arr)), np.argmin(efield_loc_arr[:, :, 2], axis=1)])
            peak_to_peak_energy = np.max(efield_loc_arr[:, :, 1:], axis=1) - np.min(efield_loc_arr[:, :, 1:], axis=1)
            peak_to_peak_energy_first = np.argmax(efield_loc_arr[:, :, 1:], axis=1)

            # we filter the signal in a smoother version and recompute features
            efield_smooth_arr = np.array([signal.filtfilt(b, a, efield_loc_arr[ant, :, 2]) for ant in range(len(efield_loc_arr))])

            time_diff_sm = - (efield_loc_arr[:, :, 0][np.arange(len(efield_smooth_arr)), np.argmax(efield_smooth_arr, axis=1)] - efield_loc_arr[:, :, 0][np.arange(len(efield_smooth_arr)), np.argmin(efield_smooth_arr, axis=1)])
            peak_to_peak_energy_sm = np.expand_dims(np.max(efield_smooth_arr, axis=1) - np.min(efield_smooth_arr, axis=1), axis=-1)
            peak_to_peak_energy_first_sm = np.expand_dims(np.argmax(efield_smooth_arr, axis=1), axis=-1)

            antenna_pos_corr = np.array([antenna_id_to_pos[id] for id in antenna_id])

            obs = np.concatenate(
                                (antenna_pos/10000,
                                ((peak_to_peak_energy) ** (1/5) / 8),
                                ((peak_to_peak_energy_sm) ** (1/5) /8),
                                efield_loc_arr[:, :, 0][np.expand_dims(np.arange(len(efield_loc_arr)), axis=-1), peak_to_peak_energy_first]/50_000,
                                efield_loc_arr[:, :, 0][np.expand_dims(np.arange(len(efield_loc_arr)), axis=-1), peak_to_peak_energy_first_sm]/50_000,
                                np.expand_dims(time_diff, axis=-1)/500,
                                np.expand_dims(time_diff_sm, axis=-1)/500,
                                ), axis=-1)

            is_test = (random.random() < 0.2)
            if is_test:
                test_graph_lst.append(G)
            else:
                for incr in range(21):
                    ants_to_keep = np.random.rand(len(antenna_id)) >= 0.05 + incr * 0.02
                    edge_index, edge_dist = compute_edges(antenna_pos_corr[ants_to_keep])

                    G = tg.data.Data(
                        x=torch.tensor(obs[ants_to_keep], dtype=torch.float32),
                        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                        edge_attr=torch.tensor(10000/edge_dist, dtype=torch.float32),
                        y=torch.tensor(energy, dtype=torch.float32)
                        )

                    train_graph_lst.append(G)

        """
        print(len(edge_index), edge_index)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(antenna_pos[:, 0], antenna_pos[:, 1], antenna_pos[:, 2], c=antenna_pos[:, 1], cmap='cividis')
        plt.title("Antennas positions")
        plt.xlabel("X")
        plt.ylabel("Z")
        """
        train_data, train_slices = self.collate(train_graph_lst)
        torch.save((train_data, train_slices), self.processed_paths[0])
        print("Train dataset saved to: ", self.processed_paths[0])

        test_data, test_slices = self.collate(test_graph_lst)
        torch.save((test_data, test_slices), self.processed_paths[1])
        print("Test dataset saved to: ", self.processed_paths[1])


# Dataset Containing the raw signals

class GrandDatasetSignal(InMemoryDataset):
    def __init__(self, root= "./GrandDatasetSignal"):
        super().__init__(root)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graph_list = []

        PATH_DATA = './GRAND_DATA/GP300Outbox/'
        PROGENITOR = 'Proton'
        ZENVAL = '_' + str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1
        list_f = glob.glob(PATH_DATA+'*'+PROGENITOR+'*'+ZENVAL+'*')
        # list_f = glob.glob(PATH_DATA+'*')
        print('Number of files = %i' % (len(list_f)))

        obs_lst = []
        label_lst = []
        for file in tqdm(range(len(list_f))):
            # We load the files
            inputfilename = glob.glob(list_f[file] + '/*' + PROGENITOR + '*' + ZENVAL + '*.hdf5')[0]
            run_info = hdf5io.GetRunInfo(inputfilename)
            event_name = hdf5io.GetEventName(run_info, 0)
            antenna_info = hdf5io.GetAntennaInfo(inputfilename, event_name)
            n_ant = hdf5io.GetNumberOfAntennas(antenna_info)  # =len(antenna_info)
            energy = run_info['Energy'][0]

            antenna_id = antenna_info["ID"].value
            antenna_pos = np.concatenate((antenna_info['X'].value[:, np.newaxis], antenna_info['Y'].value[:, np.newaxis], antenna_info['Z'].value[:, np.newaxis]), axis=-1)
            for ant in range(n_ant):
                efield_loc = hdf5io.GetAntennaEfield(inputfilename, event_name,
                                                    str(antenna_id[ant], 'UTF-8'))
                if ant == 0:
                    efield_loc_arr = np.zeros((n_ant, ) + efield_loc.shape)

                efield_loc_arr[ant] = efield_loc

            index_spike = int(np.round(np.mean(np.argmax(efield_loc_arr[:, :, 2], axis=1))))
            efield_loc_arr = efield_loc_arr[:, index_spike-250:index_spike + 518, :]  ### We normalize the length of the data

            efield_loc_arr[:, :, 0] = efield_loc_arr[:, :, 0] - np.min(efield_loc_arr[:, 0, 0]) #We want to start at t=0

            if index_spike < 250:
                print(index_spike)
                raise ValueError("Index spike is to low")

            ### We normalize the position of the antennas that are shifted from their normal positions
            antenna_id_to_pos = {}
            normalization_lst = []
            normalization = None
            for incr in range(len(antenna_id)):
                antenna = antenna_id[incr]
                if antenna in antenna_id_to_pos:
                    normalization = np.array(antenna_pos[incr]) - np.array(antenna_id_to_pos[antenna])
                    break

            if normalization is None:
                antenna_id_to_pos[antenna_id[0]] = antenna_pos[0]
                normalization = np.zeros((3,))

            for ant in range(len(antenna_id)):
                antenna = antenna_id[ant]
                if antenna in antenna_id_to_pos:
                    #print(antenna_id_to_pos[antenna] - antenna_pos[ant] - normalization)
                    if (antenna_id_to_pos[antenna] != antenna_pos[ant] - normalization).all():
                        raise Exception("It can't be normalized")
                else:
                    antenna_id_to_pos[antenna] = antenna_pos[ant] - normalization

            normalization_lst.append(normalization)
            #print(f"number of normalized antennas: {len(antenna_id_to_pos)}")

            t0 = efield_loc_arr[:, :1, 0]

            antenna_pos_corr = np.array([antenna_id_to_pos[antenna_id[i]] for i in range(len(antenna_id))])
            obs = np.concatenate((efield_loc_arr[:, :, 1:].reshape(n_ant, -1)/100, antenna_pos_corr/1000, t0/50_000), axis=-1)

            obs_lst.append(obs)
            label_lst.append(energy)

            edge_index,_ = computeNeighbors(antenna_pos_corr)
            edge_index = np.array(list(edge_index)) #Transform in array
            edge_index_mirrored = edge_index[:, [1, 0]]
            edge_index = np.concatenate((edge_index, edge_index_mirrored), axis=0) #To have the edges in the 2 ways
            edge_index = np.unique(edge_index, axis=0) #To remove the duplicates

            G = tg.data.Data(x=torch.tensor(obs, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), y=torch.tensor(energy, dtype=torch.float32))
            graph_list.append(G)
            """
            print(len(edge_index), edge_index)
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.scatter3D(antenna_pos[:, 0], antenna_pos[:, 1], antenna_pos[:, 2], c=antenna_pos[:, 1], cmap='cividis')
            plt.title("Antennas positions")
            plt.xlabel("X")
            plt.ylabel("Z")
            """

        data, slices = self.collate(graph_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Dataset saved to: ", self.processed_paths[0])


if __name__ == '__main__':
    dataset = GrandDataset("./GrandDatasetOHDeg", is_core_contained=False)
    train_dataset = dataset.train_datasets[5]
    train_dataset