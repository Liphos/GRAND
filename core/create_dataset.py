"""Define dataset class"""
from typing import Dict, List, Tuple, Union
import glob

import random
import torch
from tqdm import tqdm
import numpy as np
import torch_geometric as tg
from scipy import signal
from torch_geometric.data import InMemoryDataset

import core.hdf5fileinout as hdf5io
from core.utils import (compute_neighbors,
                        compute_neighbor_kdtree,
                        compute_time_diff,
                        compute_p2p,
                        find_dense_antennas,
                        compute_normalized_antennas
                        )

PATH_DATA = './data/GRAND_DATA/GP300Outbox/'
PROGENITOR = 'Proton'
ZENVAL = '_' + str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1

def load_file_info(filename:str) -> Tuple[float, str, np.ndarray, np.ndarray]:
    """Return needed information from the file"""
    run_info = hdf5io.GetRunInfo(filename)
    event_name = hdf5io.GetEventName(run_info, 0)
    antenna_info = hdf5io.GetAntennaInfo(filename, event_name)
    n_ant = hdf5io.GetNumberOfAntennas(antenna_info) #=len(antenna_info)
    energy = run_info['Energy'][0]

    antenna_id = antenna_info["ID"].value
    antenna_pos = np.concatenate((antenna_info['X'].value[:, np.newaxis],
                                    antenna_info['Y'].value[:, np.newaxis],
                                    antenna_info['Z'].value[:, np.newaxis]), axis=-1)

    for ant in range(n_ant):
        efield_loc = hdf5io.GetAntennaEfield(filename, event_name,
                                            str(antenna_id[ant], 'UTF-8'))
        if ant == 0:
            efield_loc_arr = np.zeros((n_ant, ) + efield_loc.shape)

        efield_loc_arr[ant] = efield_loc

    return energy, antenna_id, antenna_pos, efield_loc_arr

def load_all_files(lst_files:str) -> Dict[str, Union[float, str, np.ndarray]]:
    """Load all information from the files"""
    print(f'Number of files = {len(lst_files)}')
    all_features = {"energy": [], "antenna_id": [], "antenna_pos": [], "efield_loc": []}
    for file in tqdm(lst_files):
        # We load the files
        inputfilename = glob.glob(file + '/*' + PROGENITOR + '*' + ZENVAL + '*.hdf5')[0]
        energy, antenna_id, antenna_pos, efield_loc_arr = load_file_info(inputfilename)
        all_features["energy"].append(energy)
        all_features["antenna_id"].append(antenna_id)
        all_features["antenna_pos"].append(antenna_pos)
        all_features["efield_loc"].append(efield_loc_arr)

    return all_features

def compute_edges(antenna_pos:np.ndarray, has_fix_degree:bool) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the edges of the graph given the positions of the antennas

    Args:
        antenna_pos (np.ndarray): the positions of the antennas in 3D
    Returns:
        Tuple(np.ndarray, np.ndarray): the indicies in the array of the nodes that
        needs to be connected as well as their distance
    """
    if has_fix_degree:
        edge_index, edge_dist = compute_neighbors(antenna_pos)
        edge_index, edge_dist = tg.utils.to_undirected(
                                        torch.tensor(edge_index,
                                                     dtype=torch.long).t().contiguous(),
                                                     edge_attr=edge_dist, reduce="mean"
                                                     )
    else:
        edge_index = compute_neighbor_kdtree(antenna_pos)
        edge_index = tg.utils.to_undirected(torch.tensor(edge_index,
                                                         dtype=torch.long).t().contiguous())
        edge_dist = None

    return edge_index, edge_dist

def create_obs(efield_loc_arr:np.ndarray,
               antenna_pos:np.ndarray,
               filter_params:Dict[str, Union[float, Tuple[float, float]]]) -> np.ndarray:
    """Compute the features and create the graph"""
    # We want to start at t=0
    efield_loc_arr[:, :, 0] = efield_loc_arr[:, :, 0] - np.min(efield_loc_arr[:, 0, 0])
    # ## We compute the features
    time_diff = compute_time_diff(efield_loc_arr[:, :, 0], efield_loc_arr[:, :, 2])

    p2p_energy, p2p_energy_first = compute_p2p(efield_loc_arr[:, :, 1:])

    # we filter the signal in a smoother version and recompute features
    efield_smooth_arr = np.array([signal.filtfilt(filter_params["bA"][0],
                                                    filter_params["bA"][1],
                                                    efield_loc_arr[ant, :, 2])
                                    for ant in range(len(efield_loc_arr))])

    time_diff_sm = compute_time_diff(efield_loc_arr[:, :, 0], efield_smooth_arr)

    p2p_energy_sm, p2p_energy_first_sm = compute_p2p(efield_smooth_arr)

    obs = np.concatenate(
        (
        (((p2p_energy) ** (1/5)) / 8),
        (((p2p_energy_sm) ** (1/5)) /8),
        efield_loc_arr[:, :, 0][np.expand_dims(np.arange(len(efield_loc_arr)), axis=-1),
                                p2p_energy_first]/50_000,
        efield_loc_arr[:, :, 0][np.expand_dims(np.arange(len(efield_loc_arr)), axis=-1),
                                p2p_energy_first_sm]/50_000,
        np.expand_dims(time_diff, axis=-1)/500,
        np.expand_dims(time_diff_sm, axis=-1)/500,
        ), axis=-1)

    return obs

def find_core_antennas(antenna_id_to_pos: Dict[str, List[float]]) -> set:
    """Find the antennas in the core of the layout"""
    core_ants = set()
    for key, value in antenna_id_to_pos.items():
        droite_right = (-2 * value[0] + 4500) < value[1] or (2 * value[0] - 1000) > value[1]
        droite_left = (-2 * value[0] - 8500) > value[1] or (2*value[0] + 12000) < value[1]
        in_droites = droite_right or droite_left
        in_hexa = value[1] > 4800 or value[1] < -1200 or in_droites
        if in_hexa:
            continue
        core_ants.add(key)
    return core_ants

def compute_antennas_to_keep(antenna_id_to_pos: Dict[str, List[float]],
                             lst_pourcent:List[int],
                             is_core_contained:bool=False) -> Dict[Tuple[int, int],set]:
    """Compute the antennas that are kept between the infill, coarse, and core antennas"""
    if is_core_contained:
        core_ants = find_core_antennas(antenna_id_to_pos)
    else:
        core_ants = set(antenna_id_to_pos.keys())
    antenna_no_dense, antenna_dense = find_dense_antennas(antenna_id_to_pos)
    key_coarse_lst = list(antenna_no_dense.keys())
    key_infill_lst = list(antenna_dense.keys())
    random.shuffle(key_infill_lst)
    random.shuffle(key_coarse_lst)
    antennas_to_keep = {}
    for distrib_infill in lst_pourcent:
        for distrib_coarse in lst_pourcent:
            key = (distrib_infill, distrib_coarse)

            ants_to_keep = set()
            ants_to_keep.update(key_infill_lst[:int(len(key_infill_lst) * (1-distrib_infill*0.01))])
            ants_to_keep.update(key_coarse_lst[:int(len(key_coarse_lst) * (1-distrib_coarse*0.01))])

            ants_to_keep = ants_to_keep.intersection(core_ants)
            antennas_to_keep[key] = ants_to_keep

    return antennas_to_keep

class GrandDataset(InMemoryDataset):
    """Dataset class for the grand dataset"""
    def __init__(self, root= "GrandDataset",
                 is_core_contained:bool=False,
                 has_fix_degree:bool=False,
                 add_degree:bool=True,
                 max_degree:int=20,
                 distance:int=1500):

        self.is_core_contained = is_core_contained
        self.has_fix_degree = has_fix_degree
        self.add_degree = add_degree
        self.max_degree = max_degree
        self.all_transforms = [
            tg.transforms.RadiusGraph(r=distance, loop=False, max_num_neighbors=self.max_degree-1),
        ]
        if self.add_degree:
            self.all_transforms.append(tg.transforms.OneHotDegree(max_degree=max_degree))
        self.lst_pourcent = [0, 2, 5, 10, 15, 18, 20, 22, 25, 30, 35, 40]

        super().__init__("./data/" + root)
        self.root = "./data/" + root

        self.train_datasets = {}
        self.test_datasets = {}
        for enum_infill in enumerate(self.lst_pourcent):
            incr_infill, distrib_infill = enum_infill
            for enum_coarse in enumerate(self.lst_pourcent):
                incr_coarse, distrib_coarse = enum_coarse
                key_train = incr_infill * len(self.lst_pourcent) + incr_coarse
                key_test = key_train + len(self.lst_pourcent) ** 2
                key = (distrib_infill, distrib_coarse)
                _train_data, _train_slices = torch.load(self.processed_paths[key_train])
                _test_data, _test_slices = torch.load(self.processed_paths[key_test])

                train_dataset = InMemoryDataset()
                train_dataset.data, train_dataset.slices = _train_data, _train_slices
                self.train_datasets[key] = train_dataset

                test_dataset = InMemoryDataset()
                test_dataset.data, test_dataset.slices = _test_data, _test_slices
                self.test_datasets[key] = test_dataset


    @property
    def processed_file_names(self):
        lst_names_train = []
        lst_names_test = []
        for distrib_infill in self.lst_pourcent:
            for distrib_coarse in self.lst_pourcent:
                suffix = f'_infill_{distrib_infill}_coarse_{distrib_coarse}.pt'
                lst_names_train.append(f'train_{suffix}')
                lst_names_test.append(f'test_{suffix}')
        return lst_names_train + lst_names_test

    @property
    def num_classes(self) -> int:
        return 1

    def process(self):
        train_graph_lst = {}
        test_graph_lst = {}
        for distrib_infill in self.lst_pourcent:
            for distrib_coarse in self.lst_pourcent:
                key = (distrib_infill, distrib_coarse)
                train_graph_lst[key] = []
                test_graph_lst[key] = []

        #Parameters for the filter
        filter_params = {"N": 1, "Wn": 0.05}
        filter_params["bA"] = signal.butter(filter_params["N"], filter_params["Wn"], output='ba')

        all_features = load_all_files(glob.glob(PATH_DATA+'*'+PROGENITOR+'*'+ZENVAL+'*'))
        # We normalize the observations
        # We normalize the position of the antennas that are shifted from their normal positions
        antenna_id_to_pos = compute_normalized_antennas(all_features["antenna_pos"],
                                                        all_features["antenna_id"])

        antennas_to_keep = compute_antennas_to_keep(antenna_id_to_pos, self.lst_pourcent,self.is_core_contained)

        print("Compute features and find connections for the graphs")

        for event in tqdm(range(len(all_features["energy"]))):
            #We load the information from the files
            efield_loc_arr = all_features["efield_loc"][event]
            antenna_id = all_features["antenna_id"][event]
            energy = all_features["energy"][event]

            antenna_pos_corr = np.array([antenna_id_to_pos[id] for id in antenna_id])
            obs = create_obs(efield_loc_arr, antenna_pos_corr, filter_params)

            is_test = (random.random() < 0.2)
            for distrib_infill in self.lst_pourcent:
                for distrib_coarse in self.lst_pourcent:
                    key = (distrib_infill, distrib_coarse)
                    ants_to_keep = np.array([id in antennas_to_keep[key]
                                            for id in antenna_id])
                    graph = tg.data.Data(
                        x=torch.tensor(obs[ants_to_keep], dtype=torch.float32),
                        y=torch.tensor(energy, dtype=torch.float32),
                        ###TODO: fix the fact that num_classes is not computed correctly ###
                        pos=torch.tensor(antenna_pos_corr[ants_to_keep], dtype=torch.float32)
                        )
                    graph = tg.transforms.Compose(self.all_transforms)(graph)
                    ### We might overwrite the edges during training while loading the dataset ###

                    if is_test:
                        test_graph_lst[key].append(graph)
                    else:
                        train_graph_lst[key].append(graph)

        print("Saving")
        for enum_infill in enumerate(self.lst_pourcent):
            incr_infill, distrib_infill = enum_infill
            for enum_coarse in enumerate(self.lst_pourcent):
                incr_coarse, distrib_coarse = enum_coarse
                key = (distrib_infill, distrib_coarse)
                key_train = incr_infill * len(self.lst_pourcent) + incr_coarse
                key_test = key_train + len(self.lst_pourcent) ** 2
                train_data, train_slices = self.collate(train_graph_lst[key])
                torch.save((train_data, train_slices), self.processed_paths[key_train])
                print("Train dataset saved to: ", self.processed_paths[key_train])

                test_data, test_slices = self.collate(test_graph_lst[key])
                torch.save((test_data, test_slices), self.processed_paths[key_test])
                print("Test dataset saved to: ", self.processed_paths[key_test])


# Dataset Containing the raw signals

class GrandDatasetSignal(InMemoryDataset):
    """Old dataset to work directly with the signal"""
    def __init__(self, root= "GrandDatasetSignal"):
        super().__init__("./data/" + root)
        self.root = "./data/" + root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        graph_list = []

        list_f = glob.glob(PATH_DATA+'*'+PROGENITOR+'*'+ZENVAL+'*')
        # list_f = glob.glob(PATH_DATA+'*')
        print(f'Number of files = {len(list_f)}')

        obs_lst = []
        label_lst = []
        for file in tqdm(list_f):
            # We load the files
            inputfilename = glob.glob(file + '/*' + PROGENITOR + '*' + ZENVAL + '*.hdf5')[0]
            run_info = hdf5io.GetRunInfo(inputfilename)
            event_name = hdf5io.GetEventName(run_info, 0)
            antenna_info = hdf5io.GetAntennaInfo(inputfilename, event_name)
            n_ant = hdf5io.GetNumberOfAntennas(antenna_info)  # =len(antenna_info)
            energy = run_info['Energy'][0]

            antenna_id = antenna_info["ID"].value
            antenna_pos = np.concatenate((antenna_info['X'].value[:, np.newaxis],
                                          antenna_info['Y'].value[:, np.newaxis],
                                          antenna_info['Z'].value[:, np.newaxis]), axis=-1)
            for ant in range(n_ant):
                efield_loc = hdf5io.GetAntennaEfield(inputfilename, event_name,
                                                    str(antenna_id[ant], 'UTF-8'))
                if ant == 0:
                    efield_loc_arr = np.zeros((n_ant, ) + efield_loc.shape)

                efield_loc_arr[ant] = efield_loc

            index_spike = int(np.round(np.mean(np.argmax(efield_loc_arr[:, :, 2], axis=1))))
            ### We normalize the length of the data
            efield_loc_arr = efield_loc_arr[:, index_spike-250:index_spike + 518, :]
            #We want to start at t=0
            efield_loc_arr[:, :, 0] = efield_loc_arr[:, :, 0] - np.min(efield_loc_arr[:, 0, 0])

            if index_spike < 250:
                print(index_spike)
                raise ValueError("Index spike is to low")

            # We normalize the position of the antennas that are shifted
            # from their normal positions
            antenna_id_to_pos = {}
            normalization_lst = []
            normalization = None
            for antenna in enumerate(antenna_id):
                if antenna in antenna_id_to_pos:
                    normalization = np.array(antenna_pos[antenna[0]]) - np.array(antenna_id_to_pos[antenna[1]])
                    break

            if normalization is None:
                antenna_id_to_pos[antenna_id[0]] = antenna_pos[0]
                normalization = np.zeros((3,))

            for antenna in enumerate(antenna_id):
                if antenna in antenna_id_to_pos:
                    #print(antenna_id_to_pos[antenna] - antenna_pos[ant] - normalization)
                    if (antenna_id_to_pos[antenna[1]] != antenna_pos[antenna[0]] - normalization).all():
                        raise Exception("It can't be normalized")
                else:
                    antenna_id_to_pos[antenna] = antenna_pos[ant] - normalization

            normalization_lst.append(normalization)
            #print(f"number of normalized antennas: {len(antenna_id_to_pos)}")

            t_zero = efield_loc_arr[:, :1, 0]

            antenna_pos_corr = np.array([antenna_id_to_pos[antenna_id[i]]
                                         for i in range(len(antenna_id))])
            obs = np.concatenate((efield_loc_arr[:, :, 1:].reshape(n_ant, -1)/100,
                                  antenna_pos_corr/1000,
                                  t_zero/50_000), axis=-1)

            obs_lst.append(obs)
            label_lst.append(energy)

            edge_index,_ = compute_neighbors(antenna_pos_corr)
            edge_index = np.array(list(edge_index)) #Transform in array
            edge_index_mirrored = edge_index[:, [1, 0]]
            #To have the edges in the 2 ways
            edge_index = np.concatenate((edge_index, edge_index_mirrored), axis=0)
            edge_index = np.unique(edge_index, axis=0) #To remove the duplicates

            graph = tg.data.Data(x=torch.tensor(obs, dtype=torch.float32),
                             edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                             y=torch.tensor(energy, dtype=torch.float32))
            graph_list.append(graph)


        data, slices = self.collate(graph_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Dataset saved to: ", self.processed_paths[0])


if __name__ == '__main__':
    dataset = GrandDataset("GrandDataset",
                           add_degree=False,
                           is_core_contained=False,
                           max_degree=20,
                           distance=1500)
    print(dataset.train_datasets[(25, 25)])
