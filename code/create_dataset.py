import torch
from torch_geometric.data import InMemoryDataset, Data
import glob 
from tqdm import tqdm
import hdf5fileinout as hdf5io
import numpy as np
import torch_geometric as tg

from utils import computeNeighbors, computeNeighborsKDTree
import matplotlib.pyplot as plt


class GrandDataset(InMemoryDataset):
    def __init__(self, root= "./GrandDataset"):
        super().__init__(root)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        graph_list = []
        
        PATH_data = './GRAND_DATA/GP300Outbox/'
        progenitor = 'Proton'
        zenVal = '_' + str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1
        list_f = glob.glob(PATH_data+'*'+progenitor+'*'+zenVal+'*')
        # list_f = glob.glob(PATH_data+'*')
        print('Number of files = %i' % (len(list_f)))

        obs_lst = []
        label_lst = []
        n_samples = 0
        for file in tqdm(range(len(list_f))):
            ### We load the files
            inputfilename = glob.glob(list_f[file] + '/*' + progenitor + '*' + zenVal + '*.hdf5')[0]
            run_info = hdf5io.GetRunInfo(inputfilename)
            event_name = hdf5io.GetEventName(run_info, 0)
            antenna_info = hdf5io.GetAntennaInfo(inputfilename, event_name)
            n_ant = hdf5io.GetNumberOfAntennas(antenna_info) #=len(antenna_info)
            energy = run_info['Energy'][0]
            zenith = 180. - hdf5io.GetEventZenith(run_info, 0)
            azimuth = hdf5io.GetEventAzimuth(run_info, 0)-180.
            
            n_samples += n_ant
            antenna_id = antenna_info["ID"].value
            antenna_pos = np.concatenate((antenna_info['X'].value[:, np.newaxis], antenna_info['Y'].value[:, np.newaxis], antenna_info['Z'].value[:, np.newaxis]), axis=-1)
            for ant in range(n_ant):
                efield_loc = hdf5io.GetAntennaEfield(inputfilename, event_name,
                                                    str(antenna_id[ant], 'UTF-8'))
                if ant == 0:
                    efield_loc_arr = np.zeros((n_ant, ) + efield_loc.shape)
                    
                efield_loc_arr[ant] = efield_loc
            
            ### We compute the features
            
            time_diff = - (efield_loc_arr[:, :, 0][np.arange(len(efield_loc_arr)), np.argmax(efield_loc_arr[:, :, 2], axis=1)] - efield_loc_arr[:, :, 0][np.arange(len(efield_loc_arr)), np.argmin(efield_loc_arr[:, :, 2], axis=1)])
            peak_to_peak_energy = np.max(efield_loc_arr[:, :, 1:], axis=1) - np.min(efield_loc_arr[:, :, 1:], axis=1)
            peak_to_peak_energy_first = np.argmax(efield_loc_arr[:, :, 1:], axis=1)
            
            amplitude = np.sqrt(efield_loc_arr[:, :, 1] ** 2 + efield_loc_arr[:, :, 2] ** 2 + efield_loc_arr[:, :, 3] ** 2)
            peak_to_peak_amplitude = np.expand_dims(np.max(amplitude, axis=1) - np.min(amplitude, axis=1), axis=-1)
            
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
    
            antenna_pos_corr = np.array([antenna_id_to_pos[antenna_id[i]] for i in range(len(antenna_id))])
            obs = np.concatenate((antenna_pos_corr/1000, 
                                  peak_to_peak_energy/100,
                                  efield_loc_arr[:, :, 0][np.expand_dims(np.arange(len(efield_loc_arr)), axis=-1), peak_to_peak_energy_first]/10_000, ## TODO give relative time
                                  np.expand_dims(time_diff, axis=-1)/100
                                  ), axis=-1)
            
            obs_lst.append(obs)
            label_lst.append(energy)
            
            edge_index = computeNeighbors(antenna_pos_corr)
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
        
        PATH_data = './GRAND_DATA/GP300Outbox/'
        progenitor = 'Proton'
        zenVal = '_' + str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1
        list_f = glob.glob(PATH_data+'*'+progenitor+'*'+zenVal+'*')
        # list_f = glob.glob(PATH_data+'*')
        print('Number of files = %i' % (len(list_f)))

        obs_lst = []
        label_lst = []
        n_samples = 0
        for file in tqdm(range(len(list_f))):
            ### We load the files
            inputfilename = glob.glob(list_f[file] + '/*' + progenitor + '*' + zenVal + '*.hdf5')[0]
            run_info = hdf5io.GetRunInfo(inputfilename)
            event_name = hdf5io.GetEventName(run_info, 0)
            antenna_info = hdf5io.GetAntennaInfo(inputfilename, event_name)
            n_ant = hdf5io.GetNumberOfAntennas(antenna_info) #=len(antenna_info)
            energy = run_info['Energy'][0]
            
            n_samples += n_ant
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
            
            edge_index = computeNeighbors(antenna_pos_corr)
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
"""
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']
"""        
        

if __name__ == '__main__':
    dataset = GrandDatasetSignal()