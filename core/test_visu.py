import numpy as np
import torch
import tkinter
import matplotlib.pyplot as plt
import matplotlib
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import glob
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from tqdm import tqdm
import torch_geometric as tg
from utils import computeNeighbors, computeNeighborsKDTree

PATH_data = './GRAND_DATA/GP300Outbox/'
progenitor = 'Proton'
zenVal = '_' + str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1
gui = 'TKAgg'

matplotlib.use(gui)


list_f = glob.glob(PATH_data+'*'+progenitor+'*'+zenVal+'*')
# list_f = glob.glob(PATH_data+'*')
print('Number of files = %i' % (len(list_f)))

efield_loc_len = []
n_samples = 0
for file in tqdm(range(len(list_f))):
    inputfilename = glob.glob(list_f[file] + '/*' + progenitor + '*' + zenVal + '*.hdf5')[0]
    run_info = hdf5io.GetRunInfo(inputfilename)
    event_name = hdf5io.GetEventName(run_info, 0)
    antenna_info = hdf5io.GetAntennaInfo(inputfilename, event_name)
    n_ant = hdf5io.GetNumberOfAntennas(antenna_info) #=len(antenna_info)
    energy = run_info['Energy'][0]
    zenith = 180. - hdf5io.GetEventZenith(run_info, 0)
    azimuth = hdf5io.GetEventAzimuth(run_info, 0)-180.
    
    n_samples += n_ant
    lstPositions = []
    antenna_id = antenna_info["ID"].value
    antenna_pos = np.concatenate((antenna_info['X'].value[:, np.newaxis], antenna_info['Y'].value[:, np.newaxis], antenna_info['Z'].value[:, np.newaxis]), axis=-1)
    for ant in range(n_ant):
        efield_loc = hdf5io.GetAntennaEfield(inputfilename, event_name,
                                            str(antenna_id[ant], 'UTF-8'))
        efield_loc_len.append(len(efield_loc))
        
    efield_loc_len.append(len(efield_loc))
    
    edge_index = computeNeighbors(antenna_pos)
    edge_index = np.array(list(edge_index)) #Transform in array 
    edge_index_mirrored = edge_index[:, [1, 0]]
    edge_index = np.concatenate((edge_index, edge_index_mirrored), axis=0) #To have the edges in the 2 ways
    edge_index = np.unique(edge_index, axis=0) #To remove the duplicates
    #print(len(edge_index), edge_index)
        
    plt.plot(efield_loc[:, 0], efield_loc[:, 1], label="X")
    plt.plot(efield_loc[:, 0], efield_loc[:, 2], label="Y")
    plt.plot(efield_loc[:, 0], efield_loc[:, 3], label="Z")
    plt.legend()
    plt.title("Efield")
    plt.xlabel("time (ps)")
    plt.ylabel("Amplitude (Pev)")
    plt.show(block=True)
    fig = plt.figure()
    plt.scatter(antenna_pos[:, 0], antenna_pos[:, 1])
    plt.title("Antennas positions")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    fig = plt.figure()
    plt.scatter(antenna_pos[:, 0], antenna_pos[:, 2])
    plt.title("Antennas positions")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    fig = plt.figure()
    plt.scatter(antenna_pos[:, 1], antenna_pos[:, 2])
    plt.title("Antennas positions")
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.show()
    
    G = tg.data.Data(x=torch.tensor(antenna_pos, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), y=[zenith, azimuth])
    
    data_graphs = [G, G, G]
    loader = tg.loader.DataLoader(data_graphs, batch_size=2, shuffle=2)
    batch = next(iter(loader))
    
    fig = plt.figure()

    lines = []
    for edge in edge_index:
        lines.append([antenna_pos[edge[0]], antenna_pos[edge[1]]])
    lc = Line3DCollection(lines, linewidth=2)    
    
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
    
    ax.scatter(antenna_pos[:, 0], antenna_pos[:, 1], antenna_pos[:, 2])
    ax.add_collection(lc)
    ax.set_title('3D Graph')
    plt.show(block=True)
    
print(np.min(efield_loc_len))
print(np.max(efield_loc_len))