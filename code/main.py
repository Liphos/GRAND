import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import glob

import torch_geometric 
import torch

PATH_data = './GRAND_DATA/GP300Outbox/'
progenitor = 'Proton'
zenVal = '_'+str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1
gui = 'TKAgg'

matplotlib.use(gui)

list_f = glob.glob(PATH_data+'*'+progenitor+'*'+zenVal+'*')
# list_f = glob.glob(PATH_data+'*')
print('Number of files = %i' % (len(list_f)))

n_samples = 0
for file in range(len(list_f)):
    inputfilename = glob.glob(list_f[file] + '/*' + progenitor + '*' + zenVal + '*.hdf5')[0]
    RunInfo = hdf5io.GetRunInfo(inputfilename)
    EventName = hdf5io.GetEventName(RunInfo, 0)
    AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
    n_ant = hdf5io.GetNumberOfAntennas(AntennaInfo)
    energy = RunInfo['Energy'][0]
    zenith = 180.-hdf5io.GetEventZenith(RunInfo, 0)
    azimuth = hdf5io.GetEventAzimuth(RunInfo, 0)-180.
    
    n_samples += n_ant
    
    for ant in range(n_ant):
        AntennaID = hdf5io.GetAntennaID(AntennaInfo, ant)
        efield_loc = hdf5io.GetAntennaEfield(inputfilename, EventName, AntennaID)