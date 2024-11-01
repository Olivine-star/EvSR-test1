import glob
import os
from utils import prepare_result_dir
import configs
import NSSR
import NTSR
import evs_processing
import torch
import numpy as np



def main():
    conf = configs.Config()
    conf.result_path = prepare_result_dir(conf)

    sr = evs_processing.EVSProcessing(conf.file_dir, conf.file_name, conf.dvs_size)

    evs_voxel, t_data, evs = sr.prepare_data()
    print('Raw Events = ', evs.shape)
    np.save(conf.result_path+'/'+'raw_evs.npy', evs[:,[1,2,3,5,0,4]])
    print('Raw Event Voxel = ', evs_voxel.shape)
    print('Temporal In/Out Data = ', t_data[0].shape, t_data[1].shape,'\n')

    sr.init_seed(conf.seed)
    evs_voxel_sr = NSSR.NSSR(evs_voxel, conf).run()
    print('SR Event Voxel = ', evs_voxel_sr.shape)

    t_data_sr_in = sr.voxel2evs(evs_voxel_sr, low=conf.cutoff_low, high=conf.cutoff_high)
    print('Temporal SR In Data = ', t_data_sr_in.shape)

    sr.init_seed(conf.seed)
    t_sr = NTSR.NTSR(t_data[0], t_data[1], t_data_sr_in, conf)
    
    t_sr.run()
    t_data_sr_out = t_sr.inference()
    print('Temporal SR Out Data = ', t_data_sr_out.shape)

    evs_sr = sr.sr_evs_reshape(t_data_sr_in, t_data_sr_out, conf.scale_factors)
    print('SR Events = ', evs_sr.shape)
    np.save(conf.result_path+'/'+'sr_evs.npy', evs_sr)



if __name__ == '__main__':
    main()
