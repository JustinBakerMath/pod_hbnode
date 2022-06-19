import pandas as pd
import numpy as np
import os
import torch

DATA_DIR = './data/fiber/'
OUT_DIR = './data/'

def FIB_DAT(out_data_dir, tws_data_dir, speed_data_dir, param=None):
   
    fileout = out_data_dir.replace("out", "full")
    fileout = fileout.replace('.dat','')
    
    # non-transient data from out_pde.dat
    data_transient = pd.read_table(out_data_dir, sep="\t", index_col=2, names=["x", "h"]).to_numpy()
    end = data_transient.shape[0]//401
    data_transient = data_transient[:,1].reshape(end,401)
    data_transient = np.delete(data_transient,400,1) # Temporary - delete last column so that it is the same shape as the tws data

    raw_data = np.loadtxt(tws_data_dir)[:,1]
    
    speed_data = np.loadtxt(speed_data_dir)
    data_transient = speed_data[-1]*data_transient[:-10]/np.max(data_transient)
    # Create moving wave (down the rows of data_tensor)
    vector = speed_data[-1]*raw_data/np.max(raw_data)
    data_nonT = vector
    for i in range(140):
        vector = np.roll(vector,int(speed_data[0]/speed_data[1]))
        data_nonT = np.vstack((data_nonT,vector))
    
    data_tensor = np.vstack((data_transient,data_nonT))    
    np.savez(fileout, data_tensor)
    return 1

if __name__ == '__main__':

    for filename1 in os.listdir(DATA_DIR):
        if 'out' in filename1:
            # Found a transient dataset
            out_data_dir = os.path.join(DATA_DIR, filename1)
            
            # now identify which transient dataset this is
            identifier = filename1.split('_')[1] # number associated with the L parameter
            
            # Next, find the tws dataset that corresponds to this transient data
            matches = ['tws', identifier]
            
            for filename2 in os.listdir(DATA_DIR):
                if all(x in filename2 for x in matches):
                    # Found the corresponding tws dataset
                    tws_data_dir = os.path.join(DATA_DIR, filename2)
                    FIB_DAT(out_data_dir, tws_data_dir, DATA_DIR+'spd_'+identifier)
        data_list = []
        for filename3 in os.listdir(DATA_DIR):
            if 'full' in filename3:
                filepath = os.path.join(DATA_DIR,filename3)
                array_temp = np.load(filepath,allow_pickle=True)['arr_0']
                data_list.append(array_temp)

    fiber_param = np.stack(data_list,axis=0)
    fileout = os.path.join(OUT_DIR, 'FIB')
    np.savez(fileout, fiber_param)
