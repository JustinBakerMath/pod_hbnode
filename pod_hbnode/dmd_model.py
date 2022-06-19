import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from .datasets import *

def train(args):
    #LOAD DATA
    dmd = DMD_DATASET(args)
    if args.load_file is None:
        dmd.reduce()
        dmd.save_data(args.out_dir+'/pth/'+args.dataset.lower()+'_'+str(args.tstart)+'_'+str(args.tstop)+'_dmd_'+str(args.modes)+'.npz')
    args = dmd.args
    #INITIALIZE
    Xk = np.array(dmd.X.T[0][:dmd.domain_len*dmd.component_len])
    #GENERATE PREDICTIONS
    if args.verbose:
        pbar=trange(1,args.tpred, desc='DMD Generation')
    else:
        pbar=range(1,args.tpred)
    for k in pbar:
        Lambda_k = np.linalg.matrix_power(dmd.Lambda,k)
        xk=(dmd.Phi@Lambda_k@dmd.b)[:dmd.domain_len*dmd.component_len]
        Xk=np.vstack((Xk,xk))
    #RECONSTRUCTION
    Xk = np.array(Xk)
    dmd.data_recon = Xk
    dmd.reconstruct()
    return Xk, dmd
