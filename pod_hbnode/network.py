import numpy as np
import pandas as pd
import pickle
import torch

"""STANDARD DATA RETRIEVAL
    - Format dimensions as [t,[omega],k] e.g. [t,[xx,yy,zz],k]
"""
def VKS_DAT(data_file, param=None):
  with open(data_file, 'rb') as f:
    vks = pickle.load(f)
  vks = np.nan_to_num(vks)
  vks = np.moveaxis(vks, 2,0)
  return vks[:,:,:,:2]

def EE_DAT(data_dir, param=0):
  npzdata = np.load(data_dir)
  rho, u, E, x, params, t = npzdata['arr_0'], npzdata['arr_1'], npzdata['arr_2'], npzdata['arr_3'], npzdata['arr_4'], npzdata['arr_5']
  ee = np.array([rho[:,:,param], u[:,:,param], E[:,:,param]], dtype=np.double)
  ee=np.moveaxis(ee,0,-1)
  return ee

def FIB_DAT(data_dir, param=None):
  data = pd.read_table(data_dir, sep="\t", index_col=2, names=["x", "h"]).to_numpy()
  end = data.shape[0]//401
  return data[:,1].reshape(end,401)

def KPP_DAT(data_dir, param=None):
  npdata = np.load(data_dir,allow_pickle=True)
  xv, yv, kpp = npdata['arr_0'], npdata['arr_1'], npdata['arr_2']
  kpp = np.moveaxis(kpp, 2,0)
  return kpp

def FIB_DAT(data_dir, param=None):
    return np.load(data_dir)['arr_0']


"""PARAMETERIZED DATA RETRIEVAL
    - Format dimensions as [t,[omega],k,mu] e.g. [t,[xx,yy,zz],k,mu]
"""
def EE_PARAM(data_dir):
  npzdata = np.load(data_dir)
  rho, u, E, x, params, t = npzdata['arr_0'], npzdata['arr_1'], npzdata['arr_2'], npzdata['arr_3'], npzdata['arr_4'], npzdata['arr_5']
  ee = np.array([rho, u, E], dtype=np.double)
  return ee.T

LOADERS = {'VKS':VKS_DAT, 'EE': EE_PARAM, 'FIB' : FIB_DAT, 'KPP': KPP_DAT}
