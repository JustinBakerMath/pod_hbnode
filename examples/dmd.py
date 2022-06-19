#IMPORTS
import argparse
import numpy as np
from tqdm import trange
import warnings

import pod_hbnode
from pod_hbnode.common import set_outdir
from pod_hbnode.dmd_model import train
from pod_hbnode.visualization import  information_decay, data_reconstruct, data_animation

#SETTINGS
warnings.filterwarnings('ignore')

"""MODEL ARGUMENTS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='DMD parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset to load.')
data_parser.add_argument('--load_file', type=str, default=None,
                    help='Directory to load DMD data from.')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd.')
data_parser.add_argument('--out_dir', type=str, default='./out/dmd_examples/',
                    help='Directory of output from cwd.')
decomp_parser = parser.add_argument_group('Decomposition Parameters')
decomp_parser.add_argument('--modes', type = int, default = 64,
                    help = 'DMD reduction modes.\nNODE model parameters.')
decomp_parser.add_argument('--tstart', type = int, default=0,
                    help='Start time for reduction along time axis.')
decomp_parser.add_argument('--tstop', type=int, default=101,
                    help='Stop time for reduction along time axis.' )
decomp_parser.add_argument('--tpred', type=int, default=400,
                    help='Prediction time.' )
decomp_parser.add_argument('--lifts', type=str, default='', nargs='+',
                    choices=['sin','cos','quad','cube'],
                    help='Lifts for dmd datase.' )
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--verbose', default=False, action='store_true',
                help='To display output or not.')
args, unknown = parser.parse_known_args()

if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))
args.model ='dmd'

"""FORMATTING OUT DIR"""
set_outdir(args.out_dir, args)

Xk, dmd = train(args)

"""OUTPUT"""
if args.verbose: print("Generating Output ...\n",Xk.shape)
information_decay(dmd,args)
data_reconstruct(dmd.data_recon,args.tpred-1,args)
data_animation(dmd.data_recon,args)

