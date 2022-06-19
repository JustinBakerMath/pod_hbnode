import argparse
import numpy as np
import warnings

from pod_hbnode.param.train import train
from pod_hbnode.common import set_seed,set_outdir
from pod_hbnode.datasets import POD_DATASET, PARAM_DATASET, pod_mode_to_true
from pod_hbnode.visualization import information_decay, data_reconstruct, data_animation, plot_loss, plot_adjGrad, plot_nfe, mode_prediction, plot_stiff

warnings.filterwarnings('ignore')

"""INPUT ARGUMETNS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='[NODE] NODE parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [EE, FIB].')
data_parser.add_argument('--data_dir', type=str, default='./data/EulerEqs.npz',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/param/',
                    help='Directory of output from cwd: sci.')
pod_parser = parser.add_argument_group('Decomposition Parameters')
pod_parser.add_argument('--load_file', type=str, default=None,
                    help='Directory of pre-computed POD data.')
pod_parser.add_argument('--tstart', type = int, default=100,
                    help='Time index for data decomposition.')
pod_parser.add_argument('--tstop', type = int, default=300,
                    help='Time index for data decomposition.')
pod_parser.add_argument('--modes', type = int, default=8,
                    help='Decomposition modes.')
param_parser = parser.add_argument_group('PARAM Dataset Parameters')
param_parser.add_argument('--tpred', type = int, default=105,
                    help='Time index for data decomposition.')
param_parser.add_argument('--tr_ind', type = int, default=75,
                    help='Time index for training data.')
param_parser.add_argument('--val_ind', type=int, default=100,
                    help='Time index for validation data.' )
param_parser.add_argument('--eval_ind', type=int, default=200,
                    help='Time index for evaluation data.' )
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--model', type=str, default='HBNODE',
                    help='Model choices - GHBNODE, HBNODE, NODE.')
model_params.add_argument('--batch_size', type=int, default=20,
                help='Time index for validation data.' )
model_params.add_argument('--param_ind', type=int, default=9,
                help='Time index for validation data.' )
model_params.add_argument('--layers', type = int, default=12,
                    help = 'Number of hidden layers.')
model_params.add_argument('--corr', type=int, default=0,
                    help='Skip gate input into soft max function.')
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--epochs', type=int, default=500,
                    help='Training epochs.')
train_params.add_argument('--lr', type=float, default=0.001,
                    help = 'Initial learning rate.')
train_params.add_argument('--factor', type=float, default=0.99,
                    help = 'Initial learning rate.')
train_params.add_argument('--cooldown', type=int, default=0,
                    help = 'Initial learning rate.')
train_params.add_argument('--patience', type=int, default=5,
                    help = 'Initial learning rate.')
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--device', type=str, default='cpu',
                help='Set default torch hardware device.')
uq_params.add_argument('--seed', type=int, default=0,
                help='Set initialization seed')
uq_params.add_argument('--eeParam', type=int, default=1,
                help='Set initialization seed')
uq_params.add_argument('--verbose', default=False, action='store_true',
                help='Number of display modes.')

args, unknown = parser.parse_known_args()
if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))

args.eeParam=None

# SEED
set_seed(args.seed)
set_outdir(args.out_dir, args)

#PARAM DATASET
param = PARAM_DATASET(args)

trained_true = np.vstack((param.train_data[:args.tr_ind],param.train_label[-param.label_len:]))
validated_true = np.vstack((param.valid_data[:args.tr_ind],param.valid_label[-param.label_len:]))
data_true = np.hstack((trained_true,validated_true))*param.std_data+param.mean_data

#TRAIN AND PREDICT
predictions,rec_file = train(param,args)

#DATA PLOTS
verts = [args.tstart+args.tr_ind]
times = np.arange(args.tstart,args.tstop)
mode_prediction(predictions[:,args.param_ind+2,:4],data_true[:,args.param_ind+2,:4],times,verts,args,'_val')
mode_prediction(predictions[:,0,:4],data_true[:,0,:4],times,verts,args)
#data_animation(val_recon,args)

#MODEL PLOTS
plot_loss(rec_file, args)
plot_nfe(rec_file,'forward_nfe', args)
