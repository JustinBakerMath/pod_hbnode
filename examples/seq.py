import argparse
import numpy as np
import warnings

from pod_hbnode.seq.train import train
from pod_hbnode.common import set_seed,set_outdir
from pod_hbnode.datasets import POD_DATASET, SEQ_DATASET, pod_mode_to_true
from pod_hbnode.visualization import information_decay, data_reconstruct, data_animation, plot_loss, plot_adjGrad, plot_nfe, mode_prediction, plot_stiff

warnings.filterwarnings('ignore')

"""INPUT ARGUMETNS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='[NODE] NODE parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [VKS, KPP].')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/seq/',
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
seq_parser = parser.add_argument_group('SEQ Dataset Parameters')
seq_parser.add_argument('--tpred', type = int, default=105,
                    help='Time index for data decomposition.')
seq_parser.add_argument('--tr_ind', type = int, default=75,
                    help='Time index for training data.')
seq_parser.add_argument('--val_ind', type=int, default=100,
                    help='Time index for validation data.' )
seq_parser.add_argument('--eval_ind', type=int, default=200,
                    help='Time index for evaluation data.' )
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--model', type=str, default='HBNODE',
                    help='Model choices - GHBNODE, HBNODE, NODE.')
model_params.add_argument('--batch_size', type=int, default=20,
                help='Time index for validation data.' )
model_params.add_argument('--seq_ind', type=int, default=9,
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
uq_params.add_argument('--plt_itvl', type=int, default=20,
                help='Plot interval')
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

#POD DECOMP
if args.verbose: print("Data Decomposition ...")
pod = POD_DATASET(args)
pod.reduce()
args.load_file = args.out_dir+args.dataset.lower()+'_'+str(args.tstart)+'_'+str(args.tstop)+'_pod_'+str(args.modes)+'.npz'
pod.save_data(args.load_file)
args = pod.args
pod.reconstruct()
temp_mdl = args.model
args.model = 'pod'
information_decay(pod,args)
data_reconstruct(pod.data_recon,args.tpred-1,args)
data_animation(pod.data_recon,args)
args.model = temp_mdl

#SEQ DATASET
seq = SEQ_DATASET(args)

#TRAIN AND PREDICT
predictions,rec_file = train(seq,args)

#OUTPUT
normalized = (predictions*seq.std_data+seq.mean_data)
times = np.arange(seq.data_args.tstart+args.seq_ind,seq.data_args.tstart+args.val_ind)
#DATA PLOTS
verts = [seq.data_args.tstart+args.tr_ind]
mode_prediction(normalized[:,:4],seq.seq_label[-1,:args.val_ind],times,verts,args)
val_recon = pod_mode_to_true(seq.pod_dataset,normalized,args)
data_reconstruct(val_recon,-1,args)
#data_animation(val_recon,args)

#MODEL PLOTS
plot_loss(rec_file, args)
plot_nfe(rec_file,'forward_nfe', args)
plot_adjGrad(rec_file, args)
plot_stiff(rec_file, args)
