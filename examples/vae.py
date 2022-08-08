import argparse
import numpy as np
import warnings

from pod_hbnode.vae.train import train
from pod_hbnode.common import set_seed,set_outdir
from pod_hbnode.datasets import POD_DATASET, VAE_DATASET, pod_mode_to_true
from pod_hbnode.visualization import information_decay, data_reconstruct, data_animation, mode_prediction, plot_loss, plot_nfe
warnings.filterwarnings('ignore')

"""INPUT ARGUMETNS"""
parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='[NODE] NODE parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset types: [VKS, EE].')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd: sci.')
data_parser.add_argument('--out_dir', type=str, default='./out/vae_examples/',
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
vae_parser = parser.add_argument_group('VAE Dataset Parameters')
vae_parser.add_argument('--tpred', type = int, default=105,
                    help='Time index for data decomposition.')
vae_parser.add_argument('--tr_ind', type = int, default=75,
                    help='Time index for training data.')
vae_parser.add_argument('--val_ind', type=int, default=100,
                    help='Time index for validation data.' )
vae_parser.add_argument('--eval_ind', type=int, default=200,
                    help='Time index for evaluation data.' )
model_parser = parser.add_argument_group('Model Parameters')
model_parser.add_argument('--model', type=str, default='NODE',
                    help='Dataset types: [NODE , HBNODE].')
model_parser.add_argument('--epochs', type=int, default=2000,
                    help='Training epochs.')
model_parser.add_argument('--latent_dim', type=int, default=6,
                    help = 'Size of latent dimension')
model_parser.add_argument('--layers_enc', type=int, default=4,
                help='Encoder Layers.')
model_parser.add_argument('--units_enc', type=int, default=10,
                    help='Encoder units.')
model_parser.add_argument('--layers_node', type=int, default=[12],
                nargs='+', help='NODE Layers.')
model_parser.add_argument('--units_dec', type=int, default=41,
                    help='Training iterations.')
model_parser.add_argument('--layers_dec', type=int, default=4,
                help='Encoder Layers.')
model_parser.add_argument('--lr', type=float, default=0.00153,
                    help = 'Initial learning rate.')
model_parser.add_argument('--factor', type=float, default=0.99,
                    help = 'Factor for reducing learning rate.')
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--seed', type=int, default=1242,
                help='Set initialization seed')
uq_params.add_argument('--plt_itvl', type=int, default=20,
                help='Plot interval')
uq_params.add_argument('--verbose', default=False, action='store_true',
                help='Display full NN and all plots.')
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
if args.load_file is None:
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

#VAE DATASET
vae = VAE_DATASET(args)

#TRAIN AND PREDICT
print(args.model)
predictions, rec_file = train(vae,args)

#OUTPUT
args.modes = vae.data_args.modes
args.model = str('vae_'+args.model).lower()
normalized = (predictions*vae.std_data+vae.mean_data)
times = np.arange(vae.data_args.tstart,vae.data_args.tstart+args.val_ind)
verts = [vae.data_args.tstart+args.tr_ind]
mode_prediction(normalized[-1,:,:4],vae.data[:args.val_ind],times,verts,args)
val_recon = pod_mode_to_true(vae.pod_dataset,normalized,args)
data_reconstruct(val_recon,args.val_ind-1,args)
data_animation(val_recon,args)

plot_loss(rec_file, args)
#plot_nfe(rec_file,'forward_nfe', args)
#plot_nfe(rec_file,'backward_nfe', args)
