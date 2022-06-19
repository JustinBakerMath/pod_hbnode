import argparse
from pod_hbnode.common import set_outdir
from pod_hbnode.network import LOADERS, EE_PARAM
from pod_hbnode.visualization import data_reconstruct, data_animation
import warnings

#SETTINGS
warnings.filterwarnings('ignore')
set_outdir('./out/')


parser = argparse.ArgumentParser(prefix_chars='-+/',
    description='Base line data parameters.')
data_parser = parser.add_argument_group('Data Parameters')
data_parser.add_argument('--dataset', type=str, default='VKS',
                    help='Dataset to load.')
data_parser.add_argument('--load_file', type=str, default=None,
                    help='Directory to load DMD data from.')
data_parser.add_argument('--data_dir', type=str, default='./data/VKS.pkl',
                    help='Directory of data from cwd.')
data_parser.add_argument('--out_dir', type=str, default='./out/baseline_examples/',
                    help='Directory of output from cwd.')
uq_params = parser.add_argument_group('Unique Parameters')
uq_params.add_argument('--time', type=int, default=101,
                help='Time to plot static image.')
uq_params.add_argument('--verbose', default=False, action='store_true',
                help='To display output or not.')
args, unknown = parser.parse_known_args()
args.model = ''

if args.verbose:
    print('Parsed Arguments')
    for arg in vars(args):
        print('\t',arg, getattr(args, arg))

set_outdir(args.out_dir, args)

data = LOADERS[args.dataset](args.data_dir)
data_reconstruct(data,args.time,args)
data_animation(data,args)
