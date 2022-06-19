import csv
import numpy as np
import os
import pickle
import torch

# DIRECTORY UTILS
def set_outdir(OUTPUT_DIR, args=None):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR+'./pth/'):
        os.makedirs(OUTPUT_DIR+'./pth/')
    if not (args is None):
        top_str = args.model+'_'+args.dataset+'_'
        with open(OUTPUT_DIR + '/pth/'+top_str+'.pth', 'wb') as f:
            pickle.dump({' Model Arguments' : args}, f)

def set_seed(se):
    """ set the seeds to have reproducible results"""
    torch.manual_seed(se)
    torch.cuda.manual_seed_all(se)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(se)
    os.environ['PYTHONHASHSEED'] = str(se)

#Record Training Data
class Recorder:
    def __init__(self):
        self.store = []
        self.current = dict()

    def __setitem__(self, key, value):
        for method in ['detach', 'cpu', 'numpy']:
            if hasattr(value, method):
                value = getattr(value, method)()
        if key in self.current:
            self.current[key].append(value)
        else:
            self.current[key] = [value]

    def capture(self, verbose=False):
        for i in self.current:
            self.current[i] = np.mean(self.current[i])
        self.store.append(self.current.copy())
        self.current = dict()
        if verbose:
            for i in self.store[-1]:
                if i[0] != '_':
                    print('{}: {}'.format(i, self.store[-1][i]))
        return self.store[-1]

    def tolist(self):
        labels = set()
        labels = sorted(labels.union(*self.store))
        outlist = []
        for obs in self.store:
            outlist.append([obs.get(i, np.nan) for i in labels])
        return labels, outlist

    def writecsv(self, writer):
        labels, outlist = self.tolist()
        if isinstance(writer, str):
            outfile = open(writer, 'w')
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)
            outfile.close()
        else:
            csvwriter = writer
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)
