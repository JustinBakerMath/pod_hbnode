import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import trange

from .models import *
from ..common import *

def train(seq,args):

    MODELS = {'NODE' : NMODEL(args),'HBNODE' : HBMODEL(args, res=True, cont=True), 'GHBNODE' : GHBMODEL(args, res=True, cont=True)}

    #MODEL DIMENSIONS
    assert args.model in MODELS
    print('Generating ...\t Model: SEQ {}'.format(args.model))
    model = MODELS[args.model].to(args.device)
    if args.verbose:
        print(model.__str__())
        print('Number of Parameters: {}'.format(count_parameters(model)))

    #LEARNING UTILITIES
    gradrec = True
    torch.manual_seed(0)
    rec = Recorder()
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_meter_t = RunningAverageMeter()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                    factor=args.factor, patience=args.patience, verbose=False, threshold=1e-5,
                    threshold_mode='rel', cooldown=args.cooldown, min_lr=1e-7, eps=1e-08)

    # TRAINING
    print('Training ... \t Iterations: {}'.format(args.epochs))
    epochs = trange(1,args.epochs+1)
    for epoch in epochs:

        rec['epoch'] = epoch
        batchsize = args.batch_size
        train_start_time = time.time()

        #SCHEDULER
        for param_group in optimizer.param_groups:
            rec['lr'] = param_group['lr']
        scheduler.step(metrics=loss_meter_t.avg)

        #BATCHING
        for b_n in range(0, seq.train_data.shape[1], batchsize):
            model.cell.nfe = 0
            predict = model(seq.train_times[:, b_n:b_n + batchsize], seq.train_data[:, b_n:b_n + batchsize])
            loss = criteria(predict, seq.train_label[:, b_n:b_n + batchsize])
            loss_meter_t.update(loss.item())
            rec['tr_loss'] = loss
            rec['forward_nfe'] = model.cell.nfe
            rec['forward_stiff'] = model.cell.stiff
            epochs.set_description('loss:{:.3f}'.format(loss))

            #BACKPROP
            if gradrec is not None:
                lossf = criteria(predict[-1], seq.train_label[-1, b_n:b_n + batchsize])
                lossf.backward(retain_graph=True)
                vals = model.ode_rnn.h_rnn
                for i in range(len(vals)):
                    grad = vals[i].grad
                    rec['grad_{}'.format(i)] = 0 if grad is None else torch.norm(grad)
                model.zero_grad()
            model.cell.nfe = 0
            loss.backward()
            optimizer.step()
            rec['backward_nfe'] = model.cell.nfe
            rec['backward_stiff'] = model.cell.stiff
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        rec['train_time'] = time.time() - train_start_time

        #VALIDATION
        if epoch == 0 or (epoch + 1) % 1 == 0:
            model.cell.nfe = 0
            predict = model(seq.valid_times, seq.valid_data)
            vloss = criteria(predict, seq.valid_label)
            rec['val_nfe'] = model.cell.nfe
            rec['val_stiff'] = model.cell.stiff
            rec['val_loss'] = vloss

        #TEST
    #    if epoch == 0 or (epoch + 1) % 5 == 0:
    #        model.cell.nfe = 0
    #        predict = model(seq.eval_times, seq.eval_data)
    #        sloss = criteria(predict, seq.eval_label)
    #        sloss = sloss.detach().cpu().numpy()
    #        rec['ts_nfe'] = model.cell.nfe
    #        rec['ts_stiff'] = model.cell.stiff
    #        rec['ts_loss'] = sloss

        #OUTPUT
        rec.capture(verbose=False)
        if (epoch + 1) % 5 == 0:
            torch.save(model, args.out_dir+'/pth/{}.mdl'.format(args.model))
            rec.writecsv(args.out_dir+'/pth/{}.csv'.format(args.model))


    print("Generating Output ... ")
    rec_file = args.out_dir+ './pth/'+args.model+'.csv'
    rec.writecsv(rec_file)
    args.modes = seq.data_args.modes
    args.model = str('seq_'+args.model).lower()
    tr_pred= model(seq.train_times, seq.train_data).cpu().detach().numpy()[-1]
    val_pred = model(seq.valid_times, seq.valid_data).cpu().detach().numpy()[-1]
    predictions=np.vstack((tr_pred,val_pred))
    return predictions, rec_file
