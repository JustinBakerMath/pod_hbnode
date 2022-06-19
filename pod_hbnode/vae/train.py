import torch
import torch.optim as optim
from torchdiffeq import odeint
from tqdm import tqdm, trange
import numpy as np
import pickle

from pod_hbnode.vae.models import *
from pod_hbnode.common import Recorder

def train(vae,args):
    #MODEL
    print('Generating ...\t Model: VAE '+args.model)
    obs_dim = vae.train_data.shape[2]
    latent_dim = obs_dim - args.latent_dim
    layers_node = [latent_dim] + args.layers_node + [latent_dim]

    MODELS = {'NODE' : NODE(df = LatentODE(layers_node)),
            'HBNODE' : HBNODE(LatentODE(layers_node))}

    if args.model == "HBNODE":
        latent_dim = latent_dim*2

    rec = Recorder()

    #NETWORKS
    enc = Encoder(latent_dim, obs_dim, args.units_enc, args.layers_enc)
    node = MODELS[args.model]
    dec = Decoder(latent_dim, obs_dim, args.units_dec, args.layers_dec)
    params = (list(enc.parameters()) + list(node.parameters()) + list(dec.parameters()))

    #TRAINNG UTILS
    optimizer = optim.AdamW(params, lr= args.lr)
    loss_meter_t = RunningAverageMeter()
    meter_train = RunningAverageMeter()
    meter_valid = RunningAverageMeter()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=args.factor, patience=5, verbose=False, threshold=1e-5,
                                                    threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
    criterion = torch.nn.MSELoss()
    lossTrain = []
    lossVal = []

    #TRAINING
    print('Training ... \t Iterations: {}'.format(args.epochs))
    epochs = trange(1,args.epochs+1)
    for epoch in epochs:
        rec['epoch'] = epoch

        optimizer.zero_grad()
        #SCHEDULE
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        scheduler.step(metrics=loss_meter_t.avg)

        #FORWARD STEP
        node.nfe=0
        out_enc = enc.forward(vae.obs_t)
        qz0_mean, qz0_logvar = out_enc[:, :latent_dim], out_enc[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size())
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        zt = odeint(node, z0, vae.train_times).permute(1, 0, 2)
        output_vae_t = dec(zt)

        # LOSS
        pz0_mean = pz0_logvar = torch.zeros(z0.size())
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        kl_loss = torch.mean(analytic_kl, dim=0)
        loss = criterion(output_vae_t, vae.train_data) + kl_loss
        epochs.set_description('loss:{:.3f}'.format(loss))
        rec['tr_loss'] = loss
        rec['forward_nfe'] = node.nfe

        #BACK PROP
        node.nfe = 0
        loss.backward()
        optimizer.step()
        loss_meter_t.update(loss.item())
        meter_train.update(loss.item() - kl_loss.item())
        lossTrain.append(meter_train.avg)
        rec['backward_nfe'] = node.nfe

        #VALIDATION
        with torch.no_grad():

            enc.eval()
            node.eval()
            dec.eval()

            node.nfe = 0
            zv = odeint(node, z0, vae.valid_times).permute(1, 0, 2)
            output_vae_v = dec(zv)

            loss_v = criterion(output_vae_v[:, args.tr_ind:],
                                vae.valid_data[:, args.tr_ind:])

            meter_valid.update(loss_v.item())
            lossVal.append(meter_valid.avg)

            rec['val_nfe'] = node.nfe
            rec['val_loss'] = loss_v

            enc.train()
            node.train()
            dec.train()

        #OUTPUT
        if epoch % args.epochs == 0:
            output_vae = (output_vae_v.cpu().detach().numpy()) * vae.std_data + vae.mean_data
        if np.isnan(lossTrain[epoch - 1]):
            break
        rec.capture(verbose=False)


    #SAVE MODEL DATA
    rec_file = args.out_dir+ './pth/'+args.model+'.csv'
    rec.writecsv(rec_file)
    torch.save(enc.state_dict(), args.out_dir + './pth/enc.pth')
    torch.save(node.state_dict(), args.out_dir + './pth/node.pth')
    torch.save(dec.state_dict(), args.out_dir + './pth/dec.pth')

    #FORWARD STEP TEST DATA
    with torch.no_grad():

        enc.eval()
        node.eval()
        dec.eval()

        ze = odeint(node, z0, vae.eval_times).permute(1, 0, 2)
        output_vae_e = dec(ze)

        enc.train()
        node.train()
        dec.train()

    #SAVE TEST DATA
    data_NODE = (output_vae_e.cpu().detach().numpy()) * vae.std_data + vae.mean_data
    with open(args.out_dir + './pth/vae_'+args.model+'_modes.pth', 'wb') as f:
        pickle.dump(data_NODE, f)

    #INVERT OVER TIME
    idx = [i for i in range(vae.valid_data.size(0) - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    obs_t = vae.valid_data.index_select(0, idx)
    out_enc = enc.forward(obs_t)
    qz0_mean, qz0_logvar = out_enc[:, :latent_dim], out_enc[:, latent_dim:]
    epsilon = torch.randn(qz0_mean.size())
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    zt = odeint(node, z0, vae.valid_times).permute(1, 0, 2)
    predictions = dec(zt).detach().numpy()
    return predictions, rec_file
