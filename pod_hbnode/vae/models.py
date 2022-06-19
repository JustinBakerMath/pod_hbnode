import torch.nn as nn
import torch

# LEARNING UTILS
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
    def reset(self):
        self.val = None
        self.avg = 0
    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class NLayerNN(nn.Module):
    def __init__(self, *args, actv=nn.ReLU()):
        super().__init__()
        self.linears = nn.ModuleList()
        for i in range(len(args)):
            self.linears.append(nn.Linear(args[i], args[i+1]))
        self.actv = actv

    def forward(self, x):
        for i in range(self.layer_cnt):
            x = self.linears[i](x)
            if i < self.layer_cnt - 1:
                x = self.actv(x)
        return x


class Zeronet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

zeronet = Zeronet()

class TVnorm(nn.Module):
    def __init__(self):
        super(TVnorm, self).__init__()
        self.osize = 1

    def forward(self, t, x, v):
        return torch.norm(v, 1)

class NormAct(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = bound
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, x):
        x = x - self.bound + 1
        x = self.relu(x) * self.elu(-x) + 1
        return x


class Parameter(nn.Module):
    def __init__(self, val, frozen=False):
        super().__init__()
        val = torch.Tensor(val)
        self.val = val
        self.param = nn.Parameter(val)
        self.frozen = frozen

    def forward(self):
        if self.frozen:
            self.val = self.val.to(self.param.device)
            return self.val
        else:
            return self.param

    def freeze(self):
        self.val = self.param.detach().clone()
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return "val: {}, param: {}".format(self.val.cpu(), self.param.detach().cpu())
"""
BASE NETWORKS
    - Encoder: for VAE
    - LatentODE: for rhs ode function
    - Decoder: for VAE
"""

class Encoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(obs_dim, hidden_units, hidden_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_units, latent_dim * 2)
        
    def forward(self, x):
        y, _ = self.rnn(x)
        y = y[:, -1, :]
        y = self.h2o(y)
        return y

class LatentODE(nn.Module):
    def __init__(self, layers):
        super(LatentODE, self).__init__()
        self.act = nn.Tanh()
        self.layers = layers
        #FEED FORWARD
        arch = []
        for ind_layer in range(len(self.layers) - 2):
            layer = nn.Linear(self.layers[ind_layer], self.layers[ind_layer + 1])
            nn.init.xavier_uniform_(layer.weight)
            arch.append(layer)
        layer = nn.Linear(self.layers[-2], self.layers[-1])
        layer.weight.data.fill_(0)
        arch.append(layer)
        #LIN OUT
        self.linear_layers = nn.ModuleList(arch)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        for ind in range(len(self.layers) - 2):
            x = self.act(self.linear_layers[ind](x))
        y = self.linear_layers[-1](x)
        return y

class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Decoder, self).__init__()
        self.act = nn.Tanh()
        self.rnn = nn.GRU(latent_dim, hidden_units, hidden_layers, batch_first=True)
        self.h1 = nn.Linear(hidden_units, hidden_units - 5)
        self.h2 = nn.Linear(hidden_units - 5, obs_dim)

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.h1(y)
        y = self.act(y)
        y = self.h2(y)
        return y

"""
NEURAL ODE NETWORKS
    - Neural ODE: updates the hidden state h from h'=LatentODE
    - Heavy-Ball Neural ODE: learns hidden state h from h'+gamma m=LatentODE

"""
class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0
    def forward(self, t, x):#dy/dt = f(t,x) 
        self.nfe += 1
        return self.df(t, x) 

class HBNODE(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True):
        super().__init__(df)
        # Momentum parameter gamma
        self.gamma = Parameter([gamma_guess],frozen=False)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = Parameter([corr],frozen=False)
        self.sp = nn.Softplus()
        # Activation for dh, GHBNODE only
        self.actv_h = nn.Identity() if actv_h is None else actv_h
    def forward(self, t, x): 
        self.nfe += 1
        h, m = torch.split(x, x.shape[-1]//2, dim=1)
        dh = self.actv_h(- m) #dh/dt = m
        dm = self.df(t, h) - self.gammaact(self.gamma()) * m #dm/dt = -gamm *m + f(t, h)!(network)!
        dm = dm + self.sp(self.corr()) * h
        out = torch.cat((dh, dm), dim=1)#.to(device)
        return out
