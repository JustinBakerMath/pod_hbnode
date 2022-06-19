import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

EPS = 1e-3
device = 'cpu'

# LEARNING UTILS
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


# MODEL UTILS
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

#SUPPORT NNs
class Tinvariant_NLayerNN(NLayerNN):
    def forward(self, t, x):
        return super(Tinvariant_NLayerNN, self).forward(x)
    
class dfwrapper(nn.Module):
    def __init__(self, df, shape, recf=None):
        super(dfwrapper, self).__init__()
        self.df = df
        self.shape = shape
        self.recf = recf

    def forward(self, t, x):
        bsize = x.shape[0]
        if self.recf:
            x = x[:, :-self.recf.osize].reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dr = self.recf(t, x, dx).reshape(bsize, -1)
            dx = dx.reshape(bsize, -1)
            dx = torch.cat([dx, dr], dim=1).to(device)
        else:
            x = x.reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dx = dx.reshape(bsize, -1)
        return dx

#BASE NNs
class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + 2 * nhidden, 2 * nhidden)
        self.dense2 = nn.Linear(2 * nhidden, 2 * nhidden)
        self.dense3 = nn.Linear(2 * nhidden, 2 * out_channels)

        self.cont = cont
        self.res = res
    def forward(self, h, x):
        out = torch.cat([h[:, 0], h[:, 1], x], dim=1).to(device)
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out).reshape(h.shape)
        #out = out + h
        return out

class nodernn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + nhidden, nhidden * 2)
        self.dense2 = nn.Linear(nhidden * 2, nhidden * 2)
        self.dense3 = nn.Linear(nhidden * 2, out_channels)

    def forward(self, h, x):
        out = torch.cat([h, x], dim=1).to(device)
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out)
        out = out.reshape(h.shape)
        return out

class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Sigmoid()
        self.dense1 = nn.Linear(in_channels, out_channels)
    def forward(self, h, x):
        out = self.dense1(x)
        out = self.actv(out)
        return out

class tempout(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        out = self.dense1(x)
        return out

#ODE NNs
class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0
        self.elem_t = None
    def forward(self, t, x):
        self.nfe += 1
        if self.elem_t is None:
            return self.df(t, x)
        else:
            return self.elem_t * self.df(self.elem_t, x)
    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1)

class SONODE(NODE):
    def forward(self, t, x):
        self.nfe += 1
        v = x[:, 1:, :]
        out = self.df(t, x)
        return torch.cat((v, out), dim=1).to(device)

class HeavyBallNODE(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True):
        super().__init__(df)
        # Momentum parameter gamma
        self.gamma = Parameter([gamma_guess], frozen=False)
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = Parameter([corr], frozen=corrf)
        self.sp = nn.Softplus()
        # Activation for dh, GHBNODE only
        self.actv_h = nn.Identity() if actv_h is None else actv_h
    def forward(self, t, x): #dx/dt = f(t,x) > raised to second order system (h,m)
        self.nfe += 1
        h, m = torch.split(x, 1, dim=1)
        dh = self.actv_h(- m)
        dm = self.df(t, h) - self.gammaact(self.gamma()) * m
        dm = dm + self.sp(self.corr()) * h
        out = torch.cat((dh, dm), dim=1).to(device)
        if self.elem_t is None:
            return out
        else:
            return self.elem_t * out

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1, 1)
HBNODE = HeavyBallNODE

#MODELS
class NMODEL(nn.Module):
    def __init__(self, args):
        super(NMODEL, self).__init__()
        modes = args.modes
        nhid = modes*2
        self.cell = NODE(tempf(nhid, nhid))
        self.rnn = nodernn(modes, nhid, nhid)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, nhid, None, rnn_out=True, tol=1e-7)
        self.outlayer = tempout(nhid, modes)
    def forward(self, t, x):
        out = self.ode_rnn(t, x, retain_grad=True)[0]
        out = self.outlayer(out)[:-1]
        return out

class HBMODEL(nn.Module):
    def __init__(self,args,  res=False, cont=False):
        super(HBMODEL, self).__init__()
        modes = args.modes
        nhid = modes*4
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=args.corr, corrf=True)
        self.rnn = temprnn(modes, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=1e-7)
        self.outlayer = nn.Linear(nhid, modes)
    def forward(self, t, x):
        out = self.ode_rnn(t, x, retain_grad=True)[0]
        out = self.outlayer(out[:, :, 0])[1:]
        return out


class GHBMODEL(nn.Module):
    def __init__(self, args, res=False, cont=False):
        super(GHBMODEL, self).__init__()
        modes = args.modes
        nhid = modes*2
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=args.corr, corrf=False, actv_h=nn.Tanh())
        self.rnn = temprnn(modes, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=1e-7)
        self.outlayer = nn.Linear(nhid,modes)

    def forward(self, t, x):
        out = self.ode_rnn(t, x, retain_grad=True)[0]
        out = self.outlayer(out[:, :, 0])[1:]
        return out


class NODEintegrate(nn.Module):
    def __init__(self, df, shape=None, tol=1e-5, adjoint=True, evaluation_times=None, recf=None):
        super().__init__()
        self.df = dfwrapper(df, shape, recf) if shape else df
        self.tol = tol
        self.odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        self.evaluation_times = evaluation_times if evaluation_times is not None else torch.Tensor([0.0, 1.0])
        self.shape = shape
        self.recf = recf
        if recf:
            assert shape is not None
    def forward(self, x0):
        bsize = x0.shape[0]
        if self.shape:
            assert x0.shape[1:] == torch.Size(self.shape), \
                'Input shape {} does not match with model shape {}'.format(x0.shape[1:], self.shape)
            x0 = x0.reshape(bsize, -1)
            if self.recf:
                reczeros = torch.normallike(x0[:, :1]).to(device)
                reczeros = repeat(reczeros, 'b 1 -> b c', c=self.recf.osize)
                x0 = torch.cat([x0, reczeros], dim=1).to(device)
            out = odeint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol)
            if self.recf:
                rec = out[-1, :, -self.recf.osize:]
                out = out[:, :, :-self.recf.osize]
                out = out.reshape(-1, bsize, *self.shape)
                return out, rec
            else:
                return out
        else:
            out = odeint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol)
            return out
    @property
    def nfe(self):
        return self.df.nfe

#SUPER MODELS
class NODElayer(NODEintegrate):
    def forward(self, x0):
        out = super(NODElayer, self).forward(x0)
        if isinstance(out, tuple):
            out, rec = out
            return out[-1], rec
        else:
            return out[-1]

class ODE_RNN(nn.Module):
    def __init__(self, ode, rnn, nhid, ic, rnn_out=False, both=False, tol=1e-7):
        super().__init__()
        self.ode = ode
        self.t = torch.Tensor([0, 1]).to(device)
        self.nhid = [nhid] if isinstance(nhid, int) else nhid
        self.rnn = rnn
        self.tol = tol
        self.rnn_out = rnn_out
        self.ic = ic
        self.both = both
    def forward(self, t, x, multiforecast=None):
        n_t, n_b = t.shape
        h_ode = torch.zeros(n_t + 1, n_b, *self.nhid).to(device)
        h_rnn = torch.zeros(n_t + 1, n_b, *self.nhid).to(device)
        if self.ic:
            h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 't b c -> b (t c)')).view(h_ode[0].shape)
        if self.rnn_out:
            for i in range(n_t):
                self.ode.update(t[i])
                h_ode[i] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
                h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
            out = (h_rnn,)
        else:
            for i in range(n_t):
                self.ode.update(t[i])
                h_rnn[i] = self.rnn(h_ode[i], x[i])
                h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
            out = (h_ode,)
        if self.both:
            out = (h_rnn, h_ode)
        if multiforecast is not None:
            self.ode.update(torch.normallike((t[0])))
            forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol)
            out = (*out, forecast)
        return out

class ODE_RNN_with_Grad_Listener(nn.Module):
    def __init__(self, ode, rnn, nhid, ic, rnn_out=False, both=False, tol=1e-7):
        super().__init__()
        self.ode = ode
        self.t = torch.Tensor([0, 1])
        self.nhid = [nhid] if isinstance(nhid, int) else nhid
        self.rnn = rnn
        self.tol = tol
        self.rnn_out = rnn_out
        self.ic = ic
        self.both = both
    def forward(self, t, x, multiforecast=None, retain_grad=False):
        n_t, n_b = t.shape
        h_ode = [None] * (n_t + 1)
        h_rnn = [None] * (n_t + 1)
        h_ode[-1] = h_rnn[-1] = torch.zeros(n_b, *self.nhid).to(device)
        if self.ic:
            h_ode[0] = h_rnn[0] = self.ic(rearrange(x, 'b t c -> b (t c)')).view((n_b, *self.nhid))
        else:
            h_ode[0] = h_rnn[0] = torch.zeros(n_b, *self.nhid).to(device)
        if self.rnn_out:
            for i in range(n_t):
                self.ode.update(t[i])
                h_ode[i] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
                h_rnn[i + 1] = self.rnn(h_ode[i], x[i])
            out = (h_rnn,)
        else:
            for i in range(n_t):
                self.ode.update(t[i])
                h_rnn[i] = self.rnn(h_ode[i], x[i])
                h_ode[i + 1] = odeint(self.ode, h_rnn[i], self.t, atol=self.tol, rtol=self.tol)[-1]
            out = (h_ode,)
        if self.both:
            out = (h_rnn, h_ode)
        out = [torch.stack(h, dim=0) for h in out]
        if multiforecast is not None:
            self.ode.update(torch.normallike((t[0])).to(device))
            forecast = odeint(self.ode, out[-1][-1], multiforecast * 1.0, atol=self.tol, rtol=self.tol)
            out = (*out, forecast)
        if retain_grad:
            self.h_ode = h_ode
            self.h_rnn = h_rnn
            for i in range(n_t + 1):
                if self.h_ode[i].requires_grad:
                    self.h_ode[i].retain_grad()
                if self.h_rnn[i].requires_grad:
                    self.h_rnn[i].retain_grad()
        return out
