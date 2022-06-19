from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

plt.style.use('classic')
padding=0


#######################
### RECONSTRUCTIONS ###
#######################
def vks_plot(data,time,axis,args,index=None):
    axis.imshow(data[time,:,:,index].T, origin='upper', vmin =-.4,vmax =.4, aspect='auto')
    pass

def kpp_plot(data,time,axis,args,index=None):
    plt.style.use('default')
    xv =  np.tile(np.linspace(-2,2,data.shape[1]),(data.shape[2],1)) 
    yv = np.tile(np.linspace(0,10,data.shape[2]),(data.shape[1],1)).T
    axis.plot_surface(xv, yv, data[time,:,:], cmap=cm.coolwarm, linewidth=0)
    pass

def ee_plot(data,time,axis,param,index,args):
    x = np.linspace(-5,5,data.shape[0])
    axis.plot(x,data[:,time,index], 'k')
    pass

def fiber_plot(data,time,axis,param,index,args):
    if 'param' in args:
        data = data[args.param]
    else:
        data = data[0]

    fig, axes = plt.subplots(1,1,tight_layout=True)
    axes.set_ylim([0,2])
    x = np.linspace(0,5,data.shape[1])
    axes.plot(x,data[time,:], 'k')
    pass

def data_reconstruct(data,time,args):
    fig = plt.figure(tight_layout=True)
    if args.dataset == 'VKS':
        ax = plt.subplot(211)
        vks_plot(data,time,ax,args,index=0)
        ax.set_title('$u\'_x$')
        ax = plt.subplot(212)
        vks_plot(data,time,ax,args,index=1)
        ax.set_title('$u\'_y$')
    elif args.dataset == 'KPP':
        ax = fig.add_subplot(projection='3d')
        kpp_plot(data,time,ax,args)
    elif args.dataset=='EE':
        ax = plt.subplot(131)
        ee_plot(data,time,ax,0,0,args)
        ax = plt.subplot(132)
        ee_plot(data,time,ax,0,1,args)
        ax = plt.subplot(133)
        ee_plot(data,time,ax,0,2,args)
    elif args.dataset=='FIB':
        ax = fig.gca()
        fiber_plot(data,time,ax,0,2,args)
    else:
        raise NotImplementedError
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    return fig

######################
##### ANIMATIONS #####
######################
def vks_animate(data,args):

    fig, axes = plt.subplots(2,1,tight_layout=True)
    lines = []
    for ax in axes.flatten():
        lines = lines + [ax.imshow(np.zeros((data.shape[1:3])), origin='upper', vmin =-.4,vmax =.4, aspect='auto')]

    def run_vks_lines(vks_t):
        lines[0].set_data(data[vks_t,:,:,0].T)
        lines[1].set_data(data[vks_t,:,:,1].T)
        return lines
    ani = animation.FuncAnimation(fig, run_vks_lines, blit=False, interval=data.shape[0]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    pass
    
def kpp_animate(data,args):
    plt.style.use('default')
    xv =  np.tile(np.linspace(-2,2,data.shape[1]),(data.shape[2],1))
    yv = np.tile(np.linspace(0,10,data.shape[2]),(data.shape[1],1)).T

    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot(projection='3d')
    ax1.set_zlim(0,10)
    lines =[ax1.plot_surface(xv, yv, np.ones(data.shape[1:]), cmap=cm.coolwarm, linewidth=0, antialiased=False)]
    
    def run_kpp_lines(kpp_t):
        ax1.clear()
        ax1.set_zlim(0,10)
        lines =[ax1.plot_surface(xv, yv, data[kpp_t], cmap=cm.coolwarm, linewidth=0, antialiased=False)]
        return lines

    ani = animation.FuncAnimation(fig, run_kpp_lines, blit=False, interval=data.shape[0]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    pass

def ee_animate(data,args):
    x = np.linspace(-5,5,data.shape[0])

    fig, axes = plt.subplots(1,3,tight_layout=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    lims = [(-2,5), (-5,5),(-5,11)]
    lines = []
    for i,ax in enumerate(axes.flatten()):
        ax.set_ylim(lims[i])
        lines = lines + ax.plot(x,np.zeros(data.shape[0]), 'k')

    def run_ee_lines(ee_t):
        lines[0].set_ydata(data[:,ee_t,0])
        lines[1].set_ydata(data[:,ee_t,1])
        lines[2].set_ydata(data[:,ee_t,2])
        return lines

    ani = animation.FuncAnimation(fig, run_ee_lines, blit=False, interval=data.shape[1]-1,
        repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    pass

def fiber_animate(data,args):
    if 'param' in args:
        data = data[args.param]
    else:
        data = data[0]


    fig, axes = plt.subplots(1,1,tight_layout=True)
    axes.set_ylim([-2,2])
    x = np.linspace(0,5,data.shape[1])
    lines = axes.plot(x,np.zeros(data.shape[1]), 'k')

    def run_fiber_lines(fiber_t):
        lines[0].set_ydata(data[fiber_t,:])
        return lines

    ani = animation.FuncAnimation(fig, run_fiber_lines, frames = data.shape[0]-1, blit=False, interval=data.shape[0]-1,
            repeat=False)
    end_str = str(args.dataset+'_'+args.model+'_recon').lower()
    ani.save(args.out_dir+'/'+end_str+'.gif', "PillowWriter", fps=6)
    pass

ANIM = {'VKS':vks_animate,'KPP':kpp_animate, 'EE':ee_animate, 'FIB':fiber_animate}
def data_animation(data,args):
    ANIM[args.dataset](data,args)
    pass

##################
# DECOMPOSITIONS #
##################
def information_decay(dataset,args):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)

    total = dataset.lv.sum()
    decay=[1]
    #CALCULATE DECAY
    for eig in dataset.lv:
        val = eig/total
        decay = decay + [decay[-1]-val]
    decay = np.array(decay)
    #X-DATA
    r = np.arange(0,decay.shape[0])
    plt.figure(tight_layout=True)
    plt.plot(r,decay, 'k',linewidth=2)
    plt.xlabel('# DMD Modes $(r)$',fontsize=36)
    plt.ylabel('$1-I(r)$',fontsize=36)
    plt.yscale('log')
    y_min = max(1e-16,min(decay))
    plt.ylim(y_min,1)
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_decay').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: 
        plt.show()
        print("Relative information content is {:.5f} for {} modes.".format(1-decay[args.modes],args.modes))
    pass

def plot_mode(modes,times,args):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=16)
    plt.rc('ytick',labelsize=16)
    plt.figure(tight_layout=True)
    for i,node in enumerate(modes.T):
        plt.subplot(2,2,i+1)
        times=np.arange(0,min(1000,node.shape[0]))
        plt.plot(times,node[:1000],'k')
        plt.xlabel("Time $(t)$",fontsize=24)
        plt.ylabel("$\\alpha_{}$".format(i),fontsize=24)
        plt.xlim(times[0],times[-1])
    end_str = str(args.dataset+'_'+args.model+'_modes').lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    pass

def mode_prediction(predictions,true,times,verts,args,end_str=''):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    plt.figure(tight_layout=True)
    times=np.arange(len(predictions))
    for i,node in enumerate(predictions.T):
        plt.subplot(2,2,i+1)
        plt.plot(times,node, 'r', dashes=[1,1], label='Prediction')
        plt.plot(times,true.T[i], 'k',dashes=[1,2], label='True')
        min_1,max_1=np.min(true.T[i]),np.max(true.T[i])
        min_2,max_2=np.min(node),np.max(node)
        min_,max_=min(min_1,min_2),max(max_1,max_2)
        plt.vlines(verts,ymin=min_+.2*min_,ymax=max_+.2*max_)
        plt.xlabel("Time $(t)$",fontsize=24)
        plt.ylabel("$\\alpha_{}$".format(i),fontsize=24)
        plt.ylim(bottom=min_+.2*min_,top=max_+.2*max_)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center',mode='expand', borderaxespad=0., prop={'size': 10}, frameon=False)
    #OUTPUT
    end_str = str(args.dataset+'_'+args.model+'_mode_pred'+end_str).lower()
    plt.savefig(args.out_dir+'/'+end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    pass

############################
### MODEL UNIVERSAL PLOTS ###
############################

#INNER METHODS
def ax_nfe(epochs,nfes,plt_args):
  plt.scatter(epochs,nfes,**plt_args)
  pass

def ax_stiff(epochs,stiff,plt_args):
  plt.scatter(epochs,stiff,**plt_args)
  pass

def ax_loss(epochs,loss,plt_args):
  plt.plot(epochs,loss,**plt_args,linewidth=2)
  pass

"""  LOSS PLOT """
def plot_loss(fname,args):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['tr_loss', 'val_loss']
    color = ['k','r--']
    losses = df[index_].values
    epochs=np.arange(len(losses))
    plt.figure(tight_layout=True)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.yticks(np.logspace(-4,0,5))
    plt.ylim(1e-4,1)
    plt.xlim(epochs[0],epochs[-1])
    plt.yscale('log')
    end_str = str(args.out_dir+'/'+args.model+'_loss')
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    pass

"""  NFE PLOT """
def plot_nfe(fname,index_,args):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)

    nfes = df[index_].values
    epochs=np.arange(len(nfes))
    plt.figure(tight_layout=True)
    plt.scatter(epochs,nfes)
    plt.xlim(epochs[0],epochs[-1])
    end_str = str(args.out_dir+'/'+args.model+'_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    pass

"""  ADJOINT GRADIENT PLOT """
def plot_adjGrad(fname,args,show=False):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['grad_{}'.format(i) for i in range(args.seq_ind)]
    grad = df[index_].values
    plt.figure(tight_layout = True)
    plt.imshow(grad.T, origin='upper',vmin=0,vmax=.01, cmap='inferno', aspect='auto')
    plt.colorbar()
    plt.xlabel('Epoch',fontsize=24)
    plt.ylabel('$T-t$',fontsize=24)
    
    end_str = str(args.out_dir+'/'+args.model+'_adjGrad')
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    pass

"""  STIFFNESS PLOT """
def plot_stiff(fname,args, clip=1, show=False):
    with open(fname, 'r') as f:
        df = pd.read_csv(f, index_col=False)
    index_ = ['backward_stiff']
    stiff = df[index_].values
    epochs = np.arange(len(stiff))
    plt.figure(tight_layout=True)
    plt.scatter(epochs,stiff)
    plt.yscale('log')
    plt.xlim(0,epochs[-1])

    end_str = str(args.out_dir+'/'+args.model+'_stiffness')
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    pass

""""
COMPARISONS
"""
def compare_nfe(file_list,model_list,index_,args):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        nfes = df[index_].values[::args.epoch_freq]
        epochs=np.arange(len(nfes))*args.epoch_freq
        plt_args={'label':model_list[i],'color':args.color_list[i]}
        ax_nfe(epochs,nfes,plt_args)

    plt.xlim(epochs[0],epochs[-1])
    plt.legend(fontsize=24)
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

def compare_stiff(file_list,model_list,index_,args):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    markerlist = ['o','o']

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        stiffs = df[index_].values[::args.epoch_freq]
        epochs=np.arange(len(stiffs))*args.epoch_freq
        plt_args={'label':model_list[i],'color':args.color_list[i],'marker':markerlist[i]}
        ax_stiff(epochs,stiffs,plt_args)

    plt.legend(fontsize=24) 
    plt.ylim(2e1,1e4)
    plt.yscale('log')
    plt.xlim(epochs[0],epochs[-1])
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

def compare_loss(file_list,model_list,index_,args):
    plt.rcParams['xtick.minor.size']=0
    plt.rcParams['ytick.minor.size']=0
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)

    fig = plt.figure(tight_layout=True)
    ax=fig.axes
    for i,fname in enumerate(file_list):
        with open(fname, 'r') as f:
            df = pd.read_csv(f, index_col=False)
        losses = df[index_].values
        plt_args={'label':model_list[i],'color':args.color_list[i]}
        epochs=np.arange(len(losses[::args.epoch_freq]))*args.epoch_freq
        ax_loss(epochs,losses[::args.epoch_freq],plt_args)

    if index_=='tr_loss':
        plt.yticks(np.logspace(-4,0,5))
        plt.ylim(1e-1,1)
    else:
        plt.yticks(np.logspace(-2,0,3))
        plt.ylim(.4e-2,1)
    epochs=np.arange(len(losses[::args.epoch_freq]))*args.epoch_freq
    plt.xlim(0,epochs[-1])
    plt.yscale('log')
    plt.ylabel('Loss',fontsize=36)
    plt.xlabel('Epoch',fontsize=36)
    plt.legend(fontsize=24)
    end_str = str(args.out_dir+'/compare_'+index_)
    plt.savefig(end_str+'.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()


#################
# PARAM SPECIAL #
#################
def param_Loss(fname,args, clip=1, show=False):
    with open(fname, 'r') as file:
        df = pd.read_csv(file, index_col=False)
    index_ = ['loss', 'ts_loss']
    color = ['k','r--']
    losses = df[index_].values
    plt.figure(tight_layout=True)
    plt.title(args.dataset+' '+args.model)
    for i,loss in enumerate(losses.T):
        plt.plot(loss,color[i],label=index_[i])
    plt.legend()
    plt.yscale('log')
    # yticks = [100/(10**i) for i in range(5)]
    # plt.yticks(yticks)
    plt.savefig(args.out_dir+'/'+args.model+'/LOSS.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()
    pass


def param_ModesLong(dataloader, model, args, show=False):
    DL = dataloader
    times = DL.train_times
    stamps = [args.tstart, args.tr_win+args.tstart, args.tstop]
    xs = [np.arange(0,args.tr_win), np.arange(stamps[1], stamps[2]-1)]
    data = DL.train_data[:,:,:]
    labels = DL.train_label[:,:,:]
    plt.figure(figsize=(15,5), tight_layout=True)

    predict = model(times, data).cpu().detach().numpy()[:,0,:]

    for i,node in enumerate(predict.T):
        plt.subplot(args.modes//2,2,i+1)
        plt.plot(xs[0],node, 'r', label='Prediction')
        plt.plot(xs[0],labels[:,0,i], 'k--', label='True')
        # plt.axvspan(stamps[1]-2, stamps[1]+args.seq_win, facecolor='k', alpha=.25)
        # plt.axvspan(stamps[0], stamps[0]+args.seq_win, facecolor='k', alpha=.25)
        plt.xlabel("Time")
        plt.ylabel("$\\alpha_{}$".format(i))
        # plt.xlim(stamps[0],stamps[2]-1)
        # plt.title('Mode {}'.format(i+1))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 10}, frameon=False)
    plt.savefig(args.out_dir+'/'+args.model+'/'+'modeReconstruct.pdf', format="pdf", bbox_inches="tight")
    if args.verbose: plt.show()

    pass
