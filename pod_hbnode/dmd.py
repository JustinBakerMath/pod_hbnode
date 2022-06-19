import numpy as np

def DMD(X,Xp,modes):

    U,Sigma,Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    Ur = U[:,:modes]
    Sigmar = np.diag(Sigma[:modes])
    Vr = Vh[:modes,:].T

    invSigmar = np.linalg.inv(Sigmar)

    Atilde = Ur.T@Xp@Vr@invSigmar
    Lambda, W = np.linalg.eig(Atilde)
    Lambda = np.diag(Lambda)

    Phi = Xp@(Vr@invSigmar)@W

    alpha1 = Sigmar@(Vr[0,:].T)
    b = np.linalg.solve(W@Lambda,alpha1)

    return X, Atilde, Ur, Phi, Lambda, Sigma, b

def DMD1(data,s_ind,e_ind,modes,lifts=()):

    var = data[s_ind:e_ind,:]
    var_mean = np.mean(var, axis=0)[np.newaxis, ...]

    var_flux = var-var_mean

    X = var_flux[:-1,:].T
    Xp = var_flux[1:,:].T

    return DMD(X,Xp,modes)

""" DMD DECOMP FOR 2D, 2PARAM MODEL"""
def DMD2(data,s_ind,e_ind,modes,lifts=()):

    var1 = data[s_ind:e_ind,:,:,0]
    var2 = data[s_ind:e_ind,:,:,1]

    domain_len = var1.shape[1]*var1.shape[2]
    time_len = var1.shape[0]

    var1_mean = np.mean(var1, axis=0)[np.newaxis, ...]
    var2_mean = np.mean(var2, axis=0)[np.newaxis, ...]

    var1_flux = var1-var1_mean
    var2_flux = var2-var2_mean

    var1_flux = var1_flux.reshape(time_len,domain_len)
    var2_flux = var2_flux.reshape(time_len,domain_len)

    stacked_flux = np.hstack((var1_flux, var2_flux))

    stacked_flux = lift(stacked_flux,lifts)

    X = stacked_flux[:-1,:].T
    Xp = stacked_flux[1:,:].T

    return DMD(X,Xp,modes)


def DMD3(data,s_ind,e_ind,modes,lifts=()):

    var1 = data[s_ind:e_ind,:,0]
    var2 = data[s_ind:e_ind,:,1]
    var3 = data[s_ind:e_ind,:,2]

    var1_mean = np.mean(var1, axis=0)[np.newaxis, ...]
    var2_mean = np.mean(var2, axis=0)[np.newaxis, ...]
    var3_mean = np.mean(var3, axis=0)[np.newaxis, ...]

    var1_flux = var1-var1_mean
    var2_flux = var2-var2_mean
    var3_flux = var3-var3_mean

    stacked_flux = np.hstack((var1_flux, var2_flux, var3_flux))

    X = stacked_flux[:-1,:].T
    Xp = stacked_flux[1:,:].T

    return DMD(X,Xp,modes)

def DMDKPP(data,s_ind,e_ind,modes,lifts=()):

    var = data[s_ind:e_ind,:,:]

    var_mean = np.mean(var, axis=0)[np.newaxis, ...]
    var_flux = var-var_mean

    shape = var_flux.shape
    var_flux = var_flux.reshape(shape[0], shape[1] * shape[2])
    var_flux = lift(var_flux,lifts)

    X = var_flux[:-1,:].T
    Xp = var_flux[1:,:].T

    return DMD(X,Xp,modes)


def lift(data_init,lifts=()):
  data = data_init.copy()
  if 'sin' in lifts:
    lift = np.sin(data_init)
    data = np.hstack((data,lift))
  if 'cos' in lifts:
    lift = np.cos(data_init)
    data = np.hstack((data,lift))
  if 'quad' in lifts:
    lift = np.power(data_init,2)
    data = np.hstack((data,lift))
  if 'cube' in lifts:
    lift = np.power(data_init,3)
    data = np.hstack((data,lift))
  return data.copy()
