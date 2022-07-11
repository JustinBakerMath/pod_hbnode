import numpy as np
import scipy.linalg

def POD1(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    param = U[s_ind:e_ind, :]#[300,300,500]
    param_mean = np.mean(param, axis=0)[np.newaxis, ...]

    param_flux = param - param_mean

    # Snapshots
    snap_shots = np.matmul(param_flux, param_flux.T) #[300,300]
    eig_vals, eig_vecs = scipy.linalg.eigh(snap_shots)

    # descending order
    eig_vals = eig_vals[eig_vals.shape[0]::-1]
    eig_vecs = eig_vecs[:, eig_vals.shape[0]::-1]

    spatial_modes = np.matmul(param_flux.T, eig_vecs[:,:modes])/np.sqrt(eig_vals[:modes])
    temporal_coefficients = np.matmul(param_flux,spatial_modes)

    return spatial_modes,temporal_coefficients,eig_vals,param_flux.copy()

def POD2(data,s_ind,e_ind,modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    #Parameters
    param1 = data[s_ind:e_ind,:,:,0]
    param2 = data[s_ind:e_ind,:,:,1]

    param1_mean = np.mean(param1, axis=0)[np.newaxis, ...]
    param2_mean = np.mean(param2, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    param1_flux = param1 - param1_mean
    param2_flux = param2 - param2_mean
    
    # return copies
    flux1_copy = np.copy(param1_flux)
    flux2_copy = np.copy(param2_flux)

    #Reshape spatial_dims for Snapshots
    shape = param1_flux.shape
    param1_flux = param1_flux.reshape(shape[0], shape[1] * shape[2])
    param2_flux = param2_flux.reshape(shape[0], shape[1] * shape[2])
    stacked_flux = np.hstack((param1_flux, param2_flux))

    # Snapshot Method:
    snap_shots = np.matmul(stacked_flux, stacked_flux.T) # YY^T
    eigen_vals, eigen_vecs = scipy.linalg.eigh(snap_shots)

    # descending order
    eigen_vals = eigen_vals[eigen_vals.shape[0]::-1]
    eigen_vecs = eigen_vecs[:, eigen_vals.shape[0]::-1] #unit norm

    spatial_modes = np.matmul(stacked_flux.T, eigen_vecs[:, :modes]) / np.sqrt(eigen_vals[:modes]) # is this unit norm?
    temporal_coefficients = np.matmul(stacked_flux, spatial_modes) #"throw sqrt of Lv onto temp_coef"

    return spatial_modes, temporal_coefficients, eigen_vals, flux1_copy, flux2_copy


def POD3(data, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    var1 = data[s_ind:e_ind,:,0]
    var2 = data[s_ind:e_ind,:,1]
    var3 = data[s_ind:e_ind,:,2]

    var1_mean = np.mean(var1, axis=0)[np.newaxis, ...]
    var2_mean = np.mean(var2, axis=0)[np.newaxis, ...]
    var3_mean = np.mean(var3, axis=0)[np.newaxis, ...]

    var1_flux = var1 - var1_mean
    var2_flux = var2 - var2_mean
    var3_flux = var3 - var3_mean
        
    stacked_flux = np.hstack((var1_flux, var2_flux, var3_flux))

    # Snapshot Method:
    snap_shots = np.matmul(stacked_flux, stacked_flux.T)    

    # L:eigvals, As:eigvecs
    eigen_vals, eigen_vecs = scipy.linalg.eigh(snap_shots)

    # descending order
    eigen_vals = eigen_vals[eigen_vals.shape[0]::-1]
    eigen_vecs = eigen_vecs[:, eigen_vals.shape[0]::-1]

    spatial_modes = np.matmul(stacked_flux.T, eigen_vecs[:, :modes]) / np.sqrt(eigen_vals[:modes])    
    temporal_coefficients = np.matmul(stacked_flux, spatial_modes)

    return spatial_modes, temporal_coefficients, eigen_vals, var1_flux.copy(), var2_flux.copy(), var3_flux.copy()

def PODKPP(data, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    var = data[s_ind:e_ind,:, :]

    var_mean = np.mean(var, axis=0)[np.newaxis, ...]
    var_flux = var - var_mean

    shape = var_flux.shape
    var_flux = var_flux.reshape(shape[0], shape[1]*shape[2])

    snap_shots = np.matmul(var_flux, var_flux.T)
    eigen_vals, eigen_vecs = scipy.linalg.eigh(snap_shots)

    # descending order
    eigen_vals = eigen_vals[eigen_vals.shape[0]::-1]
    eigen_vecs = eigen_vecs[:, eigen_vals.shape[0]::-1]

    spatial_modes = np.matmul(var_flux.T, eigen_vecs[:, :modes]) / np.sqrt(eigen_vals[:modes])
    temporal_coefficients = np.matmul(var_flux, spatial_modes) #"throw sqrt of Lv onto temp_coef"

    return spatial_modes, temporal_coefficients, eigen_vals, var_flux.copy()

def PODFIB(data, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """
    # There are no parameters/components in this data

    data_mean = np.mean(data, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    data_flux = data - data_mean

    # Snapshot Method:
    snap_shots = np.matmul(data_flux, data_flux.T) # YY^T
    eigen_vals, eigen_vecs = scipy.linalg.eigh(snap_shots)

    # descending order
    eigen_vals = eigen_vals[eigen_vals.shape[0]::-1]
    eigen_vecs = eigen_vecs[:, eigen_vals.shape[0]::-1] #unit norm

    spatial_modes = np.matmul(data_flux.T, eigen_vecs[:, :modes]) / np.sqrt(eigen_vals[:modes]) # is this unit norm?
    temporal_coefficients = np.matmul(data_flux, spatial_modes) #"throw sqrt of Lv onto temp_coef"

    return spatial_modes, temporal_coefficients, eigen_vals, data_flux

"""RECONSTRUCT FROM MODES"""
def pod_mode_to_true(dataset,modes,args):
    spatial_modes = dataset.spatial_modes
    true = np.matmul(modes,spatial_modes.T)

    if args.dataset == "VKS":
        if len(true.shape)>2:
            true = true[0]
        pod_x = true[:, :dataset.domain_len]
        pod_y = true[:, dataset.domain_len:]

        shape = [true.shape[0]] + list(dataset.domain_shape)
        true_x = pod_x.reshape(shape)
        true_y = pod_y.reshape(shape)
        true = np.array([true_x.T,true_y.T])
    elif args.dataset == 'KPP':
        shape = [true.shape[0]] + list(dataset.domain_shape)
        true=true.reshape(shape).T

    elif args.dataset == 'EE':
        params = args.eeParam
        domain_len = dataset.domain_len
        shape=[true.shape[0]]+[domain_len//params,params]

        pod_rho = true[:,:domain_len].reshape(shape,order='F')
        pod_v = true[:,domain_len:2*domain_len].reshape(shape,order='F')
        pod_e = true[:,2*domain_len:].reshape(shape,order='F')

        true = np.array([pod_rho.T,pod_v.T,pod_e.T])
    return true.T
