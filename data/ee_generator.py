# Finite volume solver for the Euler equations using a Harten-Lax-van Leer flux

import numpy as np
from matplotlib import pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

gamma = 7/5.

# Grid
N = 1000
h = 10./N
dt = h**2
lam = dt/h

alpha = 1

inds = np.arange(N)

p1_range = [3, 4] # left-hand value of initial density and boundary condition
p2_range = [2, 3] # left-hand value of initial velocity and boundary condition
Nparamgrid = 10

# Canonical choices:
p1 = 3.857143 
p2 = 2.629369

p1 = 3
p2 = 2

# Downsamples in time by this factor
downsample_rate = 100

# Define the vector-valued function f (of vector valued input uu)
def f(uu):
	rho = uu[0]
	u = uu[1]/rho
	E = uu[2]
	p = (gamma-1)*(E-1/2*rho*u**2)
	return rho*u, rho*u**2 + p, (E+p)*u 

# Define HLL flux
def HLL_flux(uu):
	fu0, fu1, fu2 = f(uu)
	rho = uu[0,:]
	rhou = uu[1,:]
	u = rhou/rho
	E = uu[2,:]
	p = (gamma-1)*(E-1/2*rho*u**2)
	c = np.sqrt(gamma*p/rho)
	F = np.zeros(uu.shape)
	sm = np.zeros(len(rho))
	sp = np.zeros(len(rho))
	sm[0] = u[0] - c[0]
	sm[1:] = np.minimum(u[1:]-c[1:],u[:-1]-c[:-1])
	sp[0] = u[0] + c[0]
	sp[1:] = np.maximum(u[1:]+c[1:],u[:-1]+c[:-1])
	F[0,0] = fu0[0]*(sm[0]<=0) + fu0[1]*(sp[0]<=0) + (sp[0]*fu0[0] - sm[0]*fu0[1] + sp[0]*sm[0]*(rho[1]-rho[0]))/(sp[0]-sm[0])*(sm[0]<0)*(sp[0]>0)
	F[1,0] = fu1[0]*(sm[0]<=0) + fu1[1]*(sp[0]<=0) + (sp[0]*fu1[0] - sm[0]*fu1[1] + sp[0]*sm[0]*(rhou[1]-rhou[0]))/(sp[0]-sm[0])*(sm[0]<0)*(sp[0]>0)
	F[2,0] = fu2[0]*(sm[0]<=0) + fu2[1]*(sp[0]<=0) + (sp[0]*fu2[0] - sm[0]*fu2[1] + sp[0]*sm[0]*(E[1]-E[0]))/(sp[0]-sm[0])*(sm[0]<0)*(sp[0]>0)
	F[0,1:] = fu0[:-1]*(sm[:-1]>=0) + fu0[1:]*(sp[:-1]<=0) + (sp[:-1]*fu0[:-1] - sm[:-1]*fu0[1:] + sp[:-1]*sm[:-1]*(rho[1:]-rho[:-1]))/(sp[:-1]-sm[:-1])*(sm[:-1]<0)*(sp[:-1]>0)
	F[1,1:] = fu1[:-1]*(sm[:-1]>=0) + fu1[1:]*(sp[:-1]<=0) + (sp[:-1]*fu1[:-1] - sm[:-1]*fu1[1:] + sp[:-1]*sm[:-1]*(rhou[1:]-rhou[:-1]))/(sp[:-1]-sm[:-1])*(sm[:-1]<0)*(sp[:-1]>0)
	F[2,1:] = fu2[:-1]*(sm[:-1]>=0) + fu2[1:]*(sp[:-1]<=0) + (sp[:-1]*fu2[:-1] - sm[:-1]*fu2[1:] + sp[:-1]*sm[:-1]*(E[1:]-E[:-1]))/(sp[:-1]-sm[:-1])*(sm[:-1]<0)*(sp[:-1]>0)
	return F[:,(inds+1)%N] - F[:,inds]

# Initial conditions - Shock entropy wave
def rho_init(x, p1val, p2val):
	a = np.zeros(len(x))
	a[x<-4] = p1val
	a[x>=-4] = 1 + 0.2*np.sin(np.pi*x[x>=-4])
	return a

def u_init(x, p1val, p2val):
	a = np.zeros(len(x))
	a[x<-4] = p2val
	return a

def p_init(x, p1val, p2val):
	a = np.ones(len(x))
	a[x<-4] = 10.33333
	return a

# Initial conditions on grid
xc = np.linspace(-5.,5.,N+1)[:-1] + h/2.

# Terminal time
T = 1.8

M = 1+int(np.floor(T/dt/downsample_rate))
times = np.zeros(M)
datarho = np.zeros([N, M, Nparamgrid**2])
datau   = np.zeros([N, M, Nparamgrid**2])
dataE   = np.zeros([N, M, Nparamgrid**2])

[p1s, p2s] = np.meshgrid(np.linspace(p1_range[0], p1_range[1], Nparamgrid), np.linspace(p2_range[0], p2_range[1], Nparamgrid))
params = np.vstack([p1s.flatten(), p2s.flatten()]).T

for pind in range(params.shape[0]):

    p1, p2 = params[pind,0], params[pind,1]

    # Start time stepping
    rho_0 = rho_init(xc, p1, p2)
    u_0 = u_init(xc, p1, p2)
    p_0 = p_init(xc, p1, p2)

    rho = rho_0
    u = u_0
    p = p_0

    uu = np.zeros((3,len(rho)))
    uu[0,:] = rho
    uu[1,:] = rho*u
    uu[2,:] = p/(gamma-1) + 1/2*rho*u**2
    t = 0.

    datarho[:,0,pind] = uu[0,:]
    datau[:,0,pind]   = uu[1,:]/uu[0,:]
    dataE[:,0,pind]   = (gamma-1)*(uu[2,:] - 0.5*uu[0,:]*datau[:,0,pind]**2)

    # Data storage stuff
    counter = 0
    index = 0

    while t < T - 1e-12:
            if T-t < dt:
                    dt = T-t
                    lmbda = dt/h
            uu -= lam * HLL_flux(uu)
            # Density
            uu[0,0] = p1
            uu[0,-1] = 1+0.2*(np.sin(np.pi*5))
            # Velocity
            uu[1,0] = p1*p2
            uu[1,-1] = 0
            # Energy
            uu[2,0] = 10.33333/(gamma-1) + 1/2*p1*p2**2
            uu[2,-1] = 1/(gamma-1)
            
            t += dt

            counter += 1
            if (counter % downsample_rate) == 0:
                index += 1
                datarho[:,index,pind] = uu[0,:]
                datau[:,index,pind]   = uu[1,:]/uu[0,:]
                dataE[:,index,pind]   = (gamma-1)*(uu[2,:] - 0.5*uu[0,:]*datau[:,index,pind]**2)
                times[index] = t

np.savez('./data/EulerEqs.npz', datarho, datau, dataE, xc, params, times)

rho = uu[0,:]
u = uu[1,:]/rho
p = (gamma-1)*(uu[2,:]-1/2*rho*u**2)

fig = plt.figure(1)
plt.plot(xc,rho,'k--')
plt.plot(xc,rho_0,color='gray',linestyle='dashed')

fig = plt.figure(2)
plt.plot(xc,u,'k--')
plt.plot(xc,u_0,color='gray',linestyle='dashed')

fig = plt.figure(3)
plt.plot(xc,p,'k--')
plt.plot(xc,p_0,color='gray',linestyle='dashed')

plt.show()
