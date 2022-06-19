# Solves the 2-dimensional KPP equation with periodic boundary conditions.
# NOTE: this has periodic boundary conditions, so running the scheme long
# enough will cause things to wrap around. Mitigate by changing xl, xr, yl, yr.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import matplotlib.animation as animation

# Possible parameters to vary: 
# Nx, Ny: number of degrees of freedom
# T: terminal time (number of snapshots)
# domain parameters, xl, xr, yl, yr
# Initial condition u0

# Flux terms
f = lambda u:np.sin(u)
g = lambda u:np.cos(u)

# Initial condition 
u0 = lambda xx,yy: np.pi/4.+13.*np.pi/4.*(xx**2+yy**2<1.)

# Domain parameters: Space is the box [xl, xr] x [yl, yr]. Time is the interval [0, T].
xl = -2.0
xr = 2.0
yl = -2.5
yr = 1.5
T = 10

# Spatial discretization
Nx = 50
Ny = 50
hx = (xr - xl)/Nx
hy = (yr - yl)/Ny

# CFL condition
k = (np.minimum(hx,hy))/10.
lmbdax = k/hx
lmbday = k/hy

# Animation logging
Nsteps = int(np.ceil(T/k))
U_lf = np.zeros((Nx, Ny, 1 + Nsteps))
U_eno = np.zeros((Nx, Ny, 1 + Nsteps))

# Auxiliary variable
indrx = np.arange(Nx,dtype = int)
indry = np.arange(Ny,dtype = int)

# Set up initial condition
xc = np.linspace(xl,xr,Nx+1,dtype = float)[:-1]+hx/2.
yc = np.linspace(yl,yr,Ny+1,dtype = float)[:-1]+hy/2.
xv,yv = np.meshgrid(xc,yc)
u_eno = u0(xv,yv)
u_lf = u0(xv,yv)

U_lf[:,:,0] = u_lf
U_eno[:,:,0] = u_eno
step = 0

# Coefficients of ENO scheme
C_rl = np.array([[11./6.,-7./6.,1./3.],[1./3.,5./6.,-1./6.],[-1./6.,5./6.,1./3.],[1./3.,-7./6.,11./6.]])

###########################################################
# Lax-Friedrichs Scheme ###################################
###########################################################
# Flux on x direction
def lfx(u):
    alpha = 1.
    F = np.zeros(u.shape)
    fu = f(u)    
    F = 0.5*(fu[(indrx+1)%Nx]+fu[indrx])-0.5*alpha*(u[(indrx+1)%Nx]-u[indrx])

    return F[indrx]-F[(indrx-1)%Nx]

# Flux on y direction
def lfy(u):
    alpha = 1.
    gu = g(u)    
    F = 0.5*(gu[(indry+1)%Ny]+gu[indry])-0.5*alpha*(u[(indry+1)%Ny]-u[indry])

    return F[indry]-F[(indry-1)%Ny]

###########################################################
# ENO-LF Scheme ###########################################
###########################################################
# Flux on x direction
def ENO_lfx(u):
    # Stencil number 
    r = np.zeros(u.shape,dtype = int)

    # Step 1: Calculating the first order divided difference
    dd1 = u[(indrx+1)%Nx]-u[indrx]
    flags = (np.abs(dd1[indrx])>np.abs(dd1[(indrx-1)%Nx]))
    r[indrx[flags]] = r[indrx[flags]]+1
    
    # Step 2: Calculating the second order divided difference
    dd1 = u[(indrx-r+1)%Nx]-u[(indrx-r)%Nx] # 1st order divided difference
    dd1_r = u[(indrx-r+2)%Nx]-u[(indrx-r+1)%Nx] # 1st order divided difference (right)
    dd1_l = u[(indrx-r)%Nx]-u[(indrx-r-1)%Nx] # 1st order divided difference (left)
    flags = (np.abs(dd1_r-dd1)>np.abs(dd1-dd1_l))
    r[indrx[flags]] = r[indrx[flags]]+1

    # Step 3: Reconstruction of pointwise value according to stencil r.
    uminus = C_rl[r+1,0]*u[(indrx-r)%Nx]+C_rl[r+1,1]*u[(indrx-r+1)%Nx]+C_rl[r+1,2]*u[(indrx-r+2)%Nx]
    uplus = C_rl[r,0]*u[(indrx-r)%Nx]+C_rl[r,1]*u[(indrx-r+1)%Nx]+C_rl[r,2]*u[(indrx-r+2)%Nx] 
    fminus = f(uminus)
    fplus = f(uplus)
    F = 0.5*(fplus[indrx]+fminus[(indrx-1)%Nx])-0.5*(uplus[indrx]-uminus[(indrx-1)%Nx])

    return F[(indrx+1)%Nx]-F[indrx]

# Flux on y direction
def ENO_lfy(u):
    # Stencil number 
    r = np.zeros(u.shape,dtype = int)

    # Step 1: Calculating the first order divided difference
    dd1 = u[(indry+1)%Ny]-u[indry]
    flags = (np.abs(dd1[indry])>np.abs(dd1[(indry-1)%Ny]))
    r[indry[flags]] = r[indry[flags]]+1
    
    # Step 2: Calculating the second order divided difference
    dd1 = u[(indry-r+1)%Ny]-u[(indry-r)%Ny] # 1st order divided difference
    dd1_r = u[(indry-r+2)%Ny]-u[(indry-r+1)%Ny] # 1st order divided difference (right)
    dd1_l = u[(indry-r)%Ny]-u[(indry-r-1)%Ny] # 1st order divided difference (left)
    flags = (np.abs(dd1_r-dd1)>np.abs(dd1-dd1_l))
    r[indry[flags]] = r[indry[flags]]+1

    # Step 3: Reconstruction of pointwise value according to stencil r.
    uminus = C_rl[r+1,0]*u[(indry-r)%Ny]+C_rl[r+1,1]*u[(indry-r+1)%Ny]+C_rl[r+1,2]*u[(indry-r+2)%Ny]
    uplus = C_rl[r,0]*u[(indry-r)%Ny]+C_rl[r,1]*u[(indry-r+1)%Ny]+C_rl[r,2]*u[(indry-r+2)%Ny] 
    gminus = g(uminus)
    gplus = g(uplus)
    F = 0.5*(gplus[indry]+gminus[(indry-1)%Ny])-0.5*(uplus[indry]-uminus[(indry-1)%Ny])

    return F[(indry+1)%Ny]-F[indry]

# Time loop
t = 0.0
while t<T-1e-12:
    if T-t < k:
        k = (T-t)
        lmbdax = k/hx
        lmbday = k/hy

    t += k

    # L-F scheme
    du = np.zeros(u_lf.shape)
    for i in range(Nx):
        du[i,:] = -lmbdax*lfx(u_lf[i,:])
    for j in range(Ny):
        du[:,j] -= lmbday*lfy(u_lf[:,j])
    u_lf +=du
    U_lf[:,:,step+1] = u_lf

    # ENO-LF scheme
    du = np.zeros(u_eno.shape)
    for i in range(Nx):
        du[i,:] = -lmbdax*ENO_lfx(u_eno[i,:])
    for j in range(Ny):
        du[:,j] -= lmbday*ENO_lfy(u_eno[:,j])
    u_eno +=du    
    U_eno[:,:,step+1] = u_eno

    step += 1

np.savez('data/KPP',xv, yv, U_lf)
print(U_lf.shape)

"""
# Contour plots
fig = plt.figure()
ax1 = plt.subplot(2,2,1)
lf_contour = plt.contour(xv,yv,u_lf,20,cmap = cm.jet)
time_template = 'time = {0:1.3f}'
time_text = ax1.text(0.05, 0.90, '', transform=ax1.transAxes)
ax1.set_title("LF scheme at $T = 1$")

ax2 = plt.subplot(2,2,2)
plt.contour(xv,yv,u_eno,20,cmap = cm.jet)
ax2.set_title("ENO scheme at $T = 1$")

# Surface plots
ax3 = plt.subplot(2,2,3,projection='3d')
ax3.plot_surface(xv, yv, u_lf, cmap=cm.coolwarm, linewidth=0, antialiased=False)

ax4 = plt.subplot(2,2,4,projection='3d')
ax4.plot_surface(xv, yv, u_eno, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()

def animation_init():
    time_text.set_text('')
    return [ax1, ax2, ax3, ax4]

def animation_update(i):
    ax1.clear()
    ax1.contour(xv, yv, U_lf[:,:,i], 20, cmap=cm.jet)

    ax2.clear()
    ax2.contour(xv, yv, U_eno[:,:,i], 20, cmap=cm.jet)

    ax3.clear()
    ax3.plot_surface(xv, yv, U_lf[:,:,i], cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax4.clear()
    ax4.plot_surface(xv, yv, U_eno[:,:,i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    time_text.set_text(time_template % (i*k))
    return [ax1, ax2, ax3, ax4]

#ani = animation.FuncAnimation(fig, animation_update, np.arange(0, Nsteps, 1),
                #   interval=1, blit=True, init_func=animation_init, repeat_delay=300)

#plt.show()
"""
