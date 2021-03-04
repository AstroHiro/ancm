#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:15:54 2020

@author: hiroyasu
"""

import numpy as np
from classncm import NCM
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.models import load_model,Model
import tensorflow
import random as python_random
import control

g = 9.8
m_c = 1.0
m = 0.1
C_mu_c = 0.5
C_mu_p = 0.002
l = 0.5
tensorflow.random.set_seed(1)
python_random.seed(1)

def dynamicsf(X,p):
    """
    Enter input-affine nonlinear dynamical system of interest when u = 0
    (i.e. f of dx/dt = f(x)+g(x)u)

    Parameters
    ----------
    x : ndarray - (n, )
        current state

    Returns
    -------
    fx : ndarray - (n, )
        f(x) of dx/dt = f(x)+g(x)u

    """
    x = X[0]
    th = X[1]
    v = X[2]
    om = X[3]
    mu_c = p[0]*C_mu_c
    mu_p = p[1]*C_mu_p
    f1 = v
    f2 = om
    f4 = (g*np.sin(th)+np.cos(th)*(-m*l*om**2*np.sin(th)+mu_c*v)-mu_p*om/(m*l))/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
    f3 = (m*l*om**2*np.sin(th)-mu_c*v)/(m_c+m)-m*l*f4*np.cos(th)/(m_c+m)
    f = np.array([f1,f2,f3,f4])
    return f

def dynamicsg(X):
    """
    Enter nonlinear actuation matrix of input-affine nonlinear dynamical system 
    (i.e. g of dx/dt = f(x)+g(x)u)

    Parameters
    ----------
    x : ndarray - (n, )
        current state

    Returns
    -------
    gx : ndarray - (n, )
        g(x) of dx/dt = f(x)+g(x)u

    """
    x = X[0]
    th = X[1]
    v = X[2]
    om = X[3]
    g1 = 0
    g2 = 0
    g4 = -np.cos(th)/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
    g3 = 1/(m_c+m)-m*l*g4*np.cos(th)/(m_c+m)
    gfun = np.array([[g1],[g2],[g3],[g4]])
    return gfun

def dynamicsfNN(X,p,internal_model,Nlayers,Nunits):
    X = np.array([X])
    f = internal_model.predict(X)
    W_final = p.reshape((n,-1))
    f = W_final@f[0,:]
    return f

def sdcA(X,p):
    x = X[0]
    th = X[1]
    v = X[2]
    om = X[3]
    mu_c = p[0]*C_mu_c
    mu_p = p[1]*C_mu_p
    Mth = l*(4/3-m*np.cos(th)**2/(m_c+m))
    # sinc(x) = sin(pi*x)/(pi*x)
    Ath = np.array([0,g*np.sinc(th/np.pi)/Mth,mu_c*np.cos(th)/Mth,(-np.cos(th)*m*l*om*np.sin(th)-mu_p/m/l)/Mth])
    M = m_c+m
    Ax = np.array([0,0,-mu_c/M,m*l*om*np.sin(th)/M])
    Ax = Ax-m*l*np.cos(th)/M*Ath
    A12 = np.hstack((np.zeros((2,2)),np.identity(2)))
    A = np.vstack((A12,Ax))
    A = np.vstack((A,Ath))
    return A

def getYf(X):
    x = X[0]
    th = X[1]
    v = X[2]
    om = X[3]
    Yfth1 = np.cos(th)*v/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
    Yfth2 = (-om/(m*l))/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
    Yfx1 = -v/(m_c+m)-m*l*Yfth1*np.cos(th)/(m_c+m)
    Yfx2 = -m*l*Yfth2*np.cos(th)/(m_c+m)
    Yf = np.array([[Yfx1*C_mu_c,Yfx2*C_mu_p],[Yfth1*C_mu_c,Yfth2*C_mu_p]])
    Yf = np.vstack((np.zeros((2,2)),Yf))
    return Yf
    
def getYfNN(X,internal_model):
    n = np.size(X)
    X = np.array([X])
    yf = internal_model.predict(X)
    Nunits = np.size(yf)
    Yf = np.zeros((n,n*Nunits))
    for i in range(n):
        Yf[i,i*Nunits:(i+1)*Nunits] = yf
    return Yf

def getphi(X):
    Yf = getYf(X)
    B = dynamicsg(X)
    B_inv = np.linalg.pinv(B)
    phi = -(B_inv@Yf).T
    return phi
    
def dynamicsf1(X):
    """
    Enter input-affine nonlinear dynamical system of interest when u = 0
    (i.e. f of dx/dt = f(x)+g(x)u)

    Parameters
    ----------
    x : ndarray - (n, )
        current state

    Returns
    -------
    fx : ndarray - (n, )
        f(x) of dx/dt = f(x)+g(x)u

    """
    x = X[0]
    th = X[1]
    v = X[2]
    om = X[3]
    mu_c = 0
    mu_p = 0
    f1 = v
    f2 = om
    f4 = (g*np.sin(th)+np.cos(th)*(-m*l*om**2*np.sin(th)+mu_c*v)-mu_p*om/(m*l))/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
    f3 = (m*l*om**2*np.sin(th)-mu_c*v)/(m_c+m)-m*l*f4*np.cos(th)/(m_c+m)
    f = np.array([f1,f2,f3,f4])
    return f

def dynamics(X,Uar,p):
    x = X[0]
    th = X[1]
    v = X[2]
    om = X[3]
    U = Uar[0] 
    mu_c = p[0]*C_mu_c
    mu_p = p[1]*C_mu_p
    dxdt = v
    dthdt = om
    domdt = (g*np.sin(th)+np.cos(th)*(-U-m*l*om**2*np.sin(th)+mu_c*v)-mu_p*om/(m*l))/(l*(4/3-m*np.cos(th)**2/(m_c+m)))
    dvdt = (U+m*l*(om**2*np.sin(th)-domdt*np.cos(th))-mu_c*v)/(m_c+m)
    dXdt = np.array([dxdt,dthdt,dvdt,domdt])
    return dXdt

def adaptive_law0(x,M,C_Gm,n_p):
    phi = getphi(x)
    B = dynamicsg(x)
    Gm = np.identity(n_p)*C_Gm
    dphatdt = -Gm@phi@B.T@M@x
    return dphatdt

def adaptive_law(x,M,C_Gm,n_p):
    Y = getYf(x)
    Gm = np.identity(n_p)*C_Gm
    dphatdt = Gm@Y.T@M@x
    return dphatdt

def adaptive_law_ancm(x,M,dMx,C_Gm,n_p):
    Y = getYf(x)
    Gm = np.identity(n_p)*C_Gm
    dphatdt = Gm@(Y.T@dMx+Y.T@M)@x
    return dphatdt

def adaptive_law_l(x,C_Gm,Lam,n_p):
    M,C,G,Y = getMCGY(x)
    Gm = np.identity(n_p)*C_Gm
    q = x[0:2]
    q_dot = x[2:4]
    qr_dot = -Lam@q
    s = q_dot-qr_dot
    dphatdt = Gm@Y.T@s
    return dphatdt

def adaptive_law_NN(x,M,C_Gm,n_p,internal_model):
    Y = getYfNN(x,internal_model)
    Gm = np.identity(n_p)*C_Gm
    dphatdt = Gm@Y.T@M@x
    return dphatdt

def adaptive_law_NN_ancm(x,M,dMx,C_Gm,n_p,internal_model):
    Y = getYfNN(x,internal_model)
    Gm = np.identity(n_p)*C_Gm
    dphatdt = Gm@(Y.T@dMx+Y.T@M)@x
    return dphatdt

def lagrange_u(x,p,K,Lam):
    M,C,G,Y = getMCGY(x)
    q = x[0:2]
    q_dot = x[2:4]
    qr_dot = -Lam@q
    qr_ddot = -Lam@q_dot
    s = q_dot-qr_dot
    u = M@qr_ddot+C@qr_dot+G-Y@p-K@s
    return u
    
def getMCGY(X):
    x = X[0]
    th = X[1]
    v = X[2]
    om = X[3]
    Y = getYf(X)
    M = np.array([[m_c+m,m*l*np.cos(th)],[m*l*np.cos(th),4*m*l**2/3]])
    C = np.array([[0,-m*l*om*np.sin(th)],[0,0]])
    G = np.array([0,-m*l*g*np.sin(th)])
    Y = np.array([[-C_mu_c*v,0],[0,-C_mu_p*om]])
    return M,C,G,Y

def rk4(x,fun,Nrk,dt_rk):
        for num in range(0,Nrk):
            k1 = fun(x)
            k2 = fun(x+k1*dt_rk/2.)
            k3 = fun(x+k2*dt_rk/2.)
            k4 = fun(x+k3*dt_rk)
            x = x+dt_rk*(k1+2.*k2+2.*k3+k4)/6.
        return x
    
def getdMx(x,e,p,ncm):
    eps = 0.001
    n = x.shape[0]
    dMx = np.zeros((n,n))
    for i in range(n):
        de = np.zeros(n)
        de[i] = eps
        xn = x+de
        M = ncm.ncm(x,p)
        Mn = ncm.ncm(xn,p)
        dMx[i,:] = (Mn-M)@e/eps/2
    return dMx

def ilqr(x,p,funf,Q,R):
    Ax = getdfdX(x,p,funf)
    Bx = dynamicsg(x)
    Kc,_,_ = control.lqr(Ax,Bx,Q,R)
    u = -Kc@x
    return u

def getdfdX(X,p,funf):
    n = np.size(X)
    eps = 0.0001
    dfdX = np.zeros((n,n))
    for i in range(n):
        de = np.zeros(n)
        de[i] = eps
        Xp = X+de
        Xm = X-de
        dfdX[:,i] = (funf(Xp,p)-funf(Xm,p))/2/eps
    return dfdX


if __name__ == "__main__":
    """
    X = np.array([1,np.pi/3,1,np.pi/3])
    p = np.array([0.5,1.0])
    f1 = dynamicsf1(X)+getYf(X)@p
    f2 = dynamicsf(X,p)
    M,C,G,Y = getMCGY(X)
    f3 = np.linalg.inv(M)@(-C@X[2:4]-G+Y@p)
    print(f1)
    print(f2)
    print(f3)
    """
    # enter your choice of CV-STEM sampling period
    dt = 1
    n = 4
    Ncol = 2
    Nlayers = 3
    Nunits = 5
    Ndata = 10000
    p0 = np.ones(2)*-3.5
    p_true = np.ones(2)
    Lam = np.identity(2)*1
    K_ell = np.identity(2)*10
    B_ell = np.array([[1],[0]])    
    # specify upper and lower bounds of each state
    xlims = np.array([-np.ones(4),np.ones(4)])*np.pi/3
    pmins = np.ones(2)*-2.4
    pmaxs = np.ones(2)*2.4
    plims = np.array([pmins,pmaxs])
    # specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
    alims = np.array([0.15,30])
    # name your NCM
    np.random.seed(seed=226)
    tensorflow.random.set_seed(1)
    python_random.seed(1)
    fname0 = "NCM0cartpole"
    dynamicsf0 = lambda X: dynamicsf(X,p0)
    dynamicsg0 = lambda X: dynamicsg(X)
    ncm = NCM(dt,dynamicsf0,dynamicsg0,xlims,alims,"con",fname0,d1_over=0.0015,d2_over=1.5,da=0.01)
    ncm.Afun = lambda X: sdcA(X,p0)
    ncm.train(iTrain=0)    
    np.random.seed(seed=226)
    tensorflow.random.set_seed(1)
    python_random.seed(1)
    fname = "aNCMcartpole"
    ancm = NCM(dt,dynamicsf,dynamicsg,xlims,alims,"con",fname,d1_over=0.0015,d2_over=1.5,da=0.01,plims=plims)
    ancm.Afun = sdcA
    ancm.getY = getYf
    ancm.train(iTrain=0)
    np.random.seed(seed=88)
    tensorflow.random.set_seed(1)
    python_random.seed(1)
    fnameNN = "aNCMcartpoleNN"
    model = load_model("models/CPdynamics.h5")
    l_final = model.get_layer(index=Nlayers)
    W_final = l_final.get_weights()[0]
    p_trueNN = W_final.T.reshape(n*Nunits)
    internal_model = Model(inputs=model.input,outputs=model.get_layer(index=Nlayers-1).output)
    getf = lambda X,p: dynamicsfNN(X,p,internal_model,Nlayers,Nunits)
    pminsNN = p_trueNN+np.ones(n*Nunits)*-1.5
    pmaxsNN = p_trueNN+np.ones(n*Nunits)*2
    #pmins = p_true
    #pmaxs = p_true
    plimsNN = np.array([pminsNN,pmaxsNN])
    # specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
    ncmNN = NCM(dt,getf,dynamicsg,xlims,alims,"con",fnameNN,d1_over=0.0015,d2_over=1.5,da=0.01,plims=plimsNN)
    ncmNN.getY = lambda X: getYfNN(X,internal_model)
    ncmNN.train(iTrain=0)
    # simultion time step
    np.random.seed(seed=226)
    tensorflow.random.set_seed(1)
    python_random.seed(1)
    Q = np.identity(n)*10
    R = np.identity(1)*0.1
    Qnn = np.identity(n)*7
    Rnn = np.identity(1)*0.1
    dt = 0.1
    # terminal time
    tf = 20
    # initial state
    p0 = np.ones(2)*8.0
    p0NN = pmaxsNN
    x0 = np.random.uniform(low=-np.ones(4)*np.pi/3,high=np.ones(4)*np.pi/3)
    # simulation
    snames = [r"$x$",r"$\theta$",r"$dx/dt$",r"$d\theta/dt$"]
    #phis = np.ones((int(tf/dt),2))*-6.6
    C_Gm = 1
    C_GmNN = 15
    n_con = 8
    n_NN = p0NN.size
    XX = np.zeros((n_con,n))
    pp = np.zeros((n_con,2))
    ppNN = np.zeros((n_con,n_NN))
    if dt <= ancm.dt_rk:
        ancm.dt_rk = dt
    if dt <= ncm.dt_rk:
        ncm.dt_rk = dt
    if dt <= ncmNN.dt_rk:
        ncmNN.dt_rk = dt
    ancm.Nrk = int(dt/ancm.dt_rk)
    ncm.Nrk = int(dt/ncm.dt_rk)
    ncmNN.Nrk = int(dt/ncmNN.dt_rk)
    N = int(tf/dt)
    this = np.zeros(N+1)
    XXhis = np.zeros((n_con,n,N+1))
    pphis = np.zeros((n_con,2,N+1))
    ppNNhis = np.zeros((n_con,n_NN,N+1))
    UUhis = np.zeros((n_con,1,N))
    for c in range(n_con):
        XX[c,:] = x0
        XXhis[c,:,0] = x0 
        pp[c,:] = p0
        pphis[c,:,0] = p0 
        ppNN[c,:] = p0NN
        ppNNhis[c,:,0] = p0NN 
    t = 0
    this[0] = t
    dscale = 1e2
    for k in range(N):
        for c in range(n_con):
            p = pp[c,:]
            pNN = ppNN[c,:]
            x = XX[c,:]
            if c == 0:
                d1 = ancm.unifrand2(ancm.d1_over,np.size(ancm.Bw(x,p),1))*dscale
                print(d1)
            if c == 0:
                e = x
                Ma = ancm.ncm(x,p)
                dMx = getdMx(x,e,p,ancm)
                fun_a = lambda p: adaptive_law_ancm(x,Ma,dMx,C_Gm,2)
                p = rk4(p,fun_a,ancm.Nrk,ancm.dt_rk)
                print(p)
                Mx = ancm.ncm(x,p)
                Bx = ancm.h_or_g(x,p)
                Kx = Bx.T@Mx
                u = -Kx@x
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ancm.Nrk,ancm.dt_rk)+ancm.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                pp[c,:] = p
                XXhis[c,:,k+1] = x
                pphis[c,:,k+1] = p
                UUhis[c,:,k] = u
            if c == 1:
                #Mx = ncm.ncm(x,np.empty(0))
                Mx = ancm.ncm(x,p0)
                Bx = dynamicsg(x)
                Kx = Bx.T@Mx
                u = -Kx@x
                dEf = lambda x,p: ncm.h_or_g(x,p)@u
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ncm.Nrk,ncm.dt_rk)+ncm.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                pp[c,:] = p0
                XXhis[c,:,k+1] = x
                pphis[c,:,k+1] = p0
                UUhis[c,:,k] = u
            if c == 2:
                Mx = ncm.ncm(x,np.empty(0))
                fun_a = lambda x_upd,x_fix,dEf: adaptive_law0(x_fix,Mx,C_Gm,2)
                dEf = None
                p = ncm.rk4(p,x,dEf,fun_a)
                phi = getphi(x)
                Bx = dynamicsg(x)
                Kx = Bx.T@Mx
                u = -Kx@x+phi.T@p
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ncm.Nrk,ncm.dt_rk)+ncm.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                pp[c,:] = p
                XXhis[c,:,k+1] = x
                pphis[c,:,k+1] = p
                UUhis[c,:,k] = u
            if c == 3:
                fun_a = lambda p: adaptive_law_l(x,C_Gm,Lam,2)
                p = rk4(p,fun_a,ncm.Nrk,ncm.dt_rk)
                u = np.linalg.pinv(B_ell)@lagrange_u(x,p,K_ell,Lam)
                print(p)
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ncm.Nrk,ncm.dt_rk)+ncm.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                pp[c,:] = p
                XXhis[c,:,k+1] = x
                pphis[c,:,k+1] = p
                UUhis[c,:,k] = u
            if c == 4:
                e = x
                Ma = ncmNN.ncm(x,pNN)
                dMx = getdMx(x,e,pNN,ncmNN)
                fun_a = lambda p: adaptive_law_NN_ancm(x,Ma,dMx,C_GmNN,n_NN,internal_model)
                pNN = rk4(pNN,fun_a,ncmNN.Nrk,ncmNN.dt_rk)
                print(pNN)
                Mx = ncmNN.ncm(x,pNN)
                Bx = ncmNN.h_or_g(x,pNN)
                Kx = Bx.T@Mx
                u = -Kx@x
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ncmNN.Nrk,ncmNN.dt_rk)+ncmNN.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                ppNN[c,:] = pNN
                XXhis[c,:,k+1] = x
                ppNNhis[c,:,k+1] = pNN
                UUhis[c,:,k] = u
            if c == 5:
                Mx = ncmNN.ncm(x,p0NN)
                Bx = ncmNN.h_or_g(x,p0NN)
                Kx = Bx.T@Mx
                u = -Kx@x
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ncmNN.Nrk,ncmNN.dt_rk)+ncmNN.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                ppNN[c,:] = pNN
                XXhis[c,:,k+1] = x
                ppNNhis[c,:,k+1] = pNN
                UUhis[c,:,k] = u
            if c == 6:
                funf = dynamicsf
                u = ilqr(x,p0,funf,Q,R)
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ncm.Nrk,ncm.dt_rk)+ncm.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                pp[c,:] = p0
                XXhis[c,:,k+1] = x
                pphis[c,:,k+1] = p0
                UUhis[c,:,k] = u
            if c == 7:
                funf = lambda x,p: dynamicsfNN(x,p,internal_model,Nlayers,Nunits)
                u = ilqr(x,p0NN,funf,Qnn,Rnn)
                fun = lambda x: dynamicsf(x,p_true)+dynamicsg(x)@u
                x = rk4(x,fun,ncmNN.Nrk,ncmNN.dt_rk)+ncmNN.Bw(x,p_true)@d1*dt
                XX[c,:] = x
                ppNN[c,:] = p0NN
                XXhis[c,:,k+1] = x
                ppNNhis[c,:,k+1] = p0NN
                UUhis[c,:,k] = u
                
        t += dt
        this[k+1] = t       
        
    np.save("data/this.npy",this)
    np.save("data/XXhis.npy",XXhis)
    np.save("data/UUhis.npy",UUhis)
    np.save("data/pphis.npy",pphis)
    np.save("data/ppNNhis.npy",ppNNhis)
    
    #this,xhis,_,d1his = ancm.simulation(dt,tf,x0,p_true,C_Gm,dscale=1e2,xnames=snames,Ncol=Ncol,FigSize=(20,10),phis=phis)
    #this,xhisNN,_,d1hisNN = ancm.simulation(dt,tf,x0,p_true,C_Gm,dscale=1e2,xnames=snames,Ncol=Ncol,FigSize=(20,10),phis=phis,d1his=d1his,idx_dhis=1,idx_adaptive=1)
    
    xnames = snames
    FontSize = 20
    FigSize=(20,10)
    matplotlib.rcParams.update({"font.size": 15})
    matplotlib.rc("text",usetex=True)
    tit1 = "Performance of NCM-based Control (1)"
    tit2 = "Performance of NCM-based Control (2)"
    ly = r"tracking error: $\|x-x_d\|_2$"
    l1 = r"tracking error"
    l2 = r"optimal steady-state upper bound"
    plt.figure()
    for c in range(n_con):
        plt.plot(this,np.sqrt(np.sum((XXhis[c,:,:])**2,0)))
        #plt.plot(this,np.ones(np.size(this))*ancm.Jcv_opt)
    plt.xlabel(r"time",fontsize=FontSize)
    plt.ylabel(ly,fontsize=FontSize)
    plt.ylim([0,3.5])
    #plt.legend([l1,l2],loc="best")
    #plt.title(tit1,fontsize=FontSize)
    plt.show()