#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:06:14 2020

@author: hiroyasu
"""


import numpy as np
from classncm import NCM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model,Model
import matplotlib.pyplot as plt
import matplotlib

g = 9.8
m_c = 1.0
m = 0.1
mu_c = 0.5
mu_p = 0.002
l = 0.5

def dynamicsf(X):
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

def dynamicsfNN0(X,model):
    X = np.array([X])
    f = model.predict(X)
    return f[0,:]

def dynamicsNN2(X,internal_model,Nlayers):
    X = np.array([X])
    f = internal_model.predict(X)
    l_final = model.get_layer(index=Nlayers)
    W_final = l_final.get_weights()[0]
    f = W_final.T@f[0,:]
    return f

def dynamicsfNN(X,p,internal_model,Nlayers,Nunits):
    X = np.array([X])
    f = internal_model.predict(X)
    W_final = p.reshape((n,-1))
    f = W_final@f[0,:]
    return f
    
def getYf(X,internal_model):
    n = np.size(X)
    X = np.array([X])
    yf = internal_model.predict(X)
    Nunits = np.size(yf)
    Yf = np.zeros((n,n*Nunits))
    for i in range(n):
        Yf[i,i*Nunits:(i+1)*Nunits] = yf
    return Yf
    

if __name__ == "__main__":
    n = 4
    Nlayers = 3
    Nunits = 5
    Ndata = 10000
    """
    Nbatch = 32
    Nepochs = 10000
    ValidationSplit = 0.1
    Patience = 20
    Verbose = 2
    xmin = np.array([-5,-np.pi,-10,-2*np.pi])
    xmax = np.array([5,np.pi,10,2*np.pi])
    xs = np.random.uniform(xmin,xmax,size=(Ndata,n))
    fs = np.zeros_like(xs)
    for i in range(Ndata):
        fs[i,:] = dynamicsf(xs[i,:])
        
    model = Sequential(name="CPdynamics")
    model.add(Dense(Nunits,activation="tanh",input_shape=(n,)))
    for l in range(Nlayers-1):
        model.add(Dense(Nunits,activation="tanh"))
    model.add(Dense(n,use_bias=False))
    model.summary()
    model.compile(loss="mean_squared_error",optimizer="adam")
    es = EarlyStopping(monitor="val_loss",patience=Patience)
    model.fit(xs,fs,batch_size=Nbatch,epochs=Nepochs,verbose=Verbose,\
              callbacks=[es],validation_split=ValidationSplit)
    model.save("models/CPdynamics.h5")
    """
    model = load_model("models/CPdynamics.h5")
    l_final = model.get_layer(index=Nlayers)
    W_final = l_final.get_weights()[0]
    p_true = W_final.T.reshape(n*Nunits)
    internal_model = Model(inputs=model.input,outputs=model.get_layer(index=Nlayers-1).output)
    getf = lambda X,p: dynamicsfNN(X,p,internal_model,Nlayers,Nunits)
    
    #X = np.array([1,np.pi/6,-2,np.pi/3])
    #p = np.ones(n*Nunits)*10
    #test0 = dynamicsfNN0(X,model)
    #test1 = getf(X,p_true)
    #test2 = getYf(X)@p_true
    #print(test0-test1)
    #print(test0-test2)
    
    # enter your choice of CV-STEM sampling period
    dt = 1
    # specify upper and lower bounds of each state
    xlims = np.array([-np.ones(4),np.ones(4)])*np.pi/3
    pmins = p_true+np.ones(n*Nunits)*-1.5
    pmaxs = p_true+np.ones(n*Nunits)*2
    #pmins = p_true
    #pmaxs = p_true
    plims = np.array([pmins,pmaxs])
    # specify upper and lower bound of contraction rate (and we will find optimal alpha within this region)
    alims = np.array([0.15,30])
    # name your NCM
    np.random.seed(seed=88)
    fname = "aNCMcartpoleNN"
    ncm = NCM(dt,getf,dynamicsg,xlims,alims,"con",fname,d1_over=0.0015,d2_over=1.5,da=0.01,plims=plims)
    ncm.getY = lambda X: getYf(X,internal_model)
    ncm.train(iTrain=1)
    # simultion time step
    np.random.seed(seed=88)
    dt = 0.1
    # terminal time
    tf = 50
    # initial state
    x0 = np.random.uniform(low=-np.ones(4)*np.pi/3,high=np.ones(4)*np.pi/3)
    # simulation
    snames = [r"$x$",r"$\theta$",r"$dx/dt$",r"$d\theta/dt$"]
    phis = np.tile(pmaxs,(int(tf/dt),1))
    #phis = np.ones((int(tf/dt),n*Nunits))*0.1
    ncm.dynamicsf = lambda X,p: dynamicsf(X)
    C_Gminv = 15
    this,xhis,_,d1his = ncm.simulation(dt,tf,x0,p_true,C_Gminv,dscale=1e2,xnames=snames,Ncol=2,FigSize=(20,10),phis=phis)
    this,xhisNN,_,d1hisNN = ncm.simulation(dt,tf,x0,p_true,C_Gminv,dscale=1e2,xnames=snames,Ncol=2,FigSize=(20,10),phis=phis,d1his=d1his,idx_dhis=1,idx_adaptive=1)
    
    n = 4
    Ncol = 2
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
    plt.plot(this,np.sqrt(np.sum((xhis)**2,1)))
    plt.plot(this,np.sqrt(np.sum((xhisNN)**2,1)))
    plt.plot(this,np.ones(np.size(this))*ncm.Jcv_opt)
    plt.xlabel(r"time",fontsize=FontSize)
    plt.ylabel(ly,fontsize=FontSize)
    plt.ylim([0,3.5])
    plt.legend([l1,l2],loc="best")
    plt.title(tit1,fontsize=FontSize)
    plt.show()
        
    Nrow = int(n/Ncol)+np.remainder(n,Ncol)
    fig,ax = plt.subplots(Nrow,Ncol,figsize=FigSize)
    plt.subplots_adjust(wspace=0.25,hspace=0.25)
    if Ncol == 1:
        ax = np.reshape(ax,(n,1))
    elif Nrow == 1:
        ax = np.reshape(ax,(1,n))
    if xnames == "num":
        xnames = []
        for i in range(n):
            xnames += [r"state "+str(i+1)]
    for row in range(Nrow):
        for col in range(Ncol):
            i = Ncol*row+col
            if i+1 <= n:
                ax[row,col].plot(this,xhis[:,i])
                ax[row,col].plot(this,xhisNN[:,i])
                ax[row,col].set_xlabel(r"time",fontsize=FontSize)
                LabelName = r"tracking error: "+xnames[i]
                ax[row,col].set_ylabel(LabelName,fontsize=FontSize)
                ax[row,col].set_ylim([np.min(xhisNN[:,i])-0.2,np.max(xhisNN[:,i])+0.2])
    fig.suptitle(tit2,fontsize=FontSize)
    plt.show()
    
    