import numpy as np
from qutip import *
from math import sqrt
from scipy import *
import matplotlib.pyplot as plt
import pandas as pd
import time

N=4                                  # number of energy levels (we need just ground state)
M =10                             # Problem size
taumax = 30                        
taulist = np.linspace(0, taumax,30)
num_train=25000
num_test=1000
Seed_train=1234
Seed_test=1451
psi_list = [(basis(2,0)+basis(2,1))/np.sqrt(2) for n in range(M)]
psi0 = tensor(psi_list)
def flatten(l):
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
def adiabatic_evolution(Seed,num):
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(M):
        op_list = []
        for m in range(M):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))
    
    H0=0
    for n in range(M):
        H0 += -sx_list[n]
    




    count=0
    gap=[]                      
    Data_eachtrial=[]                 # to store Jmat,Kmat
    Data=[]                           # to store all Jmats,Kmats
    Expects=[]
    np.random.seed(Seed)                
    
    for i in range(1,num+1):
        count=count+1
        #print(count,end="      \r")
        Jmat=np.random.uniform(low=-1, high=1, size=(1,M) )
        Kmat=np.random.uniform(low=-1, high=1, size=(1,M) )
        
        H1 = 0
        for n in range(M):
            H1 += Kmat[0,n]*sz_list[n]
        



        H1 +=Jmat[0,M-1]* sz_list[M-1]*sz_list[0]
        for n1 in range(M-1) :
                H1 +=(Jmat[0,n1]*sz_list[n1]*sz_list[n1+1])
                
        
                
        args = {'t_max': max(taulist)}
        h_t = [[H0, lambda t, args :( 1-(t/args['t_max']))],
                   [H1, lambda t, args : (t/args['t_max'])]]
        
        evals_mat = np.zeros((len(taulist),N))
        P_mat = np.zeros((len(taulist),N))
        exx=np.zeros((len(taulist),3*M))
        
        idx = [0]
        def process_rho(tau, psi):
  
    # evaluate the Hamiltonian with gradually switched on interaction 
            H = qobj_list_evaluate(h_t, tau, args)

    # find the N lowest eigenvalues of the system
            evals, ekets = H.eigenstates(eigvals=N)

            evals_mat[idx[0],:] = real(evals)
    
    # find the overlap between the eigenstates and psi 
            for n, eket in enumerate(ekets):
                P_mat[idx[0],n] = abs((eket.dag().data * psi.data)[0,0])**2  
            exx[idx[0],:]=(np.hstack(np.array([[expect(sz_list[f],psi)for f in range(M)],
                          [expect(sy_list[f],psi) for f in range(M)],[expect(sx_list[f],psi) for f in range(M)]
                           ]))   
                              )
        
            idx[0] += 1
            return(psi)
    #solving schrodinger equation
        mesolve(h_t, psi0, taulist,[] ,process_rho, args,options=None, _safe_mode=True)
        gap.append(list(np.array(evals_mat[:,1]-evals_mat[:,0]).flatten()))
        
        initialdata=([Jmat.tolist(),Kmat.tolist()])
        Data_eachtrial=flatten(initialdata)
        
        Data.append(Data_eachtrial)
        Expects.append(list(exx.flatten()))
        
       
       
    
        
    
    return(gap,Data,Expects)

gap_train,Data_train,Expects_train=adiabatic_evolution(Seed_train,num_train)
gap_test,Data_test,Expects_test=adiabatic_evolution(Seed_test,num_test)
df_p_train = pd.DataFrame(gap_train)
df_p_train.to_csv("gap_train_tau=%d,M=%d,num_train=%d,seed_train=%d.dat"% (taumax, M,num_train,Seed_train), header=False, index=False)
df_p_train = pd.DataFrame(gap_test)
df_p_train.to_csv("gap_test_tau=%d,M=%d,num_test=%d,seed_test=%d.dat"% (taumax, M,num_test,Seed_test), header=False, index=False)
df_p_test = pd.DataFrame(Data_test)#, header=None)
df_p_test.to_csv("Data_test_tau=%d,M=%d,num_test=%d,seed_test=%d.dat"% (taumax, M,num_test,Seed_test), header=False, index=False)
df_p_test = pd.DataFrame(Data_train)#, header=None)
df_p_test.to_csv("Data_train_tau=%d,M=%d,num_train=%d,seed_train=%d.dat"% (taumax, M,num_train,Seed_train), header=False, index=False)
df_p_test = pd.DataFrame(Expects_test)#, header=None)
df_p_test.to_csv("Expects_test_tau=%d,M=%d,num_test=%d,seed_test=%d.dat"% (taumax, M,num_test,Seed_test), header=False, index=False)
df_p_test = pd.DataFrame(Expects_train)#, header=None)
df_p_test.to_csv("Expects_train_tau=%d,M=%d,num_train=%d,seed_train=%d.dat"% (taumax, M,num_train,Seed_train), header=False, index=False)
