#!/usr/bin/env python

from networkx import *
from vbmod import *


# read in list of edges
G=read_edgelist('../dat/football/football.edgelist')

# convert networkx graph object to sparse matrix
A=to_scipy_sparse_matrix(G)

N=A.shape[0]        # number of nodes
Kvec=range(10,15+1) # range of K values over which to search

# hyperparameters for priors
net0={}
net0['ap0']=N*1.
net0['bp0']=1.
net0['am0']=1.
net0['bm0']=N*1.

# options
opts={}
opts['NUM_RESTARTS']=25

# run vb
(net,net_K)=learn_restart(A.tocsr(),Kvec,net0,opts)

# display figures
restart_figs(A,net,net_K)
show()

