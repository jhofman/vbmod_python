#!/usr/bin/env python
"""

Copyright (C) 2007, 2008 Jake Hofman <jhofman@gmail.com>
Distributed under GPL 3.0
http://www.gnu.org/licenses/gpl.txt

jntj: todo
  - pivec in rnd
  - make to a class, object oriented
  - switch for inline usage?
  - error checking
"""

# import modules
from scipy import special, comb
from scipy.special import digamma, betaln, gammaln, exp, log
from scipy.sparse import lil_matrix
from scipy import diagonal  # for scipy < 0.7, use 'from scipy import extract_diagonal as diagonal'
from numpy import *
from pylab import spy, show, imshow, axis, plot, figure, subplot, xlabel, ylabel, title, grid, hold, legend
from matplotlib.ticker import FormatStrFormatter
import scipy.weave as weave
from time import time


def init(N,K):
    """
    returns randomly-initialized mean-field matrix Q0 for vbmod. 
    
    inputs:
      N: number of nodes
      K: (maximum) number of modules
    
    outputs:
      Q: N-by-K mean-field matrix (rows sum to 1)
    
    """
    
    Q=mat(random.random([N,K]))
    Q=Q/(Q.sum(1)*ones([1,K]))
    
    return Q


def rnd(N,K,tp,tm):
    """
    
    sample from vbmod likelihood. generates a random adjacency matrix
    sampled from a constrained stochastic block model specified by the
    given parameters.
    
    inputs:
      N: number of nodes
      K: number of modules
      tp: \theta_+, probability of edge within modules; tp=prob(Aij=1|zi=zj)
      tm: \theta_-, probability of edge between modules; tm=prob(Aij=1|zi!=zj)
    
    outputs:
      A: N-by-N adjacency matrix (logical, sparse)

    """
    # jntj: need pivec in here too
    mask=matrix(kron(eye(K),ones([N/K,N/K])))
    Qtrue=matrix(kron(eye(K),ones([N/K,1])))

    # jntj: very slow ... 
    A=multiply(tp>random.random([N,N]),mask)+multiply(tm>random.random([N,N]),1-mask)
    A=triu(A,1)
    A=A+A.transpose()
    
    A=lil_matrix(A,dtype='b')
    A=A.tocsr()

    return A

def learn(A,K,net0={},opts={}):
    """
    runs variational bayes for inference of network modules
    (i.e. community detection) under a constrained stochastic block
    model.
    
    net0 and opts inputs are optional. if provided, length(net0.a0)
    must equal K.
    
    inputs:
      A: N-by-N undirected (symmetric), binary adjacency matrix w/o
         self-edges (note: fastest for sparse and logical A)
      K: (maximum) number of modules
      net0: initialization/hyperparameter structure for network
        net0['Q0']: N-by-K initial mean-field matrix (rows sum to 1)
        net0['ap0']: alpha_{+0}, hyperparameter for prior on \theta_+
        net0['bp0']: beta_{+0}, hyperparameter for prior on \theta_+
        net0['am0']: alpha_{-0}, hyperparameter for prior on \theta_-
        net0['bm0']: beta_{-0}, hyperparameter for prior on \theta_-
        net0['a0']: alpha_{\mu0}, 1-by-K vector of hyperparameters for
                 prior on \pi
      opts: options
        opts['TOL_DF']: tolerance on change in F (outer loop)
        opts['MAX_FITER']: maximum number of F iterations (outer loop)
        opts['VERBOSE']: verbosity (0=quiet (default),1=print, 2=figures)
    
    outputs:
      net: posterior structure for network
        net['F']: converged free energy (same as net.F_iter(end))
        net['F_iter']: free energy over iterations (learning curve)
        net['Q']: N-by-K mean-field matrix (rows sum to 1)
        net['K']: K, passed for compatibility with vbmod_restart
        net['ap']: alpha_+, hyperparameter for posterior on \theta_+
        net['bp']: beta_+, hyperparameter for posterior on \theta_+
        net['am']: alpha_-, hyperparameter for posterior on \theta_-
        net['bm']: beta_-, hyperparameter for posterior on \theta_-
        net['a']: alpha_{\mu}, 1-by-K vector of hyperparameters for
                 posterior on \pi
    """
    # default options
    TOL_DF=1e-2
    MAX_FITER=30
    VERBOSE=0
    SAVE_ITER=0

    # get options from opts struct
    if (type(opts) == type({})) and (len(opts) > 0):
        if 'TOL_DF' in opts: TOL_DF=opts['TOL_DF']
        if 'MAX_FITER' in opts: MAX_FITER=opts['MAX_FITER']
        if 'VERBOSE' in opts: VERBOSE=opts['VERBOSE']
        if 'SAVE_ITER' in opts: SAVE_ITER=opts['SAVE_ITER']        

    N=A.shape[0]        # number of nodes
    M=0.5*A.sum(0).sum(1)  # total number of non-self edges
    M=M[0,0]
    C=comb(N,2)    # total number of possible edges between N nodes
    
    uk=mat(ones([K,1]))
    un=mat(ones([N,1]))
    
    # default prior hyperparameters
    ap0=2;
    bp0=1;
    am0=1;
    bm0=2;
    a0=ones([1,K]);

    # get initial Q0 matrix and prior hyperparameters from net0 struct
    if (type(net0) == type({})) and (len(net0) > 0):
        if 'Q0' in net0: Q=net0['Q0']
        if 'ap0' in net0: ap0=net0['ap0']
        if 'bp0' in net0: bp0=net0['bp0']
        if 'am0' in net0: am0=net0['am0']
        if 'bm0' in net0: bm0=net0['bm0']
        if 'a0' in net0: a0=net0['a0']

    # initialize Q if not provided
    try: Q
    except NameError: Q=init(N,K)
    #Q=init(N,K)

    # ensure a0 is a 1-by-K vector
    assert(a0.shape == (1,K))
    
    # intialize variational distribution hyperparameters to be equal to
    # prior hyperparameters
    ap=ap0
    bp=bp0
    am=am0
    bm=bm0
    a=a0
    n=Q.sum(0)

    # get indices of non-zero row/columns
    # to be passed to vbmod_estep_inline
    # jntj: must be better way
    (rows,cols)=where(A.toarray())

    # vector to store free energy over iterations
    F=[]

    for i in range(MAX_FITER):
        ####################
        # VBE-step, to update mean-field Q matrix over module assignments
        ####################
        
        # compute local and global coupling constants, JL and JG and
        # chemical potentials -lnpi
        psiap=digamma(ap)
        psibp=digamma(bp)
        psiam=digamma(am)
        psibm=digamma(bm)
        psip=digamma(ap+bp)
        psim=digamma(am+bm)
        JL=psiap-psibp-psiam+psibm
        JG=psibm-psim-psibp+psip

        lnpi=digamma(a)-digamma(sum(a))

        Q=array(Q)
        estep_inline(rows,cols,Q,float(JL),float(JG),array(lnpi),array(n))
        Q=mat(Q)
    
        """
        # local update (technically correct, but slow)
        for l in range(N):
            # exclude Q[l,:] from contributing to its own update
            Q[l,:]=zeros([1,K])
            # jntj: doesn't take advantage of sparsity
            Al=mat(A.getrow(l).toarray())
            AQl=multiply((Al.T*uk.T),Q).sum(0)
            nl=Q.sum(0)
            lnQl=JL*AQl-JG*nl+lnpi
            lnQl=lnQl-lnQl.max()
            Q[l,:]=exp(lnQl)
            Q[l,:]=Q[l,:]/Q[l,:].sum()
        """

        ####################
        # VBM-step, update distribution over parameters
        ####################

        # compute expected occupation numbers <n*>s
        n=Q.sum(0)
    
        # compute expected edge counts <n**>s
        #QTAQ=mat((Q.T*A*Q).toarray())
        #npp=0.5*trace(QTAQ)
        npp=0.5*diagonal(Q.T*A*Q).sum()
        npm=0.5*trace(Q.T*(un*n-Q))-npp
        nmp=M-npp
        nmm=C-M-npm  
    
        # compute hyperparameters for beta and dirichlet distributions over
        # theta_+, theta_-, and pi_mu
        ap=npp+ap0
        bp=npm+bp0
        am=nmp+am0
        bm=nmm+bm0
        a=n+a0
    
        # evaluate variational free energy, an approximation to the
        # negative log-evidence
        Qnz=Q
        F.append( betaln(ap,bp)-betaln(ap0,bp0)+betaln(am,bm)-betaln(am0,bm0)+sum(gammaln(a))-gammaln(sum(a))-(sum(gammaln(a0))-gammaln(sum(a0)))-sum(multiply(Qnz,log(Qnz))) )
        F[i]=-F[i]

        #print "iteration", i+1 , ": F =", F[i]

        # F should always decrease
        if (i>1) and F[i] > F[i-1]:
            print "\twarning: F increased from", F[i-1] ,"to", F[i]

        if (i>1) and (abs(F[i]-F[i-1]) < TOL_DF):
            break


    return dict(F=F[-1],F_iter=F,Q=Q,K=K,ap=ap,bp=bp,am=am,bm=bm,a=a)

def estep_inline(rows,cols,Q,JL,JG,lnpi,n):
    # following is a string of (poorly written) c code that gets
    # passed to weave below. compiles on first run, should be quite
    # fast thereafter
    # jntj: documentation needed
    code="""
    int nnz=Nrows[0];
    int N=NQ[0];
    int K=NQ[1];
    int i,rcndx,rcndxi,mu;
    double AQimu,lnQimu,lnQimumax,Zi;
    
    rcndx=0;
    /* update each node */
    for (i=0; i<N; i++) {

      rcndxi=rcndx;
      /* update each module for the i-th node */
      for (mu=0; mu<K; mu++) {

        /* calculate (A*Q)_{i,\mu} w/o the i-th node */
        AQimu=0;
        while ((rows[rcndx]<=i) && (rcndx < nnz)) {
          if (cols[rcndx] != i) {
            AQimu+=Q[K*cols[rcndx]+mu];
          }
          rcndx++;
        }
        if (mu<K-1)
          rcndx=rcndxi;
        
        /* calculate expected occupation w/o the i-th node */
        n[mu]-=Q[K*i+mu];

        /* log of updated Q_{i,\mu} entry */
        lnQimu=JL*AQimu-JG*n[mu]+lnpi[mu];
        Q[K*i+mu]=lnQimu;

        /* store max over all mu of lnQimu */
        if ((mu == 0) || (lnQimu > lnQimumax))
          lnQimumax=lnQimu;
      }

      /* calculate normalization constant */
      /* work in log-space to avoid under/over flow */
      Zi=0;
      for (mu=0; mu<K; mu++) {
        Q[K*i+mu]=exp(Q[K*i+mu]-lnQimumax);
        Zi+=Q[K*i+mu];
      }

      /* normalize */
      for (mu=0; mu<K; mu++) {
        Q[K*i+mu]=Q[K*i+mu]/Zi;
        /* adjust n[mu] */
        n[mu]+=Q[K*i+mu];
      }
    }
    """

    # run the above string through weave
    weave.inline(code,['rows','cols','Q','JL','JG','lnpi','n'])


def learn_restart(A,Kvec,net0={},opts={}):
    """
    runs vbmod with multiple restarts over a range of K values. (see
    vbmod for further documentation.) returns the best run over each
    K value (i.e. corresponding to the lowest variational free
    energy) as well as the best run over all K values.
    
    net0 and opts inputs are optional. F_K and net_K outputs are also
    optional.
    
    inputs: see learn() above
    
    outputs:
      net: posterior structure for best run over all K and
          restarts. see vbmod for further documentation.
      net_K: length-K array of posterior structures for best run over
          each K
    """
    # default options
    NUM_RESTARTS=25;
    VERBOSE=1;

    # get options if provided
    if (type(opts) == type({})) and (len(opts) > 0):
        if 'NUM_RESTARTS' in opts: NUM_RESTARTS=opts['NUM_RESTARTS']
        if 'VERBOSE' in opts: VERBOSE=opts['VERBOSE']

    N=A.shape[0]
    len_Kvec=len(Kvec)
    print "running vbmod for", NUM_RESTARTS ,"restarts"
    print "N =", N , "K =", Kvec

    net_K = []
    F_K = zeros(len_Kvec)
    # loop over K values
    for kndx in range(len_Kvec):
        # current K value
        K=Kvec[kndx];

        # perform NUM_RESTARTS of vbmod
        net_KR = []
        F_KR=zeros(NUM_RESTARTS)
        for r in range(NUM_RESTARTS):
            net_KR.append(learn(A,K,net0,opts))
            F_KR[r]=net_KR[r]['F']

        # find best run for this value of K
        (rndx,)=where(F_KR==F_KR.min())
        rndx=rndx[0]
        net_K.append(net_KR[rndx])
        F_K[kndx]=net_K[kndx]['F']

        print "best run for K =", K ,": F =", F_K[kndx]

    # find best run over all K values
    (kndx,)=where(F_K==F_K.min())
    kndx=kndx[0]
    net=net_K[kndx]
    net['K']=Kvec[kndx]
    
    print "minimum at K =", net['K'] , "of F =", net['F']

    return net, net_K



def restart_figs(A,net,net_K):
    """
    plots results from vbmod_restart
    
    inputs:
      A: N-by-N undirected (symmetric), binary adjacency matrix w/o
          self-edges (note: fastest for sparse and logical A)
      net: posterior structure for best run over all K and
          restarts. see vbmod for further documentation.
      net_K: length-K array of posterior structures for best run over
          each K; see vbmod_restart for further documentation
    """
    
    N=net['Q'].shape[0]
    K=net['Q'].shape[1]
    
    figure()
    subplot(2,2,1)
    spy(A.toarray())
    title('adjacency matrix')
    axis('off')

    subplot(2,2,2)
    Kvec=[]
    F_K=[]
    for n in net_K:
        Kvec.append(n['K'])
        F_K.append(n['F'])
    Kvec=array(Kvec)
    F_K=array(F_K)
    plot(Kvec,F_K,'b^-')
    hold(True)
    plot([K],[net['F']],'ro',label='Kvb')
    hold(False)
    legend()
    title('complexity control')
    xlabel('K')
    ylabel('F')
    grid('on')
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    subplot(2,2,3)
    imshow(array(net['Q']),interpolation='nearest',aspect=(1.0*K)/N)
    title('Qvb')
    xlabel('K')
    ylabel('N')
    
    subplot(2,2,4)
    plot(arange(1,len(net['F_iter'])+1),net['F_iter'],'bo-')
    title('learning curve')
    xlabel('iteration')
    ylabel('F')
    grid('on')

def demo():
    """
    function to demonstrate vbmod
    """
    N=128.
    K=4
    Kvec=array(range(K-2,K+2+1))
    ktot=16.
    kout=6.
    #pivec=ones(1,Ktrue)/Ktrue;  

    # determine within- and between- module edge probabilities from above
    tp=(ktot-kout)/(N/K-1)
    tm=kout/(N*(K-1)/K)

    print "generating random adjacency matrix ... "
    A=rnd(N,K,tp,tm)

    print "running variational bayes ... "
    t=time()
    (net,net_K)=learn_restart(A,Kvec)
    print "finished in", time()-t , "seconds"
    print "displaying results ... "
    restart_figs(A,net,net_K)
    show()


def demo_largeN(N=1e3,Kvec=array([4,3,5]),ktot=16,kout=6):
    """
    function to demonstrate vbmod for larger number of nodes
    """
    K=Kvec[0]
    Kvec=sort(Kvec)
    #pivec=ones(1,Ktrue)/Ktrue;

    # hyperparameters for priors
    net0={}
    net0['ap0']=N*2.
    net0['bp0']=2.
    net0['am0']=2.
    net0['bm0']=N*2.

    # options
    opts={}
    opts['NUM_RESTARTS']=1
    opts['MAX_FITER']=50

    # determine within- and between- module edge probabilities from above
    tp=(ktot-kout)/(float(N)/K-1)
    tm=kout/(float(N)*(K-1)/K)

    print "generating random adjacency matrix ... "
    # slow right now
    A=rnd(N,K,tp,tm)

    print "running variational bayes ... "
    t=time()
    (net,net_K)=learn_restart(A,Kvec,net0,opts)
    print "finished in", time()-t , "seconds"
    print "displaying results ... "
    restart_figs(A,net,net_K)
    show()


if __name__ == '__main__':
    demo()
