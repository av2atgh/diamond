#!/usr/bin/env python3
r"""
Exact q=2 RB-Ising on the diamond via HS mean-field field h_i = -eta*k_i,
solved by exact decimation. Degree by birth generation g: k=2^{t-g+1} (g>=1),
poles k=2^t.

We build W(sA,sB) = sum over internal spins of exp(-beta H_ferro + sum_i (-eta k_i) s_i),
i.e. ferromagnetic diamond with a degree-proportional field on each internal spin
(poles' field added by the caller). Self-consistency fixes eta.

Bottom-up construction with correct degrees:
A generation-1 diamond (height 1) has 1 internal spin (the mid) of degree 2.
To grow to height t, we replace each of its 2 bonds by a height-1 diamond, and
so on. A spin that is the "mid" of a sub-diamond of height h (embedded so that
it will be doubled no further) has degree 2^h. So: build recursively, a diamond
of height h has its top mid-spin with degree 2^h, and its two bonds are
diamonds of height h-1.

W_h(sA,sB): sum over all internal spins of the height-h diamond.
  h=0: single bond, W(sA,sB)=exp(K sA sB).
  h>=1: two parallel paths A->mid->B; each "bond" A-mid and mid-B is a height-
        (h-1) diamond. mid spin has degree 2^h, field b_mid=-eta*2^h.
  path weight P(sA,sB) = sum_{smid} W_{h-1}(sA,smid) exp(b_mid*smid) W_{h-1}(smid,sB)
  two paths in parallel: W_h(sA,sB)=P(sA,sB)^2   (b=2 identical bundles)
Wait: the two bundles are identical in structure & field pattern, so squared. Good.
"""
import numpy as np
from mpmath import mp, mpf, exp, log
mp.dps = 40
SP=[mpf(1),mpf(-1)]

def Wh(h, K, eta):
    """height-h diamond internal-spin-summed 2x2 weight; degrees = 2^level."""
    if h==0:
        return [[exp(K*sa*sb) for sb in SP] for sa in SP]
    Wsub = Wh(h-1, K, eta)
    bmid = -eta*(2**h)          # field on the mid spin (degree 2^h)
    P=[[mpf(0)]*2 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            P[i][j]=sum(Wsub[i][a]*exp(bmid*SP[a])*Wsub[a][j] for a in range(2))
    # two identical bundles in parallel
    return [[P[i][j]*P[i][j] for j in range(2)] for i in range(2)]

def observables(t, K, eta):
    """Given eta, compute Z, <M_k>, <m> for the height-t diamond.
    Poles have degree 2^t and field -eta*2^t each."""
    W = Wh(t, K, eta)
    bp = -eta*(2**t)   # pole field
    Z=mpf(0); Ssum=mpf(0)   # for <sum s_i> we need magnetization -> derivative trick
    # We need <M_k>=sum_i k_i<s_i>. Get it by adding a tiny probe conjugate to M_k:
    # dZ/d(source) with source coupling sum_i k_i s_i. Easiest: numerical deriv of
    # logZ wrt eta gives -<M_k> since field = -eta*k_i for ALL spins incl poles.
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            w = W[i][j]*exp(bp*(sa+sb))
            Z+=w
    return Z

def logZ_full(t, K, eta):
    W = Wh(t, K, eta); bp=-eta*(2**t)
    Z=mpf(0)
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            Z+=W[i][j]*exp(bp*(sa+sb))
    return log(Z)

def Mk_of_eta(t, K, eta):
    """<M_k> = -d logZ/d eta  (since field term is -eta*sum_i k_i s_i)."""
    d=mpf('1e-7')
    return -(logZ_full(t,K,eta+d)-logZ_full(t,K,eta-d))/(2*d)

def self_consistent(t, K, gamma):
    """Solve eta = (K*gamma/(2m)) <M_k>(eta), m=4^t. Returns eta*, m_k=<M_k>/(2m)."""
    m = mpf(4)**t
    c = K*gamma/(2*m)
    # iterate eta_{n+1} = c * Mk(eta_n); start from small symmetry-broken seed
    eta = mpf('0.001')
    for _ in range(200):
        Mk = Mk_of_eta(t,K,eta)
        eta_new = c*Mk
        if abs(eta_new-eta)<mpf('1e-20'): 
            eta=eta_new; break
        eta = mpf('0.5')*eta + mpf('0.5')*eta_new
    Mk = Mk_of_eta(t,K,eta)
    return eta, Mk/(2*m)

# ---- validate against enumeration at t=1,2 ----
import networkx as nx
from itertools import product
def diamond(t):
    G=nx.Graph();A,B=0,1;G.add_edge(A,B);nxt=2
    for gen in range(t):
        for (u,v) in list(G.edges()):
            G.remove_edge(u,v)
            for k in range(2):
                prev=u
                for st in range(2):
                    node=v if st==1 else nxt
                    if st!=1: nxt+=1
                    G.add_edge(prev,node);prev=node
    return G
def enum_mk(t,gamma,K):
    G=diamond(t);nodes=list(G.nodes());N=len(nodes);idx={n:i for i,n in enumerate(nodes)}
    edges=list(G.edges());m=G.number_of_edges();k=np.array([G.degree(n) for n in nodes]);k2=float(k@k)
    Z=mpf(0);Mk=mpf(0)
    for bits in product([1,-1],repeat=N):
        s=np.array(bits)
        Efm=-sum(s[idx[u]]*s[idx[v]] for u,v in edges)
        MkC=float(k@s)
        E=Efm+(gamma/(4*m))*(MkC*MkC-k2)
        w=exp(-K*E);Z+=w;Mk+=w*abs(MkC)
    return float(Mk/Z/(2*m))

if __name__=="__main__":
    print("Validate HS-decimation vs exact enumeration (|m_k| degree-wtd order param):")
    print(f"{'t':>2}{'gamma':>7}{'K':>5}{'enum |m_k|':>12}{'HS |m_k|':>12}")
    for t in [1,2]:
        for gamma in [0.5,1.0,2.0]:
            K=mpf(5)  # low T
            me=enum_mk(t,gamma,K)
            eta,mk=self_consistent(t,K,gamma)
            print(f"{t:>2}{gamma:>7.2f}{float(K):>5.1f}{me:>12.4f}{abs(float(mk)):>12.4f}")
