#!/usr/bin/env python3
r"""
EXACT q=2 RB-Ising on the diamond, no mean field, via a generating function that
tracks the degree-weighted magnetization M_k = sum_i k_i s_i.

The ferromagnetic part is local (decimation-friendly); the M_k^2 term is a
function of the single integer M_k. So compute the ferromagnetic partition
function RESOLVED by the value of M_k:
   G(M) = sum_{s: M_k(s)=M} exp(beta * sum_edges s_i s_j)
then
   Z = sum_M G(M) * exp(-(beta gamma/4m)(M^2 - sum_i k_i^2)).

We get G(M) exactly by carrying a 2x2 transfer object whose entries are
POLYNOMIALS (dicts) in a formal variable z, where z tracks M_k: a spin with
degree k contributes a factor z^{+k} or z^{-k}. Poles carry degree 2^t.

W(sA,sB) becomes a dict: {M_partial : weight}. Series & parallel combine by
convolution in M. Degrees follow generation (2^{t-g+1}); we assign each spin its
final degree as we build bottom-up (mid spin of a height-h sub-diamond -> deg 2^h).
"""
import numpy as np
from mpmath import mp, mpf, exp, log
mp.dps = 30
from collections import defaultdict

SP=[(1),(-1)]

def polymul(a,b):
    """Convolve two {exponent:coef} dicts (product of z-polynomials)."""
    r=defaultdict(lambda:mpf(0))
    for e1,c1 in a.items():
        for e2,c2 in b.items():
            r[e1+e2]+=c1*c2
    return r

def Wh(h, K):
    """2x2 array of dicts {M:coef}: ferromagnetic diamond of height h, each
    internal spin tagged with z^{±(final degree)}. Mid of height-h => degree 2^h."""
    if h==0:
        # single bond, no internal spin; weight exp(K sa sb), M-contribution 0
        W=[[None,None],[None,None]]
        for i,sa in enumerate(SP):
            for j,sb in enumerate(SP):
                W[i][j]={0: exp(K*sa*sb)}
        return W
    Wsub=Wh(h-1,K)
    kmid=2**h
    P=[[None,None],[None,None]]
    for i in range(2):
        for j in range(2):
            acc=defaultdict(lambda:mpf(0))
            for a,smid in enumerate(SP):
                # mid spin contributes z^{smid*kmid}
                term=polymul(Wsub[i][a], Wsub[a][j])
                for e,c in term.items():
                    acc[e+smid*kmid]+=c
            P[i][j]=dict(acc)
    # two identical bundles in parallel: convolve P with P (independent spins)
    W=[[None,None],[None,None]]
    for i in range(2):
        for j in range(2):
            W[i][j]=polymul(P[i][j],P[i][j])
    return W

def G_of_M(t, K):
    """Return dict {M: G(M)} for full height-t diamond incl pole degrees 2^t."""
    W=Wh(t,K)
    kp=2**t
    total=defaultdict(lambda:mpf(0))
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            for e,c in W[i][j].items():
                total[e+sa*kp+sb*kp]+=c   # both poles carry degree 2^t
    return dict(total)

def rb_observables(t, gamma, beta):
    m=4**t
    K=beta
    G=G_of_M(t,K)
    k2 = sum_k2(t)
    Z=mpf(0); Mkabs=mpf(0)
    for M,g in G.items():
        w=g*exp(-(beta*gamma/(4*m))*(M*M - k2))
        Z+=w; Mkabs+=w*abs(M)
    return float(Mkabs/Z/(2*m))

def sum_k2(t):
    # sum_i k_i^2 : poles 2*(2^t)^2 + interior sum
    import networkx as nx
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
    return sum(d*d for _,d in G.degree())

if __name__=="__main__":
    # validate vs enumeration at t=2
    import rb_exact2
    print("Validate exact-GF vs enumeration (m_k) at t=2, beta=5:")
    for gamma in [0.5,1.0,2.0]:
        gf=rb_observables(2,gamma,mpf(5))
        _,en,_=rb_exact2.exact_all(2,gamma,mpf(5))
        print(f"  gamma={gamma}: GF={gf:.5f}  enum={en:.5f}  {'OK' if abs(gf-en)<1e-4 else 'MISMATCH'}")
    print("\nExact-GF reaches larger t. Degree-weighted order param m_k vs gamma:")
    print(f"{'t':>2}{'N':>7}  " + "".join(f"g={g:<5}" for g in [0.5,1.0,1.5,2.0]))
    for t in [2,3,4,5,6]:
        row=[rb_observables(t,g,mpf(5)) for g in [0.5,1.0,1.5,2.0]]
        N=2+2*(4**t-1)//3
        print(f"{t:>2}{N:>7}  "+"".join(f"{v:<7.4f}" for v in row))

# ---- pole-pole correlation <s_A s_B> via GF resolved by pole states ----
def G_by_poles(t, K):
    """Return dict keyed by (sA,sB) -> {M: coef} for internal spins, so we can
    weight the M_k^2 term and read off pole correlations."""
    W=Wh(t,K); kp=2**t
    out={}
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            d=defaultdict(lambda:mpf(0))
            for e,c in W[i][j].items():
                d[e+sa*kp+sb*kp]+=c
            out[(sa,sb)]=dict(d)
    return out

def pole_corr(t, gamma, beta):
    m=4**t; K=beta; k2=sum_k2(t)
    GP=G_by_poles(t,K)
    Z=mpf(0); corr=mpf(0)
    for (sa,sb),d in GP.items():
        for M,g in d.items():
            w=g*exp(-(beta*gamma/(4*m))*(M*M-k2))
            Z+=w; corr+=w*sa*sb
    return float(corr/Z)
