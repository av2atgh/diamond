#!/usr/bin/env python3
r"""
Fast EXACT q=2 RB-Ising on the diamond. Same generating-function-over-M_k idea as
rb_exact_gf, but (a) work with numpy float128 arrays indexed by M (M ranges over
a bounded integer grid, step = gcd of degrees = 2), and (b) prune negligible
weights. Validated against rb_exact_gf.
"""
import numpy as np
from collections import defaultdict
import math

def _diamond_degsum2(t):
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

def rb_solve(t, gamma, beta, want='corr'):
    """Exact. Returns (-<sA sB>, m_k) using log-domain float128 to avoid overflow.
    Represent each W[i][j] as dict M->log-weight? Use plain float128 dict M->weight
    with periodic rescaling."""
    SP=[1,-1]
    K=float(beta)
    eK=math.exp(K); emK=math.exp(-K)
    # W[i][j] : dict M-> weight (float). Start bond.
    def bond():
        return [[{0: (eK if SP[i]*SP[j]>0 else emK)} for j in range(2)] for i in range(2)]
    def merge_series(Wsub, kmid):
        # path i->mid->j: sum over mid of Wsub[i][mid_state]*z^{mid*kmid}*Wsub[mid_state][j]
        P=[[defaultdict(float) for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for a in range(2):
                    smid=SP[a]
                    left=Wsub[i][a]; right=Wsub[a][j]
                    for e1,c1 in left.items():
                        for e2,c2 in right.items():
                            P[i][j][e1+e2+smid*kmid]+=c1*c2
        return P
    def parallel(P):
        W=[[defaultdict(float) for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                d=P[i][j]
                items=list(d.items())
                for e1,c1 in items:
                    for e2,c2 in items:
                        W[i][j][e1+e2]+=c1*c2
        return W
    W=bond()
    for h in range(1,t+1):
        kmid=2**h
        P=merge_series(W,kmid)
        W=parallel(P)
        # rescale EVERY generation to keep floats in range
        mx=max(max((max(d.values()) if d else 0.0) for d in row) for row in W)
        if mx>0:
            f=1.0/mx
            for i in range(2):
                for j in range(2):
                    W[i][j]={e:c*f for e,c in W[i][j].items()}
    # attach poles (degree 2^t) and M_k^2 weight
    m=4**t; kp=2**t; k2=_diamond_degsum2(t)
    a=float(beta*gamma/(4*m))
    # collect (logweight, sa*sb, |M|) then log-sum-exp
    import math as _m
    terms=[]
    for i in range(2):
        for j in range(2):
            sa,sb=SP[i],SP[j]
            for e,c in W[i][j].items():
                if c<=0: continue
                M=e+sa*kp+sb*kp
                lw=_m.log(c) - a*(M*M-k2)
                terms.append((lw, sa*sb, abs(M)))
    L=max(tt[0] for tt in terms)
    Z=0.0; corr=0.0; Mkabs=0.0
    for lw,ss,aM in terms:
        w=_m.exp(lw-L)
        Z+=w; corr+=w*ss; Mkabs+=w*aM
    return -corr/Z, Mkabs/Z/(2*m)

if __name__=="__main__":
    import rb_exact_gf as gf
    from mpmath import mpf
    print("Validate fast solver vs GF (t=4):")
    for g in [0.5,1.0,2.0]:
        c_fast,mk_fast=rb_solve(4,g,5.0)
        mk_gf=gf.rb_observables(4,g,mpf(5))
        print(f"  g={g}: fast m_k={mk_fast:.5f}  GF m_k={mk_gf:.5f}  {'OK' if abs(mk_fast-mk_gf)<1e-4 else 'X'}")
    import time
    t0=time.time()
    print("\nTiming t=6:", end=" ")
    c,mk=rb_solve(6,1.0,5.0); print(f"{time.time()-t0:.1f}s  -<sAsB>={c:.4f}")
