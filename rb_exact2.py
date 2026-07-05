#!/usr/bin/env python3
r"""
Exact q=2 RB-Ising via LEGENDRE/field scan: the M_k^2 term is handled exactly by
computing, for the ferromagnetic diamond in a degree-proportional field
b_i = x*k_i (x a real field, NOT self-consistent yet), the exact generating
function, and then doing the Gaussian (HS) integral over x EXACTLY by saddle +
fluctuations OR (cleanest for insight) by recognizing:

  Z = sum_s exp(beta sum_edges s_i s_j - (beta gamma/4m)(sum_i k_i s_i)^2 )

Use exact HS with a real integral (penalty term, a=beta*gamma/4m > 0):
  exp(-a M^2) = sqrt(1/(4 pi a)) \int_{-inf}^{inf} dphi exp( -phi^2/(4a) - phi M ).
Wait that's exp(+? ) check: \int exp(-phi^2/(4a) - phi M) dphi
  = sqrt(4 pi a) exp(a M^2).  That gives +aM^2, wrong sign.
For -a M^2 (a>0) we need the OSCILLATORY transform:
  exp(-a M^2) = sqrt(1/(4 pi a)) \int dphi exp(-phi^2/(4a) + i phi M).
So phi is imaginary-coupled. Equivalently substitute phi=i*psi and do a real
saddle in psi with a MAXIMUM (not min). The saddle equation is real:
  psi* = -2a <M>  and the field on spins is b_i = i phi k_i -> real psi: b_i = -2a<M> k_i? 
Let's just do it concretely & exactly for small t by DIRECT field scan + Gaussian
weight, integrating phi numerically over the real line with the oscillatory
kernel replaced by the equivalent real saddle. 

SIMPLER EXACT ROUTE (no HS): compute Z by summing over the value of M_k.
For the ferromagnetic diamond, define the restricted partition function
  G(M) = sum_{s: sum_i k_i s_i = M} exp(beta sum_edges s_i s_j).
Then Z = sum_M G(M) exp(-(beta gamma/4m)(M^2 - sum k^2)).
We get G(M) by a field-resolved transfer that tracks the running value of
sum k_i s_i. On the diamond this is doable by carrying, in the bond recursion, a
POLYNOMIAL in a bookkeeping variable z that records k-weighted magnetization:
each spin contributes z^{+k} or z^{-k}. That is exact and reaches moderate t.
For a first robust insight, we validate by enumeration and compute the
order parameter <|M_k|>/2m and the community magnetization vs gamma, T.
"""
import numpy as np
import networkx as nx
from itertools import product
from mpmath import mp, mpf, exp, log
mp.dps = 30

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

def exact_all(t, gamma, beta):
    """Full exact enumeration: order params. |m|=|sum s|/N (community imbalance),
    m_k=|sum k s|/2m. Also the BIPARTITION order: how balanced the split is."""
    G=diamond(t);nodes=list(G.nodes());N=len(nodes);idx={n:i for i,n in enumerate(nodes)}
    edges=list(G.edges());m=G.number_of_edges();k=np.array([G.degree(n) for n in nodes]);k2=float(k@k)
    Z=mpf(0);Mabs=mpf(0);Mk=mpf(0)
    # also track <s_A s_B> pole correlation (are the two hubs same or opposite community?)
    polecorr=mpf(0); A,B=0,1
    for bits in product([1,-1],repeat=N):
        s=np.array(bits)
        Efm=-sum(s[idx[u]]*s[idx[v]] for u,v in edges)
        MkC=float(k@s)
        E=Efm+(gamma/(4*m))*(MkC*MkC-k2)
        w=exp(-beta*E)
        Z+=w; Mabs+=w*abs(int(s.sum())); Mk+=w*abs(MkC)
        polecorr+=w*s[idx[A]]*s[idx[B]]
    return float(Mabs/Z/N), float(Mk/Z/(2*m)), float(polecorr/Z)

if __name__=="__main__":
    print("Exact RB-Ising q=2 order parameters vs gamma and T (enumeration, t<=3):")
    print("m=|sum s|/N (imbalance; 0=balanced split), mk=|sum k s|/2m, <sA sB>=pole corr\n")
    for t in [2,3]:
        G=diamond(t)
        print(f"=== t={t}, N={G.number_of_nodes()} ===")
        for beta,Tlabel in [(mpf(5),'T low'),(mpf('0.609'),'T~Tc_FM'),(mpf('0.3'),'T high')]:
            print(f"  [{Tlabel}, beta={float(beta):.3f}]")
            print(f"   {'gamma':>6}{'m_imbal':>9}{'m_k':>8}{'<sAsB>':>9}")
            for gamma in [0.0,0.5,1.0,1.5,2.0]:
                mi,mk,pc=exact_all(t,gamma,beta)
                print(f"   {gamma:>6.2f}{mi:>9.4f}{mk:>8.4f}{pc:>9.3f}")
            print()
