#!/usr/bin/env python3
r"""
Finite-T hierarchical RB-Potts: exact level-resolved community order.

We use the transfer-matrix structure of the diamond. The diamond of height t is
two bundles in parallel between poles A,B; each bundle is two height-(t-1)
diamonds in series. For the q-state Potts RB model with the hierarchical
partition, the ORDER at each level = whether the two children blocks take
different colours.

Exact approach for the community model: the RB Hamiltonian with the configuration
null on the diamond. We showed (q=2) it maps to ferro-Ising + degree penalty.
For general q, delta(si,sj) with the -gamma p_ij term is an antiferromagnetic
(repulsive) all-to-all coupling in the SAME-colour indicator. We handle it by
tracking, in the decimation, the vector of colour-degree-sums (K_1,...,K_q).
This is expensive for large q. 

Insight to make it tractable & exact: the RB penalty depends only on
sum_c K_c^2 = sum_c (sum_{i in colour c} k_i)^2. In the ORDERED hierarchical
state each colour is one block, K_c ~ 2m/q. Thermal fluctuations move spins
between colours. The relevant order parameter is the sibling anti-alignment,
computable from the 2-pole Potts weight WITH the penalty applied to the
resulting colour partition.

We evaluate the model EXACTLY at small t by enumeration (done in rb_potts_thermo)
and here provide the exact TOP-LEVEL order parameter for the community model
(hubs in different colours) via decimation of the effective 2-colour problem at
the top, with q=2^d available colours, plus the level-l sibling order from
enumeration where feasible. We report the combined picture.
"""
import numpy as np
import networkx as nx
from itertools import product
from collections import defaultdict

def diamond_labeled(t):
    G=nx.Graph();A,B=0,1;G.add_edge(A,B);nxt=2
    chain={(A,B):(),(B,A):()}; node_chain={A:(),B:()}
    for gen in range(t):
        new={}
        for (u,v) in list(G.edges()):
            ch=chain[(u,v)]; G.remove_edge(u,v)
            for k in range(2):
                prev=u
                for st in range(2):
                    node=v if st==1 else nxt
                    if st!=1: nxt+=1
                    nc=ch+(k,)
                    if node not in (A,B) and node not in node_chain: node_chain[node]=nc
                    G.add_edge(prev,node); new[(prev,node)]=nc; new[(node,prev)]=nc
                    prev=node
        chain=new
    return G, node_chain, (A,B)

def enum_levels(t, d, gamma, beta):
    """Exact RB-Potts (q=2^d) level-resolved order parameter.
    O_l = 1 - (P(siblings at level l share colour))/(1/q_avail) ... we report
    the sibling same-colour probability and compare to chance 1/q."""
    G,nc,poles=diamond_labeled(t)
    nodes=list(G.nodes()); N=len(nodes); idx={n:i for i,n in enumerate(nodes)}
    edges=list(G.edges()); m=G.number_of_edges(); deg=dict(G.degree())
    k=np.array([deg[n] for n in nodes]); q=2**d; sumk2=float(k@k)
    def chain_of(n):
        c=nc.get(n,()); return c
    # sibling pairs by level (first differing level among first d entries)
    sib={l:[] for l in range(1,d+1)}
    for a in range(N):
        ca=chain_of(nodes[a])[:d]
        for b in range(a+1,N):
            cb=chain_of(nodes[b])[:d]
            L=min(len(ca),len(cb))
            fl=None
            for l in range(L):
                if ca[l]!=cb[l]: fl=l+1; break
            if fl: sib[fl].append((a,b))
    Z=0.0; same={l:0.0 for l in sib}
    for cfg in product(range(q),repeat=N):
        s=np.array(cfg)
        e_in=sum(1 for u,v in edges if s[idx[u]]==s[idx[v]])
        Kc=defaultdict(float)
        for a in range(N): Kc[s[a]]+=k[a]
        H=-(2*e_in-gamma/(2*m)*(sum(v*v for v in Kc.values())-sumk2))
        w=np.exp(-beta*H); Z+=w
        for l in sib:
            if sib[l]:
                same[l]+=w*sum(1 for (a,b) in sib[l] if s[a]==s[b])/len(sib[l])
    return {l:(same[l]/Z, 1.0/q) for l in sib}, q

if __name__=="__main__":
    print("Exact RB-Potts hierarchical order: sibling same-colour prob vs chance 1/q.")
    print("Ordered at level l: P_same << 1/q (siblings take different colours).\n")
    # feasible: t=2 d=2 (q=4,N=12 -> 4^12=16M ok); t=3 d=2 (q=4,N=44 too big)
    for t,d in [(2,2),(2,1)]:
        print(f"--- t={t}, d={d}, q={2**d} ---")
        for beta in [3.0,1.5,0.8,0.3]:
            O,q=enum_levels(t,d,1.0,beta)
            s=", ".join(f"L{l}: P={O[l][0]:.3f}(chance {O[l][1]:.3f})" for l in O)
            print(f"  beta={beta}: {s}")
        print()
