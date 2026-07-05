#!/usr/bin/env python3
r"""
Finite-T thermodynamics of the growing-q hierarchical RB-Potts model on the diamond.

H = - sum_{i!=j} (A_ij - gamma k_i k_j/2m) delta(sigma_i, sigma_j),  sigma_i in {1..q}.

For the hierarchical partition at depth d we use q = 2^d states. The intended
ground state assigns each depth-d block its own Potts colour. We test whether the
HIERARCHICAL community order survives thermally by measuring LEVEL-RESOLVED
sibling correlations:

  At level l (1<=l<=d), every node of the block-tree has two children (the two
  sub-bundles). Define the level-l order parameter as the probability that two
  spins in DIFFERENT level-l siblings (but same level-(l-1) parent) are in
  DIFFERENT Potts states, minus chance. Concretely we use the pair overlap
    O_l = < delta(sigma_i,sigma_j) >  for i,j in sibling sub-bundles at level l.
  Community order at level l  <=>  O_l small (siblings avoid same colour);
  disordered <=> O_l ~ 1/q (chance).

We compute these EXACTLY by enumeration for small t (q can be up to 2^t, so
enumeration over q^N is only feasible for t<=2; for t=3 we use q up to a few).
This is the exploratory step to see the structure; then we build the exact
transfer/decimation solver.
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

def enum_potts(t, d, gamma, beta):
    """Exact q=2^d Potts RB partition function & observables by enumeration.
    Returns level-resolved sibling same-colour probabilities O_l, l=1..d."""
    G,nc,poles=diamond_labeled(t)
    nodes=list(G.nodes()); N=len(nodes); idx={n:i for i,n in enumerate(nodes)}
    edges=list(G.edges()); m=G.number_of_edges(); deg=dict(G.degree())
    k=np.array([deg[n] for n in nodes]); 
    q=2**d
    # intended block of each node = first d chain entries (pad poles with 0s)
    def blk(n):
        c=nc.get(n,())
        c=c[:d]+(0,)*(d-len(c))
        return c
    # precompute sibling pairs at each level: (i,j) with same parent chain[:l-1]
    # but differing at position l-1. We'll just measure <delta(si,sj)> averaged
    # over such node pairs.
    sib_pairs={l:[] for l in range(1,d+1)}
    for a in range(N):
        for b in range(a+1,N):
            ca=blk(nodes[a]); cb=blk(nodes[b])
            # find first differing level
            for l in range(1,d+1):
                if ca[:l-1]==cb[:l-1] and ca[l-1]!=cb[l-1]:
                    sib_pairs[l].append((a,b)); break
    Z=0.0; Odelta={l:0.0 for l in range(1,d+1)}
    for cfg in product(range(q),repeat=N):
        s=np.array(cfg)
        # energy
        e_in=sum(1 for u,v in edges if s[idx[u]]==s[idx[v]])
        Kc=defaultdict(float)
        for a in range(N): Kc[s[a]]+=k[a]
        sumKc2=sum(v*v for v in Kc.values()); sumk2=float(k@k)
        H=-(2*e_in - gamma/(2*m)*(sumKc2-sumk2))
        w=np.exp(-beta*H); Z+=w
        for l in range(1,d+1):
            if sib_pairs[l]:
                same=sum(1 for (a,b) in sib_pairs[l] if s[a]==s[b])
                Odelta[l]+=w*same/len(sib_pairs[l])
    return {l:Odelta[l]/Z for l in range(1,d+1)}, q

if __name__=="__main__":
    print("Exact hierarchical Potts sibling same-colour prob O_l (enumeration).")
    print("Community-ordered at level l <=> O_l << 1/q (siblings avoid same colour).")
    print("Disordered <=> O_l ~ 1/q (chance).\n")
    for t,d in [(2,1),(2,2),(3,2)]:
        q=2**d
        print(f"--- t={t}, d={d}, q={q}, chance 1/q={1/q:.3f} ---")
        for beta in [3.0, 1.0, 0.3]:
            O,_=enum_potts(t,d,1.0,beta)
            print(f"  beta={beta}: "+"  ".join(f"O_{l}={O[l]:.3f}" for l in O))
        print()
