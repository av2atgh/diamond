#!/usr/bin/env python3
r"""
RB-Ising on the diamond via Hubbard-Stratonovich / mean-field decoupling of the
infinite-range degree-weighted term.

  H = -sum_{edges} s_i s_j + (gamma/4m) M_k^2 + const,   M_k = sum_i k_i s_i.

The M_k^2 term is infinite-range but couples through ONE collective variable.
Its mean-field treatment is EXACT in the thermodynamic limit (infinite-range
interactions => mean field is exact).  Introduce m_k = <M_k>/(2m) (degree-
weighted magnetization per unit degree).  The self-consistent single-site field
conjugate to spin i is

   h_i = -(gamma/2m) k_i M_k * ... 
Let's be careful. Partition function Z = sum_{s} exp(-beta H). Write
  beta H = -beta sum_edges s_i s_j + (beta gamma/4m)(sum_i k_i s_i)^2 + const.
Hubbard-Stratonovich on the +a x^2 term (a>0, antiferro/penalty):
  exp(-a M_k^2) = sqrt(a/pi) int dphi exp(-a phi^2 - 2 a phi M_k)   ... sign care
Actually for exp(-a M^2) with a>0 use
  exp(-a M^2) = sqrt(1/(4 pi a)) int dphi exp(-phi^2/(4a) - i phi M)   (oscillatory)
Cleaner: since a>0 (penalty), the saddle is real; use
  exp(-a M_k^2) = const * int dphi exp(-phi^2/(4a) - phi M_k).
Then the spins couple linearly to phi via -phi sum_i k_i s_i => site field
  h_i = -phi k_i   (in units where beta absorbed).
Saddle point: phi* = -2 a <M_k> = -(beta gamma/2m)<M_k>.
Equivalently define an effective per-site field proportional to degree:
  beta h_i = -(beta gamma/2m) k_i <M_k> = -lambda k_i,  lambda = (beta gamma/2m)<M_k>.

So each spin sees the ferromagnetic diamond PLUS a field  h_i = -lambda k_i / beta
that is PROPORTIONAL TO ITS DEGREE and OPPOSES the current degree-weighted
magnetization.  Self-consistency: <M_k> = sum_i k_i <s_i> computed in that
field must reproduce lambda.

This is solvable by exact decimation IF the field is uniform, but here h_i ~ k_i
varies by site.  However on the diamond the degree is set by GENERATION of birth,
so h_i takes only t+1 distinct values.  We can carry a generation-resolved field
in the decimation.  For a FIRST insight we do the cleanest thing: solve the
self-consistency treating the two hubs (poles, degree 2^t) and the bulk, OR do
exact small-t enumeration to see the phase structure in (T, gamma).

Here: exact enumeration for t=1,2,3 to reveal the (gamma, T) phase diagram and
the ground-state partitions, before building the full decimation.
"""
import numpy as np
import networkx as nx
from itertools import product

def diamond(t,b=2,s=2):
    G=nx.Graph();A,B=0,1;G.add_edge(A,B);nxt=2
    for gen in range(t):
        for (u,v) in list(G.edges()):
            G.remove_edge(u,v)
            for k in range(b):
                prev=u
                for st in range(s):
                    node=v if st==s-1 else nxt
                    if st!=s-1: nxt+=1
                    G.add_edge(prev,node);prev=node
    return G

def rb_ground_state(t, gamma):
    """Exact minimization of RB-Ising H over all 2^N spin configs (small t)."""
    G = diamond(t); nodes=list(G.nodes()); N=len(nodes); idx={n:i for i,n in enumerate(nodes)}
    edges=list(G.edges()); m=G.number_of_edges()
    k=np.array([G.degree(n) for n in nodes])
    best=None; bestE=1e18; bestcfg=None
    for bits in product([1,-1],repeat=N):
        s=np.array(bits)
        Efm = -sum(s[idx[u]]*s[idx[v]] for u,v in edges)
        Mk = np.dot(k,s)
        E = Efm + (gamma/(4*m))*(Mk*Mk - np.dot(k,k))
        if E<bestE-1e-12:
            bestE=E; bestcfg=s.copy()
    # community sizes
    npos=int(np.sum(bestcfg==1)); nneg=N-npos
    return bestE, npos, nneg, bestcfg, N, m

if __name__=="__main__":
    print("RB-Ising exact ground state on the diamond vs gamma (q=2):")
    print("H = -sum_edges s_i s_j + (gamma/4m)[M_k^2 - sum k^2],  M_k=sum k_i s_i\n")
    for t in [1,2,3]:
        print(f"--- t={t}, N={diamond(t).number_of_nodes()} ---")
        print(f"{'gamma':>7}{'E_gs':>10}{'n+':>5}{'n-':>5}  partition")
        for gamma in [0.0,0.5,1.0,1.5,2.0,3.0,5.0]:
            E,npos,nneg,cfg,N,m = rb_ground_state(t,gamma)
            kind = "all-aligned (1 comm)" if (npos==0 or nneg==0) else "SPLIT (2 comm)"
            print(f"{gamma:>7.2f}{E:>10.3f}{npos:>5}{nneg:>5}  {kind}")
        print()
