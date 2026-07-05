#!/usr/bin/env python3
r"""
RB POTTS model on the diamond with a HIERARCHICAL nested partition, where each
nested sub-community at every level gets its OWN label => q = 2^d states at depth d.

RB Hamiltonian (Eq. 2):
  H = - sum_{i != j} (A_ij - gamma p_ij) delta(sigma_i, sigma_j),  p_ij = k_i k_j/(2m).

Rewrite with the identity sum_{i!=j} A_ij delta = 2 * (within-community edges),
and sum_{i!=j} p_ij delta = (1/2m) sum_c (K_c^2 - sum_{i in c} k_i^2), K_c = sum_{i in c} k_i.
So (dropping the i=j diagonal consistently):
  H = -2 * [ (edges within communities) - gamma/(4m) * sum_c K_c^2 ] + const,
equivalently the modularity-like energy
  H/(2) ~ - sum_c [ e_c - gamma * (K_c/2m)^2 * m ]   (standard RB/modularity form).

For a FIXED hierarchical partition we can evaluate H exactly (ground-state energy
of that labeling). But to get a THERMODYNAMIC order parameter we must let the
Potts spins fluctuate. Key question restated: with q = 2^d labels available and
the nested hierarchical partition as the intended ground state, does the
hierarchical community order persist as n -> inf?

STRATEGY:
(1) Exactly evaluate the RB energy of the nested hierarchical partition at depth
    d (q=2^d) vs the top-level (q=2) and uniform (q=1) partitions, as a function
    of gamma and size. Find which depth d minimizes the energy at each n
    (=> the diamond analogue of K_opt(n)).
(2) That tells us the OPTIMAL hierarchical depth vs n (ground-state / detection
    side). Then set up the finite-T Potts order parameter for that partition.

Here: step (1), exact closed-form energies via the block statistics we already
have. Nested level-d blocks on the diamond: removing the generation-<=d seam
edges splits the lattice into 2^d sub-diamonds (bundles-of-bundles).
"""
import numpy as np
import networkx as nx

def diamond_labeled(t):
    """Build diamond, and for each node record its nested bundle chain (which
    child 0/1 at each of the t generations of the branch it lives in)."""
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

def hier_labels(G, node_chain, poles, d):
    """Assign community label = first d entries of the node's chain (2^d blocks).
    Poles (empty chain) go to block 0."""
    lab={}
    for n in G.nodes():
        ch=node_chain.get(n,())
        key=ch[:d]
        # pad if shorter than d (poles): treat as block 0-ish; use tuple
        lab[n]=key
    # map distinct keys to 0..q-1
    keys=sorted(set(lab.values()), key=lambda x:(len(x),x))
    idx={k:i for i,k in enumerate(keys)}
    return {n:idx[lab[n]] for n in lab}, len(keys)

def rb_energy(G, labels, gamma):
    """Exact RB energy H = -sum_{i!=j}(A_ij - gamma k_i k_j/2m) delta(sig_i,sig_j)."""
    m=G.number_of_edges(); deg=dict(G.degree())
    # within-community edges
    e_in=sum(1 for u,v in G.edges() if labels[u]==labels[v])
    # sum_c K_c^2  and sum_c sum_{i in c} k_i^2
    from collections import defaultdict
    Kc=defaultdict(float)
    for n in G.nodes(): Kc[labels[n]]+=deg[n]
    sumKc2=sum(v*v for v in Kc.values())
    # H = -[ 2 e_in  - gamma/(2m) (sumKc2 - sum_i k_i^2) ]   (i!=j form)
    sumk2=sum(d*d for d in deg.values())
    H = -(2*e_in - gamma/(2*m)*(sumKc2 - sumk2))
    return H, e_in, len(Kc)

if __name__=="__main__":
    print("RB energy of nested hierarchical partitions (q=2^d) on the diamond.")
    print("Which depth d (=> q=2^d communities) MINIMIZES the RB energy at each size?\n")
    for gamma in [1.0]:
        print(f"gamma={gamma}")
        print(f"{'t':>2}{'N':>7} | "+"".join(f"d={d}(q={2**d})".ljust(11) for d in range(0,7))+"  d_opt")
        for t in range(1,8):
            G,nc,poles=diamond_labeled(t)
            row=[]
            for d in range(0,min(t+1,7)):
                labs,q=hier_labels(G,nc,poles,d)
                H,ein,qq=rb_energy(G,labs,gamma)
                row.append(H)
            dopt=int(np.argmin(row))
            N=G.number_of_nodes()
            cells="".join(f"{h:>10.1f} " for h in row)
            print(f"{t:>2}{N:>7} | {cells}  d={dopt}(q={2**dopt})")
