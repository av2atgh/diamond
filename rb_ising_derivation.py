#!/usr/bin/env python3
r"""
Reichardt-Bornholdt Hamiltonian, Eq.(2), in the q=2 (Ising) case.

  H = - sum_{i != j} (A_ij - gamma p_ij) delta(sigma_i, sigma_j),   sigma_i in {1,2}

Map to Ising spins s_i = +/-1 via  delta(sigma_i,sigma_j) = (1 + s_i s_j)/2.

  H = - sum_{i!=j} (A_ij - gamma p_ij) (1 + s_i s_j)/2
    = -1/2 sum_{i!=j}(A_ij - gamma p_ij)          [constant, drop]
      - 1/2 sum_{i!=j}(A_ij - gamma p_ij) s_i s_j

The pair sum over i!=j double counts; write as sum over unordered pairs {i,j}:
  H = const - sum_{i<j} (A_ij - gamma p_ij) s_i s_j

So the effective pairwise coupling is
  J_ij = A_ij - gamma p_ij .

With the configuration null p_ij = k_i k_j /(2m):
  - On an EDGE (A_ij=1):      J_ij = 1 - gamma k_i k_j/(2m)   (mostly ferromagnetic)
  - On a NON-EDGE (A_ij=0):   J_ij = - gamma k_i k_j/(2m)     (antiferromagnetic, all-to-all)

KEY STRUCTURE:
  H = -sum_{i<j} A_ij s_i s_j                          (ferro on the graph edges)
      + gamma/(2m) sum_{i<j} k_i k_j s_i s_j            (antiferro, fully connected, degree-weighted)

The second term is a FULLY-CONNECTED degree-weighted antiferromagnet.  Using
M_k = sum_i k_i s_i  (the degree-weighted magnetization),
  sum_{i<j} k_i k_j s_i s_j = 1/2[ (sum_i k_i s_i)^2 - sum_i k_i^2 ]
                            = 1/2 [ M_k^2 - sum_i k_i^2 ].
So
  H = -sum_{edges} s_i s_j  +  (gamma/4m)[ M_k^2 - sum_i k_i^2 ].

=> The RB-Ising model = ferromagnetic Ising on the diamond
   PLUS an infinite-range antiferromagnetic term in the *degree-weighted*
   magnetization M_k = sum_i k_i s_i, of strength gamma/(4m).

This is a ferromagnet with a global constraint that PENALIZES net
degree-weighted magnetization -- i.e. it favors states where the two spin
classes balance the total degree.  That is exactly a drive toward a
2-community (bisection) state: it frustrates the pure ferromagnet by making
"all-aligned" costly (large M_k), rewarding a split that halves the degree sum.
"""
print(__doc__)

# quick sanity: on the diamond, sum_i k_i = 2m, sum_i k_i^2 = ?
import networkx as nx
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
print("\nDiamond degree moments:")
print(f"{'t':>2}{'N':>7}{'2m':>8}{'sum k^2':>10}{'max deg':>9}")
for t in range(1,7):
    G=diamond(t); ks=[d for _,d in G.degree()]
    print(f"{t:>2}{G.number_of_nodes():>7}{sum(ks):>8}{sum(k*k for k in ks):>10}{max(ks):>9}")
