#!/usr/bin/env python3
r"""
Correct hierarchical staggered magnetization via a GAUGE (Mattis) check.

Insight: on the diamond every internal spin sits inside a unique nested chain of
bundles.  Assign eps(i) = product of the level-signs s_l along that chain, with
s_l = +/- chosen so the two children of each node are opposite.  This is a
Mattis pattern.  Under the gauge transformation sigma_i -> eps(i) sigma_i, a
ferromagnetic bond J sigma_i sigma_j becomes J eps_i eps_j sigma_i sigma_j.

If eps_i eps_j = +1 on EVERY edge, the gauge maps the FM exactly onto itself and
the hierarchical staggered magnetization = uniform magnetization in the gauged
variables => it ORDERS with the SAME (volume) eigenvalue.  If some bonds get
eps_i eps_j = -1, those bonds are frustrated and the pattern competes with FM.

So the whole question reduces to: for the level-alternating hierarchical sign
pattern, how many bonds are satisfied (eps_i eps_j=+1) vs frustrated (=-1)?
We compute this exactly by construction, then confirm by the susceptibility.
"""
import networkx as nx
from mpmath import mp, mpf, exp, log
mp.dps = 40
from ising_diamond import find_Kc, n_t

b, s = 2, 2

def build_with_signs(t):
    """Build the diamond, assigning each node a hierarchical sign via nested
    bundle membership. Returns G, sign dict, poles."""
    G = nx.Graph(); A, B = 0, 1; G.add_edge(A, B); nxt = 2
    # each edge carries a 'chain' = tuple of child-indices identifying its bundle path
    chain = {(A,B): (), (B,A): ()}
    node_chain = {A: (), B: ()}
    for gen in range(t):
        new = {}
        for (u, v) in list(G.edges()):
            ch = chain[(u,v)]
            G.remove_edge(u, v)
            for k in range(b):                     # k=0,1 : the two bundles
                prev = u
                for step in range(s):
                    node = v if step == s-1 else nxt
                    if step != s-1: nxt += 1
                    newchain = ch + (k,)            # extend chain by which child
                    if node not in (A, B):
                        node_chain[node] = newchain
                    G.add_edge(prev, node)
                    new[(prev,node)] = newchain; new[(node,prev)] = newchain
                    prev = node
        chain = new
    # hierarchical sign: eps = product of (-1)^(child index) over the chain
    def eps_of(chn):
        e = 1
        for k in chn:
            e *= (1 if k == 0 else -1)   # child 0 -> +, child 1 -> -
        return e
    sign = {n: eps_of(node_chain[n]) for n in G.nodes()}
    return G, sign, (A, B)

def bond_frustration(t):
    G, sign, poles = build_with_signs(t)
    sat = frus = 0
    for u, v in G.edges():
        if sign[u]*sign[v] == 1: sat += 1
        else: frus += 1
    return sat, frus, G.number_of_edges()

if __name__ == "__main__":
    print("Hierarchical Mattis sign pattern: satisfied vs frustrated bonds")
    print(f"{'t':>2}{'N':>8}{'E':>8}{'satisfied':>11}{'frustrated':>11}{'frus/E':>9}")
    for t in range(1, 8):
        sat, frus, E = bond_frustration(t)
        print(f"{t:>2}{n_t(t):>8}{E:>8}{sat:>11}{frus:>11}{frus/E:>9.4f}")
