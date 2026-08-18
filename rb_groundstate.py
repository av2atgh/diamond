#!/usr/bin/env python3
r"""
rb_groundstate.py -- which bipartition minimises the Reichardt-Bornholdt energy
on the diamond, and why the two hubs end up in DIFFERENT communities.

H = -sum_{i!=j} (A_ij - gamma k_i k_j / 2m) delta(sigma_i,sigma_j)
  = -sum_<ij> sigma_i sigma_j + (gamma/4m)[M_k^2 - sum_i k_i^2] + const,
    M_k = sum_i k_i sigma_i        (Ising case, sigma = +-1)

Two bipartitions cut exactly the same c_t = 2^t seam edges:

  bundle cut     block 1 = bundle 0 + {A,B},  block 2 = bundle 1
                 -> degree imbalance M_k = kappa_1 - kappa_2 = 2^{t+1}
  staggered cut  block 1 = bundle 0 + {A},    block 2 = bundle 1 + {B}
                 -> M_k = 0 exactly (each block carries one hub)

Both have the same ferromagnetic energy, so their RB energies differ by exactly
the penalty gamma M_k^2/(4m) = gamma (2^{t+1})^2/(4 . 4^t) = gamma:
the staggered cut is lower by exactly gamma for EVERY size -- an O(1), not
extensive, preference, which is why an arbitrarily small resolution gamma>0 is
enough to select it, and why the exact solution has -<sigma_A sigma_B> -> 1.

Verified here by (i) brute-force minimisation at t=2 and (ii) the closed
identity above, checked by direct construction through t=7.

Requires: networkx.
"""
import itertools
import networkx as nx
from diamond_rg import diamond


def rb_energy(G, sigma, gamma):
    deg = dict(G.degree()); m = G.number_of_edges()
    ferro = sum(sigma[u] * sigma[v] for u, v in G.edges())
    Mk = sum(deg[i] * sigma[i] for i in G)
    k2 = sum(d * d for d in deg.values())
    return -ferro + gamma / (4 * m) * (Mk * Mk - k2)


def cuts(t):
    """(G, poles, bundle-cut spins, staggered-cut spins)."""
    G, poles, part = diamond(t)
    A, B = sorted(poles)
    bundle = {v: (1 if part[v] == 0 else -1) for v in G}
    stag = dict(bundle); stag[B] = -1           # move one hub to the other block
    return G, (A, B), bundle, stag


if __name__ == "__main__":
    print("== brute force at t=2 (n=12): the RB ground state at gamma=1 ==")
    G, (A, B), bundle, stag = cuts(2)
    nodes = sorted(G.nodes()); best = None
    for bits in itertools.product([1, -1], repeat=len(nodes) - 1):
        sig = dict(zip(nodes, (1,) + bits))
        E = rb_energy(G, sig, 1.0)
        if best is None or E < best[0]: best = (E, sig)
    E0, sig0 = best
    deg = dict(G.degree())
    cut = sum(1 for u, v in G.edges() if sig0[u] != sig0[v])
    k1 = sum(deg[v] for v in G if sig0[v] == 1); k2 = sum(deg[v] for v in G if sig0[v] == -1)
    print(f"   ground state: E={E0:.3f}  cut={cut}  (kappa_1,kappa_2)=({k1},{k2})"
          f"  hubs anti-aligned: {sig0[A] != sig0[B]}")
    print(f"   staggered cut: E={rb_energy(G,stag,1.):.3f}   bundle cut: "
          f"E={rb_energy(G,bundle,1.):.3f}   one community: "
          f"E={rb_energy(G,{v:1 for v in G},1.):.3f}\n")

    print("== E(bundle) - E(staggered) = gamma exactly, all sizes and gammas ==")
    print("    t     n      cut(bundle)  cut(staggered)   gamma=0.5   1.0   3.0")
    for t in range(1, 8):
        G, (A, B), bundle, stag = cuts(t)
        cb = sum(1 for u, v in G.edges() if bundle[u] != bundle[v])
        cs = sum(1 for u, v in G.edges() if stag[u] != stag[v])
        d = [rb_energy(G, bundle, g) - rb_energy(G, stag, g) for g in (0.5, 1.0, 3.0)]
        assert cb == cs == 2 ** t
        assert all(abs(x - g) < 1e-9 for x, g in zip(d, (0.5, 1.0, 3.0)))
        print(f"   {t:>2}{G.number_of_nodes():>8}{cb:>12}{cs:>15}"
              + "".join(f"{x:>10.3f}" for x in d))
    print("   OK: identical cuts, staggered lower by exactly gamma.")
