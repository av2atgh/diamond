#!/usr/bin/env python3
r"""
Ferromagnetic (all aligned) vs community-antiferromagnetic (each community
oppositely aligned) states on the (2,2) diamond, as a function of size.

Community-AF state: block-1 spins (bundle 0 + poles) up, block-2 spins
(bundle 1) down.  The ONLY unsatisfied bonds are the seam edges (E12), which
join the two communities.  So the energy cost of the AF state relative to F is
   Delta E = 2 J * (number of seam edges) = 2 J * E12 = 2 J * 2^t.

Relative Boltzmann weight of AF vs F for the two *ordered* states:
   P(AF)/P(F) = exp(-2 K * E12) = exp(-2 K * 2^t).

But the RELEVANT comparison is the full free energy of the two DOMAIN classes:
whether a macroscopic domain wall between communities costs a free energy that
grows (F wins, wall irrelevant) or stays finite/vanishes (AF and F become
degenerate).  The domain-wall free energy is the standard interface free energy
of the hierarchical lattice.  We compute it exactly by the ratio of partition
functions with symmetric (++ ) vs antisymmetric (+-) pole boundary conditions:
   F_wall(t) = -kT ln [ Z(+-) / Z(++) ].
Z(+-)/Z(++) is obtained exactly from the SAME bond recursion: it is
   R_t = W_t[+,-] / W_t[+,+]
where W_t is the renormalized 2x2 pole weight at zero field.
"""
import math
from mpmath import mp, mpf, tanh, exp, log
mp.dps = 50
from ising_diamond import find_Kc

b, s = 2, 2
SP = [mpf(1), mpf(-1)]

def bare(K):
    return [[exp(K*si*sj) for sj in SP] for si in SP]

def series2(W):
    return [[sum(W[i][k]*W[k][j] for k in range(2)) for j in range(2)] for i in range(2)]

def grow(W):
    Wser = series2(W); Wc = Wser
    for _ in range(b-1):
        Wc = [[Wc[i][j]*Wser[i][j] for j in range(2)] for i in range(2)]
    return Wc

def domain_wall_ratio(t, K):
    """R_t = Z(+-)/Z(++): relative weight of anti-aligned vs aligned poles.
    Poles carry the two communities' orientations, so this is exactly
    P(community-AF)/P(ferromagnetic)."""
    W = bare(K)
    for _ in range(t):
        W = grow(W)
    return W[0][1]/W[0][0]   # [+,-]/[+,+]

def n_t(t): return 2 + 2*(4**t - 1)//3

if __name__ == "__main__":
    Kc = find_Kc()
    print(f"K_c = {mp.nstr(Kc,6)}  (T_c/J = {mp.nstr(1/Kc,6)})")
    print(f"r_kappa = n_3 = {n_t(3)}\n")
    print("P(community-AF)/P(ferro) = Z(+-)/Z(++), vs size, at several T:")
    print(f"{'t':>2}{'n_t':>9}   T<Tc(1.3Kc)     T=Tc          T>Tc(0.7Kc)")
    for t in range(1,13):
        rb = domain_wall_ratio(t, mpf('1.3')*Kc)
        rc = domain_wall_ratio(t, Kc)
        ra = domain_wall_ratio(t, mpf('0.7')*Kc)
        star = "  <-- r_kappa" if t==3 else ""
        print(f"{t:>2}{n_t(t):>9}   {mp.nstr(rb,5):>12}  {mp.nstr(rc,5):>12}  {mp.nstr(ra,5):>12}{star}")

    print("\nInterpretation: ratio -> 1 means the two states are EQUALLY LIKELY.")
    print("Domain-wall free energy  F_wall/kT = -ln[Z(+-)/Z(++)]:")
    for t in [1,2,3,4,5,6,8,10]:
        fb = -log(domain_wall_ratio(t, mpf('1.3')*Kc))
        fa = -log(domain_wall_ratio(t, mpf('0.7')*Kc))
        print(f"  t={t:2d} n={n_t(t):7d}:  below Tc F_wall/kT={mp.nstr(fb,5):>10}   above Tc={mp.nstr(fa,5):>10}")
