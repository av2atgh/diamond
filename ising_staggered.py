#!/usr/bin/env python3
r"""
Community-staggered magnetization M_C on the (2,2) diamond Ising model.

Structure (bundle lemma): the gen-t diamond is TWO bundles in parallel between
poles A,B. Community C1 = bundle0 + poles; C2 = bundle1. Each bundle at gen t
is two gen-(t-1) diamonds in series through one internal spin.

We carry a community-resolved field: +h on C1 spins, -h on C2 spins, and
compute the exact partition function Z(h). Then
   M_C(h) = (1/N) [ |sum_{C1}<s>| + |sum_{C2}<s>| ]
For the +h/-h staggered field, by symmetry <s> is +ve in C1 and -ve in C2, so
   M_C = (1/N) d lnZ_stag / d h            (per spin, conjugate to staggered field)
and the staggered susceptibility chi_stag = dM_C/dh|_0.
The uniform magnetization uses +h everywhere: chi_unif = dM/dh|_0.

Both are computed from the SAME exact bundle recursion at 50-digit precision.
"""
import math
from mpmath import mp, mpf, exp, log
mp.dps = 50
from ising_diamond import find_Kc, n_t

SP = [mpf(1), mpf(-1)]
K = None  # set per call

def bundle_series(t, hb):
    """Weight W(sA,sB) of ONE bundle at generation t, field hb on all its
    internal spins. A bundle = two gen-(t-1) diamonds in series (one mid spin)."""
    if t == 1:
        # A - mid - B ; internal spin 'mid' carries hb
        return [[sum(exp(K*sa*m + K*m*sb + hb*m) for m in SP) for sb in SP] for sa in SP]
    D = diamond_full(t-1, hb, hb)   # sub-diamond, uniform field hb on its internals
    return [[sum(D[i][k]*exp(hb*SP[k])*D[k][j] for k in range(2)) for j in range(2)] for i in range(2)]

def diamond_full(t, h0, h1):
    """Full diamond at gen t: bundle0 (field h0) parallel bundle1 (field h1)."""
    B0 = bundle_series(t, h0)
    B1 = bundle_series(t, h1) if h1 != h0 else B0
    return [[B0[i][j]*B1[i][j] for j in range(2)] for i in range(2)]

def logZ(t, Kval, h0, h1, hpole):
    """log partition function; bundle0 internals field h0, bundle1 internals h1,
    poles field hpole."""
    global K; K = Kval
    W = diamond_full(t, h0, h1)
    Z = mpf(0)
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            Z += W[i][j]*exp(hpole*(sa+sb))
    return log(Z)

def n_of(t): return n_t(t)

if __name__ == "__main__":
    Kc = find_Kc()
    print(f"K_c={float(Kc):.5f}  T_c/J={float(1/Kc):.5f}")
    print("r_kappa = 44 (t=3)\n")
    dh = mpf('1e-5')

    def chi_stag(t, Kv):
        # staggered field: +h on C1 (bundle0 internals + poles), -h on C2 (bundle1)
        f = lambda h: logZ(t, Kv, h, -h, h)
        return (f(dh) - 2*f(mpf(0)) + f(-dh))/(dh*dh)/n_of(t)
    def chi_unif(t, Kv):
        f = lambda h: logZ(t, Kv, h, h, h)
        return (f(dh) - 2*f(mpf(0)) + f(-dh))/(dh*dh)/n_of(t)

    print(f"{'':>6}{'t':>3}{'N':>8}  {'chi_stag':>12}{'chi_unif':>12}  {'stag/unif':>10}")
    for label,frac in [("T<Tc",mpf('1.3')),("T=Tc",mpf('1.0')),("T>Tc",mpf('0.7')),("hot",mpf('0.4'))]:
        Kv = frac*Kc
        for t in [2,3,4,5,6,7]:
            cs = chi_stag(t,Kv); cu = chi_unif(t,Kv)
            print(f"{label:>6}{t:>3}{n_of(t):>8}  {float(cs):>12.4g}{float(cu):>12.4g}  {float(cs/cu):>10.3f}")
        print()

# ---- fine temperature scan: does staggered ever beat uniform? ----
if __name__ == "__main__":
    print("="*60)
    print("Fine scan: ratio chi_stag/chi_unif at t=7 (N=10924) vs T/Tc")
    print("A ratio > 1 would mean community-AF fluctuations dominate.")
    Kc = find_Kc()
    dh = mpf('1e-5'); t = 7
    def ratio(Kv):
        fs = lambda h: logZ(t, Kv, h, -h, h)
        fu = lambda h: logZ(t, Kv, h, h, h)
        cs = (fs(dh)-2*fs(mpf(0))+fs(-dh))
        cu = (fu(dh)-2*fu(mpf(0))+fu(-dh))
        return cs/cu
    print(f"{'T/Tc':>8}{'chi_s/chi_u':>14}")
    for r in [0.3,0.5,0.7,0.85,0.95,1.0,1.05,1.2,1.5,2.0,3.0]:
        Kv = mpf(str(1.0/r))*Kc   # T/Tc = 1/(K/Kc) => K = Kc/(T/Tc)
        print(f"{r:>8.2f}{float(ratio(Kv)):>14.4f}")
