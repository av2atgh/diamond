#!/usr/bin/env python3
r"""
potts_fixedpoint.py -- critical fixed point of the K-state ferromagnetic Potts
model on the (2,2) diamond, by exact Migdal-Kadanoff decimation.

A Potts bond is (A,B) = (same-colour weight, different-colour weight); with
r = B/A in [0,1] one generation of edge replacement acts as the exact map

    r -> f(r) = [ (2r + (K-2) r^2) / (1 + (K-1) r^2) ]^2                    (*)

(series of two bonds through an internal K-state spin, then two such paths in
parallel).  Everything about the finite-temperature community order follows from
(*):

  * the unstable fixed point r_*(K) gives the critical coupling J/k_BT_c
    = -ln r_*, hence T_c(K);
  * lambda_T(K) = f'(r_*) is the thermal eigenvalue: the transition is
    CONTINUOUS whenever lambda_T is finite (a discontinuity fixed point,
    lambda_T -> bs with r_* -> a jump, is what a first-order transition would
    require).  nu = ln s / ln lambda_T;
  * the width of the finite-size crossover of any local observable shrinks as
    lambda_T^{-t}, which is why the hub-hub order parameter steepens into a step
    with size for EVERY K -- a finite-size effect, not a latent heat;
  * the energy density (fraction of same-colour bonds) is continuous through
    T_c for every K: no latent heat.  This is checked here directly.

Large-K asymptotics, exact:  r_* -> K^{-2/3}, so  J/k_BT_c -> (2/3) ln K, i.e.
k_B T_c(K)/J -> 3/(2 ln K); and lambda_T -> 4 = bs, so nu -> ln2/ln4 = 1/2.

Requires: mpmath.
"""
import math
from mpmath import mp, mpf, log, exp, diff
mp.dps = 50

b, s = 2, 2


# ------------------------------------------------------------------ the map
def f(r, K):
    """One generation of the exact Potts bond recursion, Eq. (*)"""
    K = mpf(K); r = mpf(r)
    return ((2 * r + (K - 2) * r * r) / (1 + (K - 1) * r * r)) ** 2


def fixed_point(K):
    """Unstable fixed point r_*(K) of f, by bisection on f(r)-r."""
    g = lambda r: f(r, K) - r
    xs = [mpf(10) ** (mpf(-k) / 8) for k in range(200, 0, -1)] + [mpf('0.999')]
    lo = hi = None
    for i in range(len(xs) - 1):
        if g(xs[i]) <= 0 and g(xs[i + 1]) >= 0:
            lo, hi = xs[i], xs[i + 1]; break
    for _ in range(300):
        mid = (lo + hi) / 2
        if g(mid) < 0: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def critical(K):
    """(r_*, lambda_T, nu, K_c=J/kT_c, T_c/J) for the K-state Potts model."""
    r = fixed_point(K)
    lam = diff(lambda z: f(z, K), r)
    Kc = -log(r)
    return float(r), float(lam), float(log(s) / log(lam)), float(Kc), float(1 / Kc)


# ------------------------------------------------------- order parameter / energy
def pole_order(t, K, Kcoup):
    """Colour rigidity rho_K = (P(hubs same colour) - 1/K)/(1 - 1/K) for the
    height-t lattice at coupling Kcoup: the hub-hub colour correlation in excess
    of chance, i.e. the long-range order parameter of the K-colour field.  It is
    1 in the ordered phase (the colouring is rigid across the whole lattice) and
    0 in the disordered one."""
    K = mpf(K); A = exp(mpf(Kcoup)); B = mpf(1)
    for _ in range(t):
        As = A * A + (K - 1) * B * B
        Bs = 2 * A * B + (K - 2) * B * B
        A, B = As * As, Bs * Bs
    Psame = A / (A + (K - 1) * B)
    return float((Psame - 1 / K) / (1 - 1 / K))


def eps(t, K, Kcoup):
    """Fraction of same-colour bonds, (1/m) d lnZ/dKcoup, by exact derivative
    propagation through the same recursion (log-scale, so any t is reachable)."""
    K = mpf(K); Kc = mpf(Kcoup)
    L = Kc; r = exp(-Kc); u = mpf(1); v = -exp(-Kc)      # L=lnA, r=B/A, u=L', v=r'
    for _ in range(t):
        S = 1 + (K - 1) * r * r; dS = 2 * (K - 1) * r * v
        L = 4 * L + 2 * log(S); u = 4 * u + 2 * dS / S
        n = 2 * r + (K - 2) * r * r; dn = 2 * v + 2 * (K - 2) * r * v
        rs = n / S; drs = (dn * S - n * dS) / (S * S)
        r = rs * rs; v = 2 * rs * drs
    return float((u + (K - 1) * v / (1 + (K - 1) * r)) / mpf(4) ** t)


def width(t, K, lo=0.05, hi=0.95):
    """10-90%% width of the order-parameter transition, in units of the coupling."""
    Kc = mpf(critical(K)[3])
    def solve(target):
        a, c = Kc * mpf('0.3'), Kc * mpf('3.0')
        for _ in range(120):
            mid = (a + c) / 2
            if pole_order(t, K, mid) > target: c = mid
            else: a = mid
        return (a + c) / 2
    return float(abs(solve(lo) - solve(hi)) / Kc)


# ================================================================== __main__
if __name__ == "__main__":
    KS = [2, 3, 4, 8, 16, 32, 64, 128, 256, 1024, 10 ** 5, 10 ** 7, 10 ** 10]
    print("== critical fixed point of the K-state Potts model on the diamond ==")
    print(f"{'K':>11}{'r_*':>13}{'lambda_T':>11}{'nu':>8}{'J/kT_c':>10}"
          f"{'kT_c/J':>9}{'(J/kT_c)/lnK':>14}")
    for K in KS:
        r, lam, nu, Kc, Tc = critical(K)
        print(f"{K:>11}{r:>13.7f}{lam:>11.5f}{nu:>8.4f}{Kc:>10.5f}{Tc:>9.5f}"
              f"{Kc/math.log(K):>14.5f}")
    print("   -> (J/kT_c)/ln K -> 2/3 exactly, lambda_T -> bs = 4, nu -> 1/2\n")

    print("== no latent heat: eps(K_c(1+d)) - eps(K_c(1-d)) at t=18, d -> 0 ==")
    ds = ['1e-2', '3e-3', '1e-3', '3e-4', '1e-4']
    print(f"{'K':>4}  " + "".join(f"d={d:<9}" for d in ds))
    for K in [2, 3, 4, 8, 16, 32]:
        Kc = mpf(critical(K)[3])
        row = [eps(18, K, Kc * (1 + mpf(d))) - eps(18, K, Kc * (1 - mpf(d))) for d in ds]
        print(f"{K:>4}  " + "".join(f"{v:<11.5f}" for v in row))
    print("   -> the energy density is continuous for every K: the transition is\n"
          "      continuous, and the step seen in the order parameter is finite size.\n")

    print("== finite-size width of the order-parameter step ~ lambda_T^{-t} ==")
    ts = [6, 8, 10, 12, 14]
    print(f"{'K':>4} " + "".join(f"t={t:<10}" for t in ts) + "  ratio/gen   lambda_T")
    for K in [2, 4, 8, 16]:
        w = [width(t, K) for t in ts]
        ratio = (w[0] / w[-1]) ** (1.0 / (ts[-1] - ts[0]))
        print(f"{K:>4} " + "".join(f"{v:<11.3e}" for v in w)
              + f"  {ratio:>8.3f}   {critical(K)[1]:>8.3f}")
