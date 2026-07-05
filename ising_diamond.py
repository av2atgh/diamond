#!/usr/bin/env python3
r"""
Exact Ising model on the (b,s)=(2,2) diamond hierarchical lattice.

Method: the network is defined by edge replacement, so the exact partition
function with the two poles held fixed is captured by a 2x2 "bond" object
W(sA,sB) = sum over all internal spins of exp(-beta H).  Growing one
generation is an EXACT recursion on W:

  series (s=2):  chain two bonds through one internal spin (summed, with field)
  parallel (b=2): multiply the weights of parallel bonds between the same poles

Because every internal spin carries the field, this yields the exact free
energy AND the exact magnetization of any spin via numerical field derivative.
We use it to compute:
  * K_c (unstable fixed point of the zero-field coupling map)   [thermodynamics]
  * m_pole(t,h) exactly for ALL t (no enumeration)              [order parameter]
  * chi(t) = dm/dh|_{h=0}                                        [susceptibility]
"""
import math
from mpmath import mp, mpf, tanh, log, exp, cosh, sinh, mpmathify
mp.dps = 50

b, s = 2, 2

# ---- zero-field coupling recursion (for the thermodynamic critical point) ----
def K_step(K):
    Kser = mp.atanh(tanh(K)**s)   # s=2 bonds in series
    return b*Kser                  # b=2 in parallel

def find_Kc():
    lo, hi = mpf('0.2'), mpf('1.5')
    for _ in range(120):
        mid=(lo+hi)/2; K=mid
        for _ in range(60): K=K_step(K)
        if K > mid:  hi=mid       # above Kc, flows up (K grows past mid)
        else:        lo=mid       # below, flows down
    return (lo+hi)/2

# ---- exact bond-object recursion WITH field, for magnetization ----
# Represent a bond between poles as a 2x2 matrix W[a][b], a,b in {0:+1,1:-1},
# giving the summed Boltzmann weight over all internal spins for pole spins +-.
SP = [mpf(1), mpf(-1)]

def bare_bond(K, h):
    # single edge, half the field on each endpoint so a shared spin gets full h
    W = [[mpf(0),mpf(0)],[mpf(0),mpf(0)]]
    for i,si in enumerate(SP):
        for j,sj in enumerate(SP):
            W[i][j] = exp(K*si*sj + (h/2)*(si+sj))
    return W

def series2(W, h):
    # two bond-objects in series through one internal spin carrying REMAINING field.
    # The internal spin already received h/2 from each adjacent bond end => full h. OK.
    out=[[mpf(0),mpf(0)],[mpf(0),mpf(0)]]
    for i in range(2):
        for j in range(2):
            out[i][j]=sum(W[i][k]*W[k][j] for k in range(2))
    return out

def parallel(Wa,Wb):
    return [[Wa[i][j]*Wb[i][j] for j in range(2)] for i in range(2)]

def grow(W, h):
    """One generation: series of s=2 then parallel of b=2, exact."""
    Wser = series2(W, h)          # s=2 path
    Wcell = Wser
    for _ in range(b-1):
        Wcell = parallel(Wcell, Wser)   # b paths in parallel
    return Wcell

def pole_magnetization(t, K, h):
    """Exact m = <s_A> for the t-generation diamond at coupling K, field h.
    Build the full-lattice 2x2 pole weight by growing from a bare bond, then
    the two poles are connected by Wcell plus their own field; marginalize."""
    W = bare_bond(K, h)
    for _ in range(t):
        W = grow(W, h)
    # Now W[a][b] is the summed weight over ALL internal spins for pole spins (a,b),
    # but the poles themselves still need their own field weight exp(h*s_pole).
    # Both poles equivalent; total weight P(sA,sB) = W[sA,sB]*exp(h(sA+sB))
    # (field on poles was NOT included in bare end-fields for the two outermost ends;
    #  add it here once).
    Z=mpf(0); Ma=mpf(0)
    for i,si in enumerate(SP):
        for j,sj in enumerate(SP):
            w = W[i][j]*exp(h*(si+sj))
            Z += w; Ma += w*si
    return Ma/Z

def n_t(t): return 2 + 2*(4**t - 1)//3

if __name__=="__main__":
    Kc = find_Kc()
    print(f"(1) THERMODYNAMICS")
    print(f"    Ferromagnetic critical point:  K_c = J/kT_c = {mp.nstr(Kc,8)},  T_c/J = {mp.nstr(1/Kc,8)}")
    d = math.log(b*s)/math.log(s)
    print(f"    Hausdorff dimension d = ln(bs)/ln s = {d:.4f}  (finite-T transition, as expected)\n")

    print(f"(2) EXACT POLE MAGNETIZATION m(t,h)  [tiny field h=1e-4]")
    print(f"    below Tc: K=1.30*Kc ;  above Tc: K=0.70*Kc")
    h = mpf('1e-4')
    print(f"    {'t':>2} {'n_t':>8} {'m_below':>16} {'m_above':>16}   m_below/h  m_above/h")
    for t in range(1,13):
        mb = pole_magnetization(t, mpf('1.30')*Kc, h)
        ma = pole_magnetization(t, mpf('0.70')*Kc, h)
        print(f"    {t:2d} {n_t(t):8d} {mp.nstr(mb,8):>16} {mp.nstr(ma,8):>16}   {mp.nstr(mb/h,5)}  {mp.nstr(ma/h,5)}")
