#!/usr/bin/env python3
r"""
HIERARCHICAL community-staggered magnetization on the (2,2) diamond Ising model.

Hierarchical partition (choice 1 of the plan): at each generation the two
bundles created by the last edge-replacement are the two children of a block;
recursing, depth-d gives 2^d blocks.  We assign a hierarchical staggered field
following a level-alternating (Mattis-like) sign:

   epsilon(block) = product over levels l of s_l,   s_l in {+1,-1}

chosen so that the two children of every node carry OPPOSITE sign.  This is the
sign pattern of the ground state of an ANTIFERROMAGNET on the block tree, and
it is the natural conjugate to a *hierarchical* staggered magnetization

   M_C^hier = (1/N) | sum_i epsilon(block(i)) <sigma_i> | .

Key question: does the staggered susceptibility conjugate to this field grow
with the SAME rate as the uniform one (=> genuine competing order that persists
as n->inf), rather than saturating (as the single 2-block staggering did)?

We build the field pattern by carrying a SIGN through the recursion: when an
edge is replaced by b=2 paths, the two paths get sign factors (+1,-1) times the
parent sign, and internal spins of a path inherit that path's sign.  We then
put a field h*sign on every spin and compute chi = d^2 lnZ / dh^2 at h=0 by the
exact 2-pole bond recursion, carrying the sign as we grow.
"""
import math
from mpmath import mp, mpf, exp, log
mp.dps = 50
from ising_diamond import find_Kc, n_t

SP = [mpf(1), mpf(-1)]
K = None
H = None   # global staggered field amplitude

def bundle(t, sign, depth_signs):
    """Weight W(sA,sB) of ONE bundle at generation t whose top-level sign is
    `sign`.  depth_signs: how the alternating sign propagates.  Internal spins
    carry field H*sign_of_their_path.  Recursion:
      gen1 bundle: A-mid-B, mid carries field H*sign
      gen t: two sub-diamonds in series; the mid spin carries field H*sign;
             each sub-diamond is a full diamond whose two bundles get signs
             (sign, sign) at THIS level? -- no: the alternation is BETWEEN the
             two bundles of the SAME parent edge, handled in diamond_h().
    """
    if t == 1:
        return [[sum(exp(K*sa*m + K*m*sb + H*sign*m) for m in SP) for sb in SP] for sa in SP]
    D = diamond_h(t-1, sign)          # sub-diamond inherits this bundle's sign at deeper levels
    return [[sum(D[i][k]*exp(H*sign*SP[k])*D[k][j] for k in range(2)) for j in range(2)] for i in range(2)]

def diamond_h(t, parent_sign):
    """Full diamond at gen t: its TWO bundles get opposite staggered signs
    (+parent_sign, -parent_sign) -- the hierarchical alternation."""
    B0 = bundle(t, +parent_sign, None)
    B1 = bundle(t, -parent_sign, None)
    return [[B0[i][j]*B1[i][j] for j in range(2)] for i in range(2)]

def logZ_hier(t, Kval, h, pole_sign=+1):
    """log Z with hierarchical staggered field amplitude h; poles carry +sign."""
    global K, H
    K = Kval; H = h
    W = diamond_h(t, +1)
    Z = mpf(0)
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            Z += W[i][j]*exp(H*pole_sign*(sa+sb))
    return log(Z)

def logZ_unif(t, Kval, h):
    global K, H
    K = Kval; H = mpf(0)
    # uniform: reuse bundle with H acting uniformly -> set sign structure off
    # simplest: build with H=0 in staggered slots but add uniform field via poles trick.
    # Easier: dedicated uniform builder.
    return _logZ_uniform(t, Kval, h)

def _bundle_u(t, hu):
    if t == 1:
        return [[sum(exp(K*sa*m + K*m*sb + hu*m) for m in SP) for sb in SP] for sa in SP]
    D = _diam_u(t-1, hu)
    return [[sum(D[i][k]*exp(hu*SP[k])*D[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
def _diam_u(t, hu):
    B = _bundle_u(t, hu)
    return [[B[i][j]*B[i][j] for j in range(2)] for i in range(2)]
def _logZ_uniform(t, Kval, h):
    global K
    K = Kval
    W = _diam_u(t, h)
    Z = mpf(0)
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            Z += W[i][j]*exp(h*(sa+sb))
    return log(Z)

if __name__ == "__main__":
    Kc = find_Kc()
    print(f"K_c={float(Kc):.5f}  T_c/J={float(1/Kc):.5f}   r_kappa=44 (t=3)\n")
    dh = mpf('1e-5')
    def chi_hier(t, Kv):
        f = lambda h: logZ_hier(t, Kv, h)
        return (f(dh)-2*f(mpf(0))+f(-dh))/(dh*dh)/n_t(t)
    def chi_unif(t, Kv):
        f = lambda h: _logZ_uniform(t, Kv, h)
        return (f(dh)-2*f(mpf(0))+f(-dh))/(dh*dh)/n_t(t)

    print("HIERARCHICAL staggered vs uniform susceptibility per spin:")
    print(f"{'':>6}{'t':>3}{'N':>8}  {'chi_hier':>12}{'chi_unif':>12}  {'hier/unif':>10}")
    for label,frac in [("T<Tc",mpf('1.3')),("T=Tc",mpf('1.0')),("T>Tc",mpf('0.7')),("hot",mpf('0.4'))]:
        Kv = frac*Kc
        for t in [2,3,4,5,6,7,8]:
            ch = chi_hier(t,Kv); cu = chi_unif(t,Kv)
            print(f"{label:>6}{t:>3}{n_t(t):>8}  {float(ch):>12.5g}{float(cu):>12.5g}  {float(ch/cu):>10.4f}")
        print()
