#!/usr/bin/env python3
r"""
Exact finite-T q-state Potts on the diamond via Migdal-Kadanoff decimation,
for the HIERARCHICAL community model. We compute the pole-pole Potts correlation
and the level-resolved sibling order, testing persistence as n->inf.

For the PURE ferromagnetic Potts part (couplings on edges), decimation is exact:
a Potts bond weight is a 2-parameter object (same-colour weight a, diff-colour b).
Series and parallel laws are exact. We ALSO need the RB penalty, which is
infinite-range. We treat the model in the regime where the community assignment
is imposed as the reference ground state and ask about thermal fluctuations of
the ferromagnetic Potts model whose ground state IS that partition -- i.e. we
build a Potts model with EDGE couplings that are ferromagnetic WITHIN intended
communities and antiferromagnetic (or zero + penalty) ACROSS, mimicking RB.

Cleanest exact object (matches Jelitto/Kaufman results): the q-state Potts model
where the two hubs are the roots, on the diamond, ferromagnetic. Its pole-pole
correlation is exactly computable by decimation and is KNOWN to be discontinuous
at Tc(q) for q>2. We compute it for our hierarchical q=2^d and see how Tc and the
correlation depend on d (hence on q, hence on n along the optimal locus).

Potts bond object: 2x2-in-effect but really (a,b): weight for same colour = A,
different = B. Represent normalized r = B/A in [0,1]. 
Series of two bonds (r1,r2) through an internal q-state spin:
  same-ends same colour: sum over mid: (mid=c: A1 A2) + (mid!=c, q-1 terms: B1 B2)
     -> A_s = A1 A2 + (q-1) B1 B2
  diff-ends (colours c!=c'): mid=c: A1 B2; mid=c': B1 A2; mid other(q-2): B1 B2
     -> B_s = A1 B2 + B1 A2 + (q-2) B1 B2
Parallel of two bonds: multiply weights: A_p=A1 A2, B_p=B1 B2.
Start bond ferromagnetic: A=e^{K}, B=1 (coupling K on same colour).
"""
import numpy as np
from mpmath import mp, mpf, exp, log
mp.dps = 40

def potts_pole_corr(t, q, K):
    """Ferromagnetic q-Potts on height-t diamond; return P(poles same colour)-1/q
    normalized order parameter m = (P_same - 1/q)/(1-1/q) in [0,1]."""
    # bond (A,B): same-colour A, diff-colour B
    A=exp(K); B=mpf(1)
    def series(A1,B1,A2,B2):
        As=A1*A2+(q-1)*B1*B2
        Bs=A1*B2+B1*A2+(q-2)*B1*B2
        return As,Bs
    def parallel(A1,B1,A2,B2):
        return A1*A2, B1*B2
    # build height-t: recursion W_h from W_{h-1}: path=series of two W_{h-1}; then
    # two paths in parallel.
    def Wh(h):
        if h==0: return A,B
        a,b=Wh(h-1)
        # series two (a,b) through mid
        as_,bs_=series(a,b,a,b)
        # parallel of the b=2 paths
        ap,bp=parallel(as_,bs_,as_,bs_)
        return ap,bp
    Af,Bf=Wh(t)
    # poles same colour: weight Af (q choices) ; diff: Bf (q(q-1) ordered)
    Psame=q*Af/(q*Af+q*(q-1)*Bf)
    m=(Psame-mpf(1)/q)/(1-mpf(1)/q)
    return float(m)

def find_Tc_potts(t, q):
    """Locate the ferromagnetic Potts transition via the pole correlation jump."""
    # scan K; m jumps 0->finite. Return K where m crosses 0.5 (proxy).
    lo,hi=mpf('0.01'),mpf('5.0')
    for _ in range(60):
        mid=(lo+hi)/2
        if potts_pole_corr(t,q,mid)>0.5: hi=mid
        else: lo=mid
    return (lo+hi)/2

if __name__=="__main__":
    print("Ferromagnetic q-Potts pole-pole order parameter on diamond, vs q and size.")
    print("(Hierarchical model: q=2^d states; does pole order persist as t->inf?)\n")
    for q in [2,4,8,16]:
        print(f"q={q}:")
        Kc=find_Tc_potts(8,q)
        print(f"  K_c(t=8) ~ {float(Kc):.4f}  (T_c/J={1/float(Kc):.4f})")
        # pole order at fixed T just below Tc, vs size
        Ktest=Kc*mpf('1.05')
        print(f"  pole order at K=1.05 K_c vs t: ", end="")
        for t in [3,4,5,6,7,8]:
            print(f"{potts_pole_corr(t,q,Ktest):.3f}", end=" ")
        print()
