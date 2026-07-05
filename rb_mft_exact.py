#!/usr/bin/env python3
r"""
Thermodynamic-limit-exact solution of q=2 RB-Ising on the diamond.

H = -sum_edges s_i s_j + (gamma/4m) M_k^2 + const,  M_k = sum_i k_i s_i.

The M_k^2 term is infinite-range through the single collective variable M_k, so
its Hubbard-Stratonovich / saddle treatment is EXACT as N->inf (Nishimori Ch.1).
HS (penalty, a=beta*gamma/4m>0) with oscillatory field -> real saddle at
stationary point of the free energy:
  f(psi) = psi^2/(4a')  -  (1/N) ln Z_ferro[ h_i = psi*k_i ]
where the spins live on the FERROMAGNETIC diamond with a degree-proportional
field h_i = psi * k_i (real), and we EXTREMISE over psi. The physical branch is
the free-energy MINIMUM. psi<0 penalises alignment (drives the split).

Z_ferro in a degree-proportional field is computed EXACTLY by decimation because
the field on a spin depends only on its birth generation (degree 2^{t-g+1}).
We carry the field through the bond recursion generation by generation.

Order parameter for community/hub anti-alignment: we detect the split via the
pole magnetization m_A = <s_A> in the self-consistent field: in the split phase
the two poles are oppositely polarised by the staggered saddle. Cleanest scalar:
the degree-weighted magnetization m_k = M_k/(2m) = 0 in the split phase (balanced)
and !=0 in the FM phase -> but the ORDER is the anti-alignment. We instead track
the FM<->split transition through the free-energy: two saddles (psi=0 FM after
gauge, vs psi!=0). Practically: compute chi_k = d m_k/d(field) divergence, or
directly the pole-pole correlation via a two-replica trick.

SIMPLEST robust exact-in-N diagnostic: the uniform (community) susceptibility of
the degree-weighted magnetization,
  chi_k = beta[ <M_k^2> - <M_k>^2 ]/(2m).
In the FM phase M_k is extensive (chi_k ~ N); the RB penalty suppresses it. The
community-ordered (split) phase is where the penalty has forced <M_k>~0 with the
two hubs anti-aligned. We compute the exact free energy f(gamma,T) from the GF
solver's Z at the sizes we can reach, and read the transition from the kink /
from where -<sAsB> rises. For SPEED we use decimation for Z_ferro(field) and do
the 1-D saddle over psi.
"""
import numpy as np
from mpmath import mp, mpf, exp, log, cosh, sinh, tanh
mp.dps = 30
SP=[mpf(1),mpf(-1)]

def logZ_ferro_field(t, K, psi):
    """log Z of the FERROMAGNETIC diamond, each spin field h_i=psi*k_i (degree).
    Exact decimation: build height-t weight with mid-spin of a height-h subunit
    carrying field psi*2^h; poles carry psi*2^t."""
    W=_wh(t,K,psi)
    kp=2**t
    Z=mpf(0)
    for i,sa in enumerate(SP):
        for j,sb in enumerate(SP):
            Z+=W[i][j]*exp(psi*kp*(sa+sb))
    return log(Z)

_cache={}
def _wh(h,K,psi):
    if h==0:
        return [[exp(K*sa*sb) for sb in SP] for sa in SP]
    key=(h,mp.nstr(K,20),mp.nstr(psi,20))
    if key in _cache: return _cache[key]
    sub=_wh(h-1,K,psi)
    bmid=psi*(2**h)
    P=[[mpf(0)]*2 for _ in range(2)]
    for i in range(2):
        for j in range(2):
            P[i][j]=sum(sub[i][a]*exp(bmid*SP[a])*sub[a][j] for a in range(2))
    W=[[P[i][j]*P[i][j] for j in range(2)] for i in range(2)]
    _cache[key]=W
    return W

def free_energy(t, gamma, beta, psi):
    """Total variational free energy per spin f(psi) (to be extremised).
    beta*H includes (beta gamma/4m)M_k^2; HS gives f = -a' psi^2 - (1/N)lnZ_f[psi]
    with the saddle at psi* = -(beta gamma/2m)<M_k>. We build:
      -beta F = lnZ = extremum over psi of [ -(m/(beta gamma)) psi^2 + lnZ_ferro(psi) ]
    (from exp(-a M^2)=int dphi exp(-phi^2/4a - phi M), phi=psi*(2m)... ) -- we
    calibrate the psi normalisation against the GF solver."""
    m=mpf(4)**t; N=2+2*(4**t-1)//3
    K=beta
    # -phi^2/(4a) with a=beta gamma/(4m); let field on spin = -psi*k. Then
    # matching -phi M_k with sum_i (-psi k_i) s_i = -psi M_k => phi=psi.
    # exponent E(psi) = -psi^2/(4a) + lnZ_ferro(field=-psi*k)
    a=beta*gamma/(4*m)
    return -psi*psi/(4*a) + logZ_ferro_field(t,K,-psi)

def solve(t, gamma, beta):
    """Maximise E(psi) over psi>=0 (by symmetry). Return psi*, m_k, and -lnZ."""
    # scan then refine
    best=None; bestpsi=mpf(0)
    a=beta*gamma/(4*mpf(4)**t)
    # psi ~ 2a <M_k>; <M_k> up to 2m => psi up to ~ beta gamma. scan
    import numpy as np
    grid=[mpf(x) for x in np.linspace(0,float(2*beta*gamma+2),80)]
    vals=[(free_energy(t,gamma,beta,p),p) for p in grid]
    E0,p0=max(vals,key=lambda z:float(z[0]))
    # refine
    lo,hi=max(mpf(0),p0-mpf('0.1')),p0+mpf('0.1')
    for _ in range(60):
        m1=lo+(hi-lo)/3; m2=hi-(hi-lo)/3
        if free_energy(t,gamma,beta,m1)<free_energy(t,gamma,beta,m2): lo=m1
        else: hi=m2
    psis=(lo+hi)/2
    # m_k = <M_k>/2m = psi/(2*beta*gamma/2m * ... ) from saddle psi=2a<M_k>=>><M_k>=psi/(2a)
    Mk=psis/(2*a)
    return float(psis), float(Mk/(2*mpf(4)**t))

if __name__=="__main__":
    import rb_exact_gf as gf
    from mpmath import mpf as MPF
    print("Validate MFT(exact-in-N) vs GF at t=5 (m_k degree-wtd magnetization):")
    for gamma in [0.5,1.0,2.0]:
        _,mk_mft=solve(5,gamma,MPF(5))
        mk_gf=gf.rb_observables(5,gamma,MPF(5))
        print(f"  gamma={gamma}: MFT m_k={mk_mft:.4f}  GF m_k={mk_gf:.4f}")
