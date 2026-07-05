#!/usr/bin/env python3
r"""
general_bs.py -- exact closed forms for the (b,s) diamond family, K=b bundle
partition, and their verification against direct construction.

Derivation being verified (End Matter of the manuscript):

  Bundle lemma: edges lie only within a bundle or between a pole and a bundle;
  no edge ever joins two distinct bundles (induction on edge replacement).

  Per-bundle statistics (s>=2, t>=1):
    c_t = 2 b^{t-1}                 pole-incident edges of one bundle
    i_t = b^{t-1}(s^t - 2)          interior edges of one bundle
  with exact triangular recursion
    c_{t+1} = b c_t
    i_{t+1} = bs i_t + b(s-1) c_t          => spectrum {bs, b}.

  Block statistics for block0 = bundle0 + {A,B}, block r = bundle r (r>=1):
    e_00 = i_t + c_t = b^{t-1} s^t
    e_rr = i_t                          (r>=1)
    e_0r = c_t                          (r>=1)
    e_rs = 0                            (r,s>=1, r!=s)
    kappa_r = 2 i_t + c_t = 2 b^{t-1}(s^t - 1)          (r>=1)
    kappa_0 = kappa_r + b c_t = 2 b^{t-1}(s^t - 1 + b)
    N_t = 2 + b(s-1) ((bs)^t - 1)/(bs - 1)

  Densities:  u_in -> b,  u_x = b s^t / [(s^t-1)(s^t-1+b)] -> b/s^t.
  Evidence density: log R_dc / m -> ln b.
"""
import math
import networkx as nx
from mpmath import mp, mpf, loggamma, log as mlog
mp.dps = 80

from diamond_rg import diamond, logR_dc


# ---------------------------------------------------------------- closed forms
def counts_bs(t, b, s):
    """(i, c, e00, err, e0r, kap0, kapr, N, m) for the K=b bundle partition."""
    c = 2 * b ** (t - 1)
    i = b ** (t - 1) * (s ** t - 2)
    e00 = i + c
    kapr = 2 * i + c
    kap0 = kapr + b * c
    N = 2 + b * (s - 1) * ((b * s) ** t - 1) // (b * s - 1)
    m = (b * s) ** t
    return i, c, e00, i, c, kap0, kapr, N, m


def logR_dc_bs(t, b, s, alpha=1.0):
    """Closed-form DC evidence of the K=b bundle partition."""
    a = mpf(alpha)
    i, c, e00, err, e0r, kap0, kapr, N, m = counts_bs(t, b, s)
    twom = 2 * m
    lZ = lambda e, O: loggamma(e + a) - (e + a) * mlog(O + a) + a * mlog(a) - loggamma(a)
    tot = mpf(0)
    # block-pair sums with multiplicities
    tot += lZ(e00, mpf(kap0 * kap0) / (2 * twom))                       # (0,0)
    tot += (b - 1) * lZ(err, mpf(kapr * kapr) / (2 * twom))             # (r,r), r>=1
    tot += (b - 1) * lZ(e0r, mpf(kap0 * kapr) / twom)                   # (0,r)
    tot += ((b - 1) * (b - 2) // 2) * lZ(0, mpf(kapr * kapr) / twom)    # (r,s) empty
    return float((tot - lZ(m, mpf(m))).real)


# ================================================================ verification
if __name__ == "__main__":
    print("== closed-form (i,c,e,kappa,N) and logR_dc vs direct construction ==")
    for (b, s) in [(2, 2), (3, 2), (4, 2), (2, 3), (2, 4), (3, 3), (4, 3), (2, 5)]:
        for t in range(1, 5):
            if (b * s) ** t > 40000:
                break
            G, poles, part = diamond(t, b, s)
            i, c, e00, err, e0r, kap0, kapr, N, m = counts_bs(t, b, s)
            # direct statistics
            deg = dict(G.degree())
            blocks = sorted(set(part.values()))
            kd = {r: sum(deg[v] for v in G if part[v] == r) for r in blocks}
            ed = {}
            for u, v in G.edges():
                r, q = sorted((part[u], part[v]))
                ed[(r, q)] = ed.get((r, q), 0) + 1
            assert G.number_of_nodes() == N and G.number_of_edges() == m
            assert ed.get((0, 0), 0) == e00
            for r in range(1, b):
                assert ed.get((r, r), 0) == err
                assert ed.get((0, r), 0) == e0r
                assert kd[r] == kapr
                for q in range(r + 1, b):
                    assert ed.get((r, q), 0) == 0     # bundle lemma
            assert kd[0] == kap0
            assert abs(logR_dc_bs(t, b, s) - logR_dc(G, part, 1.0)) < 1e-6
        print(f"   (b,s)=({b},{s})  OK")
    print()

    print("== per-bundle recursion i'=bs*i+b(s-1)c, c'=b*c (exact) ==")
    for (b, s) in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2)]:
        for t in range(1, 8):
            i1, c1 = counts_bs(t, b, s)[:2]
            i2, c2 = counts_bs(t + 1, b, s)[:2]
            assert i2 == b * s * i1 + b * (s - 1) * c1 and c2 == b * c1
    print("   OK: triangular map, eigenvalues {bs, b}\n")

    print("== density flow: u_in -> b,  u_x = b s^t/((s^t-1)(s^t-1+b)) ==")
    for (b, s) in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        i, c, e00, err, e0r, kap0, kapr, N, m = counts_bs(6, b, s)
        uin = err / (kapr * kapr / (4 * m))
        ux = e0r / (kap0 * kapr / (2 * m))
        ux_cf = b * s ** 6 / ((s ** 6 - 1) * (s ** 6 - 1 + b))
        print(f"   (b,s)=({b},{s}):  u_in={uin:.5f} (-> {b})   "
              f"u_x={ux:.3e} = closed {ux_cf:.3e}")
    print()

    print("== evidence density logR_dc/m -> ln b  (closed form, deep t) ==")
    for (b, s) in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 2)]:
        d = logR_dc_bs(20, b, s) / (b * s) ** 20
        print(f"   (b,s)=({b},{s}):  logR_dc/m(t=20) = {d:.6f}   ln b = {math.log(b):.6f}")
    print()

    print("== crossing formula t*=ceil(log_bs(L/ln b)) exact at large log-odds ==")
    for (b, s) in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        ok = True
        for L in [1e2, 1e3, 1e5, 1e7]:
            tf = math.ceil(math.log(L / math.log(b)) / math.log(b * s))
            tn = next(t for t in range(1, 40) if logR_dc_bs(t, b, s) >= L)
            ok &= (tf == tn)
        print(f"   (b,s)=({b},{s}):  exact for L in [1e2,1e7]: {ok}")
