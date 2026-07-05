#!/usr/bin/env python3
"""fig_ising.pdf: (a) magnetization onset; (b) ferro vs community-AF weight;
(c) staggered/uniform susceptibility ratio (no community order)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpmath import mp, mpf, log
mp.dps = 40
from ising_diamond import find_Kc, pole_magnetization, n_t
from ising_domains import domain_wall_ratio
import ising_staggered as stag

Kc = find_Kc(); rk = 44
ts = list(range(1,9)); ns = [n_t(t) for t in ts]

# (a) magnetization just below Tc
h = mpf('0.001'); K = mpf('1.10')*Kc
ms = [float(pole_magnetization(t, K, h)) for t in ts]

# (b) P(AF)/P(F)
r_below = [float(domain_wall_ratio(t, mpf('1.3')*Kc)) for t in ts]
r_at    = [float(domain_wall_ratio(t, Kc)) for t in ts]
r_above = [float(domain_wall_ratio(t, mpf('0.7')*Kc)) for t in ts]

# (c) chi_stag/chi_unif vs T/Tc at fixed large t
dh = mpf('1e-6'); tt = 7
Trs = [0.4,0.6,0.8,0.95,1.0,1.1,1.3,1.6,2.0,3.0,5.0]
ratios = []
for r in Trs:
    Kv = Kc/mpf(str(r))
    fs = lambda x: stag.logZ(tt, Kv, x, -x, x)
    fu = lambda x: stag.logZ(tt, Kv, x, x, x)
    cs = (fs(dh)-2*fs(mpf(0))+fs(-dh)); cu = (fu(dh)-2*fu(mpf(0))+fu(-dh))
    ratios.append(float(cs/cu))

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.4, 2.7))

axA.semilogx(ns, ms, 'o-', color='#b02020', mfc='white', mec='#b02020', lw=1.6, ms=5)
axA.axvline(rk, ls='--', color='0.4', lw=1.1)
axA.text(rk*1.15, 0.10, r'$r_\kappa$', color='0.3', fontsize=8)
axA.set_xlabel(r'network size $N_t$', fontsize=9)
axA.set_ylabel(r'magnetization $m$', fontsize=9)
axA.set_ylim(-0.03,1.03); axA.tick_params(labelsize=8)
axA.set_title(r'(a) $m$ onset ($T\!<\!T_c$)', fontsize=9)

axB.semilogx(ns, r_above, 'o-', color='#1f4e9b', mfc='white', mec='#1f4e9b', lw=1.5, ms=4.5, label=r'$T>T_c$')
axB.semilogx(ns, r_at, 's--', color='#2a8f5a', lw=1.2, ms=4, label=r'$T=T_c$')
axB.semilogx(ns, r_below, '^-', color='#b02020', mfc='white', mec='#b02020', lw=1.5, ms=4.5, label=r'$T<T_c$')
axB.axvline(rk, ls='--', color='0.4', lw=1.1); axB.axhline(1.0, ls=':', color='0.6', lw=0.9)
axB.set_xlabel(r'network size $N_t$', fontsize=9)
axB.set_ylabel(r'$P(\mathrm{AF})/P(\mathrm{F})$', fontsize=9)
axB.set_ylim(-0.03,1.08); axB.tick_params(labelsize=8)
axB.legend(fontsize=7, frameon=False, loc='center left')
axB.set_title(r'(b) ferro vs comm.-antiferro', fontsize=9)

axC.plot(Trs, ratios, 'o-', color='#6a3d9a', mfc='white', mec='#6a3d9a', lw=1.6, ms=4.5)
axC.axvline(1.0, ls='--', color='0.4', lw=1.1)
axC.axhline(1.0, ls=':', color='0.6', lw=0.9)
axC.text(1.05, 0.15, r'$T_c$', color='0.3', fontsize=8)
axC.text(3.2, 0.115, r'$0.108$', color='#2a8f5a', fontsize=7)
axC.plot([1.0],[0.108],'s',color='#2a8f5a',ms=5)
axC.set_xlabel(r'$T/T_c$', fontsize=9)
axC.set_ylabel(r'$\chi_{\mathrm{stag}}/\chi$', fontsize=9)
axC.set_ylim(-0.03,1.03); axC.set_xlim(0,5.2); axC.tick_params(labelsize=8)
axC.set_title(r'(c) no community order ($<1$)', fontsize=9)

fig.tight_layout()
fig.savefig('fig_ising.pdf', bbox_inches='tight')
print("wrote fig_ising.pdf (3 panels)")
print(f"  chi_stag/chi_unif: at Tc={ratios[4]:.3f}, T=5Tc={ratios[-1]:.3f}")
