#!/usr/bin/env python3
"""fig_potts.pdf: finite-temperature community order of the K-state Potts model.

(a) hub-hub colour rigidity vs T at fixed K=4 for growing size: the step is a
    finite-size crossover;
(b) its 10-90%% width shrinks as lambda_T(K)^{-t} for every K, including K=2
    -- the transition is continuous, the apparent jump is finite size;
(c) T_c(K) with the exact large-K asymptote k_BT_c/J = 3/(2 ln K).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
from mpmath import mpf
import potts_fixedpoint as pf

COLS = {2: '#1f4e9b', 4: '#6a3d9a', 8: '#b02020', 16: '#e08214'}

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.4, 2.7))

# ---- (a) order parameter vs T at K=4, several sizes
Kstates = 4
Kc = pf.critical(Kstates)[3]
Ts = np.linspace(0.90, 1.10, 61)
for t, col in [(4, '#c6a0d0'), (6, '#9a6bb0'), (8, '#6a3d9a'), (12, '#2a1050')]:
    m = [pf.pole_order(t, Kstates, Kc / T) for T in Ts]
    n = 2 + 2 * (4 ** t - 1) // 3
    axA.plot(Ts, m, '-', color=col, lw=1.5, label=fr'$n={n}$')
axA.axvline(1.0, ls='--', color='0.5', lw=1.0)
axA.set_xlabel(r'$T/T_c(K)$', fontsize=9)
axA.set_ylabel(r'colour rigidity $\rho_K$', fontsize=9)
axA.set_ylim(-0.03, 1.03); axA.tick_params(labelsize=8)
axA.legend(fontsize=6.5, frameon=False, loc='upper right')
axA.set_title(r'(a) $K=4$: the step is finite size', fontsize=9)

# ---- (b) 10-90% width vs t, compared with lambda_T^{-t}
ts = [4, 6, 8, 10, 12, 14]
for Kst in [2, 4, 8, 16]:
    w = [pf.width(t, Kst) for t in ts]
    lam = pf.critical(Kst)[1]
    axB.semilogy(ts, w, 'o', color=COLS[Kst], mfc='white', mec=COLS[Kst], ms=5,
                 label=fr'$K={Kst}$  ($\lambda_T={lam:.2f}$)')
    ref = w[0] * lam ** (-(np.array(ts) - ts[0]))
    axB.semilogy(ts, ref, '-', color=COLS[Kst], lw=1.0)
axB.set_xlabel(r'generation $t$', fontsize=9)
axB.set_ylabel(r'transition width  $\Delta K/K_c$', fontsize=9)
axB.tick_params(labelsize=8); axB.legend(fontsize=6.2, frameon=False, loc='upper right')
axB.set_title(r'(b) width $\sim\lambda_T^{-t}$ for every $K$', fontsize=9)

# ---- (c) T_c(K) and the exact asymptote
Ks = [2, 3, 4, 8, 16, 32, 64, 128, 256, 1024, 4096]
Tcs = [pf.critical(K)[4] for K in Ks]
axC.semilogx(Ks, Tcs, 'o-', color='#b02020', mfc='white', mec='#b02020',
             lw=1.5, ms=5, base=2, label=r'exact $T_c(K)$')
Kd = np.array(Ks, dtype=float)
axC.semilogx(Ks, 1.5 / np.log(Kd), '--', color='0.4', lw=1.2, base=2,
             label=r'$3/(2\ln K)$')
axC.set_xlabel(r'number of communities $K=2^d$', fontsize=9)
axC.set_ylabel(r'$k_BT_c(K)/J$', fontsize=9)
axC.tick_params(labelsize=8); axC.legend(fontsize=7, frameon=False)
axC.set_title(r'(c) $T_c$ falls as the hierarchy deepens', fontsize=9)

fig.tight_layout(); fig.savefig('fig_potts.pdf', bbox_inches='tight')
print("wrote fig_potts.pdf")
print("T_c(K):", [(K, round(T, 4)) for K, T in zip(Ks, Tcs)])
print("lambda_T:", [(K, round(pf.critical(K)[1], 4)) for K in [2, 4, 8, 16]])
