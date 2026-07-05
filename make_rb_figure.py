#!/usr/bin/env python3
"""fig_rb.pdf: (a) community order parameter -<sAsB> vs T/Tc for several gamma;
(b) size-sharpening into a step at T=Tc; (c) (gamma,T) phase diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rb_fast as rb

Kc = 0.6094

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.4, 2.7))

# (a) order parameter vs T/Tc at t=6, several gamma
Ts = np.linspace(0.5, 2.0, 16)
for g,col in [(0.2,'#1f4e9b'),(1.0,'#b02020'),(3.0,'#2a8f5a')]:
    op = [rb.rb_solve(6, g, Kc/T)[0] for T in Ts]
    axA.plot(Ts, op, 'o-', color=col, mfc='white', mec=col, lw=1.4, ms=3.5, label=fr'$\gamma={g}$')
axA.axvline(1.0, ls='--', color='0.4', lw=1.0)
axA.text(1.03, 0.05, r'$T_c$', color='0.3', fontsize=8)
axA.set_xlabel(r'$T/T_c^{\rm FM}$', fontsize=9)
axA.set_ylabel(r'community order $-\langle s_As_B\rangle$', fontsize=8.5)
axA.set_ylim(-0.03,1.03); axA.tick_params(labelsize=8)
axA.legend(fontsize=7, frameon=False, loc='upper right')
axA.set_title(r'(a) order vs temperature', fontsize=9)

# (b) size-sharpening at gamma=1
for t,col in [(3,'#c6a0d0'),(4,'#9a6bb0'),(5,'#6a3d9a'),(6,'#3a1d6a')]:
    op = [rb.rb_solve(t, 1.0, Kc/T)[0] for T in Ts]
    N = 2+2*(4**t-1)//3
    axB.plot(Ts, op, '-', color=col, lw=1.4, label=fr'$n={N}$')
axB.axvline(1.0, ls='--', color='0.4', lw=1.0)
axB.set_xlabel(r'$T/T_c^{\rm FM}$', fontsize=9)
axB.set_ylabel(r'$-\langle s_As_B\rangle$', fontsize=9)
axB.set_ylim(-0.03,1.03); axB.tick_params(labelsize=8)
axB.legend(fontsize=6.5, frameon=False, loc='upper right')
axB.set_title(r'(b) sharpening as $n\to\infty$ ($\gamma=1$)', fontsize=9)

# (c) phase diagram: order region in (gamma, T)
gammas = np.linspace(0.05, 3.0, 12)
Tgrid = np.linspace(0.6, 1.4, 12)
Z = np.zeros((len(Tgrid), len(gammas)))
for ii,T in enumerate(Tgrid):
    for jj,g in enumerate(gammas):
        Z[ii,jj] = rb.rb_solve(5, g, Kc/T)[0]
axC.contourf(gammas, Tgrid, Z, levels=np.linspace(0,1,11), cmap='RdBu_r')
axC.axhline(1.0, ls='--', color='k', lw=1.2)
axC.text(1.5, 1.03, r'$T_c^{\rm FM}$', fontsize=8)
axC.text(1.3, 0.72, 'community\nordered', fontsize=7.5, ha='center', color='white')
axC.text(1.3, 1.28, 'disordered', fontsize=7.5, ha='center')
axC.set_xlabel(r'resolution $\gamma$', fontsize=9)
axC.set_ylabel(r'$T/T_c^{\rm FM}$', fontsize=9)
axC.tick_params(labelsize=8)
axC.set_title(r'(c) $(\gamma,T)$ phase diagram', fontsize=9)

fig.tight_layout()
fig.savefig('fig_rb.pdf', bbox_inches='tight')
print("wrote fig_rb.pdf")
