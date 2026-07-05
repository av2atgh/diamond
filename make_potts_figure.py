#!/usr/bin/env python3
"""fig_potts.pdf: (a) q-Potts pole order vs T for several q (sharpening to
first-order); (b) Tc(q) ~ 1/ln q decreasing with hierarchy depth."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rb_potts_decimation as pd
from mpmath import mpf
import math

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.6,2.7))

# (a) order parameter vs T/Tc(q) for q=2,4,8,16
for q,col in [(2,'#1f4e9b'),(4,'#6a3d9a'),(8,'#b02020'),(16,'#e08214')]:
    Kc=pd.find_Tc_potts(10,q)
    fr=np.linspace(0.9,1.15,26)
    m=[pd.potts_pole_corr(10,q,Kc*mpf(str(f))) for f in fr]
    axA.plot(1/fr, m, '-', color=col, lw=1.6, label=fr'$q={q}$')
axA.axvline(1.0, ls='--', color='0.5', lw=1.0)
axA.set_xlabel(r'$T/T_c(q)$',fontsize=9)
axA.set_ylabel(r'community order $m_q$',fontsize=9)
axA.set_ylim(-0.03,1.03); axA.tick_params(labelsize=8)
axA.legend(fontsize=7,frameon=False,loc='upper right')
axA.set_title(r'(a) sharpens to first order ($q>2$)',fontsize=9)

# (b) Tc(q) vs q
qs=[2,4,8,16,32,64,128]
Tcs=[1/float(pd.find_Tc_potts(12,q)) for q in qs]
axB.semilogx(qs, Tcs, 'o-', color='#b02020', mfc='white', mec='#b02020', lw=1.5, ms=6, base=2)
# guide 1/(c ln q)
qc=np.array(qs,dtype=float); guide=1/(0.70*np.log(qc))
axB.semilogx(qs, guide, '--', color='0.4', lw=1.2, base=2, label=r'$\sim 1/\ln q$')
axB.set_xlabel(r'number of communities $q=2^d$',fontsize=9)
axB.set_ylabel(r'$T_c(q)/J$',fontsize=9)
axB.tick_params(labelsize=8); axB.legend(fontsize=8,frameon=False)
axB.set_title(r'(b) $T_c$ falls as hierarchy deepens',fontsize=9)

fig.tight_layout(); fig.savefig('fig_potts.pdf',bbox_inches='tight')
print("wrote fig_potts.pdf")
print("Tc(q):",list(zip(qs,[round(t,4) for t in Tcs])))
