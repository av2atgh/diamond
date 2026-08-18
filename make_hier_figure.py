#!/usr/bin/env python3
"""fig_hier.pdf: (a) modularity Q(d) vs depth; (b) K_opt = 2^{t/2} ~ m^{1/4} ~ n^{1/4}."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rb_potts_hier as rb
from collections import defaultdict

# cache exact S(d)=sumKc2/(2m)^2 by building graph once per t (t<=9)
def components(t):
    G,nc,poles=rb.diamond_labeled(t); m=G.number_of_edges(); deg=dict(G.degree())
    S={}
    for d in range(0,t+1):
        labs,q=rb.hier_labels(G,nc,poles,d)
        Kc=defaultdict(float)
        for n in G.nodes(): Kc[labs[n]]+=deg[n]
        S[d]=sum(v*v for v in Kc.values())/(2*m)**2
    return S,m

def Q_of_d(t,d,Scache,gamma=1.0):
    m=4**t
    e_in=m-(2**(t+d) if d>=1 else 0)
    return e_in/m - gamma*Scache[d]

fig,(axA,axB)=plt.subplots(1,2,figsize=(6.6,2.7))

comp={t:components(t)[0] for t in range(2,10)}

for t,col in [(4,'#c6a0d0'),(6,'#8a5bb0'),(8,'#3a1d6a')]:
    ds=list(range(0,t+1)); Qs=[Q_of_d(t,d,comp[t]) for d in ds]
    N=2+2*(4**t-1)//3
    axA.plot(ds,Qs,'o-',color=col,mfc='white',mec=col,lw=1.4,ms=4,label=fr'$n={N}$')
    dopt=int(np.argmax(Qs)); axA.plot([dopt],[Qs[dopt]],'*',color=col,ms=12)
axA.set_xlabel(r'hierarchy depth $d$  ($K=2^d$)',fontsize=9)
axA.set_ylabel(r'modularity $Q$',fontsize=9)
axA.tick_params(labelsize=8); axA.legend(fontsize=7,frameon=False,loc='lower center')
axA.set_title(r'(a) optimal depth grows with $n$',fontsize=9)

ts=list(range(2,10)); Ns=[]; qs=[]
for t in ts:
    Qs=[Q_of_d(t,d,comp[t]) for d in range(0,t+1)]
    dopt=int(np.argmax(Qs)); Ns.append(2+2*(4**t-1)//3); qs.append(2**dopt)
Ns=np.array(Ns,float); qs=np.array(qs,float)
slope,inter=np.polyfit(np.log(Ns),np.log(qs),1)
slope5,_=np.polyfit(np.log(Ns[-5:]),np.log(qs[-5:]),1)
axB.loglog(Ns,qs,'o',color='#b02020',mfc='white',mec='#b02020',ms=7,label=r'$K_{\rm opt}$')
axB.loglog(Ns,(1.5*Ns)**0.25,'--',color='0.4',lw=1.2,label=r'$m^{1/4}=(3n/2)^{1/4}$')
axB.loglog(Ns,np.sqrt(Ns),':',color='0.65',lw=1.2,label=r'$\sqrt{n}$ (for contrast)')
axB.set_xlabel(r'network size $n$',fontsize=9)
axB.set_ylabel(r'optimal $K_{\rm opt}=2^{d_\star}$',fontsize=9)
axB.tick_params(labelsize=8); axB.legend(fontsize=7,frameon=False,loc='upper left')
axB.set_title(r'(b) $K_{\rm opt}\sim n^{1/4}$',fontsize=9)
axB.text(0.97,0.06,f'fitted slope ${slope5:.3f}$',transform=axB.transAxes,
         ha='right',fontsize=7,color='#b02020')

fig.tight_layout(); fig.savefig('fig_hier.pdf',bbox_inches='tight')
print("wrote fig_hier.pdf; K_opt=",list(zip(ts,[int(q) for q in qs])))
print(f"  fitted slope (all sizes) = {slope:.4f}; (last five) = {slope5:.4f}; 1/4 = 0.25")
