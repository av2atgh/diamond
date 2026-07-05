#!/usr/bin/env python3
"""fig_hier.pdf: (a) modularity Q(d) vs depth; (b) q_opt ~ sqrt(N)."""
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
axA.set_xlabel(r'hierarchy depth $d$  ($q=2^d$)',fontsize=9)
axA.set_ylabel(r'modularity $Q$',fontsize=9)
axA.tick_params(labelsize=8); axA.legend(fontsize=7,frameon=False,loc='lower center')
axA.set_title(r'(a) optimal depth grows with $n$',fontsize=9)

ts=list(range(2,10)); Ns=[]; qs=[]
for t in ts:
    Qs=[Q_of_d(t,d,comp[t]) for d in range(0,t+1)]
    dopt=int(np.argmax(Qs)); Ns.append(2+2*(4**t-1)//3); qs.append(2**dopt)
axB.loglog(Ns,qs,'o',color='#b02020',mfc='white',mec='#b02020',ms=7,label=r'$q_{\rm opt}$')
axB.loglog(Ns,np.sqrt(Ns),'--',color='0.4',lw=1.2,label=r'$\sqrt{n}$')
axB.set_xlabel(r'network size $n$',fontsize=9)
axB.set_ylabel(r'optimal $q_{\rm opt}=2^{d_\star}$',fontsize=9)
axB.tick_params(labelsize=8); axB.legend(fontsize=8,frameon=False,loc='upper left')
axB.set_title(r'(b) $q_{\rm opt}\sim\sqrt{n}$',fontsize=9)

fig.tight_layout(); fig.savefig('fig_hier.pdf',bbox_inches='tight')
print("wrote fig_hier.pdf; q_opt=",list(zip(ts,qs)))
