"""Verify N_t(b,s) closed form and the r_kappa(b,s;q) crossing formula
for the symmetric b-bundle partition, numerically across (b,s)."""
import math
from diamond_rg import diamond, logR_dc

def N_closed(t,b,s):
    return 2 + b*(s-1)*((b*s)**t - 1)//(b*s - 1)

print("== N_t(b,s) closed form vs construction ==")
for (b,s) in [(2,2),(3,2),(2,3),(4,2),(2,4),(3,3)]:
    for t in range(1,5):
        G,_,_ = diamond(t,b,s)
        assert G.number_of_nodes()==N_closed(t,b,s), (b,s,t)
        assert G.number_of_edges()==(b*s)**t
print("   OK for (b,s) in {(2,2),(3,2),(2,3),(4,2),(2,4),(3,3)}, t<=4\n")

print("== r_kappa crossing: numeric t* vs formula ceil(log_bs(L/ln b)) ==")
q=0.99; L=math.log(q/(1-q))
print(f"   threshold L = ln(q/(1-q)) = {L:.3f}")
print("   (b,s)   t*_formula   t*_numeric   logR_dc at t*-1, t*   r_kappa=N_t*")
for (b,s) in [(2,2),(3,2),(2,3),(4,2),(2,4),(3,3)]:
    tf = math.ceil(math.log(L/math.log(b))/math.log(b*s))
    # numeric: K=b partition (bundle j -> block j, poles -> block 0)
    tn=None; vals={}
    for t in range(1,7):
        if (b*s)**t > 300000: break
        G,poles,part = diamond(t,b,s)
        lr = logR_dc(G,part,1.0)
        vals[t]=lr
        if tn is None and lr>=L: tn=t
    v1=vals.get((tn or 0)-1,float('nan')); v2=vals.get(tn,float('nan'))
    rk = N_closed(tn,b,s) if tn else None
    print(f"   ({b},{s})      {tf:>3}          {tn}         "
          f"{v1:>8.2f}, {v2:>8.2f}      {rk}")
