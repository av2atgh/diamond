from diamond_rg import diamond
import networkx as nx

def cut_counts(G,poles,part):
    # 2-block cut: block 0 = bundle 0 + poles ; block 1 = everything else
    b2={n:(0 if (part[n]==0) else 1) for n in G}
    e12=sum(1 for u,v in G.edges() if b2[u]!=b2[v])
    return e12, G.number_of_edges()

for (b,s) in [(2,2),(3,2),(4,2),(2,3),(2,4),(3,3)]:
    row=[]
    for t in range(1,6):
        G,poles,part=diamond(t,b,s)
        # merge bundles 1..b-1 into block1
        part2={n:(0 if part[n] in (0,) else 1) for n in G}
        e12,m=cut_counts(G,poles,part2)
        row.append((m,e12))
    # ratios
    mr=row[-1][0]/row[-2][0]; er=row[-1][1]/row[-2][1] if row[-2][1]>0 else float('nan')
    print(f"(b={b},s={s}): m ratio={mr:.3f} (bs={b*s})   seam e12 ratio={er:.3f}   b={b} s={s}   e12 seq={[e for _,e in row]}")
