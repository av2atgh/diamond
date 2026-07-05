#!/usr/bin/env python3
"""Cartoon of the diamond-lattice construction (b=s=2) by edge replacement.

Draws generations t=0..3: starting from a single bond A-B, each generation
replaces every edge by b=2 parallel paths of s=2 edges.  The positioned graph
is built by the same recursive rule and its node/edge counts are cross-checked
against diamond_rg.diamond(), so the cartoon is linked to the core module.

Writes fig_construction.pdf (Fig. 1, Sec. II).  Requires matplotlib, networkx.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import hypot
from diamond_rg import diamond, Nof

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 8,
    "pdf.fonttype": 42, "figure.dpi": 150,
})
PL, DC, DARK = "#d1495b", "#2a9d8f", "#264653"   # poles, interior nodes, edges


def subdivide(p, q, depth, amp):
    """One diamond edge replacement (b=2 parallel paths of s=2 edges), recursed
    to `depth` generations; returns a list of positioned edges."""
    if depth == 0:
        return [(p, q)]
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = hypot(dx, dy)
    ux, uy = -dy / L, dx / L                       # unit perpendicular
    up = (mx + ux * amp, my + uy * amp)            # interior node, path "up"
    dn = (mx - ux * amp, my - uy * amp)            # interior node, path "down"
    out = []
    for mid in (up, dn):                           # b=2 paths, s=2 edges each
        out += subdivide(p, mid, depth - 1, amp / 2)
        out += subdivide(mid, q, depth - 1, amp / 2)
    return out


def positioned_graph(t, amp0=0.46):
    """Recursive b=s=2 lattice with planar coordinates; poles A=(0,0), B=(1,0)."""
    A, B = (0.0, 0.0), (1.0, 0.0)
    key = lambda z: (round(z[0], 6), round(z[1], 6))
    pos, deg, E = {}, {}, []
    for p, q in subdivide(A, B, t, amp0):
        kp, kq = key(p), key(q)
        pos[kp], pos[kq] = p, q
        deg[kp] = deg.get(kp, 0) + 1
        deg[kq] = deg.get(kq, 0) + 1
        E.append((kp, kq))
    return pos, deg, E, {key(A), key(B)}


def draw(ax, t):
    pos, deg, E, poles = positioned_graph(t)
    if t >= 1:                                     # link to the reference module
        G, _, _ = diamond(t)
        assert (len(pos), len(E)) == (G.number_of_nodes(), G.number_of_edges())
    for kp, kq in E:
        (x1, y1), (x2, y2) = pos[kp], pos[kq]
        ax.plot([x1, x2], [y1, y2], color=DARK, lw=1.0, zorder=1)
    for k, (x, y) in pos.items():
        if k in poles:
            ax.scatter([x], [y], s=150, color=PL, edgecolors="white",
                       linewidths=0.7, zorder=3)
        else:
            ax.scatter([x], [y], s=7 + 5 * deg[k], color=DC,
                       edgecolors="white", linewidths=0.4, zorder=2)
    for k, lab in ((min(poles), "A"), (max(poles), "B")):
        ax.text(*pos[k], lab, color="white", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=4)
    ax.set_title(f"$t={t}$\n$n_t={Nof(t)},\\ m={4 ** t}$", fontsize=7.5)
    ax.set_xlim(-0.18, 1.18); ax.set_ylim(-0.62, 0.62)
    ax.set_aspect("equal"); ax.set_axis_off()


def arrow(ax, label=None):
    ax.set_axis_off()
    ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))
    if label:
        ax.text(0.5, 0.70, label, ha="center", va="bottom", fontsize=6.2,
                color="#555")


def main():
    fig = plt.figure(figsize=(7.0, 1.95))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 0.34, 1, 0.34, 1, 0.34, 1],
                          wspace=0.04)
    for c, t in zip((0, 2, 4, 6), (0, 1, 2, 3)):
        draw(fig.add_subplot(gs[0, c]), t)
    arrow(fig.add_subplot(gs[0, 1]),
          r"each edge $\to$" "\n" r"$b{=}2$ paths of $s{=}2$")
    arrow(fig.add_subplot(gs[0, 3]))
    arrow(fig.add_subplot(gs[0, 5]))
    fig.savefig("fig_construction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_construction.pdf")


if __name__ == "__main__":
    main()
