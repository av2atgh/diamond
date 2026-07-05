#!/usr/bin/env python3
"""Generate the three manuscript figures (vector PDF, PRE single-column width)."""
import numpy as np, networkx as nx, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diamond_rg import diamond, closed, logR_dc_cf, logR_plain_cf, densities, Nof

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "axes.linewidth": 0.7,
    "lines.linewidth": 1.3, "figure.dpi": 150,
})
COL = 3.375  # PRE single-column width (inches)
DC, PL, SEAM, DARK = "#2a9d8f", "#d1495b", "#e9c46a", "#264653"


# ---------------- Figure 1: lattice + bundle cut ----------------
def fig_lattice():
    G, poles, part = diamond(3)
    A, B = 0, 1
    pos = nx.spring_layout(G, k=0.30, iterations=700, seed=3,
                           pos={A: np.array([-1, 0.]), B: np.array([1, 0.])}, fixed=[A, B])
    fig, ax = plt.subplots(figsize=(COL, COL * 0.86))
    within = [(u, v) for u, v in G.edges() if part[u] == part[v]]
    seam = [(u, v) for u, v in G.edges() if part[u] != part[v]]
    nx.draw_networkx_edges(G, pos, edgelist=within, edge_color="#c9ccd1", width=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=seam, edge_color=SEAM, width=2.0, ax=ax)
    col = [PL if part[n] == 0 else DC for n in G]
    sz = [230 if n in poles else 42 for n in G]
    nx.draw_networkx_nodes(G, pos, node_color=col, node_size=sz,
                           edgecolors="white", linewidths=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={A: "A", B: "B"},
                            font_size=9, font_color="white", font_weight="bold", ax=ax)
    ax.set_axis_off()
    ax.margins(0.08)
    fig.tight_layout(pad=0.2)
    fig.savefig("fig_lattice.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------- Figure 2: evidence vs generation ----------------
def fig_evidence():
    ts = list(range(1, 10))
    dc = [logR_dc_cf(t, 1.0) for t in ts]
    pl = [logR_plain_cf(t, 1.0) for t in ts]
    fig, ax = plt.subplots(figsize=(COL, COL * 0.82))
    ax.axhline(0, color="#999", lw=0.7)
    thr = math.log(0.99 / 0.01)
    ax.axhline(thr, color="#bbb", ls=":", lw=0.8)
    ax.text(9, thr * 1.7, r"$q=0.99$", ha="right", va="bottom", fontsize=6.8, color="#777")
    ax.plot(ts, dc, "-o", color=DC, ms=3.4, label="degree corrected")
    ax.plot(ts, pl, "-s", color=PL, ms=3.4, label="plain")
    ax.axvline(3, color=DC, ls="--", lw=0.8, alpha=.55)
    ax.axvline(5, color=PL, ls="--", lw=0.8, alpha=.55)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"generation $t$")
    ax.set_ylabel(r"$\log R$")
    ax.set_xticks(ts)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)
    ax.text(3.05, -8, r"$r_\kappa=n_3=44$", color=DC, fontsize=6.6, rotation=90, va="center")
    ax.text(5.05, -8, r"$r_\kappa=n_5=684$", color=PL, fontsize=6.6, rotation=90, va="center")
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_evidence.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------- Figure 3: RG flow to the community fixed point ----------------
def fig_flow():
    ts = list(range(1, 9))
    uin = [densities(t)[1] for t in ts]
    ucr = [densities(t)[2] for t in ts]
    fig, ax = plt.subplots(figsize=(COL, COL * 0.82))
    ax.plot(uin, ucr, "-o", color=DARK, ms=4.2, zorder=3)
    for i, t in enumerate(ts):
        dx, dy = (4, 4) if t != 1 else (6, -2)
        ax.annotate(f"$t={t}$", (uin[i], ucr[i]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.4, color=DARK)
    ax.plot([2], [0], "*", ms=15, color=SEAM, markeredgecolor="#b8860b", mew=0.8, zorder=4)
    ax.annotate(r"$(K,0)=(2,0)$", (2, 0), textcoords="offset points",
                xytext=(-70, 20), fontsize=7.2, color="#b8860b",
                arrowprops=dict(arrowstyle="->", color="#b8860b", lw=0.8))
    ax.set_xlabel(r"within-block density $u_{\rm in}=e_{rr}/\Omega_{rr}\ \to\ K$")
    ax.set_ylabel(r"cross density $u_{\times}=e_{12}/\Omega_{12}\ \to\ 0$")
    ax.set_xlim(-0.15, 2.3)
    ax.set_ylim(-0.12, 1.5)
    ax.grid(alpha=0.25, lw=0.5)
    # inset: seam eigenvalue gap, u_cross ~ (1/s)^t
    axin = fig.add_axes([0.60, 0.55, 0.34, 0.34])
    axin.semilogy(ts, ucr, "o-", color=SEAM, ms=2.6, mec="#b8860b", lw=1.0)
    axin.semilogy(ts, [ucr[0] * 0.5 ** (t - 1) for t in ts], "--", color="#999", lw=0.8)
    axin.set_title(r"$u_{\times}\sim(1/s)^{t}$", fontsize=6.6, pad=2)
    axin.tick_params(labelsize=5.6, length=2)
    axin.set_xlabel(r"$t$", fontsize=6, labelpad=1)
    fig.tight_layout(pad=0.3)
    fig.savefig("fig_flow.pdf", bbox_inches="tight")
    plt.close(fig)


# -------- Combined double-column data figure (evidence | flow) for PRL --------
def fig_data():
    ts = list(range(1, 10))
    dc = [logR_dc_cf(t, 1.0) for t in ts]
    pl = [logR_plain_cf(t, 1.0) for t in ts]
    uin = [densities(t)[1] for t in ts]
    ucr = [densities(t)[2] for t in ts]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.55))

    # (a) evidence vs generation
    ax1.axhline(0, color="#999", lw=0.7)
    thr = math.log(0.99 / 0.01)
    ax1.axhline(thr, color="#bbb", ls=":", lw=0.8)
    ax1.text(9, thr * 1.7, r"$q=0.99$", ha="right", va="bottom", fontsize=6.6, color="#777")
    ax1.plot(ts, dc, "-o", color=DC, ms=3.2, label="degree corrected")
    ax1.plot(ts, pl, "-s", color=PL, ms=3.2, label="plain")
    ax1.axvline(3, color=DC, ls="--", lw=0.8, alpha=.55)
    ax1.axvline(5, color=PL, ls="--", lw=0.8, alpha=.55)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_xlabel(r"generation $t$"); ax1.set_ylabel(r"$\log R$")
    ax1.set_xticks(ts)
    ax1.legend(loc="upper left", frameon=False, handlelength=1.5, fontsize=6.8)
    ax1.text(3.05, -8, r"$r_\kappa=44$", color=DC, fontsize=6.2, rotation=90, va="center")
    ax1.text(5.05, -8, r"$r_\kappa=684$", color=PL, fontsize=6.2, rotation=90, va="center")
    ax1.set_title("(a)", loc="left", fontsize=8, fontweight="bold")

    # (b) RG flow
    ax2.plot(uin, ucr, "-o", color=DARK, ms=3.6, zorder=3)
    for i, t in enumerate(ts):
        dx, dy = (4, 3) if t != 1 else (5, -3)
        ax2.annotate(f"$t={t}$", (uin[i], ucr[i]), textcoords="offset points",
                     xytext=(dx, dy), fontsize=5.8, color=DARK)
    ax2.plot([2], [0], "*", ms=13, color=SEAM, markeredgecolor="#b8860b", mew=0.8, zorder=4)
    ax2.annotate(r"$(K,0)$", (2, 0), textcoords="offset points",
                 xytext=(-42, 16), fontsize=6.8, color="#b8860b",
                 arrowprops=dict(arrowstyle="->", color="#b8860b", lw=0.8))
    ax2.set_xlabel(r"$u_{\rm in}=e_{rr}/\Omega_{rr}\ \to\ K$")
    ax2.set_ylabel(r"$u_{\times}=e_{12}/\Omega_{12}\ \to\ 0$")
    ax2.set_xlim(-0.15, 2.35); ax2.set_ylim(-0.12, 1.5)
    ax2.grid(alpha=0.25, lw=0.5)
    axin = ax2.inset_axes([0.58, 0.55, 0.38, 0.40])
    axin.semilogy(ts, ucr, "o-", color=SEAM, ms=2.2, mec="#b8860b", lw=0.9)
    axin.semilogy(ts, [ucr[0] * 0.5 ** (t - 1) for t in ts], "--", color="#999", lw=0.7)
    axin.set_title(r"$u_{\times}\sim(1/s)^{t}$", fontsize=6, pad=1.5)
    axin.tick_params(labelsize=5, length=2); axin.set_xlabel(r"$t$", fontsize=5.5, labelpad=0.5)
    ax2.set_title("(b)", loc="left", fontsize=8, fontweight="bold")

    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig.savefig("fig_data.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_lattice(); fig_evidence(); fig_flow(); fig_data()
    print("wrote fig_lattice.pdf, fig_evidence.pdf, fig_flow.pdf, fig_data.pdf")
