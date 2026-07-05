# Build the figures and compile the manuscript (Physica A).
# Requires: python3 (numpy, networkx, mpmath, matplotlib) and TeX Live with
# Elsevier elsarticle (Debian/Ubuntu package texlive-publishers).

TEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all figures pdf verify clean

all: pdf

# ----------------------------------------------------------------- figures
# Six figures enter the manuscript:
#   Fig. 1  fig_construction.pdf  (make_construction_figure.py -> diamond_rg.py)
#   Fig. 2  fig_lattice.pdf       Fig. 3  fig_data.pdf   (make_figures.py)
#   Fig. 4  fig_rb.pdf        (make_rb_figure.py    -> rb_fast.py, rb_exact_gf.py)
#   Fig. 5  fig_hier.pdf      (make_hier_figure.py  -> rb_potts_hier.py)
#   Fig. 6  fig_potts.pdf     (make_potts_figure.py -> rb_potts_decimation.py)
figures: fig_construction.pdf fig_lattice.pdf fig_data.pdf fig_rb.pdf fig_hier.pdf fig_potts.pdf

fig_construction.pdf: make_construction_figure.py diamond_rg.py
	python3 make_construction_figure.py

# make_figures.py also emits the standalone fig_evidence.pdf / fig_flow.pdf,
# which are not used by the manuscript.
fig_lattice.pdf fig_data.pdf: make_figures.py diamond_rg.py
	python3 make_figures.py

fig_rb.pdf: make_rb_figure.py rb_fast.py rb_exact_gf.py
	python3 make_rb_figure.py

fig_hier.pdf: make_hier_figure.py rb_potts_hier.py
	python3 make_hier_figure.py

fig_potts.pdf: make_potts_figure.py rb_potts_decimation.py
	python3 make_potts_figure.py

# ----------------------------------------------------------------- manuscript
pdf: manuscript.pdf

manuscript.pdf: manuscript.tex refs.bib figures
	$(TEX) manuscript.tex
	bibtex manuscript
	$(TEX) manuscript.tex
	$(TEX) manuscript.tex

# ----------------------------------------------------------------- verify
# Reproduce and check every table and closed form against direct construction.
#   diamond_rg.py  Tables I-III (counts, evidence, r_kappa) for the (2,2) diamond
#   general_bs.py  the general-(b,s) bundle lemma, RG map, density ln b
#   check_rk_bs.py the r_kappa(b,s;q) crossing formula
#   probe_bs.py    seam eigenvalue = b across the (b,s) family
verify:
	python3 diamond_rg.py
	python3 general_bs.py
	python3 check_rk_bs.py
	python3 probe_bs.py

# ----------------------------------------------------------------- clean
clean:
	rm -f *.aux *.log *.out *.blg *.bbl manuscript.pdf \
	      fig_construction.pdf fig_lattice.pdf fig_data.pdf \
	      fig_evidence.pdf fig_flow.pdf \
	      fig_rb.pdf fig_hier.pdf fig_potts.pdf
