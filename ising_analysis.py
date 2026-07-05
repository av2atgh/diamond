#!/usr/bin/env python3
"""Definitive Ising-vs-r_kappa analysis on the (2,2) diamond."""
import math
from mpmath import mp, mpf, tanh, exp, log
mp.dps = 50
from ising_diamond import find_Kc, pole_magnetization, n_t

Kc = find_Kc()
Tc = 1/Kc
print(f"Critical point: K_c = {mp.nstr(Kc,8)},  T_c/J = {mp.nstr(Tc,8)}")
print(f"r_kappa = n_3 = {n_t(3)} nodes (degree-corrected, q>=0.9)\n")

# Susceptibility chi(t) = dm/dh|_0  ~  m(h)/h for tiny h
h = mpf('1e-6')
print("Susceptibility chi(t)=m/h and magnetization m(t) at fixed small field, vs generation:")
print(" Below (K=1.15 Kc), near (K=Kc), above (K=0.85 Kc)")
print(f" {'t':>2} {'n_t':>8} {'chi_below':>14} {'chi_near':>14} {'chi_above':>14}")
rows=[]
for t in range(1,13):
    cb = pole_magnetization(t, mpf('1.15')*Kc, h)/h
    cn = pole_magnetization(t, Kc, h)/h
    ca = pole_magnetization(t, mpf('0.85')*Kc, h)/h
    rows.append((t,cb,cn,ca))
    print(f" {t:2d} {n_t(t):8d} {mp.nstr(cb,6):>14} {mp.nstr(cn,6):>14} {mp.nstr(ca,6):>14}")

print("\nMagnetization m(t) at a FIXED small symmetry-breaking field h=0.001,")
print("just below Tc (K=1.10 Kc) -- watch it switch on across r_kappa (t=3, n=44):")
h2=mpf('0.001')
for t in range(1,10):
    m = pole_magnetization(t, mpf('1.10')*Kc, h2)
    star = "   <-- r_kappa (n=44)" if t==3 else ""
    print(f"   t={t}  n={n_t(t):7d}   m = {mp.nstr(m,6):>12}{star}")

# growth rate of chi below Tc: ratio per generation
print("\nchi growth ratio chi(t+1)/chi(t) below Tc (K=1.15Kc):")
for i in range(len(rows)-1):
    t=rows[i][0]; r=rows[i+1][1]/rows[i][1]
    print(f"   t={t}->{t+1}: {mp.nstr(r,5)}")
