This mini toolkit computes and plots ideal metal–semiconductor (M|S) band diagrams,
including intrinsic junctions and doped cases with depletion-region band bending and
image-force barrier lowering. Energies are referenced to vacuum on the metal side
(Evac,metal = 0 eV); work functions are positive numbers; Fermi levels are negative
relative to vacuum (e.g., EF = –Φ).

Repo layout

band_diagrams.py Core physics: material/metal databases, intrinsic/doped
level calculations, Schottky barriers, depletion solve,
IFBL, and junction computation functions.

bandDiagram_plotting.py Plotly plotting routines and a driver with a SETTINGS dict
for selecting metal/semiconductor, doping, and figure options.

Quick start

Install dependencies (Python 3.10+ recommended):

pip install numpy pandas plotly

Run the plotting script:

python bandDiagram_plotting.py

It reads SETTINGS (metal, semiconductor, dopings, figure toggles) and opens
interactive Plotly windows.

Key concepts & conventions

• Vacuum level reference: Evac (metal) = 0 eV.
EF,metal = –ΦM.
Ec = –χ, Ev = Ec – Eg, Ei = (Ec + Ev)/2 + (kT/2)*ln(Nv/Nc).

• Alignment offset: Semiconductor is rigidly shifted by
Δ = ΦS – ΦM so that EF,semi = EF,metal.

• Schottky–Mott barriers:
φBn = ΦM – χ
φBp = Eg – φBn

• Band bending: Ec,v,i(x) = Ec,v,i0 – ψ(x),
where ψ(x) is the depletion potential.

• Image-force barrier lowering (IFBL):
Δφ = sqrt(qE / (4πϵ)), applied as φBn_corr = φBn – Δφ.

What you can plot

• Intrinsic M|S band diagram (flat semiconductor bands, global EF and vacuum refs)
via compute_msj_intrinsic().

• Overlay Ec/Ev for multiple dopings (n or p type) with optional band bending and IFBL
annotations.

• Δ, depletion width W, corrected barrier, and regime_class (kT/E00) vs doping.

API (core functions)

intrinsic_levels(chi_eV, Eg_eV, Nc, Nv)
Returns Ec, Ev, Ei, and intrinsic work function ΦS = –Ei.

semiconductor_work_function_doped(chi_eV, Eg_eV, dop_type, N_cm3, ...)
Applies EF shift with doping:
EF – Ei = ±kT ln(N/n_i)
and returns ΦS = –EF.

compute_msj_intrinsic(metal, semi, Lm_nm, Ls_nm, npts)
Builds intrinsic diagram data with Δ = ΦS – ΦM alignment.

compute_msj_doped(metal, semi, dop_type, N_cm3, Lm_nm, Ls_nm, npts)
Doped case without spatial bending (flat bands).

compute_msj_doped_with_bending(metal, semi, dop_type, N_cm3, Lm_nm, Ls_nm, npts)
Doped case with depletion solution and IFBL; returns Ec(x), Ev(x), Ei(x),
W, Vbi, φBn(ideal/corr), etc.

Configure plots

Edit SETTINGS in bandDiagram_plotting.py:

SETTINGS = {
"metal": "Ti", # pick from Metal enum
"semiconductor": "AlN", # "Si", "AlN", or "GaN"
"dop_type": "n", # "n" or "p"
"overlay_dopings_cm3": (1e16, 1e18),
"use_band_bending": True,
"show_intrinsic_plot": True,
"show_overlay_plot": True,
"show_delta_vs_doping": True,
}

Built-in materials and metals

Semiconductors:
Si, AlN, GaN (Eg, me*/m0, mp*/m0, Ks, χ)

Metals:
Pt, Al, Mg, Sc, Y, Be, Ti, Ni, ThO2, Th, Cs, Cr, V (ΦM)

Assumptions and limits

• Ideal Schottky–Mott interface (no Fermi-level pinning)
φBn = ΦM – χ

• Non-degenerate statistics (Maxwell–Boltzmann)
for the doping-induced Fermi shift. Very heavy doping
requires Fermi–Dirac and bandgap-narrowing models.

• 1-D depletion approximation on the semiconductor side.

Troubleshooting

• Bands shift the “wrong” way:
Check Δ = ΦS – ΦM sign; the code adds +Δ to semiconductor energies.

• Flat vs bent bands:
Toggle use_band_bending between False (flat) and True (depletion).

• Constants (kT/q, eps0, etc.) live in the Constants dataclass
near the top of band_diagrams.py.

Core formulas implemented

• Intrinsic energy levels & effective DOS (Nc, Nv)
• Schottky–Mott barrier heights φBn, φBp
• Image-force barrier lowering Δφ
• Depletion potential and width in 1-D Poisson solution

Summary

Choose a metal/semiconductor pair, sweep doping,
and visualize how work function, electron affinity,
and depletion physics control injection barriers and
band bending. This toolkit links material parameters
to contact behavior—ideal for Schottky and ohmic contact analysis.
