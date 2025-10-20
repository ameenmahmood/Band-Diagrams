import os
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, Mapping, Dict, Any
import pandas as pd
import numpy.typing as npt

# Physical constants used in calculations
@dataclass(frozen=True)
class Constants:
    hbar: float = 1.054571817e-34   # [J*s] (B15)
    h_eV: float = 4.135667696e-15  # [eV*s] (B15)
    h_j: float = 6.62607015e-34      # [J*s] (B15)
    #k: float    = 1.3806503e-23     # [J/K] (B16)
    q: float    = 1.602e-19          # [C] (B17)
    k_eV: float     = 8.617333e-5        # [eV/K] (B16)
    k_j: float     = 1.380649e-23       # [J/K] (B16)
    T: float    = 300.0             # [K] (B18)
    p: float    = 5.95519e-09       # [ohm*cm^2]  (B19 in sheet used later as a prefactor)
    eps0: float = 8.854e-14         # [F/cm] (B21)
    ti: float = 1.0e-7          # interface thickness [cm] (B10)
    Ki: float = 8.0               # interface dielectric constant [unitless] (A11)
    Nss: float = 1.0e12           # [#/(eV*cm^2)]  (B13)
    mass_e: float = 9.10938356e-31  # electron mass [kg]
    Vt = k_j * T / q  # Thermal voltage at temperature T
# Create an instance of Constants for easy access
c = Constants()


# ---- Material / Metal enums and databases ----
# Enum class for different metal types with their symbols
class Metal(Enum): 
    Pt = "Pt"
    Al = "Al" 
    Mg = "Mg"
    Sc = "Sc"
    Y = "Y"
    Be = "Be"
    Ti = "Ti"
    Ni = "Ni"
    ThO2 = "ThO2"
    Thorium = "Th"
    Cs = "Cs"
    Cr = "Cr"
    V = "V"

# Enum class for semiconductor materials
class Material(Enum):
    SI  = "Si"
    ALN = "AlN"
    GAN = "GaN"

# Data structure to hold material properties
@dataclass(frozen=True)
class MaterialProps:
    Eg_eV: float         # bandgap [eV] (B2)
    me_over_m0: float    # electron effective mass [unitless] (B3)
    mp_over_m0: float    # hole effective mass [unitless] (B4)
    Ks: float            # relative permittivity [unitless] (B6)
    chi_eV: float        # electron affinity [eV] (B7)

# Data structure for metal properties
@dataclass(frozen=True)
class MetalProps:
    phi_m_eV: float   # work function [eV]

def props(material: str | Material) -> MaterialProps:
    """Convenience accessor (accepts either Enum or string like 'Si')."""
    key = material if isinstance(material, Material) else Material(material)
    return MATERIAL_DB[key]

# material database
MATERIAL_DB: Mapping[Material, MaterialProps] = {
    # Silicon properties
    Material.SI:  MaterialProps(
        Eg_eV=1.12,
        me_over_m0=1.09,
        mp_over_m0=1.15,
        Ks=11.9,
        chi_eV=4.05
    ),
    # Aluminum Nitride properties
    Material.ALN: MaterialProps(
        Eg_eV=6.015,
        me_over_m0=0.40,
        mp_over_m0=0.30,
        Ks=9.0,
        chi_eV=1.0
    ),
    # Gallium Nitride properties
    Material.GAN: MaterialProps(
        Eg_eV=3.39,
        me_over_m0=0.20,
        mp_over_m0=0.80,
        Ks=10.4,
        chi_eV=4.1
    ),
}

# Database of metal properties (work functions)
METAL_DB = {
    Metal.Pt:      MetalProps(phi_m_eV=5.29),
    Metal.Al:      MetalProps(phi_m_eV=3.74),
    Metal.Mg:      MetalProps(phi_m_eV=3.74),
    Metal.Sc:      MetalProps(phi_m_eV=3.5),
    Metal.Y:       MetalProps(phi_m_eV=3.5),
    Metal.Be:      MetalProps(phi_m_eV=3.37),
    Metal.Ti:      MetalProps(phi_m_eV=4.09),
    Metal.Ni:      MetalProps(phi_m_eV=4.84),
    Metal.ThO2:    MetalProps(phi_m_eV=2.63),   
    Metal.Thorium: MetalProps(phi_m_eV=3.4),   # placeholder
    Metal.Cs:      MetalProps(phi_m_eV=1.89),
    Metal.Cr:      MetalProps(phi_m_eV=4.51),
    Metal.V:       MetalProps(phi_m_eV=4.11),
}


# ---------- helpers ----------
exp_max = 700  # to avoid overflow in exp()
def coth(x: np.ndarray | float) -> np.ndarray | float:
    """
    Numerically stable hyperbolic cotangent implementation.
    Handles both scalar and array inputs safely.
    Args:
        x: Input value or array
    Returns:
        Hyperbolic cotangent of input
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-12
    out[small] = np.inf
    out[~small] = np.cosh(x[~small]) / np.sinh(x[~small])
    return out if out.shape != () else float(out)

# effective densities of states
def effective_dos(mstar_over_m0: float, T: float) -> float:
    """
    Nc or Nv [cm^-3] using:
        N = 2.51e19 * (mT/300)^1.5
    """
    #val_m3 = (2 * ((2* np.pi * c.mass_e * c.k_j * c.T)/(c.h_j)**2)**1.5) * (mstar_over_m0 * T / 300.0) ** 1.5
    val_m3 = 2.51e19 * (mstar_over_m0 * T / 300.0) ** 1.5
    return float(val_m3)   # cm^-3

# material effective densities of states
def material_effective_densities(mat: MaterialProps, T: float) -> tuple[float, float]:
    """Return (Nc, Nv) [cm^-3] for the material at temperature T."""
    Nc = effective_dos(mat.me_over_m0, T)
    Nv = effective_dos(mat.mp_over_m0, T)
    return Nc, Nv

# intrinsic levels bands 
def intrinsic_levels(chi_eV: float, Eg_eV: float, Nc: float, Nv: float) -> Tuple[float, float, float, float]:
    """
    Semiconductor band edges and intrinsic EF relative to vacuum.
    Vacuum is 0 eV. Ec = -chi; Ev = Ec - Eg; Ei above Ev by midgap + ln(Nv/Nc) term.
    """
    Ec = -chi_eV
    Ev = Ec - Eg_eV
    # intrinsic level formula: Ei = (Ec + Ev)/2 + (kT/2q) * ln(Nv/Nc)
    Ei_offset = ((c.k_j * c.T * 0.5) / c.q) * np.log(max(Nv, 1e-300)/max(Nc, 1e-300))
    Ei = (Ec + Ev)/2 + Ei_offset
    phi_S = -Ei
    return Ec, Ev, Ei, phi_S

def _depletion_profile(
    Ks: float,
    N_cm3: float,
    dop_type: str,
    Vbi_eV: float,
    Ls_nm: float,
    npts: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    1-D depletion approximation for a Schottky contact (semiconductor side).
    Solves Poisson in the depletion region to get electrostatic potential psi(x).

    Returns
    -------
    x_s_nm : np.ndarray
        Position grid on the semiconductor side [nm], from 0 to Ls_nm.
    psi_x  : np.ndarray
        Electrostatic potential [V] with psi(0) = 0 at the interface,
        psi(x) increases to Vbi at the depletion edge.
    W_cm   : float
        Depletion width [cm].
    Vbi_eV : float
        Built-in potential [eV] used in the solve (same numeric as volts).
    """
    # units & constants
    eps_s = Ks * c.eps0                 # [F/cm]
    Vbi_V = float(max(Vbi_eV, 0.0))     # treat eV numerically as V
    N = float(max(N_cm3, 1.0))          # avoid division-by-zero

    # depletion width
    W_cm = np.sqrt(2.0 * eps_s * Vbi_V / (c.q * N))

    # grid on semiconductor side
    x_s_nm = np.linspace(0.0, Ls_nm, max(2, npts // 2))
    x_cm = x_s_nm * 1e-7                # nm -> cm

    # piecewise potential
    psi_x = np.empty_like(x_cm)
    inside = x_cm <= W_cm
    
    # Poisson solution in depletion region (0 <= x <= W)
    # psi(x) = (q N / (eps)) * (W x - x^2 / 2)  for 0 <= x <= W
    psi_x[inside] = (c.q * N / (eps_s)) * (W_cm * x_cm[inside] - 0.5 * x_cm[inside]**2)
    # psi(x) = Vbi for x >= W (flat in neutral region)
    psi_x[~inside] = Vbi_V

    return x_s_nm, psi_x, W_cm, Vbi_eV

def image_force_lowering(Ks: float, N_cm3: float, W_cm: float) -> float:
    """
    Image force barrier lowering Δφ [eV].
    Ks : relative permittivity
    N_cm3 : doping concentration [cm^-3]
    W_cm : depletion width [cm]
    """
    eps = Ks * c.eps0 
    # Electric field at inteface  
    Emax = c.q * N_cm3 * W_cm / eps   # [V/cm]
    imageforce_dphi = np.sqrt(c.q * Emax / (4 * np.pi * eps)) # [eV]
    return imageforce_dphi

def intrinsic_ni(Eg_eV: float, Nc: float, Nv: float) -> float:
    """Intrinsic carrier concentration [cm^-3] using Eg (eV), Nc, Nv at the same T."""
    return (Nc * Nv) ** 0.5 * np.exp(-Eg_eV / (2 * c.Vt))

# --- Semiconductor work functions ---
def semiconductor_work_function_doped(
    chi_eV: float, Eg_eV: float, dop_type: str, N_cm3: float,
    Nc: float | None = None, Nv: float | None = None,
    me_over_m0: float | None = None, mp_over_m0: float | None = None
) -> tuple[float, float, float, float]:
    """
    Returns (Phi_S, Ec, Ev, Ei) for a doped semiconductor (vacuum = 0 eV).
    If Nc/Nv are None, they are computed from me_over_m0/mp_over_m0 at temperature c.T.
    The actual work function the junction sees, after the Fermi level moves with doping
    """
    if Nc is None or Nv is None:
        if me_over_m0 is None or mp_over_m0 is None:
            raise ValueError("Need me_over_m0/mp_over_m0 to compute Nc, Nv.")
        Nc = effective_dos(me_over_m0, c.T)
        Nv = effective_dos(mp_over_m0, c.T)

    # Thermal voltage
    Ec, Ev, Ei, _ = intrinsic_levels(chi_eV, Eg_eV, Nc=Nc, Nv=Nv)
    ni = intrinsic_ni(Eg_eV, Nc, Nv)

    if dop_type.lower().startswith("n"):
        # fermi level shift for n-type doping
        dE = c.Vt * np.log(max(N_cm3, 1e-300) / max(ni, 1e-300))      
    else:
        # fermi level shift for p-type doping
        dE = -c.Vt * np.log(max(N_cm3, 1e-300) / max(ni, 1e-300))     

    EF = Ei + dE
    Phi_S = -EF
    return Phi_S, Ec, Ev, Ei


# --- Junction computation functions ---
def compute_msj_doped_with_bending(
    metal: Metal,
    semi: Material,
    dop_type: str,
    N_cm3: float,
    Lm_nm: float,
    Ls_nm: float,
    npts: int,
) -> Dict[str, Any]:
    """
    Metal|Semiconductor junction with doping and depletion band-bending (semiconductor side).
    Ideal Schottky–Mott, non-degenerate, depletion approximation.
    Returns arrays Ec(x), Ev(x), Ei(x) on the semiconductor side.
    """
    mp = METAL_DB[metal]
    sp = props(semi)

    # Metal side (vacuum on metal = 0 eV)
    phi_M = mp.phi_m_eV
    EF_M  = -phi_M
    E0_M  = 0.0

    # Semiconductor bulk levels & work function 
    Nc_S, Nv_S = material_effective_densities(sp, c.T)
    phi_S, Ec_S0, Ev_S0, Ei_S0 = semiconductor_work_function_doped(
        sp.chi_eV, sp.Eg_eV, dop_type, N_cm3, Nc=Nc_S, Nv=Nv_S
    )

    # Global alignment offset
    Delta = phi_S - phi_M
    Ec0 = Ec_S0 + Delta
    Ev0 = Ev_S0 + Delta
    Ei0 = Ei_S0 + Delta
    E0_S = E0_M + Delta

    # Ideal barrier(s)
    phiBn = phi_M - sp.chi_eV                 # n-type electron barrier
    phiBp = sp.Eg_eV - phiBn                  # p-type hole barrier (Eg - phiBn)
 
    # Built-in potential (depletion approx)
    if dop_type.lower().startswith("n"):
        # n type Vbi 
        Vbi_eV = max(0.0, phiBn + c.Vt * np.log(max(N_cm3, 1.0) / max(Nc_S, 1e-300)))
    else:
        # p type Vbi
        Vbi_eV = max(0.0, phiBp + c.Vt * np.log(max(N_cm3, 1.0) / max(Nv_S, 1e-300)))

    # Solve Poisson in depletion region to get psi(x) on semiconductor side
    x_s_nm, psi_x, W_cm, _ = _depletion_profile(sp.Ks, N_cm3, dop_type, Vbi_eV, Ls_nm, npts)
    
    # Image force lowering correction
    IFBL = image_force_lowering(sp.Ks, N_cm3, W_cm)

    # Corrected barrier heights
    phiBn_corr = phiBn - IFBL
    phiBp_corr = sp.Eg_eV - phiBn_corr

    # Bend bands with Ec(x) = Ec0 - psi(x), etc. (psi in V; energies in eV)
    Ec_x = Ec0 - psi_x
    Ev_x = Ev0 - psi_x
    Ei_x = Ei0 - psi_x

    # Metal-side axis (still flat)
    x_m = np.linspace(-Lm_nm, 0.0, max(2, npts // 2))
    W_nm = W_cm * 1e7 # cm -> nm
    
    # contact resistance
    h_bj = c.h_j / (2 * np.pi)  # reduced Planck's constant [eV*s]
    # 10^6 to convert cm^-3 to m^-3, 100 to convert F/cm to F/m
    Eoo =  (c.q *h_bj / 2) * np.sqrt((10**6) / (c.mass_e * c.eps0 * 100)) * np.sqrt(N_cm3 / (sp.me_over_m0 * sp.Ks))  
    regime_class = (c.k_j * c.T) / (Eoo)
    
    return {
        "x_m": x_m, "x_s": x_s_nm,
        "E0_M": E0_M, "EF_M": EF_M,
        "E0_S": E0_S,
        "Ec_S": Ec_x, "Ev_S": Ev_x, "Ei_S": Ei_x,
        "phi_M": phi_M, "phi_S": phi_S, "Delta": Delta,
        "phi_Bn_ideal": phiBn, "phi_Bp_ideal": phiBp,
        "Vbi_eV": Vbi_eV, "W_cm": W_cm,
        "metal": metal, "semi": semi, "dop_type": dop_type, "N_cm3": N_cm3,"phi_Bn_corr": phiBn_corr, "phi_Bp_corr": phiBp_corr, 
        "IFBL": IFBL, "W_nm": W_nm, "Eoo": Eoo, "regime_class": regime_class,
    }

def compute_msj_intrinsic(
    metal: Metal,
    semi: Material,
    Lm_nm: float,
    Ls_nm: float,
    npts: int,
) -> Dict[str, Any]:
    """
    Compute energies/axes for a metal–semiconductor junction (intrinsic semiconductor)
    at equilibrium. Returns data only (no plotting).
    """
    mp = METAL_DB[metal]
    sp = props(semi)

    # Metal work function and Fermi level (vacuum on metal is 0 eV)
    phi_M = mp.phi_m_eV
    EF_M = -phi_M               # eV
    E0_M = 0.0                  # vacuum reference on metal side

    # calculate Nc and Nv automatically for this semiconductor
    Nc_S, Nv_S = material_effective_densities(sp, c.T)

    # use those values in the intrinsic level calculation
    Ec_S, Ev_S, Ei_S, phi_S = intrinsic_levels(sp.chi_eV, sp.Eg_eV, Nc=Nc_S, Nv=Nv_S)

    # Align Fermi levels: Delta = phi_S - phi_M
    Delta = phi_S - phi_M

    # Shift semiconductor energies so semiconductor EF aligns to metal EF
    Ec_S_aligned = Ec_S + Delta
    Ev_S_aligned = Ev_S + Delta
    Ei_S_aligned = Ei_S + Delta
    E0_S = E0_M + Delta

    # Position grids (nm)
    x_m = np.linspace(-Lm_nm, 0.0, npts//2)
    x_s = np.linspace(0.0,  Ls_nm, npts//2)

    return {
        "x_m": x_m, "x_s": x_s,
        "E0_M": E0_M, "EF_M": EF_M,
        "E0_S": E0_S, "Ec_S": Ec_S_aligned, "Ev_S": Ev_S_aligned, "Ei_S": Ei_S_aligned,
        "phi_M": phi_M, "phi_S": phi_S, "Delta": Delta,
        "metal": metal, "semi": semi,
    }

def compute_msj_doped(
    metal: Metal,
    semi: Material,
    dop_type: str, 
    N_cm3: float,
    Lm_nm: float,
    Ls_nm: float,
    npts: int,
) -> Dict[str, Any]:
    mp = METAL_DB[metal]
    sp = props(semi)

    phi_M = mp.phi_m_eV
    EF_M  = -phi_M
    E0_M  = 0.0

    # 🔧 Auto-compute Nc, Nv for this semiconductor at current T
    Nc_S, Nv_S = material_effective_densities(sp, c.T)

    # Use doped work-function with Nc, Nv provided
    phi_S, Ec_S0, Ev_S0, Ei_S0 = semiconductor_work_function_doped(
        sp.chi_eV, sp.Eg_eV, dop_type, N_cm3,
        Nc=Nc_S, Nv=Nv_S
    )
    EF_S0 = -phi_S

    # Align Fermi levels across the junction
    Delta = phi_S - phi_M
    Ec_S = Ec_S0 + Delta
    Ev_S = Ev_S0 + Delta
    Ei_S = Ei_S0 + Delta
    E0_S = E0_M + Delta

    x_m = np.linspace(-Lm_nm, 0.0, npts//2)
    x_s = np.linspace(0.0,  Ls_nm, npts//2)

    phi_Bn_ideal = phi_M - sp.chi_eV

    return {
        "x_m": x_m, "x_s": x_s,
        "E0_M": E0_M, "EF_M": EF_M,
        "E0_S": E0_S, "Ec_S": Ec_S, "Ev_S": Ev_S, "Ei_S": Ei_S,
        "phi_M": phi_M, "phi_S": phi_S, "Delta": Delta, "phi_Bn_ideal": phi_Bn_ideal,
        "metal": metal, "semi": semi, "dop_type": dop_type, "N_cm3": N_cm3
    }


# --- Sweeps / Analysis helpers ---
def sweep_doping(
    Ns_cm3: list[float],
    dop_type: str,
    metal: Metal,
    semi: Material,
    Lm_nm: float,
    Ls_nm: float,
    npts: int,
) -> pd.DataFrame:
    rows = []
    for N in Ns_cm3:
        d = compute_msj_doped(metal, semi, dop_type, N, Lm_nm, Ls_nm, npts)
        rows.append({
            "N_cm3": N,
            "dop_type": dop_type,
            "Phi_S_eV": d["phi_S"],
            "Delta_eV": d["Delta"],
            "EF_global_eV": d["EF_M"],
            "Ec_bulk_eV": d["Ec_S"],
            "Ev_bulk_eV": d["Ev_S"],
        })
    return pd.DataFrame(rows)




