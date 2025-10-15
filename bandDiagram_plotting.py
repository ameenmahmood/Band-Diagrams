# plot_msj_intrinsic.py
import numpy as np
import plotly.graph_objects as go

# Import required enums and constants from the band_diagrams module
from band_diagrams import (
    Metal, Material,
    compute_msj_intrinsic,
    compute_msj_doped,
    compute_msj_doped_with_bending,   
    sweep_doping,)

# ===== USER SETTINGS =====
SETTINGS = {
    # Metal and emiconductor selection
    "metal": "Ti",              # "Al","Pt","Mg","Sc","Y","Be","Ti","Ni","ThO2","Th","Cs","Cr","V"
    "semiconductor": "AlN",      # "Si","AlN","GaN"

    # Doping controls
    "dop_type": "n",            # "n" or "p"
    "doping_scale": "log",      # "log" or "lin"
    "doping_min_cm3": 1e13,     # range start
    "doping_max_cm3": 1e20,     # range end
    "doping_points": 13,        # number of points for the sweep

    # Which plots to show
    "show_intrinsic_plot": False,
    "show_overlay_plot":  True, # overlay Ec/Ev for three sample dopings
    "overlay_dopings_cm3": (1e16, 1e18), # doping values to overlay
    "show_delta_vs_doping": False,

    # Band bending
    "use_band_bending": True,   # True, False
    
    # what metrics to plot vs doping in plot 3 (see return in compute_msj_doped_with_bending for options)
    "delta_vs_doping_metrics": [
    "Delta",               
    "phi_S",
    "W_nm",
    "regime_class",  
    
    ],
    
    # labeling for delta vs doping (optional)
    "delta_vs_doping_labels": {
    "Delta": "Δ = Φ_S - Φ_M (eV)",
    "IFBL": "IFBL (eV)",
    "phi_Bn_corr": "φ_Bn (corrected, eV)",
    "phi_Bn_ideal": "φ_Bn (ideal, eV)",
    "W_nm": "Depletion width (nm)",
    
    },

    # geometry/resolution
    "L_metal_nm": 10.0, # metal side extent (nm)
    "L_sc_nm": 100.0, # semiconductor side extent (nm)
    "npts": 400, # total points (metal + semiconductor)
    
    # show legend 
    "show_legend": True,
    
    # font size
    "font_sizes": {
        "title":20,
        "axis": 16,
        "ticks": 12,
        "legend": 12,
        "annotation": 12
    },
    
    # line width 
    "line_wdith": 3,
    
    # show IFBL 
    "show_ifbl": False,
} 

# --- Helper functions ---
def _resolve_metal(name: str) -> Metal:
    """
    Map a user string to a Metal enum member.

    Accepts exact enum keys (e.g., "Al") or case-insensitive value matches.
    Raises:
        ValueError if the name does not match a known metal.
    """
    try:
        return Metal[name] 
    except KeyError:
        for m in Metal:
            if m.value.lower() == name.lower():
                return m
    raise ValueError(f"Unknown metal '{name}'. Valid: {[m.value for m in Metal]}")

def _resolve_material(name: str) -> Material:
    """
    Map a user string to a Material enum member.

    Accepts exact enum keys (e.g., "SI", "ALN", "GAN") or case-insensitive value matches.
    Raises:
        ValueError if the name does not match a known semiconductor.
    """
    try:
        return Material[name.upper()]
    except KeyError:
        for s in Material:
            if s.value.lower() == name.lower():
                return s
    raise ValueError(f"Unknown semiconductor '{name}'. Valid: {[s.value for s in Material]}")

def _make_doping_grid(cfg: dict) -> np.ndarray:
    """
    Build a 1D array of doping values from SETTINGS.

    Returns:
        np.ndarray of length cfg["doping_points"] on either a log or linear scale.
    """
    nmin, nmax, npts = cfg["doping_min_cm3"], cfg["doping_max_cm3"], cfg["doping_points"]
    if cfg["doping_scale"].lower().startswith("log"):
        return np.logspace(np.log10(nmin), np.log10(nmax), int(npts))
    else:
        return np.linspace(nmin, nmax, int(npts))

# Allows for a movable legend
SHOW_CONFIG = {
    "editable": True,
    "edits": {
        "legendPosition": True,
        "annotationPosition": False,
        "shapePosition": False,
        "axisTitleText": False,
        "titleText": False,
        "legendText": False
    }
}


# --- Figure Functions ---
# ---------- INSTRINSIC M/S JUNCTION PLOT ----------
def make_msj_band_diagram_intrinsic(
    metal: Metal,
    semi: Material,
    Lm_nm: float,
    Ls_nm: float,
    npts: int,
) -> go.Figure:
    """
    Intrinsic metal|semiconductor band diagram at equilibrium.

    Parameters
    ----------
    metal : Metal
        Contact metal selection.
    semi : Material
        Semiconductor selection.
    Lm_nm, Ls_nm : float
        Extents of the metal and semiconductor regions (nm) used for plotting.
    npts : int
        Total sample count (metal half + semiconductor half).

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with vacuum and Fermi on the metal side, and Ec/Ev/Ei on the semiconductor side.
        All energies are referenced to vacuum = 0 on the metal side.
    """

    # Compute the band structure data for the intrinsic semiconductor
    data = compute_msj_intrinsic(metal, semi, Lm_nm, Ls_nm, npts)

    # Extract relevant data from the computed results
    x_m = data["x_m"]; x_s = data["x_s"]
    E0_M = data["E0_M"]; EF_M = data["EF_M"]
    E0_S = data["E0_S"]; Ec = data["Ec_S"]; Ev = data["Ev_S"]; Ei = data["Ei_S"]
    phi_M = data["phi_M"]; phi_S = data["phi_S"]; Delta = data["Delta"]
    phi_Bn_ideal = data["phi_Bn_ideal"]
    metal = data["metal"]; semi = data["semi"]

    # Create a new figure for the band diagram
    fig = go.Figure()

    # Add traces for the semiconductor side of the band diagram
    fig.add_trace(go.Scatter(x=x_m, y=np.full_like(x_m, E0_M),
                             mode="lines", name="Vacuum (metal)", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=x_m, y=np.full_like(x_m, EF_M),
                             mode="lines", name="Fermi (metal)"))

    # Semiconductor side
    fig.add_trace(go.Scatter(x=x_s, y=np.full_like(x_s, E0_S),
                             mode="lines", name="Vacuum (semi)", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=x_s, y=np.full_like(x_s, Ec),
                             mode="lines", name="Ec")) # Conduction band
    fig.add_trace(go.Scatter(x=x_s, y=np.full_like(x_s, Ev),
                             mode="lines", name="Ev")) # Valence band
    fig.add_trace(go.Scatter(x=x_s, y=np.full_like(x_s, Ei),
                             mode="lines", name="Ei", line=dict(dash="dot")))

    # Add a global Fermi level line across the diagram
    #fig.add_trace(go.Scatter(x=[x_m[0], x_s[-1]], y=[EF_M, EF_M],
    #                        mode="lines", name="Fermi (global)", line=dict(width=3)))

    # Add a vertical line at x=0 for reference
    fig.add_vline(x=0.0, line_width=1, line_dash="dot", line_color="gray")

    # Update the layout of the figure with titles and annotations
    fs = SETTINGS.get("font_sizes", {})
    lw = SETTINGS.get("line_wdith", 2)
    showlegend = SETTINGS.get("show_legend", True)
    fig.update_traces(line=dict(width=lw))
    fig.update_layout(
        showlegend = SETTINGS.get("show_legend", True),
        title=f"Band Diagram: {metal.value} | {semi.value} (Intrinsic, Equilibrium)",
        xaxis_title="Position x (nm)",  # kept for readability; font set below
        yaxis_title="Energy (eV, vacuum = 0 on metal side)",  # kept for readability; font set below
        template="plotly_white",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=fs.get("legend", 12))
        ),
        # global/base fonts
        font=dict(size=fs.get("ticks", 12)),
        title_font=dict(size=fs.get("title", 18)),
        # axis fonts (titles + ticks)
        xaxis=dict(
            title=dict(text="Position x (nm)", font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12))
        ),
        yaxis=dict(
            title=dict(text="Energy (eV, vacuum = 0 on metal side)", font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12))
        ),
        annotations=[
            dict(
                x=0, y=-.12, xref="paper", yref="paper",
                text=(f"Phi_M = {phi_M:.2f} eV, "
                    f"Phi_S ≈ {phi_S:.2f} eV, "
                    f"Delta = Phi_S - Phi_M = {Delta:.2f} eV<br>"
                    f"Phi_Bn (ideal) = Phi_M - chi = {phi_Bn_ideal:.2f} eV"),
                showarrow=False, font=dict(size=fs.get("anno", 12)), align="left"
            )
        ]
    )
    return fig

# ---------- DOPING OVERLAY PLOTS ----------
def overlay_band_diagrams_by_doping(
    Ns_cm3: tuple | list,
    dop_type: str,
    metal: Metal,
    semi: Material,
    Lm_nm: float,
    Ls_nm: float,
    npts: int,
    use_bending: bool,           # ← no default here
) -> go.Figure:
    fig = go.Figure()
    dashes = ["solid", "dot", "dash", "dashdot", "longdash"]

    # --- first trace just to draw the global Fermi line
    if use_bending:
        first = compute_msj_doped_with_bending(
            metal, semi, dop_type, float(Ns_cm3[0]), Lm_nm, Ls_nm, npts
        )
    else:
        first = compute_msj_doped(
            metal, semi, dop_type, float(Ns_cm3[0]), Lm_nm, Ls_nm, npts
        )

    fig.add_trace(go.Scatter(
        x=[-Lm_nm, Ls_nm], y=[first["EF_M"], first["EF_M"]],
        mode="lines", name="Fermi (global)", line=dict(width=3)
    ))
    fig.add_vline(x=0.0, line_width=1, line_dash="dot", line_color="gray")

    # --- overlay Ec/Ev for each doping value
    for i, N in enumerate(Ns_cm3):
        dash = dashes[i % len(dashes)]
        if use_bending:
            d = compute_msj_doped_with_bending(
                metal, semi, dop_type, float(N), Lm_nm, Ls_nm, npts
            )
            x_s = d["x_s"]
            y_ec = d["Ec_S"]
            y_ev = d["Ev_S"]
            
            # Grab the barrier corrections (image lowering)  
            phiBn_corr = d.get("phi_Bn_corr")
            IFBL  = d.get("IFBL")

            # Optional: annotate the interface with Δφ
            if SETTINGS.get("show_ifbl", True) and (IFBL is not None):
                fig.add_trace(go.Scatter(
                x=[x_s[0]], y=[y_ec[0]], mode="markers+text",
                text=[f"Δφ_B={IFBL:.2f} eV"],
                textposition="top center",
                name=f"Barrier lowering (N={float(N):.1e})",
                marker=dict(symbol="circle", size=6, color="black"),
                showlegend=SETTINGS.get("show_legend", True)
            ))

        else:
            d = compute_msj_doped(
                metal, semi, dop_type, float(N), Lm_nm, Ls_nm, npts
            )
            x_s = d["x_s"]
            y_ec = np.full_like(x_s, d["Ec_S"])
            y_ev = np.full_like(x_s, d["Ev_S"])

        fig.add_trace(go.Scatter(
            x=x_s, y=y_ec, mode="lines",
            name=f"Ec ({dop_type}, N={float(N):.1e} cm⁻³)", line=dict(dash=dash)
        ))
        fig.add_trace(go.Scatter(
            x=x_s, y=y_ev, mode="lines",
            name=f"Ev ({dop_type}, N={float(N):.1e} cm⁻³)", line=dict(dash=dash)
        ))

    # Update the layout of the figure with titles and annotations
    fs = SETTINGS.get("font_sizes", {})
    lw = SETTINGS.get("line_wdith", 2)
    showlegend = SETTINGS.get("show_legend", True)
    fig.update_traces(line=dict(width=lw))
    fig.update_layout(
        showlegend = SETTINGS.get("show_legend", True),
        title=f"{metal.value} | {semi.value}: Band edges vs doping ({dop_type}-type, 300 K)",
        xaxis_title="Position x (nm)",
        yaxis_title="Energy (eV, vacuum = 0 on metal side)",
        template="plotly_white",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=fs.get("legend", 12))
        ),
        # global/base fonts
        font=dict(size=fs.get("ticks", 12)),
        title_font=dict(size=fs.get("title", 18)),
        # axis fonts (titles + ticks)
        xaxis=dict(
            title=dict(text="Position x (nm)", font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12))
        ),
        yaxis=dict(
            title=dict(text="Energy (eV, vacuum = 0 on metal side)", font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12))
        ),
    )
    return fig

# ---------- SUMMARY CURVE: Δ vs DOPING ----------
def plot_delta_vs_doping(
    Ns_cm3: np.ndarray,
    dop_type: str,
    metal: Metal,
    semi: Material,
    Lm_nm: float,
    Ls_nm: float,
    npts: int,
) -> go.Figure:
    """
    User-controlled curves vs doping N.

    The list of metrics to plot is taken from SETTINGS["delta_vs_doping_metrics"].
    For each metric m:
      1) If m is a column in the DataFrame returned by sweep_doping(...),
         we use that column directly.
      2) Otherwise we compute it pointwise from compute_msj_doped_with_bending(...).
    """
    # 1) Run the sweep once to get dataframe-based metrics
    df = sweep_doping(
        Ns_cm3.tolist(), dop_type, metal, semi,
        Lm_nm=Lm_nm, Ls_nm=Ls_nm, npts=npts
    )

    # Ensure numeric x for plotting (and preserve user’s x scale choice outside)
    xN = np.array(df["N_cm3"], dtype=float)

    # 2) Prepare figure
    fig = go.Figure()
    want = SETTINGS.get("delta_vs_doping_metrics", ["Delta_eV", "delta_phi"])
    labels = SETTINGS.get("delta_vs_doping_labels", {})

    # Helper: try dataframe first, else compute pointwise
    def _series_for_metric(metric: str) -> tuple[np.ndarray | None, str]:
        # Case A: metric found directly in df
        if metric in df.columns:
            y = np.array(df[metric], dtype=float)
            return y, labels.get(metric, metric)

        # Case B: compute per-doping using the bending-aware solver
        values = []
        ok = True
        for N in xN:
            try:
                d = compute_msj_doped_with_bending(
                    metal, semi, dop_type, float(N), Lm_nm, Ls_nm, npts
                )
                if metric in d:
                    values.append(float(d[metric]))
                elif metric == "delta_phi" and "delta_phi" in d:
                    values.append(float(d["delta_phi"]))
                else:
                    ok = False
                    break
            except Exception:
                ok = False
                break

        if ok and values:
            return np.array(values, dtype=float), labels.get(metric, metric)

        # Not available—silently skip
        return None, labels.get(metric, metric)

    # 3) Add one trace per requested metric (skip those that arent provide)
    added_any = False
    for m in want:
        y, name = _series_for_metric(m)
        if y is not None:
            fig.add_trace(go.Scatter(
                x=xN, y=y, mode="lines+markers", name=name
            ))
            added_any = True

    # 4) Always add Δφ if user asked for it but it wasn't in df and not computed
    # (handled by _series_for_metric already, so nothing extra needed)

    # 5) Layout
    if not added_any:
        # Fallback: add Δ as at least one series if everything else failed
        if "Delta_eV" in df.columns:
            fig.add_trace(go.Scatter(
                x=xN, y=df["Delta_eV"], mode="lines+markers",
                name=labels.get("Delta_eV", "Δ = Φ_S − Φ_M (eV)")
            ))

    # Update the layout of the figure with titles and annotations
    fs = SETTINGS.get("font_sizes", {})
    lw = SETTINGS.get("line_wdith", 2)
    showlegend = SETTINGS.get("show_legend", True)
    fig.update_traces(line=dict(width=lw))
    fig.update_layout(
        showlegend = SETTINGS.get("show_legend", True),
        title=(f"{metal.value} | {semi.value}: Selected metrics vs doping "
               f"({dop_type}-type, 300 K)"),
        xaxis_type="log",
        xaxis_title="Doping N (cm⁻³)",
        yaxis_title="Value",
        template="plotly_white",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=fs.get("legend", 12))
        ),
        # global/base fonts
        font=dict(size=fs.get("ticks", 12)),
        title_font=dict(size=fs.get("title", 18)),
        # axis fonts (titles + ticks)
        xaxis=dict(
            title=dict(text="Position x (nm)", font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12))
        ),
        yaxis=dict(
            title=dict(text="Energy (eV, vacuum = 0 on metal side)", font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12))
        ),
    )
    return fig


# ===== MAIN SCRIPT =====
if __name__ == "__main__":
    metal = _resolve_metal(SETTINGS["metal"])
    semi  = _resolve_material(SETTINGS["semiconductor"])
    dop_type = SETTINGS["dop_type"].lower()
    Lm = SETTINGS["L_metal_nm"]
    Ls = SETTINGS["L_sc_nm"]
    npts = SETTINGS["npts"]
    use_bending = SETTINGS["use_band_bending"]

    if SETTINGS["show_intrinsic_plot"]:
        fig = make_msj_band_diagram_intrinsic(metal, semi, Lm, Ls, npts)
        fig.show(config=SHOW_CONFIG)          

    if SETTINGS["show_overlay_plot"]:
        fig_overlay = overlay_band_diagrams_by_doping(
        Ns_cm3=SETTINGS["overlay_dopings_cm3"],
        dop_type=dop_type, metal=metal, semi=semi,
        Lm_nm=Lm, Ls_nm=Ls, npts=npts,
        use_bending=use_bending
    )
        fig_overlay.show(config=SHOW_CONFIG)
 

    if SETTINGS["show_delta_vs_doping"]:
        Ns = _make_doping_grid(SETTINGS)
        fig_delta = plot_delta_vs_doping(
            Ns_cm3=Ns, dop_type=dop_type, metal=metal, semi=semi,
            Lm_nm=Lm, Ls_nm=Ls, npts=npts
        )
        fig_delta.show(config=SHOW_CONFIG)     



