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
    "doping_points": 20,        # number of points for the sweep

    # Which plots to show
    "show_intrinsic_plot": False,
    "show_overlay_plot":  False, # overlay Ec/Ev for three sample dopings
    "overlay_dopings_cm3": (1e16, 1e18), # doping values to overlay
    "show_delta_vs_doping": True,

    # Band bending
    "use_band_bending": True,   # True, False
    
    # what metrics to plot vs doping in plot 3 (see return in compute_msj_doped_with_bending for options)
    "delta_vs_doping_metrics": [
    "W_nm",
    "regime_class",  
    
    ],
    
    # labeling for delta vs doping (optional)
    "delta_vs_doping_labels": {
    "Delta": "Δ = Φ_S - Φ_M (eV)",
    "IFBL": "IFBL (eV)",
    "phi_Bn_corr": "φ_Bn (corrected, eV)",
    "phi_Bn_ideal": "φ_Bn (ideal, eV)",
    "W_nm": "Depletion width",
    
    },

    # geometry/resolution
    "L_metal_nm": 10.0, # metal side extent (nm)
    "L_sc_nm": 100.0, # semiconductor side extent (nm)
    "npts": 600, # total points (metal + semiconductor)
    
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
    "show_ifbl": True, 
    
    # Unit labels per metric key (defaults to "value" if missing)
    "metric_yaxis": {
        "W_nm": "nm",
        "Wd_nm": "nm",
        "phi_Bn_corr": "eV",
        "phi_Bn_ideal": "eV",
        "Delta": "eV",
        "delta_phi": "eV",
        "Emax_MVcm": "MV/cm",
        "regime_class": "kT/Eoo",   # no unit
        # add more as needed
    },
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
                text=[f"IFBL={IFBL:.2f} eV"],
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
    Plot selected scalar metrics vs doping with multi-y-axis support.
    User chooses metrics in SETTINGS["delta_vs_doping_metrics"] and assigns
    y-axis labels in SETTINGS["metric_yaxis"].
    Metrics with the same label share the same y-axis.
    """
    xN = np.asarray(Ns_cm3, dtype=float)

    # compute once per doping
    cache = [
        compute_msj_doped_with_bending(metal, semi, dop_type, float(N), Lm_nm, Ls_nm, npts)
        for N in xN
    ]

    want      = [str(k).strip() for k in SETTINGS.get("delta_vs_doping_metrics", ["Delta", "delta_phi"])]
    labels    = SETTINGS.get("delta_vs_doping_labels", {})
    axis_map  = SETTINGS.get("metric_yaxis", {})  # metric -> axis label (e.g., "eV", "nm")

    fig = go.Figure()
    any_added = False

    # Collect unique axis labels in order of first use
    axis_labels_ordered: list[str] = []

    # Build and add traces
    for key in want:
        yvals = []
        for d in cache:
            if key in d:
                val = d[key]
            elif key == "W_nm" and ("W_cm" in d):
                val = float(d["W_cm"]) * 1e7  # cm -> nm convenience
            else:
                val = None

            if isinstance(val, (int, float, np.integer, np.floating)):
                yvals.append(float(val))
            else:
                yvals = []
                break  # skip metric if any value missing/non-numeric

        if not yvals:
            continue

        # which y-axis label does this metric use?
        axis_label = axis_map.get(key, "Value")
        if axis_label not in axis_labels_ordered:
            axis_labels_ordered.append(axis_label)

        # assign to y, y2, y3...
        axis_idx = axis_labels_ordered.index(axis_label)
        yaxis_name = "y" if axis_idx == 0 else f"y{axis_idx+1}"

        name = labels.get(key, key) + f" [{axis_label}]"
        fig.add_trace(go.Scatter(
            x=xN, y=np.asarray(yvals, float),
            mode="lines+markers", name=name,
            yaxis=yaxis_name
        ))
        any_added = True

    if not any_added:
        fig.add_trace(go.Scatter(x=xN, y=np.zeros_like(xN),
                                 mode="lines+markers", name="(no valid metrics)"))

    # ---- styling (same as your other plots) ----
    fs = SETTINGS.get("font_sizes", {})
    lw = SETTINGS.get("line_wdith", 2)
    fig.update_traces(line=dict(width=lw))
    fig.update_layout(
        showlegend=SETTINGS.get("show_legend", True),
        title=f"{metal.value} | {semi.value}: Selected metrics vs doping ({dop_type}-type, 300 K)",
        xaxis_type="log",
        xaxis_title="Doping N (cm⁻³)",
        yaxis_title=axis_labels_ordered[0] if axis_labels_ordered else "Value",
        template="plotly_white",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=fs.get("legend", 12))
        ),
        font=dict(size=fs.get("ticks", 12)),
        title_font=dict(size=fs.get("title", 18)),
        xaxis=dict(
            title=dict(text="Doping N (cm⁻³)", font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12))
        ),
        yaxis=dict(  # primary left axis
            title=dict(text=axis_labels_ordered[0] if axis_labels_ordered else "Value",
                       font=dict(size=fs.get("axis", 14))),
            tickfont=dict(size=fs.get("ticks", 12)),
            rangemode="tozero"
        ),
    )

    # Define extra right-side axes for remaining labels
    if len(axis_labels_ordered) > 1:
        # space extra axes slightly to the left of the right edge
        positions = np.linspace(1.0, 0.86, num=len(axis_labels_ordered)-1)
        for i, lab in enumerate(axis_labels_ordered[1:], start=2):
            fig.update_layout({
                f"yaxis{i}": dict(
                    title=dict(text=lab, font=dict(size=fs.get("axis", 14))),
                    tickfont=dict(size=fs.get("ticks", 12)),
                    overlaying="y",
                    side="right",
                    position=float(positions[i-2]),
                    rangemode="tozero"
                )
            })

    # Optional grid toggle
    g = SETTINGS.get("grid", None)
    if isinstance(g, bool):
        fig.update_xaxes(showgrid=g); fig.update_yaxes(showgrid=g)
    elif isinstance(g, dict):
        fig.update_xaxes(
            showgrid=g.get("x", True),
            gridwidth=g.get("width", 1),
            gridcolor=g.get("color", "rgba(0,0,0,0.15)"),
            griddash=g.get("dash", None),
            minor=dict(showgrid=g.get("minor", False))
        )
        fig.update_yaxes(
            showgrid=g.get("y", True),
            gridwidth=g.get("width", 1),
            gridcolor=g.get("color", "rgba(0,0,0,0.15)"),
            griddash=g.get("dash", None),
            minor=dict(showgrid=g.get("minor", False))
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



