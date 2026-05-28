"""
Arnold Weak Resonance Ornament Explorer

Single-file Streamlit app for exploring phase portraits inspired by:
Berezovskaya & Karev, "Arnold's Weak Resonance Equation as the Model of Greek Ornamental Design."

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import ast
import io
import json
import math
import time
from dataclasses import dataclass, asdict, field
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Preset:
    id: int
    fig: str
    label: str
    n: int
    eps1: float
    eps2: float
    A2: List[float]
    B: float
    Rmax: float
    stopR: float
    notes: List[str] = field(default_factory=list)

    def parameter_dict(self) -> dict:
        return {
            "n": int(self.n),
            "eps1": float(self.eps1),
            "eps2": float(self.eps2),
            "A2": [float(x) for x in self.A2],
            "B": float(self.B),
            "Rmax": float(self.Rmax),
            "stopR": float(self.stopR),
        }


PRESETS: List[Preset] = [
    Preset(1, "Fig 1.1", "n=5 classic: flower ring + 5-star + spider-net", 5, 0, 1, [-1], 0.5, 3, 3.6,
           ["Introductory paper figure.", "Moderate B gives ring and star; eps1=0 is Hamiltonian."]),
    Preset(2, "A2.1-1", "n=4 centroid only", 4, 0, 1, [1], 0.7, 3, 3.6,
           ["A2(1)>0 blocks peripheral-ring roots.", "Even n below spider threshold."]),
    Preset(3, "A2.1-2a", "n=4 centroid + spider-net", 4, 0, 1, [1], 1.2, 3, 3.6,
           ["B exceeds the even-n spider threshold.", "No flower ring because A2(1)>0."]),
    Preset(4, "A2.1-2b", "n=4 centroid + spider-net, rotated", 4, 0, 1, [-1], 1.2, 3, 3.6,
           ["A2(1)<0 with larger B.", "Orientation differs from related n=4 cases."]),
    Preset(5, "A2.1-3", "n=4 flower ring + spider-net", 4, 0, 1, [-1], 0.7, 3, 3.6,
           ["A2(1)<0 supports peripheral equilibria.", "Four-petal flower-ring structure."]),
    Preset(6, "A2.2-1", "n=5 centroid + spider-net", 5, 0, 1, [1], 2.0, 3, 3.6,
           ["Odd n gives spider-net behavior for nonzero B.", "A2(1)>0 suppresses ring."]),
    Preset(7, "A2.2-2a", "n=5 flower ring + 5-star + spider-net", 5, 0, 1, [-1], 0.3, 3, 3.6,
           ["Canonical n=5 flower-ring example.", "Ring condition 27 eps2 B^2 + 4 A2(1)^3 < 0 holds."]),
    Preset(8, "A2.2-2b", "n=5 flower ring + 5-star, smaller B", 5, 0, 1, [-1], 0.2, 3, 3.6,
           ["Topologically close to preset 7.", "Smaller B gives rounder petals."]),
    Preset(9, "A2.3-1a", "n=6 spider-net only", 6, 0, 0, [1, 0], 0.5, 4, 5,
           ["Even n, A2(end)=0, B nonzero.", "No inner flower ring."]),
    Preset(10, "A2.3-1b", "n=6 spider-net, B negative", 6, 0, 0, [1, 0], -0.5, 4, 5,
           ["Same topology as preset 9 with sign(B) changing orientation."]),
    Preset(11, "A2.3-2", "n=6 centers + flower band", 6, 0, 1, [-1, 0.1], 0.04, 3, 3.6,
           ["Two-term A2 vector enables richer radial structure.", "Small B keeps orbits comparatively rounded."]),
    Preset(12, "A2.3-3", "n=6 center + star + two flower rings", 6, 0, 1, [-1, 0.1], 0.06, 3, 3.6,
           ["Two-ring example.", "Increasing B sharpens the separatrix star."]),
    Preset(13, "A2.3-4", "n=6 large centroid + outer spider-net", 6, 0, 1, [1, 0], 0.5, 5, 6,
           ["Wide Rmax shows outer structure.", "No peripheral ring from A2(1)>0."]),
    Preset(14, "A2.3-5", "n=6 annular band", 6, 0, 1, [-1, 0], 0.5, 3, 3.6,
           ["Inner centroid, annular band, and outer structure."]),
    Preset(15, "A2.3-6", "n=6 star + two rings, wider view", 6, 0, 1, [-1, 0.1], 0.06, 3.5, 4.2,
           ["Wider view of preset 12 parameters."]),
    Preset(16, "A2.4-1", "n=7 two flower rings, B=-1.6", 7, 0, -0.56, [3, -3.5, 0], -1.6, 5, 6,
           ["Alternating A2 signs support multiple rings.", "Negative eps2 changes rotation direction near origin."]),
    Preset(17, "A2.4-2", "n=7 two flower rings, B=-1.0", 7, 0, -0.56, [3, -3.5, 0], -1.0, 5, 6,
           ["Same family as preset 16 with smaller |B|."]),
    Preset(18, "A2.5-a", "n=9 B=0 pure centroid", 9, 0, 8, [-33, 23.765, -3.5], 0.0, 4, 4.8,
           ["B=0 removes the resonance term.", "Useful baseline before turning on 9-fold structure."]),
    Preset(19, "A2.5-b", "n=9 B=0.01 near-circular", 9, 0, 8, [-33, 23.765, -3.5], 0.01, 4, 4.8,
           ["Small B begins to reveal 9-fold perturbation."]),
    Preset(20, "A2.5-c1", "n=9 B=0.045 outer view", 9, 0, 8, [-33, 23.765, -3.5], 0.045, 8, 9.5,
           ["Outer view of nested 9-fold structure.", "Use zoom presets or smaller Rmax to inspect inner rings."]),
    Preset(21, "A2.5-c2", "n=9 B=0.045 mid zoom", 9, 0, 8, [-33, 23.765, -3.5], 0.045, 2.5, 3,
           ["Mid/inner zoom for preset 20 parameters."]),
    Preset(22, "A2.5-c4", "n=9 B=0.045 innermost zoom", 9, 0, 8, [-33, 23.765, -3.5], 0.045, 0.8, 1.0,
           ["Innermost centroid region."]),
    Preset(23, "A2.6-a", "n=11 two flower rings, B=0.05", 11, 0, 14.4, [-55.6, 54.6, 14.4, 1], 0.05, 7, 8.5,
           ["Large coefficients push structure outward.", "Odd n has spider-net behavior for nonzero B."]),
    Preset(24, "A2.6-b", "n=11 two flower rings, B=-0.01", 11, 0, 14.4, [-55.6, 54.6, 14.4, 1], -0.01, 7, 8.5,
           ["Negative smaller B rotates and softens the pattern relative to preset 23."]),
    Preset(25, "A2.7-a", "n=5 non-Hamiltonian: flower ring outside limit cycle", 5, 0.005, 1, [-0.01, -1], 0.1, 4, 5,
           ["eps1>0 makes origin unstable.", "Non-Hamiltonian perturbation converts centers to foci."]),
    Preset(26, "A2.7-b", "n=5 non-Hamiltonian: flower ring inside limit cycle", 5, 0.005, -0.1, [0.045, 1], 1.0, 4, 5,
           ["Flower ring is inside the limit-cycle region."]),
    Preset(27, "A2.7-c", "n=5 non-Hamiltonian: all spirals", 5, 0.005, -0.1, [-0.045, 1], 1.0, 4, 5,
           ["No visible limit cycle in the paper panel.", "Nearby starts can approach different peripheral foci."]),
    Preset(28, "A2.8-a", "n=6 non-Hamiltonian: ring outside unstable LC", 6, -0.001, -0.1, [1.3, 0.1], 0.05, 5, 6,
           ["eps1<0 makes origin stable.", "Paper describes ring outside an unstable limit cycle."]),
    Preset(29, "A2.8-b", "n=6 non-Hamiltonian: ring inside unstable LC", 6, -0.001, 0.1, [1, -0.1], 0.05, 5, 6,
           ["Ring is inside the unstable limit-cycle region."]),
]

PRESET_BY_ID = {p.id: p for p in PRESETS}

QUALITY = {
    # Restored from v1: these were the prettier/fast-enough render settings.
    # Labels kept from the newer interface: Regular = old Balanced; Detailed = old Dense.
    "Fast": {"Nseed": 5, "NangPerN": 2, "Tmax": 90, "max_step_div": 450},
    "Regular": {"Nseed": 8, "NangPerN": 3, "Tmax": 180, "max_step_div": 700},
    "Detailed": {"Nseed": 13, "NangPerN": 5, "Tmax": 350, "max_step_div": 1100},
}

SWEEP_DEFAULTS = {
    "B": [-1.5, -0.5, -0.15, 0.15, 0.5, 1.5],
    "A2_1": [1.0, 0.3, -0.3, -0.8, -1.5, -2.5],
    "n": [4, 5, 6, 7, 9, 11],
    "eps2": [-2, -0.5, 0, 0.5, 1, 2],
    "eps1": [-0.01, -0.003, 0, 0.003, 0.01, 0.02],
}


def pad_A2(n: int, A2: Sequence[float]) -> List[float]:
    s_needed = max(math.floor(n / 2) - 1, 1)
    values = list(A2)
    if len(values) < s_needed:
        values = values + [0.0] * (s_needed - len(values))
    return [float(x) for x in values[:s_needed]]


def parse_A2(text: str, n: int) -> List[float]:
    text = text.strip()
    if not text:
        return pad_A2(n, [])
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (int, float)):
            return pad_A2(n, [float(value)])
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise ValueError
        return pad_A2(n, [float(x) for x in value])
    except Exception as exc:
        raise ValueError("A2 must look like [-1] or [-1, 0.1].") from exc


# -----------------------------
# Arnold model
# -----------------------------

def polar_rhs(_t: float, RT: Sequence[float], P: dict) -> Tuple[float, float]:
    r = max(float(RT[0]), 0.0)
    phi = float(RT[1])
    n = int(P["n"])
    A2 = P["A2"]

    r2 = r * r
    Ar = 0.0
    Aphi = 0.0
    rpow = r2
    for a in A2:
        Ar += a * rpow * r
        Aphi += a * rpow
        rpow *= r2

    if r < 1e-12:
        return (0.0, float(P["eps2"]))

    theta = n * phi
    dr = float(P["eps1"]) * r + Ar + float(P["B"]) * (r ** (n - 1)) * math.cos(theta)
    dphi = float(P["eps2"]) + Aphi - float(P["B"]) * (r ** (n - 2)) * math.sin(theta)
    return (dr, dphi)


def radial_polynomial(r: np.ndarray, P: dict, sign: int) -> np.ndarray:
    y = np.full_like(r, float(P["eps2"]), dtype=float)
    for k, a in enumerate(P["A2"], start=1):
        y += float(a) * r ** (2 * k)
    y += sign * float(P["B"]) * r ** max(int(P["n"]) - 2, 0)
    return y


def eq_radii(P: dict, rmax: float | None = None, samples: int = 2400) -> List[float]:
    stopR = float(P.get("stopR", P.get("Rmax", 3.0) * 1.25))
    upper = float(rmax) if rmax is not None else stopR * 0.93
    if upper <= 0:
        return []
    r = np.linspace(0.01, upper, samples)
    roots: List[float] = []
    for sign in (+1, -1):
        y = radial_polynomial(r, P, sign)
        idx = np.where(y[:-1] * y[1:] < 0)[0]
        for i in idx:
            denom = y[i + 1] - y[i]
            if abs(denom) < 1e-12:
                continue
            rs = r[i] - y[i] * (r[i + 1] - r[i]) / denom
            if np.isfinite(rs) and rs > 0:
                roots.append(float(rs))
    roots = sorted(set(round(x, 6) for x in roots))
    return roots


def classify_pattern(P: dict) -> str:
    n = int(P["n"])
    A2 = P["A2"]
    radii = eq_radii(P, rmax=min(float(P.get("stopR", 5)), 8), samples=1200)
    has_eq = len(radii) > 0
    if n % 2 == 1:
        has_spider = abs(float(P["B"])) > 1e-12
    else:
        has_spider = len(A2) >= 1 and abs(float(P["B"])) > abs(float(A2[-1]))

    if not has_eq and not has_spider:
        label = "centroid"
    elif has_eq and not has_spider:
        label = "flower ring / annular band"
    elif not has_eq and has_spider:
        label = "spider-net"
    else:
        label = "flower ring + spider-net"
    if abs(float(P["eps1"])) > 1e-12:
        label += " [non-Hamiltonian]"
    return label


def seed_radii(P: dict, Nseed: int) -> np.ndarray:
    Rmax = float(P["Rmax"])
    base = np.linspace(Rmax * 0.035, Rmax * 0.92, Nseed)
    extras: List[float] = []
    for re in eq_radii(P):
        if re < Rmax * 0.97:
            extras.extend([re * x for x in [0.78, 0.86, 0.92, 0.97, 1.0, 1.03, 1.08, 1.15, 1.24]])
    all_r = np.array([x for x in list(base) + extras if 0 < x < Rmax * 0.985], dtype=float)
    if all_r.size == 0:
        return base
    return np.unique(np.round(all_r, 5))


def integrate_one(r0: float, phi0: float, P: dict, Tmax: float, stopR: float, max_step: float) -> np.ndarray | None:
    def event_stop(_t, rt):
        return stopR - rt[0]

    event_stop.terminal = True
    event_stop.direction = -1
    try:
        sol = solve_ivp(
            lambda t, y: polar_rhs(t, y, P),
            (0.0, Tmax),
            (float(r0), float(phi0)),
            rtol=2e-5,
            atol=1e-7,
            max_step=max_step,
            events=event_stop,
        )
        if sol.y.shape[1] < 2:
            return None
        return sol.y.T
    except Exception:
        return None


def plot_orbits(
    ax,
    P: dict,
    quality: str,
    use_sector: bool,
    backward: bool,
    show_eq: bool,
    background: str,
    line_alpha: float,
    line_width: float,
    seed_override: int | None = None,
    time_override: float | None = None,
):
    q = QUALITY[quality]
    Nseed = int(seed_override if seed_override is not None else q["Nseed"])
    Nang = int(q["NangPerN"])
    Tmax = float(time_override if time_override is not None else q["Tmax"])
    max_step = max(Tmax / float(q["max_step_div"]), 0.01)
    n = int(P["n"])
    Rmax = float(P["Rmax"])
    stopR = float(P["stopR"])

    radii = seed_radii(P, Nseed)
    if use_sector:
        phis = np.linspace(0, 2 * np.pi / n, max(n * Nang, 4) + 1)[:-1]
    else:
        phis = np.linspace(0, 2 * np.pi, max(n * Nang, 4) + 1)[:-1]

    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(len(radii), 1)) for i in range(len(radii))]

    for ir, r0 in enumerate(radii):
        color = colors[ir]
        for phi0 in phis:
            for direction in ([1.0, -1.0] if backward else [1.0]):
                RT = integrate_one(r0, phi0, P, Tmax * direction, stopR, max_step)
                if RT is None:
                    continue
                r = RT[:, 0]
                phi = RT[:, 1]
                valid = np.isfinite(r) & np.isfinite(phi) & (r >= 0) & (r <= stopR * 1.02)
                if valid.sum() < 2:
                    continue
                r = r[valid]
                phi = phi[valid]
                if use_sector:
                    for ks in range(n):
                        dp = ks * 2 * np.pi / n
                        ax.plot(r * np.cos(phi + dp), r * np.sin(phi + dp), color=color, alpha=line_alpha, lw=line_width)
                else:
                    ax.plot(r * np.cos(phi), r * np.sin(phi), color=color, alpha=line_alpha, lw=line_width)

    if show_eq:
        mark_equilibria(ax, P, background)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-Rmax, Rmax)
    ax.set_ylim(-Rmax, Rmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)


def mark_equilibria(ax, P: dict, background: str):
    n = int(P["n"])
    Rmax = float(P["Rmax"])
    stopR = float(P["stopR"])
    origin_color = "white" if background == "Dark" else "black"
    ax.plot(0, 0, marker="+", color=origin_color, markersize=9, mew=2.0, linestyle="None")

    r = np.linspace(0.005, stopR, 4000)
    for pol_index, sign in enumerate([+1, -1]):
        y = radial_polynomial(r, P, sign)
        idx = np.where(y[:-1] * y[1:] < 0)[0]
        p0 = 0.0 if sign == +1 else np.pi / n
        for root_index, i in enumerate(idx):
            denom = y[i + 1] - y[i]
            if abs(denom) < 1e-12:
                continue
            r_star = r[i] - y[i] * (r[i + 1] - r[i]) / denom
            for ke in range(n):
                pe = p0 + 2 * np.pi * ke / n
                xq = r_star * np.cos(pe)
                yq = r_star * np.sin(pe)
                if abs(xq) > Rmax * 1.05 or abs(yq) > Rmax * 1.05:
                    continue
                # Approximation copied from MATLAB visual convention: alternate roots.
                if root_index % 2 == 0:
                    ax.plot(xq, yq, "x", color="red", markersize=6, mew=1.8)
                else:
                    ax.plot(xq, yq, "o", color="limegreen", markersize=5, fillstyle="none", mew=1.6)


def make_figure(
    P: dict,
    quality: str = "Fast",
    use_sector: bool = True,
    backward: bool = True,
    show_eq: bool = True,
    show_info: bool = True,
    background: str = "Dark",
    line_alpha: float = 0.65,
    line_width: float = 0.55,
    seed_override: int | None = None,
    time_override: float | None = None,
    title: str | None = None,
):
    dark = background == "Dark"
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=140)
    fig.patch.set_facecolor("#020617" if dark else "white")
    ax.set_facecolor("#020617" if dark else "white")
    ax.tick_params(colors="white" if dark else "black")
    ax.xaxis.label.set_color("white" if dark else "black")
    ax.yaxis.label.set_color("white" if dark else "black")
    for spine in ax.spines.values():
        spine.set_color("white" if dark else "black")
        spine.set_alpha(0.45)

    plot_orbits(ax, P, quality, use_sector, backward, show_eq, background, line_alpha, line_width, seed_override, time_override)

    ttl = title or f"n={P['n']} | B={P['B']:.4g} | eps1={P['eps1']:.4g} | eps2={P['eps2']:.4g} | A2={P['A2']}"
    ax.set_title(ttl, color="white" if dark else "black", fontsize=9)

    if show_info:
        txt = (
            f"Pattern heuristic: {classify_pattern(P)}\n"
            f"n={P['n']}  B={P['B']:.4g}  eps1={P['eps1']:.4g}  eps2={P['eps2']:.4g}\n"
            f"A2={np.array2string(np.asarray(P['A2']), precision=3, separator=', ')}  Rmax={P['Rmax']:.3g}"
        )
        ax.text(
            0.01,
            0.01,
            txt,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=7,
            color="white" if dark else "black",
            bbox=dict(facecolor="#111827" if dark else "#f8fafc", edgecolor="white" if dark else "black", alpha=0.76, pad=4),
        )
    fig.tight_layout()
    return fig


def figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def params_to_key(P: dict) -> str:
    """Stable JSON key for caching rendered plots."""
    cleaned = {
        "n": int(P["n"]),
        "B": float(P["B"]),
        "eps1": float(P["eps1"]),
        "eps2": float(P["eps2"]),
        "A2": [float(x) for x in P["A2"]],
        "Rmax": float(P["Rmax"]),
        "stopR": float(P["stopR"]),
    }
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


@st.cache_data(show_spinner=False, max_entries=160)
def cached_figure_png(
    params_key: str,
    quality: str,
    use_sector: bool,
    backward: bool,
    show_eq: bool,
    show_info: bool,
    background: str,
    line_alpha: float,
    line_width: float,
    seed_override: int | None,
    time_override: float | None,
    title: str | None,
) -> bytes:
    """Cache the expensive part: numerical integration + Matplotlib rendering."""
    P = json.loads(params_key)
    fig = make_figure(
        P,
        quality=quality,
        use_sector=use_sector,
        backward=backward,
        show_eq=show_eq,
        show_info=show_info,
        background=background,
        line_alpha=line_alpha,
        line_width=line_width,
        seed_override=seed_override,
        time_override=time_override,
        title=title,
    )
    png = figure_to_png_bytes(fig)
    plt.close(fig)
    return png


def show_cached_figure(P: dict, **kwargs) -> bytes:
    """Render once per unique parameter/settings combination, then reuse from cache."""
    defaults = {"seed_override": None, "time_override": None, "title": None}
    defaults.update(kwargs)
    png = cached_figure_png(params_to_key(P), **defaults)
    st.image(png, use_container_width=True)
    return png


# -----------------------------
# Streamlit state helpers
# -----------------------------

def ensure_state():
    if "params" not in st.session_state:
        p = PRESET_BY_ID[7]
        st.session_state.params = p.parameter_dict()
        st.session_state.A2_text = str(p.A2)
        st.session_state.loaded_preset_id = 7


def set_params_from_preset(preset_id: int):
    p = PRESET_BY_ID[int(preset_id)]
    P = p.parameter_dict()
    st.session_state.params = P
    st.session_state.A2_text = str(p.A2)
    st.session_state.loaded_preset_id = int(preset_id)
    # Keep widget-backed values synchronized when loading presets from Start, Paper Presets, or Sweep.
    for key in ["n", "B", "eps1", "eps2", "Rmax", "stopR"]:
        st.session_state[key] = P[key]


def current_params_from_widgets() -> dict:
    n = int(st.session_state.get("n", st.session_state.params["n"]))
    A2 = parse_A2(st.session_state.get("A2_text", "[-1]"), n)
    P = {
        "n": n,
        "B": float(st.session_state.get("B", st.session_state.params["B"])),
        "eps1": float(st.session_state.get("eps1", st.session_state.params["eps1"])),
        "eps2": float(st.session_state.get("eps2", st.session_state.params["eps2"])),
        "A2": A2,
        "Rmax": float(st.session_state.get("Rmax", st.session_state.params["Rmax"])),
        "stopR": float(st.session_state.get("stopR", st.session_state.params["stopR"])),
    }
    return P


def render_plot_with_timer(P: dict, **kwargs):
    # Kept for compatibility, but now routes through cache and returns PNG bytes.
    t0 = time.time()
    png = show_cached_figure(P, **kwargs)
    elapsed = time.time() - t0
    return png, elapsed


def pattern_vocab_card():
    st.markdown(
        """
        **Pattern vocabulary**

        **Centroid**: center or spiral equilibrium with surrounding orbits.  
        **n-flower ring**: n centroids and n saddles arranged with n-fold symmetry.  
        **n-star / separatrix cycle**: separatrix boundary connecting saddle structure.  
        **Spider-net**: large-radius saddle/separatrix structure with hyperbolic sectors.
        """
    )


# -----------------------------
# App UI
# -----------------------------

st.set_page_config(page_title="Greek Ornament Simulator", page_icon="✦", layout="wide")
ensure_state()

st.title("Greek ornamental design simulator")
st.caption("Arnold weak-resonance phase-portrait explorer. Local prototype; verify outputs before GitHub deployment.")

with st.sidebar:
    st.header("Render settings")
    quality = st.selectbox(
        "Run mode",
        ["Fast", "Regular", "Detailed"],
        index=0,
        help="Fast is the default preview mode, tuned to stay responsive while still looking reasonably filled-in. Use Regular for clearer previews and Detailed only after a pattern looks promising.",
    )
    background = st.radio("Background", ["Dark", "Light"], horizontal=True)
    use_sector = st.toggle("Use sector symmetry", value=True, help="Integrate one angular sector and rotate copies. Faster for n-fold symmetric cases.")
    backward = st.toggle(
        "Forward + backward integration",
        value=True,
        help="On by default because the original pretty/fast version used forward + backward integration. Turn off if you need a very quick sparse scan.",
    )
    show_eq = st.toggle("Show equilibrium markers", value=True)
    st.caption("Fast uses the original v1 render settings. Forward + backward is on by default because it makes the flower/separatrix structure fuller.")
    if st.button("Clear plot cache"):
        st.cache_data.clear()
        st.rerun()
    show_info = st.toggle("Show info box", value=True)
    line_alpha = st.slider("Line opacity", 0.05, 1.0, 0.65, 0.05)
    line_width = st.slider("Line width", 0.15, 2.0, 0.55, 0.05)



PAGES = ["Paper presets", "Sweep / zoom", "Explore", "Render / export", "About"]
st.info("First load on Streamlit Cloud may be slow because figures are computed and cached. After that, repeated presets/settings should be faster.")
st.caption("Cached version: the original tab interface is restored, while expensive plots are cached by parameter set and render settings.")

tab_presets, tab_sweep, tab_explore, tab_render, tab_about = st.tabs(PAGES)

with tab_presets:
    st.subheader("Paper preset selector")
    options = {f"{p.id}. {p.fig} — {p.label}": p.id for p in PRESETS}
    selection = st.selectbox("Preset", list(options.keys()), index=0)
    selected = PRESET_BY_ID[options[selection]]
    c1, c2 = st.columns([0.42, 0.58])
    with c1:
        st.markdown(f"**Figure:** {selected.fig}")
        st.markdown(f"**Pattern:** {selected.label}")
        st.table({
            "Parameter": ["n", "B", "eps1", "eps2", "A2", "Rmax", "stopR"],
            "Value": [selected.n, selected.B, selected.eps1, selected.eps2, str(selected.A2), selected.Rmax, selected.stopR],
            "Meaning": [
                "rotational symmetry order",
                "resonance coupling strength",
                "non-Hamiltonian perturbation",
                "angular frequency near the origin",
                "radial polynomial coefficients",
                "plot window radius",
                "integration cutoff radius",
            ],
        })
        if selected.notes:
            st.markdown("**Notes**")
            for note in selected.notes:
                st.markdown(f"- {note}")
        if st.button("Load selected preset into Explore", type="primary"):
            set_params_from_preset(selected.id)
            st.success(f"Loaded preset {selected.id}: {selected.fig}")
    with c2:
        P = selected.parameter_dict()
        png, elapsed = render_plot_with_timer(
            P,
            quality=quality,
            use_sector=use_sector,
            backward=backward,
            show_eq=show_eq,
            show_info=show_info,
            background=background,
            line_alpha=line_alpha,
            line_width=line_width,
            title=f"Preset {selected.id}: {selected.fig} — {selected.label}",
        )
        st.caption(f"Rendered in {elapsed:.1f} seconds. Pattern heuristic: {classify_pattern(P)}")

with tab_sweep:
    st.subheader("One-parameter sweep")
    st.markdown("This is the MATLAB sweep workflow translated into the app: hold the current explorer parameters fixed and vary one parameter over exact values.")
    try:
        base = current_params_from_widgets()
    except ValueError:
        base = PRESET_BY_ID[7].parameter_dict()
        st.warning("Using preset 7 as sweep base because the current A2 field could not be parsed.")
    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        sweep_param = st.selectbox("Parameter to vary", ["B", "A2_1", "n", "eps2", "eps1"])
        sweep_quality = st.selectbox("Sweep quality", ["Fast", "Regular"], index=0)
    with c2:
        st.markdown("**Exact test values**")
        st.caption("These are the individual values that will be rendered as separate panels, not the endpoints of a continuous range.")
        vals_text = st.text_input(
            "Enter values as a Python-style list",
            value=str(SWEEP_DEFAULTS[sweep_param]),
            help="Example: [-1.5, -0.5, -0.15, 0.15, 0.5, 1.5]. Each entry becomes one sweep panel.",
        )
    try:
        vals = ast.literal_eval(vals_text)
        if not isinstance(vals, list) or len(vals) < 2:
            raise ValueError
        if sweep_param == "n":
            vals = [int(round(float(v))) for v in vals]
        else:
            vals = [float(v) for v in vals]
    except Exception:
        vals = SWEEP_DEFAULTS[sweep_param]
        st.warning("Could not parse exact test values; using defaults.")

    vals_display = ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals)
    st.caption(f"Will render {len(vals)} panels using these exact {sweep_param} values: [{vals_display}]. Only `{sweep_param}` changes; all other parameters come from the current Explore settings.")
    run_sweep = st.button("Run sweep", type="primary")

    def build_sweep_results(base_params: dict, param_name: str, test_values: list, quality_name: str) -> dict:
        """Build a session-state record of the requested sweep.

        This stores the parameter sets and the render settings used at the time the
        sweep was run. The images themselves remain in st.cache_data, but this
        record lets the panels stay visible when the user changes tabs and returns.
        """
        panels = []
        for idx, v in enumerate(test_values):
            P = dict(base_params)
            P["A2"] = list(base_params["A2"])
            if param_name == "B":
                P["B"] = float(v)
            elif param_name == "A2_1":
                if not P["A2"]:
                    P["A2"] = [0.0]
                P["A2"][0] = float(v)
                P["A2"] = pad_A2(int(P["n"]), P["A2"])
            elif param_name == "n":
                P["n"] = int(round(v))
                P["A2"] = pad_A2(int(P["n"]), P["A2"])
            elif param_name == "eps2":
                P["eps2"] = float(v)
            elif param_name == "eps1":
                P["eps1"] = float(v)
            panels.append({
                "idx": idx,
                "value": v,
                "params": P,
                "pattern": classify_pattern(P),
            })
        return {
            "sweep_param": param_name,
            "values_display": ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in test_values),
            "quality": quality_name,
            "use_sector": use_sector,
            "backward": backward,
            "show_eq": show_eq,
            "background": background,
            "line_alpha": line_alpha,
            "line_width": line_width,
            "panels": panels,
        }

    if run_sweep:
        st.session_state.sweep_results = build_sweep_results(base, sweep_param, vals, sweep_quality)

    sweep_results = st.session_state.get("sweep_results")
    if sweep_results is None:
        st.info("Click Run sweep to render the panels. The most recent sweep will stay visible until you run a new sweep.")
    else:
        st.success(
            f"Showing saved sweep: {len(sweep_results['panels'])} panels for "
            f"{sweep_results['sweep_param']} = [{sweep_results['values_display']}]. "
            "These panels will stay here until you run a new sweep."
        )
        ncols = 3
        rows = math.ceil(len(sweep_results["panels"]) / ncols)
        idx = 0
        for _row in range(rows):
            cols = st.columns(ncols)
            for col in cols:
                if idx >= len(sweep_results["panels"]):
                    continue
                panel = sweep_results["panels"][idx]
                v = panel["value"]
                P = panel["params"]
                with col:
                    st.markdown(f"**{sweep_results['sweep_param']} = {v:g}**  ")
                    show_cached_figure(
                        P,
                        quality=sweep_results["quality"],
                        use_sector=sweep_results["use_sector"],
                        backward=sweep_results["backward"],
                        show_eq=sweep_results["show_eq"],
                        show_info=False,
                        background=sweep_results["background"],
                        line_alpha=sweep_results["line_alpha"],
                        line_width=sweep_results["line_width"],
                        seed_override=None,
                        time_override=None,
                        title=f"{sweep_results['sweep_param']}={v:g} | {panel['pattern']}",
                    )
                    st.caption(panel["pattern"])
                    st.download_button(
                        "Params JSON",
                        data=json.dumps(P, indent=2),
                        file_name=f"sweep_{sweep_results['sweep_param']}_{idx+1}.json",
                        mime="application/json",
                        key=f"sweep_json_saved_{idx}",
                    )
                idx += 1

with tab_explore:
    st.subheader("Manual exploration")
    st.caption("A Fast preview is shown by default. Edit parameters in the form, then click Update preview; slider changes are not applied until you press the button.")
    p0 = st.session_state.params
    c1, c2 = st.columns([0.36, 0.64])
    with c1:
        with st.form("explore_controls"):
            st.number_input("n-fold symmetry", min_value=4, max_value=15, value=int(p0["n"]), step=1, key="n")
            st.slider("B resonance coupling", min_value=-2.5, max_value=2.5, value=float(p0["B"]), step=0.005, key="B")
            st.slider("eps1 non-Hamiltonian perturbation", min_value=-0.05, max_value=0.05, value=float(p0["eps1"]), step=0.0005, format="%.4f", key="eps1")
            st.slider("eps2 angular frequency", min_value=-15.0, max_value=20.0, value=float(p0["eps2"]), step=0.05, key="eps2")
            st.text_input("A2 coefficients", value=st.session_state.get("A2_text", str(p0["A2"])), key="A2_text", help="Use Python-list syntax, e.g. [-1] or [-1, 0.1]. Length is padded/truncated to s=floor(n/2)-1.")
            st.slider("Plot window Rmax", min_value=0.25, max_value=12.0, value=float(p0["Rmax"]), step=0.05, key="Rmax")
            st.slider("Stop integration radius", min_value=0.5, max_value=15.0, value=float(p0["stopR"]), step=0.05, key="stopR")
            run_button = st.form_submit_button(f"Update {quality.lower()} preview", type="primary", use_container_width=True)
        if st.button("Reset to preset 7", use_container_width=True):
            set_params_from_preset(7)
            st.rerun()
        st.info("Fast mode uses the original v1 render settings. Use Regular or Detailed when you want more density, or turn off forward + backward if you need a sparse quick scan.")

    with c2:
        try:
            if run_button:
                P_new = current_params_from_widgets()
                st.session_state.params = P_new
                st.success("Preview updated.")
            P = st.session_state.params
            png, elapsed = render_plot_with_timer(
                P,
                quality=quality,
                use_sector=use_sector,
                backward=backward,
                show_eq=show_eq,
                show_info=show_info,
                background=background,
                line_alpha=line_alpha,
                line_width=line_width,
            )
            st.caption(f"Rendered in {elapsed:.1f} seconds. Pattern heuristic: {classify_pattern(P)}")
            st.markdown("**Current parameter set**")
            st.table({
                "Parameter": ["n", "B", "eps1", "eps2", "A2", "Rmax", "stopR", "Run mode"],
                "Value": [P["n"], P["B"], P["eps1"], P["eps2"], str(P["A2"]), P["Rmax"], P["stopR"], quality],
                "Meaning": [
                    "rotational symmetry order",
                    "resonance coupling strength",
                    "non-Hamiltonian perturbation; 0 gives Hamiltonian case",
                    "angular frequency near the origin",
                    "radial polynomial coefficients",
                    "plot window radius",
                    "integration cutoff radius",
                    "controls seed density and integration time",
                ],
            })
            st.download_button(
                "Download current parameters JSON",
                data=json.dumps(P, indent=2),
                file_name="arnold_ornament_parameters.json",
                mime="application/json",
            )
        except ValueError as exc:
            st.error(str(exc))

with tab_render:
    st.subheader("Render and export")
    st.markdown("Use this after the fast or regular preview is promising. Detailed mode uses larger seed counts and longer integration times.")
    try:
        P = current_params_from_widgets()
    except ValueError:
        P = PRESET_BY_ID[7].parameter_dict()
        st.warning("Using preset 7 because the current A2 field could not be parsed.")

    c1, c2 = st.columns([0.32, 0.68])
    with c1:
        dense_quality = st.selectbox("Render mode", ["Regular", "Detailed"], index=1)
        seed_override = st.slider("Seed density override", 5, 40, 16, 1)
        time_override = st.slider("Integration time override", 50, 1000, 350, 25)
        export_format = st.radio("Export", ["PNG", "Parameters JSON"], horizontal=True)
        st.table({"Parameter": ["n", "B", "eps1", "eps2", "A2", "Rmax", "stopR"], "Value": [P["n"], P["B"], P["eps1"], P["eps2"], str(P["A2"]), P["Rmax"], P["stopR"]]})
        run_render = st.button("Render export figure", type="primary")
    with c2:
        if not run_render:
            st.info("Click Render export figure when you are ready. This prevents detailed rendering during the first cloud load.")
        else:
            png, elapsed = render_plot_with_timer(
                P,
                quality=dense_quality,
                use_sector=use_sector,
                backward=backward,
                show_eq=show_eq,
                show_info=show_info,
                background=background,
                line_alpha=line_alpha,
                line_width=line_width,
                seed_override=seed_override,
                time_override=time_override,
                title=None,
            )
            st.caption(f"Displayed in {elapsed:.1f} seconds; repeated renders are cached. Pattern heuristic: {classify_pattern(P)}")
            if export_format == "PNG":
                st.download_button("Download PNG", data=png, file_name="arnold_ornament_render.png", mime="image/png")
            else:
                st.download_button("Download parameters JSON", data=json.dumps(P, indent=2), file_name="arnold_ornament_parameters.json", mime="application/json")

with tab_about:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("About / reference")
        st.markdown(
            """
            Reference: **Berezovskaya & Karev, “Arnold’s Weak Resonance Equation as the Model of Greek Ornamental Design.”**

            This app is meant to reproduce the working MATLAB workflow in a single Python/Streamlit interface:
            paper-preset inspection, one-parameter sweeps, custom parameter exploration, zooms, and slower dense rendering.

            The current version is a **first functional prototype**. It uses the polar-coordinate system from the MATLAB scripts,
            with optional sector symmetry and optional backward integration. The rendered figures should be treated as qualitative
            phase-portrait previews until we compare selected presets directly against the MATLAB outputs.
            """
        )
        st.subheader("Model integrated by the app")
        st.latex(r"\dot r = \epsilon_1 r + \sum_{k=1}^{s} A^2_k r^{2k+1} + B r^{n-1}\cos(n\phi)")
        st.latex(r"\dot \phi = \epsilon_2 + \sum_{k=1}^{s} A^2_k r^{2k} - B r^{n-2}\sin(n\phi)")
        st.markdown(r"where $s=\lfloor n/2\rfloor-1$.")
    with right:
        st.subheader("Workflow")
        st.markdown(
            """
            1. Start with **Paper presets** to reproduce known regimes.  
            2. Use **Sweep / zoom** to scan exact values for one parameter.  
            3. Move into **Explore** for manual adjustment.  
            4. Use **Render / export** only after the fast preview is promising.
            """
        )
        pattern_vocab_card()
        st.info("Marker convention: white/black + = origin, red × ≈ saddle-like peripheral point, green ○ ≈ center-like peripheral point. The current saddle/center labeling follows the MATLAB visual convention and should be verified for edge cases.")
