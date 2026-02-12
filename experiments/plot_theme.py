from __future__ import annotations

import colorsys
import re
from typing import Dict

import matplotlib.pyplot as plt


FAMILY_BASE_COLORS: Dict[str, str] = {
    "fp16": "#1F3552",      # deep navy
    "uniform": "#D55E00",   # vermillion
    "mixed": "#009E73",     # bluish green
    "enc_retention": "#5E3C99",  # purple
    "asymmetric": "#7A7A7A",     # neutral gray
    "other": "#4D4D4D",
}


def apply_paper_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#BBBBBB",
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
        }
    )


def family_from_variant(name: str) -> str:
    if name == "fp16":
        return "fp16"
    if name.startswith("uniform_"):
        return "uniform"
    if name.startswith("mixed_"):
        return "mixed"
    if name.startswith("encfp16_"):
        return "enc_retention"
    if re.match(r"^enc\d+_pred\d+$", name):
        return "asymmetric"
    return "other"


def bit_from_variant(name: str) -> int | None:
    m = re.search(r"int(\d+)$", name)
    if not m:
        return None
    return int(m.group(1))


def _adjust_lightness(hex_color: str, factor: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(rr*255):02x}{int(gg*255):02x}{int(bb*255):02x}"


def variant_color(name: str) -> str:
    fam = family_from_variant(name)
    base = FAMILY_BASE_COLORS.get(fam, FAMILY_BASE_COLORS["other"])
    bits = bit_from_variant(name)
    if bits is None:
        return base
    lightness_map = {
        8: 0.85,
        6: 1.00,
        4: 1.15,
        3: 1.28,
    }
    return _adjust_lightness(base, lightness_map.get(bits, 1.0))


def variant_marker(name: str) -> str:
    fam = family_from_variant(name)
    return {
        "fp16": "D",
        "uniform": "o",
        "mixed": "s",
        "enc_retention": "^",
        "asymmetric": "v",
    }.get(fam, "o")
