"""MCP server exposing LightshowAI XANES prediction tools."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

# --- paths -------------------------------------------------------------------
# This file may live in either of two layouts:
#   (a) <repo>/mcp/lightshowai_server.py         (broader agentic_workflows repo)
#   (b) <repo>/examples/LightshowAI/mcp/...      (self-contained LightshowAI example)
# Detect (b) first by checking for a sibling `lightshowai` package directory.
_HERE = Path(__file__).resolve().parent
_VENDORED_LIGHTSHOWAI_DIR = _HERE.parent  # examples/LightshowAI/ in layout (b)
if (_VENDORED_LIGHTSHOWAI_DIR / "lightshowai").is_dir():
    _LIGHTSHOWAI_DIR = _VENDORED_LIGHTSHOWAI_DIR
    _REPO_DIR = _LIGHTSHOWAI_DIR
else:
    _REPO_DIR = _HERE.parent
    _LIGHTSHOWAI_DIR = _REPO_DIR / "examples" / "LightshowAI"

_ENV_FILE = _REPO_DIR / ".env"

# Add LightshowAI package to path
sys.path.insert(0, str(_LIGHTSHOWAI_DIR))

# --- env / API key -----------------------------------------------------------
def _load_env() -> None:
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

_load_env()

# --- energy grid (per-element K-edge, 141 pts over 35 eV) -------------------
_ENE_START = {
    "Ti": 4964.504, "V": 5464.097, "Cr": 5989.168, "Mn": 6537.886,
    "Fe": 7111.23,  "Co": 7709.282, "Ni": 8332.181, "Cu": 8983.173,
}
_ENE_GRID = {el: np.linspace(s, s + 35, 141) for el, s in _ENE_START.items()}

# --- lazy model cache --------------------------------------------------------
_model_cache: dict[str, object] = {}

def _get_model(element: str, spectroscopy_type: str):
    from lightshowai.models import XASBlockModule
    key = f"{element}_{spectroscopy_type}"
    if key not in _model_cache:
        _model_cache[key] = XASBlockModule.load(element=element, spectroscopy_type=spectroscopy_type)
    return _model_cache[key]

def _fetch_structure(material_id: str):
    from mp_api.client import MPRester
    api_key = os.environ.get("MP_API_KEY")
    with MPRester(api_key) as mpr:
        return mpr.get_structure_by_material_id(material_id)

def _predict(material_id: str, absorbing_element: str, spectroscopy_type: str) -> dict:
    from lightshowai.models import predict
    structure = _fetch_structure(material_id)
    spectra = predict(structure, absorbing_element, spectroscopy_type)
    energy = _ENE_GRID[absorbing_element].tolist()
    return {
        "material_id": material_id,
        "absorbing_element": absorbing_element,
        "spectroscopy_type": spectroscopy_type,
        "formula": structure.formula,
        "energy_eV": energy,
        "site_spectra": {str(k): v.tolist() for k, v in spectra.items()},
        "mean_spectrum": np.mean(list(spectra.values()), axis=0).tolist(),
        "n_sites": len(spectra),
    }

# --- MCP server --------------------------------------------------------------
mcp = FastMCP("lightshowai")

XASBLOCKS_PATH = _LIGHTSHOWAI_DIR / "model_checkpoints" / "xasblock" / "v1.1.1"


@mcp.tool()
def list_available_models() -> str:
    """List all available element / spectroscopy-type combinations."""
    combos = sorted(f.stem for f in XASBLOCKS_PATH.glob("*.ckpt"))
    elements = sorted(_ENE_START.keys())
    return json.dumps({
        "available_models": combos,
        "supported_elements": elements,
        "spectroscopy_types": ["FEFF", "VASP"],
        "note": "VASP models available for Ti and Cu only; FEFF for all 8 elements.",
    }, indent=2)


@mcp.tool()
def predict_xanes(
    material_id: str,
    absorbing_element: str,
    spectroscopy_type: str = "FEFF",
) -> str:
    """Predict XANES spectrum for a Materials Project structure.

    Args:
        material_id: MP material ID, e.g. 'mp-149'.
        absorbing_element: Element symbol for the absorbing site (Ti, V, Cr, Mn, Fe, Co, Ni, Cu).
        spectroscopy_type: 'FEFF' (default, all elements) or 'VASP' (Ti and Cu only).
    """
    try:
        result = _predict(material_id, absorbing_element, spectroscopy_type)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def plot_xanes(
    material_id: str,
    absorbing_element: str,
    spectroscopy_type: str = "FEFF",
    output_path: str = "",
    open_browser: bool = True,
) -> str:
    """Predict XANES and save an interactive HTML plot.

    Args:
        material_id: MP material ID, e.g. 'mp-149'.
        absorbing_element: Absorbing element symbol.
        spectroscopy_type: 'FEFF' or 'VASP'.
        output_path: Where to save the HTML file. Defaults to ~/tmp/<material_id>_xanes.html.
        open_browser: Automatically open the plot in the default browser.
    """
    import plotly.graph_objects as go

    try:
        data = _predict(material_id, absorbing_element, spectroscopy_type)
    except Exception as e:
        return json.dumps({"error": str(e)})

    energy = data["energy_eV"]
    fig = go.Figure()

    # per-site spectra (lighter traces)
    for site_idx, spectrum in data["site_spectra"].items():
        fig.add_trace(go.Scatter(
            x=energy, y=spectrum,
            mode="lines",
            name=f"Site {site_idx}",
            line=dict(width=1),
            opacity=0.5,
        ))

    # mean spectrum (bold)
    fig.add_trace(go.Scatter(
        x=energy, y=data["mean_spectrum"],
        mode="lines",
        name="Mean",
        line=dict(width=2.5, color="black"),
    ))

    title = (f"{data['formula']} ({material_id}) — "
             f"{absorbing_element} K-edge XANES ({spectroscopy_type})")
    fig.update_layout(
        title=title,
        xaxis_title="Energy (eV)",
        yaxis_title="Intensity (arb. units)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if not output_path:
        out_dir = Path.home() / "tmp"
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / f"{material_id}_{absorbing_element}_{spectroscopy_type}_xanes.html")

    fig.write_html(output_path)

    if open_browser:
        subprocess.Popen(["open", output_path])

    return json.dumps({
        "saved_to": output_path,
        "material_id": material_id,
        "formula": data["formula"],
        "absorbing_element": absorbing_element,
        "spectroscopy_type": spectroscopy_type,
        "n_sites": data["n_sites"],
    }, indent=2)


@mcp.tool()
def compare_xanes(
    material_ids: str,
    absorbing_element: str,
    spectroscopy_type: str = "FEFF",
    output_path: str = "",
    open_browser: bool = True,
) -> str:
    """Compare mean XANES spectra for multiple materials on one plot.

    Args:
        material_ids: Comma-separated MP material IDs, e.g. 'mp-149,mp-22862'.
        absorbing_element: Absorbing element symbol (must be present in all structures).
        spectroscopy_type: 'FEFF' or 'VASP'.
        output_path: Where to save the HTML file.
        open_browser: Automatically open the plot in the default browser.
    """
    import plotly.graph_objects as go

    ids = [m.strip() for m in material_ids.split(",") if m.strip()]
    if len(ids) < 2:
        return json.dumps({"error": "Provide at least 2 comma-separated material IDs."})

    fig = go.Figure()
    results = []

    for mid in ids:
        try:
            data = _predict(mid, absorbing_element, spectroscopy_type)
            fig.add_trace(go.Scatter(
                x=data["energy_eV"],
                y=data["mean_spectrum"],
                mode="lines",
                name=f"{data['formula']} ({mid})",
                line=dict(width=2),
            ))
            results.append({"material_id": mid, "formula": data["formula"], "n_sites": data["n_sites"]})
        except Exception as e:
            results.append({"material_id": mid, "error": str(e)})

    fig.update_layout(
        title=f"{absorbing_element} K-edge XANES comparison ({spectroscopy_type})",
        xaxis_title="Energy (eV)",
        yaxis_title="Intensity (arb. units)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if not output_path:
        out_dir = Path.home() / "tmp"
        out_dir.mkdir(exist_ok=True)
        label = "_".join(ids)[:60]
        output_path = str(out_dir / f"compare_{absorbing_element}_{label}_xanes.html")

    fig.write_html(output_path)

    if open_browser:
        subprocess.Popen(["open", output_path])

    return json.dumps({"saved_to": output_path, "results": results}, indent=2)


if __name__ == "__main__":
    mcp.run()
