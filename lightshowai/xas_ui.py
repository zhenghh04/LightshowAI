"""
OmniXAS@Lightshow.ai — Interactive XANES Spectrum Prediction & Comparison Dashboard
====================================================================================

This Dash web application allows researchers to:
  1. Upload crystal structures (CIF, VASP/POSCAR, JSON) — single or batch.
  2. Predict K-edge XANES spectra for transition-metal absorbers using ML models.
  3. Upload experimental spectra (.csv, .dat, .mat, .xdi) and overlay them on predictions.
  4. Quantitatively compare predicted vs. experimental spectra with multiple
     correlation metrics (cosine-derivative, Pearson, Spearman, etc.).
  5. Rank candidate structures by how well their predicted spectra match experiment.

Key dependencies:
  - Dash / Plotly          — web framework & interactive plots
  - pymatgen               — crystal structure I/O & manipulation
  - mp_api                 — Materials Project REST client (structure search)
  - crystal_toolkit        — 3-D structure viewer component for Dash
  - lightshow              — ML spectrum prediction & postprocessing utilities
  - numpy / pandas / scipy — numerical processing

Environment requirements:
  - Set the ``MP_API_KEY`` environment variable to a valid Materials Project API key.
  - The app listens on **port 8443** by default (all interfaces).

Architecture overview (single-file app):
  ┌────────────────────────────────────────────────────────────────────┐
  │  §1  Platform Compatibility Patch                                 │
  │  §2  Imports                                                      │
  │  §3  App Initialisation & Core UI Components                      │
  │  §4  Constants & Shared Configuration                             │
  │  §5  Spectrum Comparison Helpers                                  │
  │  §6  Client-Side Data Stores                                      │
  │  §7  Experimental-Spectrum Upload Components                      │
  │  §8  Shared Style Definitions                                     │
  │  §9  Page Layout                                                  │
  │  §10 File-Parsing Utilities                                       │
  │  §11 Dash Callbacks — Experimental Spectrum Workflow               │
  │  §12 Dash Callbacks — Structure Loading (MP search / file upload) │
  │  §13 Dash Callbacks — Batch Upload & Scoring                      │
  │  §14 Dash Callbacks — XAS Prediction & Plotting                   │
  │  §15 Dash Callbacks — UI Helpers (shift, sort, display)           │
  │  §16 Matching-Results Table Builder                                │
  │  §17 Crystal Toolkit Registration & Entry Point                   │
  └────────────────────────────────────────────────────────────────────┘

Author / Maintainer: Qu, Xiaohui, Sairam Sri Vatsavai, Christopher
"""

# =============================================================================
# §1  Platform Compatibility Patch
# =============================================================================
# On Windows, pymatgen's Cython neighbor-finding routine expects ``int64``
# arrays for the periodic-boundary-condition (pbc) argument, but sometimes
# receives platform-native ``int32``.  This monkey-patch coerces the dtype
# *before* calling the original C extension so the app works on all OSes.

import numpy as np


def _patch_pymatgen_neighbors():
    """Wrap ``pymatgen.optimization.neighbors.find_points_in_spheres`` so that
    the ``pbc`` argument is always cast to ``int64``."""
    try:
        from pymatgen.optimization import neighbors as pmg_neighbors

        _original_find_points = pmg_neighbors.find_points_in_spheres

        def _patched_find_points_in_spheres(
            all_coords, center_coords, r, pbc, lattice, tol=1e-8
        ):
            pbc = np.asarray(pbc, dtype=np.int64)
            return _original_find_points(
                all_coords, center_coords, r, pbc, lattice, tol
            )

        pmg_neighbors.find_points_in_spheres = _patched_find_points_in_spheres
        print("Applied Windows int64 compatibility patch for pymatgen")
    except Exception as e:
        print(f"Warning: Could not apply pymatgen patch: {e}")


# Apply the patch at import time — before any pymatgen neighbour lookups.
_patch_pymatgen_neighbors()


# =============================================================================
# §2  Imports
# =============================================================================

from base64 import b64encode, b64decode
import io
import json
import os
import pathlib
import re
import tempfile
from zipfile import ZipFile

import pandas as pd

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

from pymatgen.core.structure import Structure
from mp_api.client import MPRester

import crystal_toolkit.components as ctc
from crystal_toolkit.helpers.layouts import Box, Column, Columns, Loading

from lightshowai.models import predict
from lightshowai import compare_utils


# =============================================================================
# §3  App Initialisation & Core UI Components
# =============================================================================

# Create the Dash app.  ``prevent_initial_callbacks=True`` stops every
# callback from firing once on page load (we trigger them explicitly).
app = dash.Dash(
    prevent_initial_callbacks=True,
    title="OmniXAS@Lightshow.ai",
    url_base_pathname="/omnixas/",
)

# Expose the underlying Flask server (useful for WSGI deployment, e.g. gunicorn).
server = app.server

# --- Crystal Toolkit widgets --------------------------------------------------
# These are pre-built Dash components from the ``crystal_toolkit`` library.
# They handle 3-D structure rendering, MP-ID search, and file upload natively.

struct_component = ctc.StructureMoleculeComponent(
    id="st_vis",
    show_image_button=False,
    show_export_button=False,
)
search_component = ctc.SearchComponent(id="mpid_search")
upload_component = ctc.StructureMoleculeUploadComponent(id="file_loader")


# =============================================================================
# §4  Constants & Shared Configuration
# =============================================================================

# Supported absorbing elements and the corresponding K-edge onset energies (eV).
ALL_ELEMENTS = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]

EDGE_ONSET_ENERGIES = {
    "Ti": 4964.504, "V": 5464.097, "Cr": 5989.168, "Mn": 6537.886,
    "Fe": 7111.23,  "Co": 7709.282, "Ni": 8332.181, "Cu": 8983.173,
}

# Pre-compute a uniform energy grid (141 points spanning 35 eV) for each element.
# All ML predictions are reported on this grid.
ENERGY_GRID = {
    el: np.linspace(start, start + 35, 141)
    for el, start in EDGE_ONSET_ENERGIES.items()
}

# Dropdown options for the XAS model selector.
# Each entry combines an element with the simulation code used for training.
XAS_MODEL_NAMES = [f"{el} FEFF" for el in ALL_ELEMENTS] + ["Ti VASP", "Cu VASP"]

# --- Metric definitions for spectrum comparison --------------------------------
# These are the statistical measures used by ``compare_utils`` to quantify
# similarity between a predicted and an experimental spectrum.
ALL_METRICS = [
    "coss_deriv",          # Cosine similarity of the first derivative
    "pearson",             # Pearson correlation coefficient
    "spearman",            # Spearman rank correlation
    "coss",                # Cosine similarity (raw spectra)
    "kendalltaub",         # Kendall's tau-b rank correlation
    "normed_wasserstein",  # Normalised Wasserstein (earth-mover) distance
]

# Short display names used in the compact comparison table.
METRIC_SHORT_NAMES = {
    "coss_deriv":          "Cos(∂)",
    "pearson":             "Pearson",
    "spearman":            "Spearman",
    "coss":                "Cosine",
    "kendalltaub":         "Kendall",
    "normed_wasserstein":  "Wasser.",
}


# =============================================================================
# §5  Spectrum Comparison Helpers
# =============================================================================

def get_spectrum_match_score(predicted_spectrum, exp_spectrum, element):
    """Compare a predicted XANES spectrum to an experimental one.

    Uses ``lightshow.postprocess.compare_utils.compare_between_spectra`` to
    find the optimal energy shift that aligns the two spectra, then evaluates
    multiple correlation / distance metrics.

    Parameters
    ----------
    predicted_spectrum : np.ndarray
        1-D array of predicted absorption values on the standard energy grid.
    exp_spectrum : dict
        Must contain ``'energy'`` and ``'absorption'`` keys (lists of floats).
    element : str
        Absorbing element symbol (e.g. ``'Ti'``) — used to look up the grid.

    Returns
    -------
    dict
        Keys: ``score`` (float — primary metric value), ``correlations`` (dict
        of all metric values), ``shift`` (float — optimal energy shift in eV),
        ``comparison_range`` (tuple[float, float] | None — energy window used).
    """
    try:
        ene = ENERGY_GRID[element]

        # Build 2-column [energy, absorption] arrays expected by the comparator.
        ml_spectrum = np.column_stack((ene, predicted_spectrum))
        expt_spectrum = np.column_stack((
            np.array(exp_spectrum["energy"]),
            np.array(exp_spectrum["absorption"]),
        ))

        # --- Comparison hyper-parameters ---
        opt_metric = "coss_deriv"                # Metric to optimise the shift for
        other_metrics = ALL_METRICS               # All metrics to report
        erange = 35                               # Comparison window width (eV)
        erange_threshold = 0.04                   # Normalised-intensity edge threshold
        truncation_strategy = "from_spect2"       # Truncate from the ML spectrum side
        erange_lbound_delta = 5                   # Lower-bound offset from the edge

        correlations, shift = compare_utils.compare_between_spectra(
            expt_spectrum,
            ml_spectrum,
            erange=erange,
            erange_threshold=erange_threshold,
            erange_lbound_delta=erange_lbound_delta,
            truncation_strategy=truncation_strategy,
            grid_interpolator=compare_utils.gridInterpolatorFixedSpacing(0.25),
            output_correlations=other_metrics,
            opt_strategy="grid_search_and_local_opt",
            accuracy=0.1,
            method=opt_metric,
            norm_y_axis=True,
        )

        # --- Derive the comparison energy range in the *experimental* frame ---
        ml_y_norm = (ml_spectrum[:, 1] - ml_spectrum[:, 1].min()) / (
            ml_spectrum[:, 1].max() - ml_spectrum[:, 1].min()
        )
        ml_edge_idx = np.argmax(ml_y_norm > erange_threshold)
        ml_edge_energy = ml_spectrum[ml_edge_idx, 0]
        comparison_start = ml_edge_energy + shift
        comparison_end = comparison_start + erange

        print(
            f"=== Comparison Range Debug ===\n"
            f"  ML edge energy : {ml_edge_energy:.1f} eV\n"
            f"  Optimal shift  : {shift:.2f} eV\n"
            f"  Comparison     : {comparison_start:.1f} – {comparison_end:.1f} eV"
        )

        # Sanitise the primary score.
        score = correlations.get(opt_metric, 0.0)
        if np.isnan(score) or np.isinf(score):
            score = 0.0

        return {
            "score": round(float(score), 3),
            "correlations": {
                k: round(float(v), 3) if not (np.isnan(v) or np.isinf(v)) else 0.0
                for k, v in correlations.items()
            },
            "shift": round(float(shift), 2),
            "comparison_range": (
                round(float(comparison_start), 1),
                round(float(comparison_end), 1),
            ),
        }

    except Exception as e:
        print(f"Error in spectrum matching: {e}")
        import traceback
        traceback.print_exc()
        return {
            "score": 0.0,
            "correlations": {},
            "shift": 0.0,
            "comparison_range": None,
        }


# =============================================================================
# §6  Client-Side Data Stores (dcc.Store)
# =============================================================================
# ``dcc.Store`` components hold JSON-serialisable data on the *client* side
# (browser).  They let us pass state between Dash callbacks without global
# variables.  Each store is referenced by its ``id`` in callback I/O lists.

# Scores & ranking for all structures that have been compared to the
# experimental spectrum.  Each entry is a dict with keys like
# ``structure_id``, ``score``, ``correlations``, ``spectrum``, etc.
structure_scores_store = dcc.Store(id="structure_scores_store", data=[])

# The energy window (eV) over which the most recent comparison was made.
comparison_range_store = dcc.Store(id="comparison_range_store", data=None)

# List of spectra the user has ticked for overlay in the plot.
selected_spectra_store = dcc.Store(id="selected_spectra_store", data=[])

# Which metric column is currently used to sort the ranking table.
sort_metric_store = dcc.Store(id="sort_metric_store", data="coss_deriv")

# Batch-upload progress tracker (not actively used for polling, but available).
batch_processing_store = dcc.Store(
    id="batch_processing_store",
    data={"status": "idle", "processed": 0, "total": 0},
)

# Energy shift value (eV) applied to the predicted spectrum in the plot.
energy_shift_store = dcc.Store(id="energy_shift_store", data=0)

# Experimental-spectrum pipeline stores:
#   raw_data  → column info + arrays right after file parsing
#   columns   → column metadata (possibly with user-edited names)
#   spectrum  → final {energy, absorption} dict ready for comparison
exp_raw_data_store = dcc.Store(id="exp_raw_data_store", data=None)
exp_columns_store = dcc.Store(id="exp_columns_store", data=None)
exp_spectrum_store = dcc.Store(id="exp_spectrum_store", data=None)


# =============================================================================
# §7  Experimental-Spectrum Upload Components
# =============================================================================
# These Dash components form the left-panel UI for uploading and configuring
# an experimental reference spectrum.

# --- Drag-and-drop file uploader for experimental data ------------------------
exp_upload_component = dcc.Upload(
    id="exp_spectrum_upload",
    children=html.Div([
        html.Div([
            "Drag and Drop or ",
            html.A("Select File", style={
                "color": "#333", "cursor": "pointer",
                "fontWeight": "500", "textDecoration": "underline",
            }),
        ]),
    ]),
    style={
        "width": "100%", "height": "50px", "lineHeight": "50px",
        "borderWidth": "1px", "borderStyle": "dashed", "borderColor": "#d0d0d0",
        "borderRadius": "6px", "textAlign": "center", "backgroundColor": "#fafafa",
        "cursor": "pointer", "color": "#666", "fontSize": "12px",
    },
    multiple=False,
    accept=".dat,.mat,.csv,.xdi",
)

# --- Material-name text input (optional label for the experimental curve) -----
exp_material_name_input = dcc.Input(
    id="exp_material_name",
    type="text",
    placeholder="e.g., Anatase TiO2",
    style={
        "width": "100%", "padding": "10px 12px", "borderRadius": "6px",
        "border": "1px solid #ddd", "fontSize": "12px", "boxSizing": "border-box",
    },
)

# --- Column-selection dropdowns (populated after file upload) -----------------
exp_x_axis_dropdown = dcc.Dropdown(
    id="exp_x_axis_dropdown", options=[], placeholder="Select X-axis column",
    style={"marginBottom": "8px"},
)
exp_y_axis_dropdown = dcc.Dropdown(
    id="exp_y_axis_dropdown", options=[], placeholder="Select Y-axis column",
    style={"marginBottom": "8px"},
)

# --- Action buttons -----------------------------------------------------------
exp_apply_btn = html.Button("Apply & Plot", id="exp_apply_btn", style={
    "padding": "8px 16px", "fontSize": "12px", "border": "none",
    "borderRadius": "6px", "backgroundColor": "#333", "color": "white",
    "cursor": "pointer", "fontWeight": "500", "marginRight": "8px",
})
clear_exp_btn = html.Button("Clear", id="clear_exp_btn", style={
    "fontSize": "12px", "padding": "8px 16px", "border": "1px solid #ddd",
    "borderRadius": "6px", "backgroundColor": "white", "color": "#666",
    "cursor": "pointer",
})

# --- Dynamic areas that update after file upload ------------------------------
exp_column_definition_area = html.Div(id="exp_column_definition_area", children=[],
                                       style={"marginTop": "10px"})
exp_file_info = html.Div(id="exp_file_info", children="No experimental spectrum loaded",
                          style={"fontSize": "11px", "color": "#888", "marginTop": "10px"})

# --- Combined single / multiple structure upload (drag-and-drop) --------------
batch_upload_component = dcc.Upload(
    id="batch_structure_upload",
    children=html.Div([html.Div([
        "Drag & Drop or ",
        html.A("Select File(s)", style={
            "color": "#333", "cursor": "pointer",
            "fontWeight": "500", "textDecoration": "underline",
        }),
    ])]),
    style={
        "width": "100%", "height": "50px", "lineHeight": "50px",
        "borderWidth": "1px", "borderStyle": "dashed", "borderColor": "#d0d0d0",
        "borderRadius": "6px", "textAlign": "center", "backgroundColor": "#fafafa",
        "cursor": "pointer", "color": "#666", "fontSize": "12px",
    },
    multiple=True,                               # Accept one or many files
    accept=".cif,.vasp,.poscar,.json",
)

# --- Absorber / model selector ------------------------------------------------
absorber_dropdown = dcc.Dropdown(
    XAS_MODEL_NAMES, clearable=False, value="Ti VASP", id="absorber",
)

# --- The main spectrum plot ----------------------------------------------------
xas_plot = dcc.Graph(id="xas_plot")

# --- Small heading showing which structure is currently loaded -----------------
st_source = html.H1(id="st_source", children="No structure loaded yet")


# =============================================================================
# §8  Shared Style Definitions
# =============================================================================
# Centralised style dicts prevent repetition and make theming easier.

BASE_FONT = (
    "system-ui, -apple-system, BlinkMacSystemFont, "
    "'Segoe UI', Roboto, sans-serif"
)

SECTION_HEADER_STYLE = {
    "fontWeight": "600", "fontSize": "13px", "color": "#333",
    "marginBottom": "14px", "paddingBottom": "10px",
    "borderBottom": "1px solid #eee", "fontFamily": BASE_FONT,
    "letterSpacing": "0.2px",
}

COLUMN_HEADER_STYLE = {**SECTION_HEADER_STYLE}  # Same as section header for now.

INPUT_LABEL_STYLE = {
    "fontSize": "12px", "color": "#666", "marginBottom": "6px",
    "fontWeight": "500", "fontFamily": BASE_FONT,
}

CARD_STYLE = {
    "backgroundColor": "white", "borderRadius": "8px", "padding": "18px",
    "marginBottom": "12px", "border": "1px solid #e8e8e8",
}

BUTTON_PRIMARY_STYLE = {
    "padding": "10px 20px", "fontSize": "13px", "border": "none",
    "borderRadius": "6px", "backgroundColor": "#333", "color": "white",
    "cursor": "pointer", "fontWeight": "500", "marginRight": "8px",
    "fontFamily": BASE_FONT,
}

BUTTON_SECONDARY_STYLE = {
    "padding": "8px 16px", "fontSize": "12px", "border": "1px solid #ddd",
    "borderRadius": "6px", "backgroundColor": "white", "color": "#666",
    "cursor": "pointer", "fontFamily": BASE_FONT,
}


# =============================================================================
# §9  Page Layout
# =============================================================================
# The page is a three-column layout built with Crystal Toolkit's ``Columns``
# helper (which wraps Bulma CSS grid):
#
#   Column 1 (narrow) — Input controls:  experimental upload, structure search,
#                        file upload, model selector.
#   Column 2           — 3-D crystal structure viewer.
#   Column 3           — Spectrum plot, energy-shift slider, download button,
#                        and the structure-ranking table.

omnixas_layout = html.Div([
    Columns([
        # ── Column 1: Input Controls ─────────────────────────────────────
        Column(
            html.Div([
                # Card: Experimental Spectrum Upload
                html.Div([
                    html.Div("Upload Experimental Spectrum", style=SECTION_HEADER_STYLE),
                    html.Div("Material Name (optional):", style=INPUT_LABEL_STYLE),
                    exp_material_name_input,
                    html.Div("Accepted formats: .csv, .dat, .mat, .xdi",
                             style={"fontSize": "11px", "color": "#999",
                                    "marginTop": "10px", "marginBottom": "8px"}),
                    exp_upload_component,
                    exp_column_definition_area,
                    # Column-selection controls (shown after file is parsed)
                    html.Div(
                        id="exp_column_selection_area",
                        children=[
                            html.Div("Select columns to plot:",
                                     style={**INPUT_LABEL_STYLE, "marginTop": "12px"}),
                            html.Div([
                                html.Div([
                                    html.Span("X-axis:", style={"fontSize": "11px",
                                              "display": "block", "marginBottom": "4px", "color": "#666"}),
                                    exp_x_axis_dropdown,
                                ], style={"display": "inline-block", "width": "48%",
                                          "marginRight": "4%", "verticalAlign": "top"}),
                                html.Div([
                                    html.Span("Y-axis:", style={"fontSize": "11px",
                                              "display": "block", "marginBottom": "4px", "color": "#666"}),
                                    exp_y_axis_dropdown,
                                ], style={"display": "inline-block", "width": "48%",
                                          "verticalAlign": "top"}),
                            ]),
                            html.Div([exp_apply_btn, clear_exp_btn],
                                     style={"marginTop": "12px"}),
                        ],
                        style={"display": "none"},       # Hidden until file is loaded
                    ),
                    exp_file_info,
                    exp_raw_data_store,
                    exp_columns_store,
                    exp_spectrum_store,
                ], style=CARD_STYLE),

                # Card: Load Structure
                html.Div([
                    html.Div("Load Structure", style=SECTION_HEADER_STYLE),
                    html.Div("Search by Materials Project ID:",
                             style={**INPUT_LABEL_STYLE, "marginBottom": "8px"}),
                    Loading(search_component.layout()),
                    html.Hr(style={"margin": "15px 0", "border": "none",
                                   "borderTop": "1px solid #eee"}),
                    html.Div("Upload structure file(s):",
                             style={**INPUT_LABEL_STYLE, "marginBottom": "4px"}),
                    html.Div("Single or multiple files • Supported: .cif, .vasp, .poscar, .json",
                             style={"fontSize": "10px", "color": "#999", "marginBottom": "8px"}),
                    batch_upload_component,
                    batch_processing_store,
                    html.Div(id="batch_status", children="",
                             style={"fontSize": "11px", "color": "#666", "marginTop": "8px",
                                    "fontFamily": BASE_FONT}),
                    html.Div(st_source, style={"marginTop": "10px"}),
                ], style=CARD_STYLE),

                # Card: XAS Model Prediction
                html.Div([
                    html.Div("XAS Model Prediction", style=SECTION_HEADER_STYLE),
                    Loading(absorber_dropdown),
                ], style=CARD_STYLE),
            ], style={"width": "100%"}),
            narrow=True,
        ),

        # ── Column 2: Crystal Structure Viewer ────────────────────────────
        Column(
            html.Div([html.Div([
                html.Div("Crystal Structure Viewer", style=COLUMN_HEADER_STYLE),
                Loading(struct_component.layout(size="100%")),
            ], style={
                "backgroundColor": "white", "borderRadius": "8px",
                "padding": "18px", "border": "1px solid #e8e8e8",
                "minHeight": "500px",
            })]),
            style={"flex": "1", "minWidth": "400px", "padding": "0 6px"},
        ),

        # ── Column 3: Spectrum Analysis ───────────────────────────────────
        Column(
            html.Div([html.Div([
                html.Div("XANES Spectrum Analysis", style=COLUMN_HEADER_STYLE),
                xas_plot,

                # Energy-shift slider
                html.Div([
                    html.Div([
                        html.Span("Shift Predicted Spectrum: ",
                                  style={"fontSize": "12px", "color": "#666",
                                         "fontFamily": BASE_FONT}),
                        html.Span(id="energy_shift_display", children="0.0 eV",
                                  style={"fontSize": "12px", "fontWeight": "600",
                                         "color": "#333", "fontFamily": BASE_FONT}),
                    ], style={"marginTop": "15px", "marginBottom": "8px"}),
                    dcc.Slider(
                        id="energy_shift_slider",
                        min=-50, max=50, step=0.01, value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="drag", included=False,
                    ),
                    html.Div([
                        html.Span("-50 eV", style={"fontSize": "10px", "color": "#999"}),
                        html.Span("0", style={"fontSize": "10px", "color": "#999",
                                              "position": "absolute", "left": "50%",
                                              "transform": "translateX(-50%)"}),
                        html.Span("+50 eV", style={"fontSize": "10px", "color": "#999"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "position": "relative", "marginTop": "-5px"}),
                    html.Button("Reset Shift", id="reset_shift_btn",
                                style={**BUTTON_SECONDARY_STYLE, "marginTop": "10px",
                                       "fontSize": "11px", "padding": "6px 14px"}),
                ], id="energy_shift_container", style={"padding": "0 10px"}),

                html.Hr(style={"margin": "20px 0", "border": "none",
                                "borderTop": "1px solid #eee"}),

                # Download button
                html.Button("Download POSCAR and Spectrum", id="download_btn",
                            style={**BUTTON_PRIMARY_STYLE, "width": "100%",
                                   "padding": "12px", "fontSize": "12px",
                                   "marginRight": "0", "borderRadius": "6px"}),
                dcc.Download(id="download_sink"),

                # Matching-results section
                html.Div([
                    html.Div([
                        html.Span("Structure Matching Scores", style={
                            "fontWeight": "600", "fontSize": "13px", "color": "#333"}),
                        html.Button("Clear All", id="clear_scores_btn", style={
                            "fontSize": "10px", "padding": "4px 10px",
                            "border": "1px solid #ddd", "borderRadius": "4px",
                            "backgroundColor": "white", "color": "#666",
                            "cursor": "pointer", "marginLeft": "10px"}),
                    ], style={
                        "display": "flex", "alignItems": "center",
                        "justifyContent": "space-between", "marginTop": "20px",
                        "marginBottom": "12px", "paddingBottom": "10px",
                        "borderBottom": "1px solid #eee",
                    }),
                    html.Div(id="matching_results_table", children=[
                        html.Div(
                            "Upload experimental spectrum and load structures "
                            "to see matching scores",
                            style={"color": "#999", "fontSize": "12px",
                                   "textAlign": "center", "padding": "20px"}),
                    ]),
                    structure_scores_store,
                    comparison_range_store,
                    selected_spectra_store,
                    sort_metric_store,
                ]),
            ], style={
                "backgroundColor": "white", "borderRadius": "8px",
                "padding": "18px", "border": "1px solid #e8e8e8",
            })]),
            style={"flex": "1", "minWidth": "400px", "padding": "0 6px"},
        ),
    ], desktop_only=False, centered=False),
], style={
    "backgroundColor": "#f5f5f5", "minHeight": "100vh",
    "padding": "12px", "fontFamily": BASE_FONT,
})


# =============================================================================
# §10  File-Parsing Utilities
# =============================================================================

def parse_file_columns(contents, filename):
    """Parse an uploaded experimental data file and return column metadata + data.

    Supports:
      - **.csv / .dat / .txt / .xdi** — whitespace- or comma-delimited text,
        with optional ``# Column.N: name`` XDI-style headers.
      - **.mat** — MATLAB ``.mat`` files (via ``scipy.io.loadmat``).

    Parameters
    ----------
    contents : str
        Base64-encoded file contents in the Dash upload format
        (``"data:<mime>;base64,<payload>"``).
    filename : str
        Original filename (used to determine the extension).

    Returns
    -------
    dict or None
        On success: ``{'columns': [...], 'data': [[...], ...],
        'filename': str, 'auto_x_col': int, 'auto_y_col': int}``.
        On failure: ``{'error': str}``.
    """
    if contents is None:
        return None

    _content_type, content_string = contents.split(",")
    decoded = b64decode(content_string)

    try:
        filename = filename or "unknown.dat"
        ext = pathlib.Path(filename).suffix.lower()
        print(f"=== DEBUG: Parsing file '{filename}' with extension '{ext}'")

        columns = []
        data = []
        auto_x_col = 0
        auto_y_col = 1

        if ext in (".csv", ".dat", ".txt", ".xdi"):
            columns, data, auto_x_col, auto_y_col = _parse_text_file(decoded)
        elif ext == ".mat":
            columns, data, auto_x_col, auto_y_col = _parse_mat_file(decoded)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if len(columns) < 2:
            raise ValueError("File must have at least 2 columns for X and Y axes")

        # Clamp auto-selections to valid column indices.
        auto_x_col = min(auto_x_col, len(columns) - 1)
        auto_y_col = min(auto_y_col, len(columns) - 1)
        if auto_x_col == auto_y_col and len(columns) > 1:
            auto_y_col = 1 if auto_x_col == 0 else 0

        print(f"=== DEBUG: Found {len(columns)} columns")
        for col in columns:
            print(f"  Column {col['index']}: {col['name']} ({col['num_values']} values)")
        print(f"=== DEBUG: Auto-selected X={auto_x_col}, Y={auto_y_col}")

        return {
            "columns": columns,
            "data": data,
            "filename": filename,
            "auto_x_col": auto_x_col,
            "auto_y_col": auto_y_col,
        }

    except Exception as e:
        print(f"Error parsing file columns: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def _parse_text_file(decoded_bytes):
    """Parse a text-based experimental data file (CSV / DAT / XDI / TXT).

    Returns
    -------
    tuple
        ``(columns, data, auto_x_col, auto_y_col)``
    """
    text = decoded_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    comment_lines = []
    data_lines = []
    for line in lines:
        if line.startswith(("#", "%", "!")):
            comment_lines.append(line)
        else:
            data_lines.append(line)

    if not data_lines:
        raise ValueError("No data lines found in file")

    # --- Extract XDI-style column headers (``# Column.N: name``) -----------
    xdi_columns = {}
    energy_col_candidates = []
    absorption_col_candidates = []

    for comment in comment_lines:
        xdi_match = re.match(r"#\s*Column\.(\d+):\s*(.+)", comment, re.IGNORECASE)
        if xdi_match:
            col_num = int(xdi_match.group(1)) - 1       # XDI is 1-indexed
            col_name = xdi_match.group(2).strip()
            xdi_columns[col_num] = col_name
            print(f"=== DEBUG: Found XDI column {col_num}: '{col_name}'")

            col_lower = col_name.lower()
            if any(t in col_lower for t in ("energy", " e ", "ev", "photon")):
                energy_col_candidates.append(col_num)
            if any(t in col_lower for t in ("norm", "absorption", "abs", "mu", "flat")):
                absorption_col_candidates.append(col_num)

    # Fallback: try to interpret the last comment line as a header row.
    if comment_lines and not xdi_columns:
        header_text = comment_lines[-1].lstrip("#").strip()
        header_parts = header_text.split()
        if len(header_parts) >= 2 and ":" not in header_text:
            for i, name in enumerate(header_parts):
                xdi_columns[i] = name
                name_lower = name.lower()
                if name_lower in ("e", "energy", "ev"):
                    energy_col_candidates.append(i)
                if name_lower in ("norm", "flat", "abs", "mu", "absorption"):
                    absorption_col_candidates.append(i)

    # --- Detect delimiter & header row --------------------------------------
    first_line = data_lines[0]
    delimiter = "," if "," in first_line else None
    first_parts = first_line.split(delimiter) if delimiter else first_line.split()
    num_columns = len(first_parts)

    try:
        float(first_parts[0].strip())
        header = None
        start_idx = 0
    except ValueError:
        header = [p.strip() for p in first_parts]
        start_idx = 1
        if not xdi_columns:
            for i, name in enumerate(header):
                xdi_columns[i] = name

    # --- Read numeric data into column-major lists --------------------------
    data = [[] for _ in range(num_columns)]
    for line in data_lines[start_idx:]:
        parts = line.split(delimiter) if delimiter else line.split()
        for i, part in enumerate(parts):
            if i < num_columns:
                try:
                    data[i].append(float(part.strip()))
                except ValueError:
                    pass

    # --- Build column metadata list -----------------------------------------
    columns = []
    for i in range(num_columns):
        if i in xdi_columns:
            col_name = xdi_columns[i]
        elif header and i < len(header):
            col_name = header[i]
        else:
            col_name = f"Column {i + 1}"
        sample_values = data[i][:5] if len(data[i]) >= 5 else data[i]
        columns.append({
            "index": i, "name": col_name,
            "num_values": len(data[i]), "sample_values": sample_values,
        })

    # --- Determine best auto-selected X / Y columns ------------------------
    auto_x_col = energy_col_candidates[0] if energy_col_candidates else 0
    auto_y_col = 1  # default

    if absorption_col_candidates:
        # Prefer 'norm' or 'flat' columns for the Y axis.
        for candidate in absorption_col_candidates:
            col_name = xdi_columns.get(candidate, "").lower()
            if "norm" in col_name or "flat" in col_name:
                auto_y_col = candidate
                break
        else:
            auto_y_col = absorption_col_candidates[0]

    return columns, data, auto_x_col, auto_y_col


def _parse_mat_file(decoded_bytes):
    """Parse a MATLAB ``.mat`` file.

    Returns
    -------
    tuple
        ``(columns, data, auto_x_col, auto_y_col)``
    """
    from scipy.io import loadmat

    mat_data = loadmat(io.BytesIO(decoded_bytes))
    data_keys = [k for k in mat_data.keys() if not k.startswith("__")]

    columns = []
    data = []
    auto_x_col = 0
    auto_y_col = 1

    for i, key in enumerate(data_keys):
        arr = mat_data[key]
        if isinstance(arr, np.ndarray) and arr.size > 1:
            flat_arr = arr.flatten().astype(float).tolist()
            sample_values = flat_arr[:5] if len(flat_arr) >= 5 else flat_arr
            columns.append({
                "index": i, "name": key,
                "num_values": len(flat_arr), "sample_values": sample_values,
            })
            data.append(flat_arr)

            key_lower = key.lower()
            if any(t in key_lower for t in ("energy", "e", "ev")):
                auto_x_col = i
            if any(t in key_lower for t in ("absorption", "abs", "mu", "norm")):
                auto_y_col = i

    return columns, data, auto_x_col, auto_y_col


def parse_structure_file(contents, filename):
    """Parse an uploaded crystal-structure file into a pymatgen ``Structure``.

    Supports CIF, VASP / POSCAR, and pymatgen JSON formats.  Falls back to
    auto-detection if the extension is unrecognised.

    Parameters
    ----------
    contents : str
        Base64-encoded Dash upload string.
    filename : str
        Original filename.

    Returns
    -------
    Structure or None
        The parsed structure, or ``None`` on failure.
    """
    try:
        _content_type, content_string = contents.split(",")
        decoded = b64decode(content_string)
        text = decoded.decode("utf-8")
        ext = pathlib.Path(filename).suffix.lower()

        if ext == ".cif":
            from pymatgen.io.cif import CifParser
            return CifParser.from_str(text).parse_structures()[0]

        if ext in (".vasp", ".poscar", ""):
            from pymatgen.io.vasp import Poscar
            return Poscar.from_str(text).structure

        if ext == ".json":
            return Structure.from_dict(json.loads(text))

        # Unknown extension — try CIF, then POSCAR.
        try:
            from pymatgen.io.cif import CifParser
            return CifParser.from_str(text).parse_structures()[0]
        except Exception:
            from pymatgen.io.vasp import Poscar
            return Poscar.from_str(text).structure

    except Exception as e:
        print(f"Error parsing structure file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# §11  Dash Callbacks — Experimental Spectrum Workflow
# =============================================================================

@app.callback(
    # Outputs: populate stores, dropdowns, UI visibility, file info text.
    Output("exp_raw_data_store", "data"),
    Output("exp_columns_store", "data"),
    Output("exp_x_axis_dropdown", "options"),
    Output("exp_y_axis_dropdown", "options"),
    Output("exp_x_axis_dropdown", "value"),
    Output("exp_y_axis_dropdown", "value"),
    Output("exp_column_selection_area", "style"),
    Output("exp_column_definition_area", "children"),
    Output("exp_file_info", "children", allow_duplicate=True),
    Output("exp_spectrum_upload", "contents"),
    Output("exp_spectrum_upload", "filename"),
    Output("exp_material_name", "value"),
    # Inputs / state
    Input("exp_spectrum_upload", "contents"),
    Input("clear_exp_btn", "n_clicks"),
    State("exp_spectrum_upload", "filename"),
    prevent_initial_call=True,
)
def handle_file_upload(contents, clear_clicks, filename):
    """Parse an uploaded experimental spectrum file and show column-selection UI.

    Triggered by either a new file upload or the "Clear" button.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    hidden_style = {"display": "none"}
    visible_style = {"display": "block"}

    # --- Clear button pressed → reset everything ---
    if trigger_id == "clear_exp_btn":
        return (
            None, None, [], [], None, None,
            hidden_style, [],
            "No experimental spectrum loaded",
            None, None, "",
        )

    if contents is None:
        raise PreventUpdate

    # --- Parse the file ---
    result = parse_file_columns(contents, filename)

    if result is None or "error" in result:
        error_msg = result.get("error", "Failed to parse file") if result else "Failed to parse file"
        return (
            None, None, [], [], None, None,
            hidden_style, [],
            html.Span(f"Error: {error_msg}", style={"color": "red"}),
            dash.no_update, dash.no_update, dash.no_update,
        )

    columns = result["columns"]
    options = [
        {"label": f"{col['name']} ({col['num_values']} pts)", "value": col["index"]}
        for col in columns
    ]
    default_x = result.get("auto_x_col", 0)
    default_y = result.get("auto_y_col", 1 if len(columns) > 1 else 0)

    # --- Build the editable column-name table ---
    max_visible_rows = 5
    table_height = "auto" if len(columns) <= max_visible_rows else f"{max_visible_rows * 40 + 30}px"

    col_definition = html.Div([
        html.Div(f"Detected {len(columns)} columns (edit names if needed):",
                 style={"fontSize": "12px", "marginBottom": "6px", "marginTop": "10px"}),
        html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th("#",           style={"padding": "4px 8px", "fontSize": "11px", "width": "30px",
                                                   "position": "sticky", "top": "0", "backgroundColor": "#fafafa", "zIndex": "1"}),
                    html.Th("Column Name", style={"padding": "4px 8px", "fontSize": "11px",
                                                   "position": "sticky", "top": "0", "backgroundColor": "#fafafa", "zIndex": "1"}),
                    html.Th("Points",      style={"padding": "4px 8px", "fontSize": "11px", "width": "50px",
                                                   "position": "sticky", "top": "0", "backgroundColor": "#fafafa", "zIndex": "1"}),
                    html.Th("Sample Values", style={"padding": "4px 8px", "fontSize": "11px",
                                                     "position": "sticky", "top": "0", "backgroundColor": "#fafafa", "zIndex": "1"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(col["index"] + 1, style={"padding": "4px 8px", "fontSize": "11px", "verticalAlign": "middle"}),
                        html.Td(dcc.Input(
                            id={"type": "col-name-input", "index": col["index"]},
                            type="text", value=col["name"],
                            style={"width": "100%", "padding": "4px", "fontSize": "11px",
                                   "border": "1px solid #ccc", "borderRadius": "3px"},
                        ), style={"padding": "4px"}),
                        html.Td(col["num_values"], style={"padding": "4px 8px", "fontSize": "11px", "verticalAlign": "middle"}),
                        html.Td(
                            ", ".join(f"{v:.2f}" for v in col["sample_values"][:3]) + "...",
                            style={"padding": "4px 8px", "fontSize": "10px", "color": "#666", "verticalAlign": "middle"},
                        ),
                    ]) for col in columns
                ]),
            ], style={"borderCollapse": "collapse", "width": "100%"}),
        ], style={"maxHeight": table_height,
                  "overflowY": "auto" if len(columns) > max_visible_rows else "visible",
                  "border": "1px solid #ddd", "marginBottom": "10px"}),
        html.Button("Update Column Names", id="exp_update_col_names_btn",
                     style={"fontSize": "11px", "padding": "4px 8px", "marginBottom": "10px"}),
    ])

    x_name = columns[default_x]["name"] if default_x < len(columns) else "Column 1"
    y_name = columns[default_y]["name"] if default_y < len(columns) else "Column 2"
    info_text = f"File loaded: {filename} (auto-selected: X={x_name}, Y={y_name})"
    material_name_from_file = pathlib.Path(filename).stem if filename else ""

    return (
        result, columns, options, options, default_x, default_y,
        visible_style, col_definition,
        html.Span(info_text, style={"color": "blue"}),
        dash.no_update, dash.no_update, material_name_from_file,
    )


@app.callback(
    Output("exp_columns_store", "data", allow_duplicate=True),
    Output("exp_x_axis_dropdown", "options", allow_duplicate=True),
    Output("exp_y_axis_dropdown", "options", allow_duplicate=True),
    Output("exp_file_info", "children", allow_duplicate=True),
    Input("exp_update_col_names_btn", "n_clicks"),
    State({"type": "col-name-input", "index": ALL}, "value"),
    State("exp_columns_store", "data"),
    prevent_initial_call=True,
)
def update_column_names(n_clicks, new_names, columns):
    """Persist user-edited column names and refresh the dropdown labels."""
    if n_clicks is None or columns is None:
        raise PreventUpdate

    for i, new_name in enumerate(new_names):
        if i < len(columns):
            columns[i]["name"] = new_name.strip() if new_name else f"Column {i + 1}"

    options = [
        {"label": f"{col['name']} ({col['num_values']} pts)", "value": col["index"]}
        for col in columns
    ]
    return columns, options, options, html.Span("Column names updated!", style={"color": "green"})


@app.callback(
    Output("exp_spectrum_store", "data"),
    Output("exp_file_info", "children", allow_duplicate=True),
    Input("exp_apply_btn", "n_clicks"),
    State("exp_raw_data_store", "data"),
    State("exp_columns_store", "data"),
    State("exp_x_axis_dropdown", "value"),
    State("exp_y_axis_dropdown", "value"),
    State("exp_material_name", "value"),
    prevent_initial_call=True,
)
def apply_column_selection(n_clicks, raw_data, columns, x_col_idx, y_col_idx, material_name):
    """Finalise the experimental spectrum from the selected X/Y columns.

    Sorts by energy, stores the result in ``exp_spectrum_store``, and updates
    the file-info label.
    """
    if n_clicks is None or raw_data is None:
        raise PreventUpdate
    if x_col_idx is None or y_col_idx is None:
        return None, html.Span("Please select both X and Y axis columns", style={"color": "red"})

    try:
        data = raw_data["data"]
        filename = raw_data["filename"]
        x_data = np.array(data[x_col_idx])
        y_data = np.array(data[y_col_idx])

        min_len = min(len(x_data), len(y_data))
        x_data, y_data = x_data[:min_len], y_data[:min_len]
        if len(x_data) < 2:
            return None, html.Span("Not enough data points", style={"color": "red"})

        # Sort by ascending energy.
        sort_idx = np.argsort(x_data)
        x_data, y_data = x_data[sort_idx], y_data[sort_idx]

        x_label = columns[x_col_idx]["name"]
        y_label = columns[y_col_idx]["name"]
        display_name = material_name.strip() if material_name and material_name.strip() else filename

        result = {
            "energy": x_data.tolist(),
            "absorption": y_data.tolist(),
            "filename": filename,
            "material_name": display_name,
            "x_label": x_label,
            "y_label": y_label,
        }
        info_text = f"✓ {display_name} ({len(x_data)} points, {x_label}: {x_data.min():.1f}-{x_data.max():.1f})"
        return result, html.Span(info_text, style={"color": "green"})

    except Exception as e:
        print(f"Error applying column selection: {e}")
        return None, html.Span(f"Error: {e}", style={"color": "red"})


# =============================================================================
# §12  Dash Callbacks — Structure Loading (MP Search / Single File Upload)
# =============================================================================

def decorate_structure_with_xas(st: Structure, el_type: str) -> dict:
    """Add predicted XAS spectra to a pymatgen Structure dict.

    If the structure contains the absorbing element, calls the ML prediction
    model and stores the result under the ``'xas'`` key.

    Parameters
    ----------
    st : Structure
        The crystal structure.
    el_type : str
        E.g. ``'Ti VASP'`` — first token is the element, second is the code.

    Returns
    -------
    dict
        ``st.as_dict()`` augmented with an ``'xas'`` key mapping
        site-index strings to predicted absorption arrays.
    """
    absorbing_element, spectroscopy_type = el_type.split(" ")
    st_dict = st.as_dict()

    if absorbing_element in st.composition:
        specs = predict(st, absorbing_element, spectroscopy_type)
        st_dict["xas"] = specs
    else:
        st_dict["xas"] = {}

    return st_dict


@app.callback(
    Output(struct_component.id(), "data", allow_duplicate=True),
    Output(upload_component.id("upload_data"), "contents"),
    Output("st_source", "children", allow_duplicate=True),
    Input(search_component.id(), "data"),
    State("absorber", "value"),
)
def update_structure_by_mpid(search_mpid: str, el_type):
    """Fetch a structure from the Materials Project by MP-ID and predict XAS."""
    if not search_mpid:
        raise PreventUpdate

    with MPRester() as mpr:
        st = mpr.get_structure_by_material_id(search_mpid)
        if not isinstance(st, Structure):
            raise Exception(
                "mp_api MPRester.get_structure_by_material_id did not return "
                "a pymatgen Structure object."
            )

    st_dict = decorate_structure_with_xas(st, el_type)
    return st_dict, None, f"Current structure: {search_mpid}"


@app.callback(
    Output(struct_component.id(), "data", allow_duplicate=True),
    Output("st_source", "children", allow_duplicate=True),
    Input(upload_component.id(), "data"),
    State(upload_component.id("upload_data"), "filename"),
    State("absorber", "value"),
)
def update_structure_by_file(upload_data: dict, fn, el_type):
    """Handle single-file upload via Crystal Toolkit's built-in uploader."""
    if not upload_data:
        raise PreventUpdate
    st = Structure.from_dict(upload_data["data"])
    st_dict = decorate_structure_with_xas(st, el_type)
    return st_dict, f"Current structure: {fn}"


@app.callback(
    Output(struct_component.id(), "data", allow_duplicate=True),
    Input("absorber", "value"),
    State(struct_component.id(), "data"),
)
def update_structure_by_absorber(el_type, st_data):
    """Re-predict XAS when the user changes the absorber / model dropdown."""
    if st_data is None:
        raise PreventUpdate
    st = Structure.from_dict(st_data)
    return decorate_structure_with_xas(st, el_type)


# =============================================================================
# §13  Dash Callbacks — Batch Upload & Scoring
# =============================================================================

@app.callback(
    Output("structure_scores_store", "data", allow_duplicate=True),
    Output("matching_results_table", "children", allow_duplicate=True),
    Output("comparison_range_store", "data", allow_duplicate=True),
    Output("batch_status", "children"),
    Output("batch_structure_upload", "contents"),
    Output(struct_component.id(), "data", allow_duplicate=True),
    Output("st_source", "children", allow_duplicate=True),
    # Inputs / state
    Input("batch_structure_upload", "contents"),
    State("batch_structure_upload", "filename"),
    State("exp_spectrum_store", "data"),
    State("absorber", "value"),
    State("structure_scores_store", "data"),
    State("sort_metric_store", "data"),
    prevent_initial_call=True,
)
def handle_batch_upload(contents_list, filenames_list, exp_data, el_type,
                        existing_scores, sort_metric):
    """Process one or more uploaded structure files: predict XAS, score vs. experiment.

    For each file:
      1. Parse the crystal structure.
      2. Check it contains the absorbing element.
      3. Run the ML prediction.
      4. Compare the average predicted spectrum to the experimental spectrum
         (if available).
      5. Append the result to the ranking table.

    The last successfully parsed structure is displayed in the 3-D viewer.
    """
    if contents_list is None or len(contents_list) == 0:
        raise PreventUpdate

    existing_scores = existing_scores or []
    sort_metric = sort_metric or "coss_deriv"
    has_exp_data = (
        exp_data is not None
        and "energy" in exp_data
        and "absorption" in exp_data
    )

    element, theory = el_type.split(" ")

    successful = 0
    failed = 0
    failed_files = []
    last_st_dict = None
    last_filename = None
    comparison_range = None

    for contents, filename in zip(contents_list, filenames_list):
        try:
            st = parse_structure_file(contents, filename)
            if st is None:
                failed += 1
                failed_files.append(filename)
                continue

            if element not in st.composition:
                print(f"Structure {filename} does not contain {element}, skipping...")
                failed += 1
                failed_files.append(f"{filename} (no {element})")
                continue

            # Predict XAS for every absorbing site.
            specs = predict(st, element, theory)
            if len(specs) == 0:
                failed += 1
                failed_files.append(f"{filename} (no spectrum)")
                continue

            # Average over all equivalent absorbing sites.
            specs_array = np.array(list(specs.values()))
            predicted_spectrum = specs_array.mean(axis=0)
            energy = ENERGY_GRID[element].tolist()

            structure_id = pathlib.Path(filename).stem

            # Score against experiment (if loaded).
            if has_exp_data:
                match_result = get_spectrum_match_score(predicted_spectrum, exp_data, element)
            else:
                match_result = {"score": 0.0, "correlations": {}, "shift": 0.0,
                                "comparison_range": None}

            # Preserve checkbox selection state if re-uploading the same file.
            old_entry = next(
                (s for s in existing_scores if s["structure_id"] == structure_id), None
            )
            was_selected = old_entry.get("selected", False) if old_entry else False
            existing_scores = [s for s in existing_scores if s["structure_id"] != structure_id]

            existing_scores.append({
                "structure_id": structure_id,
                "score": match_result["score"],
                "shift": match_result["shift"],
                "correlations": match_result["correlations"],
                "comparison_range": match_result["comparison_range"],
                "spectrum": predicted_spectrum.tolist(),
                "energy": energy,
                "element": element,
                "selected": was_selected,
            })

            if match_result["comparison_range"] is not None:
                comparison_range = match_result["comparison_range"]

            # Keep the last structure for the 3-D viewer.
            st_dict = st.as_dict()
            st_dict["xas"] = specs
            last_st_dict = st_dict
            last_filename = filename
            successful += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_files.append(filename)

    # Sort the full ranking by the active metric.
    existing_scores = sort_scores_by_metric(existing_scores, sort_metric)

    # --- Build the status message ---
    if successful > 0 and failed == 0:
        status_msg = html.Span(
            f"✓ Processed {successful} structure(s) successfully",
            style={"color": "green"},
        )
    elif successful > 0:
        fail_summary = ", ".join(failed_files[:3]) + ("..." if len(failed_files) > 3 else "")
        status_msg = html.Span([
            html.Span(f"✓ Processed {successful} structure(s). ", style={"color": "green"}),
            html.Span(f"✗ Failed: {failed} ({fail_summary})", style={"color": "orange"}),
        ])
    else:
        status_msg = html.Span(
            f"✗ Failed to process all {failed} file(s)", style={"color": "red"}
        )

    if successful == 1:
        source_text = f"Current structure: {last_filename}"
    elif successful > 1:
        source_text = f"Batch loaded: {successful} structures"
    else:
        source_text = "No structures loaded"

    return (
        existing_scores,
        build_scores_table(existing_scores, sort_metric),
        comparison_range,
        status_msg,
        None,                                                       # Clear upload widget
        last_st_dict if last_st_dict else dash.no_update,
        source_text,
    )


# =============================================================================
# §14  Dash Callbacks — XAS Prediction & Plotting
# =============================================================================

def build_figure_with_exp(
    predicted_spectrum, exp_data, el_type,
    is_average, no_element, sel_mismatch,
    energy_shift=0, comparison_range=None,
    selected_spectra=None, current_structure_id=None,
):
    """Construct the main Plotly figure for the XANES spectrum panel.

    Handles several display modes:
      - A single predicted spectrum (optionally energy-shifted).
      - Multiple "selected" spectra overlaid from the ranking table.
      - An experimental reference spectrum (markers).
      - Edge-case messages (element not present, wrong atom selected).

    When an experimental spectrum is present, all predicted curves are
    normalised to match its vertical range so they overlay meaningfully.

    Parameters
    ----------
    predicted_spectrum : np.ndarray or None
        1-D absorption array on the standard grid.
    exp_data : dict or None
        Experimental spectrum with ``energy`` / ``absorption`` keys.
    el_type : str
        ``"<element> <code>"`` string.
    is_average : bool
        Whether the prediction is an average over all sites.
    no_element : bool
        True when the structure lacks the absorbing element entirely.
    sel_mismatch : bool
        True when the user clicked a non-absorbing atom in the 3-D viewer.
    energy_shift : float
        Manual energy shift applied to the predicted curve.
    comparison_range : tuple[float, float] or None
        If given, the x-axis is zoomed to this energy window.
    selected_spectra : list[dict] or None
        Multiple spectra selected via checkboxes in the ranking table.
    current_structure_id : str or None
        Label for the currently active structure.
    """
    element = el_type.split(" ")[0]
    fig = go.Figure()

    has_exp = exp_data is not None and "energy" in exp_data and "absorption" in exp_data
    has_selected = selected_spectra is not None and len(selected_spectra) > 0

    # --- Determine the plot title ---
    if has_selected:
        n = len(selected_spectra)
        title = f"Comparing {n} Structure{'s' if n > 1 else ''} with Experimental"
    elif predicted_spectrum is None and has_exp:
        name = exp_data.get("material_name", exp_data.get("filename", "Experimental"))
        title = f"Experimental Spectrum: {name}"
    elif no_element:
        title = f"This structure doesn't contain {element}"
    elif sel_mismatch:
        title = f"The selected atom is not a {element} atom"
    elif is_average:
        title = f"Average K-edge XANES Spectrum of {el_type}"
        if has_exp:
            title += " (with Experimental)"
    else:
        title = f"K-edge XANES Spectrum for the selected {element} atom"
        if has_exp:
            title += " (with Experimental)"

    # Pre-extract experimental data for normalisation.
    exp_energy = np.array(exp_data["energy"]) if has_exp else None
    exp_absorption = np.array(exp_data["absorption"]) if has_exp else None

    # Colour palette for multi-spectrum overlay.
    palette = [
        "#636EFA", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]

    def _normalise_to_exp(spectrum):
        """Scale a predicted spectrum to match the experimental range."""
        if has_exp and len(exp_absorption) > 0:
            pred_range = spectrum.max() - spectrum.min()
            exp_range = exp_absorption.max() - exp_absorption.min()
            if pred_range > 0 and exp_range > 0:
                normed = (spectrum - spectrum.min()) / pred_range
                return normed * exp_range + exp_absorption.min()
        return spectrum

    # --- Draw selected spectra (multi-overlay mode) -------------------------
    if has_selected:
        for idx, entry in enumerate(selected_spectra):
            spec = np.array(entry["spectrum"])
            ene = np.array(entry["energy"]) + entry.get("shift", 0.0)
            fig.add_trace(go.Scatter(
                x=ene, y=_normalise_to_exp(spec),
                mode="lines", name=entry["structure_id"],
                line=dict(color=palette[idx % len(palette)], width=2),
            ))

    # --- Draw single predicted spectrum -------------------------------------
    elif predicted_spectrum is not None:
        ene_shifted = ENERGY_GRID[element] + energy_shift
        scaled = _normalise_to_exp(predicted_spectrum)
        was_normalised = (scaled is not predicted_spectrum) if has_exp else False

        name = current_structure_id or "Predicted"
        if was_normalised:
            name += " (normalized)"
        if energy_shift != 0:
            name += f" [{energy_shift:+.1f} eV]"

        fig.add_trace(go.Scatter(
            x=ene_shifted, y=scaled,
            mode="lines", name=name,
            line=dict(color="#636EFA", width=2),
        ))

    # --- Draw experimental spectrum as markers ------------------------------
    if has_exp:
        exp_name = exp_data.get("material_name", exp_data.get("filename", "Experimental"))
        fig.add_trace(go.Scatter(
            x=exp_energy, y=exp_absorption,
            mode="markers", name=f"Exp: {exp_name}",
            marker=dict(color="#EF553B", size=4),
        ))

    # --- Axis labels --------------------------------------------------------
    x_label = exp_data.get("x_label", "Energy (eV)") if has_exp else "Energy (eV)"
    y_label = exp_data.get("y_label", "Absorption") if has_exp else "Absorption"

    layout_cfg = dict(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(
            yanchor="bottom", y=0.01, xanchor="right", x=0.99,
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    # Zoom to the comparison range if applicable.
    if has_exp and comparison_range and len(comparison_range) == 2:
        x_start, x_end = comparison_range
        if x_start < x_end and x_end - x_start > 5:
            padding = (x_end - x_start) * 0.1
            layout_cfg["xaxis"] = dict(
                range=[x_start - padding, x_end + padding], title=x_label,
            )

    fig.update_layout(**layout_cfg)
    return fig


@app.callback(
    Output("xas_plot", "figure", allow_duplicate=True),
    Input(struct_component.id(), "data"),
    Input("exp_spectrum_store", "data"),
    Input("energy_shift_slider", "value"),
    Input("comparison_range_store", "data"),
    Input("structure_scores_store", "data"),
    State("absorber", "value"),
    State("st_source", "children"),
)
def predict_average_xas(st_data, exp_data, energy_shift, comparison_range,
                        structure_scores, el_type, structure_source):
    """Recalculate and display the average predicted XANES spectrum.

    This is the *main* plot-update callback.  It fires whenever the structure,
    experimental data, energy shift, comparison range, or selected-spectra
    checkboxes change.
    """
    if st_data is None and exp_data is None:
        raise PreventUpdate

    # Extract a human-readable ID for the current structure.
    current_id = None
    if structure_source and isinstance(structure_source, str):
        current_id = structure_source.split(":")[-1].strip() if ":" in structure_source else structure_source

    # Gather checked spectra from the ranking table.
    selected = None
    if structure_scores:
        selected = [s for s in structure_scores if s.get("selected") and "spectrum" in s]
        if not selected:
            selected = None

    predicted = None
    no_element = False
    if selected is None and st_data is not None:
        specs = st_data.get("xas", {})
        if not specs:
            no_element = True
        else:
            predicted = np.array(list(specs.values())).mean(axis=0)

    return build_figure_with_exp(
        predicted, exp_data, el_type,
        is_average=True, no_element=no_element, sel_mismatch=False,
        energy_shift=energy_shift or 0, comparison_range=comparison_range,
        selected_spectra=selected, current_structure_id=current_id,
    )


@app.callback(
    Output("xas_plot", "figure", allow_duplicate=True),
    Input(struct_component.id("scene"), "selectedObject"),
    State(struct_component.id(), "data"),
    State("exp_spectrum_store", "data"),
    State("absorber", "value"),
    State("energy_shift_slider", "value"),
    State("comparison_range_store", "data"),
    State("st_source", "children"),
)
def predict_site_specific_xas(sel, st_data, exp_data, el_type, energy_shift,
                               comparison_range, structure_source):
    """Show the spectrum for a *specific* atom when the user clicks it in the 3-D viewer.

    Falls back to the average spectrum if nothing is selected or if the
    selected atom is not the absorbing element.
    """
    if st_data is None:
        raise PreventUpdate

    current_id = None
    if structure_source and isinstance(structure_source, str):
        current_id = structure_source.split(":")[-1].strip() if ":" in structure_source else structure_source

    specs = st_data["xas"]
    element = el_type.split(" ")[0]
    shift = energy_shift or 0

    if not specs:
        # Structure has no absorbing element.
        return build_figure_with_exp(
            None, exp_data, el_type, False, True, False,
            energy_shift=shift, comparison_range=comparison_range,
            current_structure_id=current_id,
        )

    if sel is None or len(sel) == 0:
        # No atom selected → show average.
        spectrum = np.array(list(specs.values())).mean(axis=0)
        return build_figure_with_exp(
            spectrum, exp_data, el_type, True, False, False,
            energy_shift=shift, comparison_range=comparison_range,
            current_structure_id=current_id,
        )

    # Identify which crystallographic site was clicked.
    st = Structure.from_dict(st_data)
    spheres = list(st._get_sites_to_draw())
    i_sphere = int(sel[0]["id"].split("--")[-1])
    i_site = spheres[i_sphere][0]

    if st[i_site].specie.symbol != element:
        # Clicked atom is not the absorbing element.
        return build_figure_with_exp(
            None, exp_data, el_type, False, False, True,
            energy_shift=shift, comparison_range=comparison_range,
            current_structure_id=current_id,
        )

    spectrum = np.array(specs[str(i_site)])
    site_label = f"{current_id} (site {i_site})" if current_id else None
    return build_figure_with_exp(
        spectrum, exp_data, el_type, False, False, False,
        energy_shift=shift, comparison_range=comparison_range,
        current_structure_id=site_label,
    )


# =============================================================================
# §15  Dash Callbacks — UI Helpers (Shift, Sort, Display, Download)
# =============================================================================

@app.callback(
    Output("energy_shift_slider", "value"),
    Input("reset_shift_btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_energy_shift(n_clicks):
    """Reset the energy-shift slider to zero."""
    if n_clicks is None:
        raise PreventUpdate
    return 0


@app.callback(
    Output("energy_shift_display", "children"),
    Input("energy_shift_slider", "value"),
)
def update_shift_display(value):
    """Update the numeric label next to the shift slider."""
    return f"{(value or 0):+.1f} eV"


@app.callback(
    Output("download_sink", "data"),
    Input("download_btn", "n_clicks"),
    State(struct_component.id(), "data"),
    State("absorber", "value"),
)
def download_xas_prediction(n_clicks, st_data, el_type):
    """Bundle the current structure (POSCAR) and predicted spectrum (CSV) into a ZIP."""
    if st_data is None:
        raise PreventUpdate

    element, theory = el_type.split(" ")
    st = Structure.from_dict(st_data)
    d_xas = st_data["xas"]

    # Build a DataFrame: first row = energies, subsequent rows = per-site spectra.
    specs = np.stack([ENERGY_GRID[element]] + list(d_xas.values()))
    row_labels = ["Energy"] + [f"Atom #{int(i) + 1}" for i in d_xas.keys()]
    df = pd.DataFrame(specs, index=row_labels)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = pathlib.Path(td)
        fn_spec = tmpdir / ("no_spectrum.csv" if len(d_xas) == 0 else "spectrum.csv")
        fn_poscar = tmpdir / "POSCAR"

        st.to(fn_poscar, fmt="poscar")
        df.to_csv(fn_spec, float_format="%.3f", header=False)

        zip_fn = tmpdir / f"OmniXAS_{element}_{theory}_Prediction_{n_clicks}.zip"
        with ZipFile(zip_fn, mode="w") as zf:
            zf.write(fn_poscar, arcname=fn_poscar.name)
            zf.write(fn_spec, arcname=fn_spec.name)

        encoded = b64encode(zip_fn.read_bytes()).decode("ascii")

    return {
        "content": encoded,
        "base64": True,
        "type": "application/zip",
        "filename": zip_fn.name,
    }


@app.callback(
    Output("sort_metric_store", "data"),
    Input({"type": "sort-metric-btn", "metric": ALL}, "n_clicks"),
    State("sort_metric_store", "data"),
    prevent_initial_call=True,
)
def handle_sort_click(n_clicks_list, current_sort_metric):
    """Change the active sort metric when a column header is clicked."""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"]
    try:
        id_str = trigger_id.rsplit(".", 1)[0]
        return json.loads(id_str)["metric"]
    except Exception:
        raise PreventUpdate


@app.callback(
    Output("structure_scores_store", "data"),
    Output("matching_results_table", "children"),
    Output("comparison_range_store", "data"),
    Input(struct_component.id(), "data"),
    Input("exp_spectrum_store", "data"),
    Input("clear_scores_btn", "n_clicks"),
    Input({"type": "spectrum-checkbox", "index": ALL}, "value"),
    Input("sort_metric_store", "data"),
    State("structure_scores_store", "data"),
    State("st_source", "children"),
    State("absorber", "value"),
    prevent_initial_call=True,
)
def update_matching_results(st_data, exp_data, clear_clicks, checkbox_values,
                            sort_metric, existing_scores, structure_source, el_type):
    """Central callback that keeps the ranking table in sync.

    Fires when:
      - A new structure is loaded (single-search mode).
      - The experimental spectrum changes.
      - The user clicks "Clear All".
      - A checkbox is toggled.
      - The sort metric changes.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"]
    existing_scores = existing_scores or []
    sort_metric = sort_metric or "coss_deriv"

    placeholder = lambda msg: html.Div(
        msg, style={"color": "#999", "fontSize": "12px", "textAlign": "center", "padding": "20px"}
    )

    # --- Clear button -------------------------------------------------------
    if "clear_scores_btn" in trigger_id:
        return [], placeholder("Upload experimental spectrum and load structures to see matching scores"), None

    # --- Checkbox toggled ---------------------------------------------------
    if "spectrum-checkbox" in trigger_id:
        for i, entry in enumerate(existing_scores):
            if i < len(checkbox_values):
                entry["selected"] = bool(checkbox_values[i])
        existing_scores = sort_scores_by_metric(existing_scores, sort_metric)
        return existing_scores, build_scores_table(existing_scores, sort_metric), dash.no_update

    # --- Sort metric changed ------------------------------------------------
    if "sort_metric_store" in trigger_id:
        existing_scores = sort_scores_by_metric(existing_scores, sort_metric)
        return existing_scores, build_scores_table(existing_scores, sort_metric), dash.no_update

    # --- Check prerequisites ------------------------------------------------
    has_exp = exp_data is not None and "energy" in exp_data and "absorption" in exp_data

    if not has_exp:
        if not existing_scores:
            return existing_scores, placeholder("Upload experimental spectrum first to enable matching"), None
        return existing_scores, build_scores_table(existing_scores, sort_metric), dash.no_update

    if st_data is None:
        if not existing_scores:
            return existing_scores, placeholder("Load a structure to see matching scores"), None
        return existing_scores, build_scores_table(existing_scores, sort_metric), dash.no_update

    specs = st_data.get("xas", {})
    if not specs:
        if not existing_scores:
            return existing_scores, placeholder("No spectrum available for matching"), None
        return existing_scores, build_scores_table(existing_scores, sort_metric), dash.no_update

    # --- Score the current structure ----------------------------------------
    element = el_type.split(" ")[0]
    predicted = np.array(list(specs.values())).mean(axis=0)
    energy = ENERGY_GRID[element].tolist()

    structure_id = "unknown"
    if structure_source and isinstance(structure_source, str):
        structure_id = structure_source.split(":")[-1].strip() if ":" in structure_source else structure_source

    match_result = get_spectrum_match_score(predicted, exp_data, element)

    # Preserve selection state on re-score.
    old = next((s for s in existing_scores if s["structure_id"] == structure_id), None)
    was_selected = old.get("selected", False) if old else False
    updated = [s for s in existing_scores if s["structure_id"] != structure_id]

    updated.append({
        "structure_id": structure_id,
        "score": match_result["score"],
        "shift": match_result["shift"],
        "correlations": match_result["correlations"],
        "comparison_range": match_result["comparison_range"],
        "spectrum": predicted.tolist(),
        "energy": energy,
        "element": element,
        "selected": was_selected,
    })

    updated = sort_scores_by_metric(updated, sort_metric)
    return updated, build_scores_table(updated, sort_metric), match_result["comparison_range"]


# =============================================================================
# §16  Matching-Results Table Builder
# =============================================================================

def sort_scores_by_metric(scores, metric):
    """Sort the scores list by a given metric.

    For ``normed_wasserstein`` lower is better (ascending), for all others
    higher is better (descending).
    """
    if not scores:
        return scores

    ascending = (metric == "normed_wasserstein")

    def key_fn(entry):
        val = entry.get("correlations", {}).get(metric, 0.0)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return -999 if not ascending else 999
        return val

    return sorted(scores, key=key_fn, reverse=not ascending)


def build_scores_table(scores, sort_metric="coss_deriv"):
    """Render the structure-ranking table as Dash HTML components.

    Each column header is a clickable button that changes the sort metric.
    Metric values are colour-coded green / yellow / red by quality.
    """
    if not scores:
        return html.Div(
            "No scores yet",
            style={"color": "#999", "fontSize": "12px", "textAlign": "center", "padding": "20px"},
        )

    # --- Style templates ---
    base_hdr = {
        "padding": "5px 4px", "textAlign": "right", "fontWeight": "600",
        "fontSize": "10px", "color": "#666", "borderBottom": "2px solid #e8e8e8",
        "backgroundColor": "#fafafa", "whiteSpace": "nowrap",
    }
    active_hdr = {
        **base_hdr, "color": "#333",
        "borderBottom": "2px solid #333", "backgroundColor": "#f0f0f0",
    }
    cell = {
        "padding": "5px 4px", "fontSize": "11px", "color": "#333",
        "borderBottom": "1px solid #eee", "textAlign": "right",
    }

    # --- Header row ---------------------------------------------------------
    header_cells = [
        html.Th("", style={**base_hdr, "width": "28px", "textAlign": "center"}),
        html.Th("#", style={**base_hdr, "width": "22px", "textAlign": "center"}),
        html.Th("Structure", style={**base_hdr, "textAlign": "left", "minWidth": "70px"}),
        html.Th("Shift", style={**base_hdr, "width": "50px"}),
    ]

    for metric in ALL_METRICS:
        is_active = metric == sort_metric
        arrow = ""
        if is_active:
            arrow = " ▲" if metric == "normed_wasserstein" else " ▼"

        header_cells.append(html.Th(
            html.Button(
                METRIC_SHORT_NAMES[metric] + arrow,
                id={"type": "sort-metric-btn", "metric": metric},
                style={
                    "border": "none", "background": "none", "cursor": "pointer",
                    "fontWeight": "700" if is_active else "600", "fontSize": "10px",
                    "color": "#333" if is_active else "#666", "padding": "0",
                    "fontFamily": BASE_FONT, "whiteSpace": "nowrap",
                },
                title=(f"Sort by {metric}"
                       + (" (lower is better)" if metric == "normed_wasserstein" else " (higher is better)")),
            ),
            style=active_hdr if is_active else base_hdr,
        ))

    # --- Data rows ----------------------------------------------------------
    rows = []
    for rank, entry in enumerate(scores):
        corr = entry.get("correlations", {})
        shift = entry.get("shift", 0.0)

        row = [
            # Checkbox
            html.Td(
                dcc.Checklist(
                    id={"type": "spectrum-checkbox", "index": rank},
                    options=[{"label": "", "value": True}],
                    value=[True] if entry.get("selected") else [],
                    style={"margin": "0", "padding": "0"},
                    inputStyle={"marginRight": "0"},
                ),
                style={**cell, "textAlign": "center", "padding": "3px"},
            ),
            # Rank
            html.Td(rank + 1, style={**cell, "color": "#999", "fontWeight": "500", "textAlign": "center"}),
            # Structure ID
            html.Td(entry["structure_id"], style={
                **cell, "fontFamily": "monospace", "fontSize": "10px", "textAlign": "left",
                "maxWidth": "90px", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
            }),
            # Shift
            html.Td(f"{shift:+.1f}", style={**cell, "fontSize": "10px", "color": "#666"}),
        ]

        # One cell per metric.
        for metric in ALL_METRICS:
            val = corr.get(metric)
            is_sort = metric == sort_metric

            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                display, colour = "—", "#999"
            else:
                display = f"{val:.3f}"
                if metric == "normed_wasserstein":
                    colour = "#28a745" if val <= 0.1 else ("#ffc107" if val <= 0.3 else "#dc3545")
                else:
                    colour = "#28a745" if val >= 0.9 else ("#ffc107" if val >= 0.7 else "#dc3545")

            row.append(html.Td(display, style={
                **cell,
                "fontWeight": "700" if is_sort else "400",
                "color": colour,
                "fontSize": "11px" if is_sort else "10px",
                "backgroundColor": "#f8f8f8" if is_sort else "transparent",
            }))

        rows.append(html.Tr(row))

    table = html.Table(
        [html.Thead(html.Tr(header_cells)), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse",
               "fontFamily": BASE_FONT, "tableLayout": "auto"},
    )
    return html.Div(table, style={"overflowX": "auto", "fontSize": "11px"})


# =============================================================================
# §17  Crystal Toolkit Registration & Entry Point
# =============================================================================

# Crystal Toolkit requires explicit registration so its internal callbacks
# (structure rendering, file upload parsing, etc.) are wired into the Dash app.
ctc.register_crystal_toolkit(app=app, layout=omnixas_layout)


def serve():
    """Launch the Dash development server.

    Requires the ``MP_API_KEY`` environment variable to be set for
    Materials Project queries.
    """
    if "MP_API_KEY" not in os.environ:
        print(
            "Environment variable MP_API_KEY not found. "
            "Please set your Materials Project API key to this "
            "environment variable before running this app."
        )
        exit()
    app.run(debug=False, port=8443, host="0.0.0.0")


if __name__ == "__main__":
    serve()