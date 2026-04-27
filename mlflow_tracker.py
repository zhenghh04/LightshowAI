"""
MLflow tracker for LightshowAI XANES predictions.

Usage (CLI):
    python mlflow_tracker.py --material-id mp-19306 --element Fe --type FEFF

Usage (Python):
    from mlflow_tracker import predict_and_log, log_mcp_result

Auth:
    Set MLFLOW_TRACKING_TOKEN in your environment (AmSC API key).
"""

import argparse
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import mlflow

# ── paths ──────────────────────────────────────────────────────────────────────
PARENT = pathlib.Path(__file__).parent
sys.path.insert(0, str(PARENT))

# Energy grids (141 points, edge → edge + 35 eV)
ENE_START = {
    "Ti": 4964.504, "V": 5464.097, "Cr": 5989.168, "Mn": 6537.886,
    "Fe": 7111.23,  "Co": 7709.282, "Ni": 8332.181, "Cu": 8983.173,
}

MLFLOW_URI      = "https://mlflow.american-science-cloud.org"
EXPERIMENT_NAME = "LightshowAI-XANES"


# ── MLflow setup ───────────────────────────────────────────────────────────────
def _load_env(env_path: pathlib.Path | None = None) -> dict:
    """Load key=value pairs from a .env file without requiring python-dotenv."""
    candidates = [
        env_path,
        PARENT.parent.parent / ".env",  # agentic_workflows/.env
        pathlib.Path.home() / ".env",
    ]
    for path in candidates:
        if path and path.exists():
            env = {}
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
            return env
    return {}


def _patch_mlflow_x_api_key(api_key: str):
    """Inject X-Api-Key header into every MLflow REST request (mirrors the MCP server's auth patch)."""
    import mlflow.utils.rest_utils as rest_utils
    original = rest_utils.http_request

    def patched(host_creds, endpoint, method, *args, **kwargs):
        headers = dict(kwargs.get("extra_headers") or {})
        headers["X-Api-Key"] = api_key
        kwargs["extra_headers"] = headers
        kwargs.pop("headers", None)
        return original(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = patched


def _setup_mlflow(env_path: pathlib.Path | None = None):
    env = _load_env(env_path)
    api_key = os.environ.get("AM_SC_API_KEY") or env.get("AM_SC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No AmSC API key found. Set AM_SC_API_KEY in your .env file."
        )
    _patch_mlflow_x_api_key(api_key)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


# ── core prediction + logging ──────────────────────────────────────────────────
def predict_and_log(
    material_id: str,
    absorbing_element: str,
    spectroscopy_type: str = "FEFF",
    mp_api_key: str | None = None,
    output_dir: pathlib.Path | None = None,
) -> str:
    """Run LightshowAI prediction and log the run to MLflow. Returns run_id."""
    from mp_api.client import MPRester
    from lightshowai.models import predict

    output_dir = output_dir or pathlib.Path(os.environ.get("HOME", ".")) / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = mp_api_key or os.environ.get("MP_API_KEY")
    with MPRester(api_key) as mpr:
        doc = mpr.summary.get_data_by_id(material_id)
        structure = doc.structure
        formula   = doc.formula_pretty
        symmetry  = doc.symmetry
        e_hull    = doc.energy_above_hull

    spectra = predict(structure, absorbing_element, spectroscopy_type)
    energy  = np.linspace(
        ENE_START[absorbing_element],
        ENE_START[absorbing_element] + 35,
        141,
    )

    # Build per-site + mean dataframe
    df = pd.DataFrame({"energy_eV": energy})
    for site_idx, mu in spectra.items():
        df[f"site_{site_idx}"] = mu
    site_cols = [c for c in df.columns if c.startswith("site_")]
    df["mean"] = df[site_cols].mean(axis=1)

    spectrum_csv = output_dir / f"{material_id}_{absorbing_element}_{spectroscopy_type}.csv"
    df.to_csv(spectrum_csv, index=False)

    _setup_mlflow()
    with mlflow.start_run(run_name=f"{material_id}_{absorbing_element}_{spectroscopy_type}") as run:
        # Parameters
        mlflow.log_params({
            "material_id":       material_id,
            "absorbing_element": absorbing_element,
            "spectroscopy_type": spectroscopy_type,
        })
        # Tags (structure metadata)
        mlflow.set_tags({
            "formula":        formula,
            "crystal_system": symmetry.crystal_system,
            "space_group":    symmetry.symbol,
            "n_sites":        str(len(spectra)),
            "energy_above_hull_eV_per_atom": f"{e_hull:.4f}",
        })
        # Artifacts
        try:
            mlflow.log_artifact(str(spectrum_csv), artifact_path="spectra")
        except Exception as e:
            print(f"Warning: artifact upload skipped ({e.__class__.__name__})")

        run_id = run.info.run_id

    print(f"Logged run {run_id} → {MLFLOW_URI}/#/experiments/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/runs/{run_id}")
    return run_id


MODEL_NAME      = "XASBlock"
MODEL_VERSION   = "v1.1.1"
FEATURIZER_NAME = "M3GNet-MP-2021.2.8-PES"


def log_mcp_result(
    material_id: str,
    absorbing_element: str,
    spectroscopy_type: str,
    formula: str,
    n_sites: int,
    html_path: str | None = None,
    extra_tags: dict | None = None,
    spectrum_data: dict | None = None,
) -> str:
    """
    Log metadata (and optionally spectrum) from an already-completed MCP call.

    spectrum_data: the dict returned by mcp__lightshowai__predict_xanes, containing
                   'energy_eV', 'mean_spectrum', and 'site_spectra'.
    """
    checkpoint = f"model_checkpoints/xasblock/{MODEL_VERSION}/{absorbing_element}_{spectroscopy_type}.ckpt"
    _setup_mlflow()
    with mlflow.start_run(run_name=f"{material_id}_{absorbing_element}_{spectroscopy_type}") as run:
        mlflow.log_params({
            "material_id":       material_id,
            "absorbing_element": absorbing_element,
            "spectroscopy_type": spectroscopy_type,
            "model_name":        MODEL_NAME,
            "model_version":     MODEL_VERSION,
            "featurizer":        FEATURIZER_NAME,
            "checkpoint":        checkpoint,
        })
        tags = {"formula": formula, "n_sites": str(n_sites)}
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)

        if spectrum_data:
            energy  = spectrum_data["energy_eV"]
            mean_mu = spectrum_data["mean_spectrum"]
            # Step = point index (0-140); energy axis stored as params below
            mlflow.log_params({
                "energy_start_eV": f"{energy[0]:.3f}",
                "energy_end_eV":   f"{energy[-1]:.3f}",
                "energy_n_points": str(len(energy)),
                "energy_step_eV":  f"{(energy[-1] - energy[0]) / (len(energy) - 1):.3f}",
            })
            for i, (e, mu) in enumerate(zip(energy, mean_mu)):
                mlflow.log_metric("mu_mean", mu, step=i)
            for site_idx, site_mu in spectrum_data.get("site_spectra", {}).items():
                for i, mu in enumerate(site_mu):
                    mlflow.log_metric(f"mu_site_{site_idx}", mu, step=i)

        if html_path and pathlib.Path(html_path).exists():
            try:
                mlflow.log_artifact(html_path, artifact_path="plots")
            except Exception as e:
                print(f"Warning: artifact upload skipped ({e.__class__.__name__})")

        run_id = run.info.run_id

    print(f"Logged MCP result as run {run_id}")
    return run_id


# ── Model Card ─────────────────────────────────────────────────────────────────
MODEL_CARD_NAME = "LightshowAI-XASBlock"

MODEL_CARD_DESCRIPTION = """\
## LightshowAI XASBlock — Model Card

### Overview
XASBlock predicts site-resolved XANES (X-ray Absorption Near Edge Structure) spectra
from crystal structures using a two-stage pipeline: a graph neural network featurizer
(M3GNet) extracts per-site structural embeddings, which are then passed to a small MLP
(XASBlock) that outputs a 141-point absorption spectrum spanning 35 eV above the
absorption edge.

### Architecture
| Component | Details |
|-----------|---------|
| Featurizer | M3GNet-MP-2021.2.8-PES — graph neural network, 64-dim node embeddings |
| Predictor  | XASBlock MLP: Linear(64→500)→BN→SiLU→Dropout(0.5) × 3 layers → Linear(→141)→Softplus |
| Output     | 141-point μ(E) spectrum, 35 eV range from absorption edge, 0.25 eV spacing |

### Supported Elements & Edges
| Element | Edge Energy (eV) | Spectroscopy Types |
|---------|------------------|--------------------|
| Ti      | 4964.504         | FEFF, VASP         |
| V       | 5464.097         | FEFF               |
| Cr      | 5989.168         | FEFF               |
| Mn      | 6537.886         | FEFF               |
| Fe      | 7111.230         | FEFF               |
| Co      | 7709.282         | FEFF               |
| Ni      | 8332.181         | FEFF               |
| Cu      | 8983.173         | FEFF, VASP         |

### Energy Grid
`np.linspace(E_edge, E_edge + 35, 141)` — 0.25 eV spacing per point

### Checkpoints
`model_checkpoints/xasblock/v1.1.1/{element}_{spectroscopy_type}.ckpt`

### Training Data
Crystal structures from the Materials Project (mp-*) with FEFF-simulated XANES spectra.

### Usage
```python
from lightshowai.models import predict
spectra = predict(structure, absorbing_element="Fe", spectroscopy_type="FEFF")
# returns {site_idx: np.array(141)} for every absorbing site
```
"""


def create_model_card() -> str:
    """Register the LightshowAI XASBlock model card in the MLflow Model Registry."""
    from mlflow import MlflowClient
    _setup_mlflow()
    client = MlflowClient()

    try:
        client.create_registered_model(MODEL_CARD_NAME, description=MODEL_CARD_DESCRIPTION)
        print(f"Created registered model: {MODEL_CARD_NAME}")
    except Exception as e:
        if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
            client.update_registered_model(MODEL_CARD_NAME, description=MODEL_CARD_DESCRIPTION)
            print(f"Updated registered model: {MODEL_CARD_NAME}")
        else:
            raise

    tags = {
        "framework":    "PyTorch Lightning",
        "featurizer":   "M3GNet-MP-2021.2.8-PES",
        "version":      MODEL_VERSION,
        "task":         "XANES spectrum prediction",
        "elements":     "Ti,V,Cr,Mn,Fe,Co,Ni,Cu",
        "spectroscopy": "FEFF,VASP",
        "output_dim":   "141",
        "energy_range": "35 eV above edge",
    }
    for key, val in tags.items():
        client.set_registered_model_tag(MODEL_CARD_NAME, key, val)

    url = f"{MLFLOW_URI}/#/models/{MODEL_CARD_NAME}"
    print(f"Model card → {url}")
    return url


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict XANES and log to MLflow")
    parser.add_argument("--material-id",  required=True)
    parser.add_argument("--element",      required=True)
    parser.add_argument("--type",         default="FEFF", choices=["FEFF", "VASP"])
    parser.add_argument("--mp-api-key",   default=None)
    parser.add_argument("--output-dir",   default=None)
    args = parser.parse_args()

    run_id = predict_and_log(
        material_id       = args.material_id,
        absorbing_element = args.element,
        spectroscopy_type = args.type,
        mp_api_key        = args.mp_api_key,
        output_dir        = pathlib.Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Run ID: {run_id}")
