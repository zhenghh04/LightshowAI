from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math
from scipy import optimize
import numpy as np
import scipy
from typing import Tuple
import pandas as pd

def _find_columns(df_columns, aliases):
    # detects columns
    return [col for col in df_columns if str(col).strip().lower() in aliases]

def bin_region(energy, mu, step, reducer="mean"):
    # helper function to bin a specific region
    ebin = np.round(energy / step) * step
    return pd.DataFrame({"E": ebin, "mu": mu}).groupby("E", as_index=False).agg(reducer)

def bin_xas(energy, mu):
    # bins data similar to how Bruce's data looks

    # create a rough binned version of raw data for edge detection
    rough_dict = {'E': np.round(energy), 'mu': mu}
    df_rough = pd.DataFrame(rough_dict)
    df_rough = df_rough.groupby('E', as_index=False).mean()

    # find derivative of the above binned version
    mu_smoothed = df_rough['mu']
    energy_smoothed = df_rough['E']
    derivative = np.gradient(mu_smoothed, energy_smoothed)

    # find the max of the derivative
    max_slope_index = np.argmax(derivative)
    e0 = df_rough['E'].iloc[max_slope_index]

    regions = [
        (energy <  e0 - 30, 5.0),
        ((energy >= e0 - 30) & (energy < e0 - 15), 1.0),
        ((energy >= e0 - 15) & (energy < e0 + 30), 0.25),
        ((energy >= e0 + 30) & (energy < e0 + 100), 0.5),
        (energy >= e0 + 100, 2.0),
    ]

    parts = []
    for mask, step in regions:
        if mask.any():
            filtered_energy = energy[mask]
            filtered_mu = mu[mask]
            binned_part = bin_region(filtered_energy, filtered_mu, step)
            parts.append(binned_part)
            
    spec_df = pd.concat(parts, ignore_index=True)
    
    spec_df['E'] = np.round(spec_df['E'], 3)
    final_grouped_df = spec_df.groupby("E", as_index=False).mean()
    
    return final_grouped_df.to_numpy()

def spectrum_from_new_csv(df: pd.DataFrame, mode: str = "transmission", apply_binning: bool = True):
    # from a csv file which contains raw data
    energy_cols = _find_columns(df.columns, ["energy", "e", "ev"])
    i0_cols = _find_columns(df.columns, ["i0", "io"])
    if_cols = _find_columns(df.columns, ["iff", "if", "fluor"])
    it_cols = _find_columns(df.columns, ["it", "trans"])
    
    if not energy_cols or not i0_cols:
        raise ValueError("CSV must contain at least 'energy' and 'i0' columns.")

    energy = pd.to_numeric(df[energy_cols[0]], errors="coerce").to_numpy(float)
    i0 = pd.to_numeric(df[i0_cols[0]], errors="coerce").to_numpy(float)
 
    valid_i0 = i0 > 1e-6
    mode = (mode or "fluorescence").lower()
    
    if mode == "fluorescence" and if_cols:
        signal = pd.to_numeric(df[if_cols[0]], errors="coerce").to_numpy(float)
        mu = np.divide(signal, i0, out=np.full_like(signal, np.nan), where=valid_i0)
        y_label = "Fluorescence (iff / i0)"
        
    elif mode == "transmission" and it_cols:
        it = pd.to_numeric(df[it_cols[0]], errors="coerce").to_numpy(float)
        ratio = np.divide(it, i0, out=np.full_like(it, np.nan), where=valid_i0)
        mu = -np.log(np.clip(ratio, 1e-12, None))
        y_label = "Transmission -ln(it / i0)"
    else:
        raise ValueError(f"Could not find valid columns for mode: {mode}")

    mask = np.isfinite(energy) & np.isfinite(mu)
    energy, mu = energy[mask], mu[mask]

    if apply_binning:
        spec = bin_xas(energy, mu)
    else:
        spec = pd.DataFrame({'E': np.round(energy, 2), 'mu': mu}).groupby('E', as_index=False).mean().to_numpy()

    return spec, {"x_label": "Energy (eV)", "y_label": y_label, "mode": mode}
    
def smoothSpectrum(spect, smooth_width_eV=1.0):
    """
    Smooth the spectrum over the provided smoothing window size
    """
    e = spect[:,0]
    mu = spect[:,1]

    #Filter takes a number of points in the smoothing window, not a smoothing size
    #As the expt spectrum does not have a uniform energy spacing we must first interpolate to a 
    #fixed-spacing grid
    dE = e[1]-e[0]
    for i in range(1,len(e)):
        dE_cur = e[i] - e[i-1]
        dE = min(dE, dE_cur)
    #print("dE smallest", dE)
    
    num = int(math.floor( (e[-1] - e[0])/dE ) ) + 1
    grid = np.linspace(e[0],e[-1],num=num, endpoint=True)
    dE = grid[1]-grid[0]
    #print("grid dE",dE,"endpoints",grid[0],grid[-1],"vs orig endpoints", e[0],e[-1])

    ius = interp1d(
        e, mu,        
        assume_sorted=True,
        kind="cubic",
        bounds_error=False,
    )
    mu_grid = ius(grid)
        
    window_pts = int(round(smooth_width_eV / dE))
    window_pts = window_pts + (1 if (window_pts %2 == 0) else 0)

    mu_smooth = savgol_filter(mu_grid, window_pts, polyorder=3)
    return np.vstack( (grid, mu_smooth) ).T    

def normalizeSpectrum(spec, output_curves=False, preedge_range: Tuple[float,float] | None = None, postedge_range: Tuple[float,float] | None = None, flatten=False):    
    """
    Normalize the spectrum
    Inputs:
    spec: The input spectrum as a 2d array
    output_curves: Output the pre and post edge fit curves
    preedge_range: The fit range for the pre-edge fit (2-tuple). If None, this will be inferred automatically using the xraylarch rules of thumb
    postedge_range: The fit range for the post-edge fit (2-tuple). If None, this will be inferred automatically using the xraylarch rules of thumb
    flatten: flatten the spectrum

    Return:
    For 
    norm_spec : the normalized spectrum
    pre : the pre-edge curve
    post : the post-edge curve
    flat_quad : a quadratic fit to the post-edge fit range performed after normalization to perform flattening on the post-edge region
    
    if !output_curves: norm_spec   
    elif output_curves and !flatten:  norm_spec, pre, post
    else norm_spec, pre, post, flat_quad
    """
    
    energy = spec[:,0]

    #Determine the edge location
    spec_smooth = smoothSpectrum(spec, 1)
    dmu_dE = np.gradient(spec_smooth[:,1], spec_smooth[:,0]) #note, not on the same grid as energy
    edge_E = spec_smooth[np.argmax(dmu_dE),0]
    edge_idx = np.argmin( np.abs(energy - edge_E) )

    #Pre-edge fit - linear fit to region from start to halfway between start and edge
    lbound_idx = 0 if preedge_range == None else np.argmin( np.abs(energy - preedge_range[0]) ) 
    
    ubound_E = (energy[0] + edge_E)/2 if preedge_range == None else preedge_range[1]
    ubound_idx = np.argmin( np.abs(energy - ubound_E) )
    p = np.polyfit(spec[lbound_idx:ubound_idx,0], spec[lbound_idx:ubound_idx,1], 1)
    preedge = p[0] * energy + p[1]
    print("Pre-edge fit range",energy[lbound_idx],energy[ubound_idx], "width", energy[ubound_idx]-energy[lbound_idx],"degree 1")
    
    #Post-edge fit for normalization
    #xraylarch uses the lower bound as lbound = min(e0+25, e0+  (emax-e0)/3 ) and
    #polynomial order nnorm = 2 in emax-lbound>300, 1 if emax-lbound>30, or 0 if less.
    #e0 is the edge location
    dlbound = (energy[-1] - edge_E)/3
    dlbound = min(dlbound, 25)
    
    lbound_E = edge_E + dlbound if postedge_range == None else postedge_range[0]
    lbound_idx=np.argmin( np.abs(energy - lbound_E) )
    lbound_E = energy[lbound_idx]

    ubound_idx = len(energy)-1 if postedge_range == None else np.argmin( np.abs(energy - postedge_range[1]) )
    
    frange = energy[ubound_idx] - lbound_E
    
    if frange >= 300:
        nnorm=2
    elif frange >= 30:
        nnorm=1
    else:
        nnorm=0
    
    print("Post-edge fit range",lbound_E, energy[ubound_idx], "width", energy[ubound_idx] - lbound_E, "degree", nnorm)
    
    p = np.polyfit(spec[lbound_idx:ubound_idx,0], spec[lbound_idx:ubound_idx,1], nnorm)
    
    if nnorm == 0:
        postedge = p[0]
    elif nnorm == 1:
        postedge = p[0]*energy + p[1]
    else:
        postedge = p[0]*energy**2 + p[1]*energy + p[2]        

    #Normalize
    edge_idx = np.argmin( np.abs(energy - edge_E) )
    edge_step = postedge[edge_idx] - preedge[edge_idx]
    edge_step = max(1.e-12, abs(float(edge_step)))
    print("Edge step", edge_step)
    
    spec_sub = spec.copy()
    spec_sub[:,1] = (spec_sub[:,1] - preedge)/edge_step 

    #Flatten
    if flatten:
        #For flattening xraylarch fits a quadratic (always) to the post-edge region and subtracts the linear and quadratic terms from the post-edge region (in a really clunky way!)
        epost = spec_sub[lbound_idx:ubound_idx,0]
        p = np.polyfit(epost, spec_sub[lbound_idx:ubound_idx,1], 2)
        flat_quad = p[0]*energy**2 + p[1]*energy + p[2]
        print("Post-edge quadratic fit range",lbound_E, energy[ubound_idx], "width", energy[ubound_idx] - lbound_E)
        spec_sub[edge_idx:,1] = spec_sub[edge_idx:,1] - flat_quad[edge_idx:] + flat_quad[edge_idx]
    
    if output_curves:
        if flatten:
            return spec_sub, preedge, postedge, flat_quad
        else:
            return spec_sub, preedge, postedge
    else:
        return spec_sub
