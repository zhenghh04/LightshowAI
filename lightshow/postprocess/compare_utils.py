# This is a simple tool to compare to plots
# Chuntian Cao developed based on Fanchen Meng's 2022 code


import numpy as np
import scipy
from scipy import interpolate
from scipy.stats import pearsonr, spearmanr, kendalltau, wasserstein_distance
import math
from typing import Callable

class gridInterpolatorExt:
    """Use a predefined external array as the interpolation grid"""
    def __init__(self,grid : np.ndarray | list):
        """Initialize with the provided grid"""
        self.grid = grid
    def __call__(self, range1: np.ndarray, range2: np.ndarray, shift2 : float = 0):
        """Return the provided grid"""
        return self.grid

class gridInterpolatorFixedN:
    """Create an interpolation grid with a fixed number of points over the window of overlap between the spectra"""
    def __init__(self, n : int):
        """Initialize with the number of points, n"""
        self.n = n
    def __call__(self, spectrum1: np.ndarray, spectrum2: np.ndarray, shift2 : float = 0):
        """Return the created grid with spectrum2 shifted by shift2 (default 0)"""
        start = max(spectrum1[0, 0], spectrum2[0, 0] + shift2),
        end = min(spectrum1[-1, 0], spectrum2[-1, 0] + shift2)
        return np.linspace(start, end, self.n)

class gridInterpolatorFixedSpacing:
    """Create an interpolation grid over the window of overlap between the spectra with a number of points chosen such that the grid spacing is as close as possible to the provided value"""
    def __init__(self, delta : float):
        """Initialize with the number of points, n"""
        assert isinstance(delta, float) and delta > 0.
        self.delta = delta
    def __call__(self, spectrum1: np.ndarray, spectrum2: np.ndarray, shift2 : float = 0):
        """Return the created grid with spectrum2 shifted by shift2 (default 0)"""
        start = max(spectrum1[0, 0], spectrum2[0, 0] + shift2),
        end = min(spectrum1[-1, 0], spectrum2[-1, 0] + shift2)
        n=int(math.ceil( (end-start)/self.delta))
        return np.linspace(start, end, n)

def compare_between_spectra(
    spectrum1 : np.ndarray,
    spectrum2 : np.ndarray,
    erange : float =35,
    erange_threshold : float = 0.02,
    grid_interpolator = gridInterpolatorFixedN(300), 
    output_correlations = ["pearson","spearman"],
    opt_strategy : str = "grid_search",
    accuracy=0.01,
    method="coss"
):
    """Automatic align the spectra and calculate the correlation coefficients.
    The spectra are first truncated to a comparison window defined by a provided range and starting at a threshold where the spectrum reaches some fraction of its peak
    An optimization is then performed to locate the maximum overlap within that window    

    Parameters
    ----------
    spectrum1, spectrum2 : 2d-array
        Two-column arrays of energy vs. intensity XAS data.
    method : any method supported by 'similarity'
        The correlation metric for spectra comparison.
        Empirically 'coss' works well.
    output_correlations : a list of correlation metrics computed at the optimal point to output along with that used for the optimization

    erange : float, default=35
        Energy range for comparison. Unit: eV.
    erange_threshold : float, default=0.02
        The threshold fraction of the spectrum maximum at which to define the start the comparison range

    grid_interpolator: a callable that constructs the interpolation grid for computing the similarity, cf spectra_corr

    opt_strategy : an optimization strategy supported by "max_corr". Default is "grid_search"; set the grid spacing using 'accuracy'

    accuracy : float, default=0.01
        Accuracy for spectra alignment. Relevant only to opt_strategy == "grid_search"  Unit: eV.

    Returns
    -------
    correlations: dict mapping similarity metrics to their values computed at the optimal point. Control which metrics are included alongside the optimization metric using 'output_correlations'

    shift : float
        Relative shift between the two spectra, sign is meaningful.
        Spectrum2 should be shifted to spectrum2+shift for alignment.

    """

    #Truncate spectra and re-zero the x-axis
    start1, end1 = truncate_spectrum(spectrum1, erange, threshold=erange_threshold)
    plot1 = np.column_stack(
        (
            spectrum1[start1:end1, 0] - spectrum1[start1][0],
            spectrum1[start1:end1, 1],
        )
    )
    start2, end2 = truncate_spectrum(spectrum2, erange, threshold=erange_threshold)
    plot2 = np.column_stack(
        (
            spectrum2[start2:end2, 0] - spectrum2[start2][0],
            spectrum2[start2:end2, 1],
        )
    )

    shift_prior = spectrum1[start1, 0] - spectrum2[start2, 0]

    #Optimize the shift
    _, shift = max_corr(plot1, plot2, step=accuracy, method=method, grid=grid_interpolator, opt_strategy=opt_strategy)

    #Calculate correlation metrics at optimal point
    correlations = {}
    op = output_correlations.copy()
    if method not in op:
        op.append(method)
        
    cm = spectra_corr(plot1, plot2, omega=shift, verbose=True, method=op, grid=grid_interpolator)
    for i, method in enumerate(op):
        correlations[method] = cm[i]

    #Account for re-zeroing the x-axis
    shift += shift_prior

    return correlations, shift


def truncate_spectrum(spectrum, erange=35, threshold=0.02):
    """Truncate XAS spectrum to desired energy range.

    Parameters
    ----------
    spectrum : 2d-array
        Column stacked spectrum, energy vs. intensity.
    erange : float, default=35
        Truncation energy range in eV.
    threshold : float, default=0.02
        Start truncation at threshold * maximum intensity.

    Returns
    -------
    start, end : int
        Indices of truncated spectrum in the input spectrum.

    """
    x = spectrum[:, 0]
    y = spectrum[:, 1] / np.max(spectrum[:, 1])

    logic = y > threshold
    seq = x == x[logic][0]
    start = seq.argmax()

    logic = x < x[start] + erange
    seq = x == x[logic][-1]
    end = seq.argmax()

    return start, end


def cos_similar(v1, v2):
    """Calculates the cosine similarity between two vectors.

    Parameters
    ----------
    v1, v2 : 1d-array

    Returns
    -------
    cosSimilarity : float

    """
    norm1 = np.sqrt(np.dot(v1, v1))
    norm2 = np.sqrt(np.dot(v2, v2))
    cosSimilarity = np.dot(v1, v2) / (norm1 * norm2)
    return cosSimilarity

def similarity(grid: np.ndarray, spect1: np.ndarray, spect2: np.ndarray, sim_type: str):
    """Return the similarity between two XAS spectra using the provided metric
    It is assumed that the spectra are aligned to the same grids prior to calling this function
    The similarity is always defined such that maximization results in the most similar spectra

    Parameters
    ----------
    grid: 1d array - the common grid over which the spectra are defined
    spect1, spect2: 1d arrays of the absorbtion. 
    sim_type: The similarity metric
              "pearson" - The Pearson correlation
              "spearman" - The Spearman correlation
              "coss" - The cosine similarity v1 \dot v2 / |v1||v2|
              "kendalltaub" - Kendall's tau-b metric
              "normed_wasserstein" - The Wasserstein (aka earth-mover's) distance. We normalize the y-axis by its sum then treat each spectrum as a discrete 
                                     probability distribution (cf https://lilianweng.github.io/posts/2017-08-20-gan/#what-is-wasserstein-distance).
                                     The distance is then normalized to [0,1] by dividing by the full range of the grid, then subtracted from 1 as a similarity metric
              "coss_deriv" - The cosine similarity applied to the gradient of the two curves
    
    Output: the similarity metric
    """
    
    if sim_type == "pearson":
        metric = pearsonr(spect1, spect2)[0]        
    elif sim_type == "spearman":
        metric = spearmanr(spect1, spect2)[0]
    elif sim_type == "coss":
        metric = cos_similar(spect1,spect2)
    elif sim_type == "kendalltaub":
        metric = kendalltau(scipy.stats.rankdata(spect1), scipy.stats.rankdata(spect2))[0]
    elif sim_type == "normed_wasserstein":
        metric = 1 - wasserstein_distance(grid,grid,u_weights=spect1/np.sum(spect1), v_weights=spect2/np.sum(spect2))/(grid[-1] - grid[0])
    elif sim_type == "coss_deriv":
        metric = cos_similar(np.gradient(spect1,grid), np.gradient(spect2,grid))        
    else:
        raise Exception("Unknown sim_type")
    
    return metric
    

    
def spectra_corr(
        spectrum1: np.ndarray, spectrum2: np.ndarray, omega: float=0,
        grid: np.ndarray | Callable[ [np.ndarray, np.ndarray, float], np.ndarray ] = gridInterpolatorFixedN(300),
        verbose=True, method: str | list[str] = ["pearson","spearman","coss"]
)-> float | list[float] :
    """Calculate one or more similarity metrics for two spectra.
    Prior to computing the correlation, the spectra are interpolated to a common grid which can either be provided or determined automatically via the provided callable

    Parameters
    ----------
    spectrum1, spectrum2 : 2d-array
        Two-column arrays of energy vs. intensity XAS data.
    omega : float
        Shift between two spectra. spectrum2 shifted to spectrum2 + omega.
    grid : Common grid for interpolation. 
           If an array it will use this grid directly
           Otherwise it is treated as a callable that accepts the two spectra plus a shift and returns a grid
    method : A method or list of methods chosen from those supported by 'similarity' above

    Returns
    -------
    correlation : the correlation metric, or list of correlations if > 1 methods are provided
    """
    if not isinstance(grid, (list,np.ndarray)):
        grid = grid(spectrum1,spectrum2,omega)
        
    interp1 = interpolate.interp1d(
        spectrum1[:, 0],
        spectrum1[:, 1],
        assume_sorted=False,
        kind="cubic",
        bounds_error=False,
    )
    interp2 = interpolate.interp1d(
        spectrum2[:, 0] + omega,
        spectrum2[:, 1],
        assume_sorted=False,
        kind="cubic",
        bounds_error=False,
    )
    curve1 = interp1(grid)
    curve2 = interp2(grid)
    indices = ~(np.isnan(curve1) | np.isnan(curve2))

    correlation = np.array([ similarity(grid[indices], curve1[indices], curve2[indices], sim_type) for sim_type in (method if isinstance(method,list) else [method]) ])

    width = 0.5 * min(
        spectrum1[-1, 0] - spectrum1[0, 0], spectrum2[-1, 0] - spectrum2[0, 0]
    )
    # require 50% overlap

    if grid[indices][-1] - grid[indices][0] < width:
        decay = 0.9 ** (width / (grid[indices][-1] - grid[indices][0]))
        if verbose:
            print(
                "Overlap less than 50%%. Similarity values decayed by %0.4f"
                % decay
            )
        correlation *= decay

    return correlation if len(correlation) > 0 else correlation[0]


def max_corr(
    spectrum1 : np.ndarray,
    spectrum2 : np.ndarray,
    opt_strategy : str = "grid_search",
    start : float=-12,
    stop : float=12,
    step : float=0.01,
    grid: np.ndarray | Callable[ [np.ndarray, np.ndarray, float], np.ndarray ] = gridInterpolatorFixedN(300),
    method: str ="coss",
    shgo_iters : int = 10
):
    """Calculate the correlation between two spectra,
        and the amount of shift to obtain maximum correlation.
    
    This method uses a simple grid optimization of the shift

    Parameters
    ----------
    spectrum1, spectrum2 : 2d-array
        Two-column arrays of energy vs. intensity XAS.
    opt_strategy : str
         "grid_search" - compute the similarity at fixed steps between start and stop. Use 'step' to control the interval.
         "grid_search_and_local_opt" - perform a grid search first with interval 'step' then a local optimization within a window of +/- 3*step around the best value. Empirically it appears optimal to use this with a somewhat coarser step
         "shgo" - use the Simplicial Homology Global Optimization algorithm with simplicial sampling. Control the number of iterations with 'shgo_iters' (cf scipy.optimize.shgo documentation for more information)
    start, stop : float
        Shift of spectrum2 ranges from start to stop with start < stop
    step : float
        Step size used for the "grid_search" method
    grid : Common grid for interpolation. 
           If an array it will use this grid directly
           Otherwise it is treated as a callable that accepts the two spectra plus a shift and returns a grid
    method : One of the methods supported by 'similarity' above
        Empirically 'coss' (cosine similarity) works well.
    shgo_iters : The number of refinement iterations of the simplicial complex for the shgo algorithm

    Returns
    -------
    correlation : float
        The maximized value of the correlation
    m_shift : float
        Shift value at which the correlation is max.

    """

    if start >= stop:
        raise Exception("WARNING: Start {} is larger than stop {}]".format(start, stop))

    def opt_target(params):
        x = params[0] if isinstance(params,list) else params
        
        metric = -spectra_corr(spectrum1,spectrum2,omega=x,grid=grid,verbose=False,method=method)
        print(x,metric)
        return metric
    
    if opt_strategy in ("grid_search", "grid_search_and_local_opt"):
        correlation = {}

        #iterate from top of range to bottom (for some reason)
        i = stop
        while i > start:
            correlation[i] = spectra_corr(
                spectrum1,
                spectrum2,
                omega=i,
                grid=grid,
                verbose=False,
                method=method,
            )
            i -= step

        # find index at maximum correlation        
        max_corr_val = 0
        for i, j in correlation.items():
            if j > max_corr_val:
                max_corr_val = j
                m_shift = i

        if opt_strategy == "grid_search_and_local_opt":
            result = scipy.optimize.minimize_scalar(opt_target, bounds=(m_shift - 3*step, m_shift + 3*step))
            m_shift = result.x[0]
            max_corr_val = result.fun
                
    elif opt_strategy == "shgo":
        result = scipy.optimize.shgo(opt_target,
                                     [(start,stop)], sampling_method='simplicial', iters=10 )
        max_corr_val = result.fun
        m_shift = result.x[0]
    else:
        raise Exception("Invalid optimization strategy")
       
    # check if the gradient makes sense
    gplot1 = np.vstack(
        (
            spectrum1[:, 0],
            np.gradient(spectrum1[:, 1], spectrum1[1, 0] - spectrum1[0, 0]),
        )
    ).T
    gplot2 = np.vstack(
        (
            spectrum2[:, 0],
            np.gradient(spectrum2[:, 1], spectrum2[1, 0] - spectrum2[0, 0]),
        )
    ).T
    x1 = peak_loc(gplot1)
    x2 = peak_loc(gplot2)
    if abs(x1 - m_shift - x2) < 2:
        pass
    else:
        print(
            "XAS edge positions might not align. "
            "Better to plot and check the spectrum."
        )
    return max_corr_val, m_shift


def peak_loc(plot):
    """Locate the peak positon of a spectrum.

    Parameters
    ----------
    plot : 2d-array

    Returns
    -------
    position of the peak

    """
    return plot[plot[:, 1].argmax(), 0]
