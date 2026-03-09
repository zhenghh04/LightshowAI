import numpy as np
import scipy
import math

def loadShakeupKernel(filename, invert_w=True):
    """
    Load the kernel A(w) from a file
    The energy and kernel values are assumed to be in the 1st and 2nd columns, respectively

    Inputs:
      filename : The name of the file containing the kernel data
      invert_w : The kernel is often stored and plotted such that the secondary peaks are to the left of the origin, to indicate that the convolution pulls in values of the spectrum
                 from lower energies. However, the actual convolution is  Q'(w) = \int_{-oo}^{oo} dw' Q(w') A(w-w'),   hence the convolution pulls from lower values when w > w', i.e. at positive values of A(w)
                 Set invert_w = True to invert the x-axis of the input kernel if its mass is below the origin.

    Return the kernel as a 2D tensor with energy on the 1st dimension and A(w) on the 2nd
    """

    kern = np.loadtxt(filename)[:,0:2]
    
    if invert_w:
        kern[:,0] = -kern[:,0]
        kern = np.flip(kern, axis=0)
    return kern

def convolveSamples(fsamples, gsamples, dy):
    """
    Compute 
    (f * g)(x) = \int_{-\infty}^{\infty} dy f(y) g(x-y)   
               = \int_{-\infty}^{\infty} dy g(y) f(x-y)

    fsamples: function f sampled on  y0 + n*dy  for n \in [0,N)
    gsamples: function g samples on  -(N-1) + n'*dy for  n' \in [0,2N-1)

    returns (f * g) sampled on y0 + n*dy  for n \in [0,N)
    """
    N=len(fsamples)
    assert len(gsamples) == 2*N-1
    c = np.convolve(fsamples, gsamples, mode="full")*dy
    c = c[N-1 : N-1+N]
    return c


def shakeup(spectrum, Aw, pad_right=0, pad_left=0, truncate_right=0, dw=None):
    """
    Perform the shakeup correction
    Inputs:
      spectrum: The XAS spectrum as a 2D tensor with energy on the 1st dimension and the value on the 2nd
      Aw : the shakeup convolution kernel as a 2D tensor with energy on the 1st dimension and the value on the 2nd
      pad_left, pad_right : add zero padding to the left/right of the XAS data prior to convolution to this size in energy units (eV). This can help if the span of the XAS data is smaller than that of the kernel, resulting in the kernel becoming truncated when interpolated onto the internal grid
      truncate_right: If the kernel has a small portion in the negative energy region, the convolution pulls a small amount of the spectrum from higher energies. This can result in a very steep, unphysical dropoff at the end of the convolved spectrum. Use truncate_right (value in energy units) to truncate this off.
      dw: None (default) | float. The spacing of the output energy grid. By default it will use the spacing of the Aw kernel but can be overridden by setting this value.
    """

    if dw == None:
        dw = Aw[1,0] - Aw[0,0]
    
    y0 = spectrum[0,0] - pad_left
    y1 = spectrum[-1,0] + pad_right

    N = math.floor( (y1-y0)/dw ) + 1
    grid = y0 + dw * np.arange(N)
    print("Max",grid[-1], "target", y1, "samples", N, "dw",dw)

    #interpolate onto the fine grid and pad with zeroes as appropriate
    ius = scipy.interpolate.CubicSpline(spectrum[:,0], spectrum[:,1], extrapolate=False)
    spectrum_interp = np.nan_to_num(ius(grid), nan=0)

    #Interpolate Aw onto a symmetrized grid of size 2N-1 and the same spacing
    Aw_grid = np.linspace( -(N-1)*dw, (N-1)*dw, num=2*N-1, endpoint=True)
    assert abs(Aw_grid[1]-Aw_grid[0] - dw) < 1e-6
    ius = scipy.interpolate.CubicSpline(Aw[:,0], Aw[:,1], extrapolate=False) #this will leave nans if it has to extrapolate
    Aw_interp = ius(Aw_grid)

    print(f"Truncated/interpolated Aw from range {Aw[0,0],Aw[-1,0],Aw[1,0]-Aw[0,0]} to range {Aw_grid[0], Aw_grid[1], dw}")
    
    #Need to truncate back to original range
    start = np.argmin( np.abs(grid-spectrum[0,0]) )
    lessthan = np.argmin( np.abs(grid-spectrum[-1,0]) ) + 1
    assert lessthan <= len(grid)
    
    grid = grid[start:lessthan]
    conv = convolveSamples(spectrum_interp, Aw_interp, dw)[start:lessthan]

    #Remove the downwards turn
    if truncate_right != 0:
        targ = grid[-1]-truncate_right
        idx = np.argmin(np.abs(grid-targ))
        print(f"Truncate convolved spectrum to idx {idx} (w={grid[idx]}) vs original len {len(grid)} (w={grid[-1]})")
        grid=grid[:idx+1]
        conv=conv[:idx+1]
    
    return np.vstack((grid,conv)).T
