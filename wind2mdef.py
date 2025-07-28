import xarray as xr
import numpy as np
###############
def getexp_xarray(cosz, wind, ssrd):
    # Assuming cosz, wind, ssrd are xarray DataArrays
    result = wind * 0 + 999  # Initialize result array with same shape and coords as wind
    c11 = xr.where((cosz > 0) & (ssrd >= 925) & (wind >= 5), True, False)
    c12 = xr.where((cosz > 0) & (ssrd >= 675) & (ssrd < 925) & (wind >= 5) & (wind < 6), True, False)
    c13 = xr.where((cosz > 0) & (ssrd >= 175) & (ssrd < 675) & (wind >= 2) & (wind < 5), True, False)
    c1 = c11 | c12 | c13
    c21 = xr.where((cosz > 0) & (ssrd >= 675) & (ssrd < 925) & (wind >= 6), True, False)
    c22 = xr.where((cosz > 0) & (ssrd >= 175) & (ssrd < 675) & (wind >= 5), True, False)
    c23 = xr.where((cosz > 0) & (ssrd < 175), True, False)
    c24 = xr.where((cosz <= 0) & (wind >= 2.5), True, False)
    c2 = c21 | c22 | c23 | c24
    c31 = xr.where((cosz > 0) & (ssrd >= 925) & (wind < 5), True, False)
    c32 = xr.where((cosz > 0) & (ssrd >= 675) & (ssrd < 925) & (wind < 5), True, False)
    c33 = xr.where((cosz > 0) & (ssrd >= 175) & (ssrd < 675) & (wind < 2), True, False)
    c3 = c31 | c32 | c33
    c4 = xr.where((cosz <= 0) & (wind < 2.5), True, False)
    result = xr.where(c1, 0.2, result)
    result = xr.where(c2, 0.25, result)
    result = xr.where(c3, 0.15, result)
    result = xr.where(c4, 0.3, result)
    result = xr.where(~(c1 | c2 | c3 | c4), np.NaN, result)
    return result

#exp_values = getexp_xarray(coszda.coszda, df_ws10.ws10.values, df_ssrd.ssrd)

def getwind2m(wind10m,cosz,ssrd):    # obtain 2m wind from 10m wind    
   wind2m=wind10m * ((2.0/10.0)** getexp_xarray(cosz, wind10m, ssrd))
   xr.where(wind2m<0.13,0.13,wind2m)
   #wind2m=0.13 if wind2m<0.13 else wind2m
   return wind2m
