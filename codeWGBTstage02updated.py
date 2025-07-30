# conda activate PyWBGT
import os, sys, glob
import numpy as np
from datetime import datetime, timedelta
from metpy.calc import wind_speed
from metpy.units import units
import xarray as xr
from coszenith import coszda, cosza
from WBGT import WBGT_Liljegren, WBGT_GCM
from WBGT import Tg_Liljegren, Tnwb_Liljegren
from WBGT import Tg_GCM, Tnwb_GCM
import dask
import dask.array as da
import thermofeel # thermofeel.calculate_heat_index_adjusted_v2 (https://github.com/masabhathini/thermofeel)
import pandas as pd
from dask_mpi import initialize
from dask.distributed import Client, wait
from wind2mdef import getexp_xarray, getwind2m  


##############################
resultdir = '/RESULTS/era5-land_results/'
era5_land = '/INPUTDATA/era5-land'
processingyear = sys.argv[1]
#processingyear = '1951'
processingyearP1 = str(int(processingyear) + 1) 
print('Processing year: ', processingyear)
###################
var_2d = glob.glob(era5_land + '/2d/hourlync/era5land.global.2d.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/2d/hourlync/era5land.global.2d.' + processingyearP1 + '.netcdf.nc')
var_2t = glob.glob(era5_land + '/2t/hourlync/era5land.global.2t.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/2t/hourlync/era5land.global.2t.' + processingyearP1 + '.netcdf.nc')
var_snsr = glob.glob(era5_land + '/snsr/hourlync/era5land.global.snsr.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/snsr/hourlync/era5land.global.snsr.' + processingyearP1 + '.netcdf.nc')
var_sntr = glob.glob(era5_land + '/sntr/hourlync/era5land.global.sntr.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/sntr/hourlync/era5land.global.sntr.' + processingyearP1 + '.netcdf.nc')
var_sp = glob.glob(era5_land + '/sp/hourlync/era5land.global.sp.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/sp/hourlync/era5land.global.sp.' + processingyearP1 + '.netcdf.nc')
var_ssrd = glob.glob(era5_land + '/ssrd/hourlync/era5land.global.ssrd.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/ssrd/hourlync/era5land.global.ssrd.' + processingyearP1 + '.netcdf.nc')
var_strd = glob.glob(era5_land + '/strd/hourlync/era5land.global.strd.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/strd/hourlync/era5land.global.strd.' + processingyearP1 + '.netcdf.nc')
var_tsdsrs = glob.glob(era5_land + '/tsdsrs/hourly/era5.global.tsdsrs.' + processingyear + '.nc') + glob.glob(era5_land + '/tsdsrs/hourly/era5.global.tsdsrs.' + processingyearP1 + '.nc')
var_u = glob.glob(era5_land + '/u/hourlync/era5land.global.u.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/u/hourlync/era5land.global.u.' + processingyearP1 + '.netcdf.nc')
var_v = glob.glob(era5_land + '/v/hourlync/era5land.global.v.' + processingyear + '.netcdf.nc') + glob.glob(era5_land + '/v/hourlync/era5land.global.v.' + processingyearP1 + '.netcdf.nc')
#######################
#######################
df_2d = xr.open_mfdataset(var_2d)
df_2t = xr.open_mfdataset(var_2t)
df_snsr = xr.open_mfdataset(var_snsr)
df_snsr['time'] = df_snsr['time'] + pd.Timedelta(hours=-1)
df_sntr = xr.open_mfdataset(var_sntr)
df_sntr['time'] = df_sntr['time'] + pd.Timedelta(hours=-1)
df_sp = xr.open_mfdataset(var_sp)
df_ssrd = xr.open_mfdataset(var_ssrd)
df_ssrd['time'] = df_ssrd['time'] + pd.Timedelta(hours=-1)
df_strd = xr.open_mfdataset(var_strd)
df_strd['time'] = df_strd['time'] + pd.Timedelta(hours=-1)
df_tsdsrs = xr.open_mfdataset(var_tsdsrs)
df_tsdsrs['time'] = df_tsdsrs['time'] + pd.Timedelta(hours=-1)
df_u = xr.open_mfdataset(var_u)
df_v = xr.open_mfdataset(var_v)
#######################
startDateF = pd.to_datetime(processingyear + '-12-01')
endDateF   = pd.to_datetime(processingyear + '-12-31')
#for datestamp in pd.date_range(startDateF,endDateF,freq='D')[:-2:-1]:
for datestamp in pd.date_range(startDateF,endDateF,freq='D'):
   datestampstr = datestamp.strftime('%Y-%m-%d')
   print(datestamp, datestampstr)
   outpath = datestamp.strftime( resultdir + '%Y/')
   print(outpath)
   os.makedirs(outpath,exist_ok=True)
   outfile = datestamp.strftime( outpath + 'result%Y-%m-%d.nc')
   ddf_2d = df_2d.sel(time=slice(datestamp, datestamp + timedelta(hours=23)))
   ddf_2t = df_2t.sel(time=slice(datestamp, datestamp + timedelta(hours=23)))
   ddf_snsr = (df_snsr.sel(time=slice(datestamp + timedelta(hours=-1), datestamp + timedelta(hours=0) + timedelta(hours=23))).diff('time')/3600).compute()
   ddf_snsr.loc[{'time':datestamp}] = (df_snsr.sel(time=datestamp+timedelta(hours=0))/3600).compute()
   ddf_sntr = (df_sntr.sel(time=slice(datestamp + timedelta(hours=-1), datestamp + timedelta(hours=0) + timedelta(hours=23))).diff('time')/3600).compute()
   ddf_sntr.loc[{'time':datestamp}] = (df_sntr.sel(time=datestamp+timedelta(hours=0))/3600).compute()
   ddf_sp = df_sp.sel(time=slice(datestamp, datestamp + timedelta(hours=23)))
   ddf_ssrd = (df_ssrd.sel(time=slice(datestamp + timedelta(hours=-1), datestamp + timedelta(hours=0) + timedelta(hours=23))).diff('time')/3600).compute()
   ddf_ssrd.loc[{'time':datestamp}] = (df_ssrd.sel(time=datestamp+timedelta(hours=0))/3600).compute()
   #ddf_ssrd = (ddf_ssrd.shift(time=-1) + ddf_ssrd)/2
   ddf_strd = (df_strd.sel(time=slice(datestamp + timedelta(hours=-1), datestamp + timedelta(hours=0) + timedelta(hours=23))).diff('time')/3600).compute()
   ddf_strd.loc[{'time':datestamp}] = (df_strd.sel(time=datestamp+timedelta(hours=0))/3600).compute()
   ddf_tsdsrs = (df_tsdsrs.sel(time=slice(datestamp, datestamp + timedelta(hours=23)))/3600).compute()
   ddf_tsdsrs = ddf_tsdsrs.interp(longitude=ddf_2t.longitude, latitude=ddf_2t.latitude)
   #ddf_tsdsrs = (ddf_tsdsrs.shift(time=-1) + ddf_tsdsrs)/2 # not needed
   ddf_u = df_u.sel(time=slice(datestamp, datestamp + timedelta(hours=23)))
   ddf_v = df_v.sel(time=slice(datestamp, datestamp + timedelta(hours=23)))
   ########
   ddf_ws10 = wind_speed(ddf_u.u10, ddf_v.v10).to_dataset(name='ws10')
   ddf_rh = thermofeel.calculate_relative_humidity_percent(ddf_2t.t2m,ddf_2d.d2m).to_dataset(name='rh')
   ddf_rh_org = ddf_rh.copy()
   ddf_rh = xr.where((ddf_rh <0) |(ddf_rh >105), np.nan, ddf_rh)
   ddf_rh = xr.where((ddf_rh >100) |(ddf_rh <=105), 100, ddf_rh)
   ddf_svp = thermofeel.calculate_saturation_vapour_pressure(ddf_2t).rename({'t2m':'svp'})
   #### Cosz ########
   ## https://github.com/QINQINKONG/PyWBGT/blob/v1.0.0/Jupyter_notebooks/Calculate_WBGT_with_CMIP6_data.ipynb
   #######################
   # create meshgrid of latitude and longitude, and we will calculate cosine zenith angle on these grids
   lon,lat=np.meshgrid(ddf_2t.longitude,ddf_2t.latitude)
   lat=lat*np.pi/180
   lon=lon*np.pi/180
   # specifiy the time seris for which we want to calculate cosine zenith angle 
   date=xr.DataArray(ddf_2t.time.values,dims=('time'),coords={'time':ddf_2t.time}).chunk({'time':len(ddf_2t.time)})
   # use dask.array map_blocks to calculate cosine zenith angle lazily and parallelly
   czda=da.map_blocks(coszda,date.data,lat,lon,1,chunks=(len(date),lat.shape[0],lat.shape[1]),new_axis=[1,2])
   # transfer to xarray DataArray
   ddf_coszda = xr.DataArray(czda,dims={'time': len(ddf_2t.time), 'latitude': len(ddf_2t.latitude), 'longitude': len(ddf_2t.longitude)}, coords=ddf_2t.coords).to_dataset(name = 'coszda')
   ddf_coszda = xr.where(ddf_coszda.coszda<=0.01,0.01,ddf_coszda.coszda).to_dataset(name="coszda")  ### added Aditionally.
   ddf_coszda = xr.where(np.isnan(ddf_2t.t2m),np.NaN,ddf_coszda.coszda).to_dataset(name="coszda")   ## added Aditionally.
   ddf_coszdaOri = ddf_coszda.copy()
   shifted_ds = ddf_coszda.assign_coords(time=ddf_coszda['time'] + pd.Timedelta(hours=1))
   shifted_ds['time'] = ddf_coszda.time
   ddf_coszda = shifted_ds.copy()
   #from wind2mdef import getexp_xarray, getwind2m  # personally defined.
   ddf_wind2m = getwind2m(ddf_ws10.ws10.transpose('time', 'latitude', 'longitude').values,ddf_coszda.coszda, ddf_ssrd.ssrd).to_dataset(name='ws2m')
   ddf_wind2m_org = ddf_wind2m.copy()
   ddf_wind2m = xr.where(ddf_wind2m.ws2m <= 1, 1, ddf_wind2m.ws2m).to_dataset()
   ddf_dsrp = (ddf_tsdsrs.fdir / ddf_coszda.where(ddf_coszda.coszda >= 0, ddf_coszda.coszda)).rename({'coszda':'dsrp'})
   #calculate_mean_radiant_temperature(ssrd, ssr, dsrp, strd, fdir, strr, cossza)
   ddf_coszda = ddf_coszdaOri.copy()
   ddf_mrt = thermofeel.calculate_mean_radiant_temperature(ddf_ssrd.ssrd, ddf_snsr.ssr, ddf_dsrp.dsrp, ddf_strd.strd, ddf_tsdsrs.fdir, ddf_sntr.str, ddf_coszda.coszda).to_dataset(name='mrt')
   ddf_utci = thermofeel.calculate_utci(ddf_2t.t2m, ddf_ws10.ws10, ddf_mrt.mrt, ddf_2d.d2m, ehPa=None).to_dataset(name='utci')
   f = (ddf_tsdsrs.fdir/ddf_ssrd.ssrd).to_dataset(name='f')
   f = xr.where(ddf_coszda.coszda <=0.01,0,f.f)
   f = xr.where(f>0.9,0.9,f)
   f = xr.where(f<0,0,f)
   f = xr.where(ddf_ssrd.ssrd <=0,0,f).to_dataset(name='f')
   f = xr.where(np.isnan(ddf_2t.t2m),np.NaN,f.f).to_dataset(name = 'f')
   ddf_rh = ddf_rh_org.copy()
   ddf_tnwb = xr.apply_ufunc(Tnwb_GCM,ddf_2t.t2m,ddf_rh.rh,ddf_sp.sp,ddf_wind2m.ws2m,ddf_ssrd.ssrd,ddf_ssrd.ssrd - ddf_snsr.ssr,ddf_strd.strd,(ddf_strd -ddf_sntr.str),f.f,ddf_coszda.coszda,True,dask="parallelized",output_dtypes=[float]).rename({'strd':'tnwb'})
   ddf_wbgt = thermofeel.calculate_wbgt(ddf_2t.t2m, ddf_mrt.mrt, ddf_wind2m.ws2m, ddf_2d.d2m).to_dataset(name='wbgt')
   ddf_tg = xr.apply_ufunc(Tg_GCM,ddf_2t.t2m,ddf_sp.sp,ddf_wind2m.ws2m,ddf_ssrd.ssrd,ddf_ssrd.ssrd - ddf_snsr.ssr,ddf_strd.strd,ddf_strd.strd - ddf_sntr.str,f.f,ddf_coszda.coszda,True,dask="parallelized",output_dtypes=[float]).to_dataset(name='tg')
   ddf_wbtindoor = thermofeel.calculate_wbt(ddf_2t.t2m, ddf_rh.rh).to_dataset(name='wbtindoor')
   ddf_wbtindoor_lsi = (ddf_wbtindoor.wbtindoor + 4.5*(1 - (ddf_rh.rh/100)**2)).to_dataset(name='lsi')
   ddf_humidex = thermofeel.calculate_humidex(ddf_2t.t2m, ddf_2d.d2m).to_dataset(name='humidex')
   ddf_net = thermofeel.calculate_normal_effective_temperature(ddf_2t.t2m, ddf_ws10.ws10, ddf_rh.rh).to_dataset(name='net')
   ddf_apparenttemp = thermofeel.calculate_apparent_temperature(ddf_2t.t2m, ddf_ws10.ws10, ddf_rh.rh).to_dataset(name='apparenttemp')
   ddf_windchill = thermofeel.calculate_wind_chill(ddf_2t.t2m, ddf_wind2m.ws2m).to_dataset(name='windchill')
   ddf_bgt = thermofeel.calculate_bgt(ddf_2t.t2m, ddf_mrt.mrt, ddf_wind2m.ws2m).to_dataset(name='bgt')
   ddf_heatindexadjusted = thermofeel.calculate_heat_index_adjusted_v2(ddf_2t.t2m, ddf_2d.d2m).to_dataset(name='heatindexadjusted')
   #### ATTRS
   longitudeAttrs = {'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'}
   latitudeAttrs = {'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'}
   tsdsrsAttrs = {'units': 'W m**-2', 'long_name': 'Total sky direct solar radiation at surface'}
   ssrdAttrs = {'units': 'W m**-2', 'long_name': 'Surface solar radiation downwards', 'standard_name': 'surface_downwelling_shortwave_flux_in_air'}
   strdAttrs = {'units': 'W m**-2', 'long_name': 'Surface thermal radiation downwards'}
   snsrAttrs = {'units': 'W m**-2', 'long_name': 'Surface net solar radiation', 'standard_name': 'surface_net_downward_shortwave_flux'}
   sntrAttrs = {'units': 'W m**-2', 'long_name': 'Surface net thermal radiation', 'standard_name': 'surface_net_thermal_radiation'}
   coszdaAttrs = {'units': 'unitless', 'long_name': 'average cosine zenith angle during only sunlit part', 'standard_name': 'coszda'} 
   wind2mAttrs = {'units': 'm s*-1', 'long_name': 'wind at 2 meter', 'standard_name': 'ws2m'}
   ws10Attrs = {'units': 'm s*-1', 'long_name': 'wind speed at 10 meter', 'standard_name': 'ws10m'}
   rhAttrs = {'units': '%', 'long_name': 'relative humidity', 'standard_name': 'rh2'}
   tnwbAttrs = {'units': 'K', 'long_name': 'Natural Wet Bulb Temperature', 'standard_name': 'tnw'}
   wbgtAttrs = {'units': 'K', 'long_name': 'Wet Bulb Globe Temperature', 'standard_name': 'wbgt'}
   tgAttrs = {'units': 'K', 'long_name': 'Globe Temperature', 'standard_name': 'tg'}
   mrtbgtAttrs = {'units': 'K', 'long_name': 'Mean Radiant Temperature from Globe Temperature', 'standard_name': 'mrtgt'}
   wbtindoorAttrs = {'units': 'K', 'long_name': 'Indoor Wet Bulb Temperature', 'standard_name': 'wbt'}
   lsiAttrs = {'units': 'K', 'long_name': 'Leathal Heat Stress Index', 'standard_name': 'lsi'}
   humidexAttrs = {'units': 'K', 'long_name': 'Humidex', 'standard_name': 'humidex'}
   netAttrs = {'units': 'K', 'long_name': 'Normal Effective Temperature ', 'standard_name': 'net'}
   apparenttempAttrs = {'units': 'K', 'long_name': 'Apparent Temperature', 'standard_name': 'apparenttemp'}
   windchillAttrs = {'units': 'K', 'long_name': 'Wind Chill Temperature', 'standard_name': 'windchill'}
   bgtAttrs = {'units': 'K', 'long_name': 'Black Globe Temperature', 'standard_name': 'Black Globe Temperature'}
   utciAttrs = {'units': 'K', 'long_name': 'Universal Thermal Climate Index', 'standard_name': 'Universal Thermal Climate Index'}
   heatindexadjustedAttrs = {'units': 'K', 'long_name': 'Heat Index Adjusted', 'standard_name': 'heatindexadjusted'}
   
   #### SAVE
   ddf_tsdsrs.longitude.attrs = longitudeAttrs
   ddf_tsdsrs.latitude.attrs = latitudeAttrs
   ddf_tsdsrs.fdir.attrs = tsdsrsAttrs
   ddf_tsdsrs.to_netcdf(outpath + datestampstr + 'tsdsrs.nc')
   ###
   ddf_ssrd.longitude.attrs = longitudeAttrs
   ddf_ssrd.latitude.attrs = latitudeAttrs
   ddf_ssrd.ssrd.attrs = ssrdAttrs
   ddf_ssrd.to_netcdf(outpath + datestampstr + 'ssrd.nc')
   ###
   ddf_strd.longitude.attrs = longitudeAttrs
   ddf_strd.latitude.attrs = latitudeAttrs
   ddf_strd.strd.attrs = strdAttrs
   ddf_strd.to_netcdf(outpath + datestampstr + 'strd.nc')
   ###
   ddf_snsr.longitude.attrs = longitudeAttrs
   ddf_snsr.latitude.attrs = latitudeAttrs
   ddf_snsr.ssr.attrs = snsrAttrs
   ddf_snsr.to_netcdf(outpath + datestampstr + 'snsr.nc')
   ###
   ddf_sntr.longitude.attrs = longitudeAttrs
   ddf_sntr.latitude.attrs = latitudeAttrs
   ddf_sntr.str.attrs = sntrAttrs
   ddf_sntr.to_netcdf(outpath + datestampstr + 'stsr.nc')
   ###
   ddf_coszda.longitude.attrs = longitudeAttrs
   ddf_coszda.latitude.attrs = latitudeAttrs
   ddf_coszda.coszda.attrs = coszdaAttrs
   ddf_coszda.to_netcdf(outpath + datestampstr + 'coszda.nc')
   ###
   ddf_wind2m = ddf_wind2m_org.copy()
   ddf_wind2m.longitude.attrs = longitudeAttrs
   ddf_wind2m.latitude.attrs = latitudeAttrs
   ddf_wind2m.ws2m.attrs = wind2mAttrs
   ddf_wind2m.to_netcdf(outpath + datestampstr + 'wind2m.nc')
   ###
   ddf_rh.longitude.attrs = longitudeAttrs
   ddf_rh.latitude.attrs = latitudeAttrs
   ddf_rh.rh.attrs = rhAttrs
   ddf_rh.to_netcdf(outpath + datestampstr + 'rh.nc')
   ###
   ddf_tnwb.longitude.attrs = longitudeAttrs
   ddf_tnwb.latitude.attrs = latitudeAttrs
   ddf_tnwb.tnwb.attrs = tnwbAttrs
   ddf_tnwb.to_netcdf(outpath + datestampstr + 'tnwb.nc')
   ###
   ddf_wbgt.longitude.attrs = longitudeAttrs
   ddf_wbgt.latitude.attrs = latitudeAttrs
   ddf_wbgt.wbgt.attrs = wbgtAttrs
   ddf_wbgt.to_netcdf(outpath + datestampstr + 'wbgt.nc')
   ###
   ddf_tg.longitude.attrs = longitudeAttrs
   ddf_tg.latitude.attrs = latitudeAttrs
   ddf_tg.tg.attrs = tgAttrs
   ddf_tg.to_netcdf(outpath + datestampstr + 'tg.nc')
   ###
   ddf_mrt.longitude.attrs = longitudeAttrs
   ddf_mrt.latitude.attrs = latitudeAttrs
   ddf_mrt.mrt.attrs = mrtbgtAttrs
   ddf_mrt.to_netcdf(outpath + datestampstr + 'mrt.nc')
   ###
   ddf_wbtindoor.longitude.attrs = longitudeAttrs
   ddf_wbtindoor.latitude.attrs = latitudeAttrs
   ddf_wbtindoor.wbtindoor.attrs = wbtindoorAttrs
   ddf_wbtindoor.to_netcdf(outpath + datestampstr + 'wbtindoor.nc')
   ###
   ddf_wbtindoor_lsi.longitude.attrs = longitudeAttrs
   ddf_wbtindoor_lsi.latitude.attrs = latitudeAttrs
   ddf_wbtindoor_lsi.lsi.attrs = lsiAttrs
   ddf_wbtindoor_lsi.to_netcdf(outpath + datestampstr + 'lsi.nc')
   ###
   ddf_humidex.longitude.attrs = longitudeAttrs
   ddf_humidex.latitude.attrs = latitudeAttrs
   ddf_humidex.humidex.attrs = humidexAttrs
   ddf_humidex.to_netcdf(outpath + datestampstr + 'humidex.nc')
   ###
   ddf_net.longitude.attrs = longitudeAttrs
   ddf_net.latitude.attrs = latitudeAttrs
   ddf_net.net.attrs = netAttrs
   ddf_net.to_netcdf(outpath + datestampstr + 'net.nc')
   ###
   ddf_apparenttemp.longitude.attrs = longitudeAttrs
   ddf_apparenttemp.latitude.attrs = latitudeAttrs
   ddf_apparenttemp.apparenttemp.attrs = apparenttempAttrs
   ddf_apparenttemp.to_netcdf(outpath + datestampstr + 'apparenttemp.nc')
   ###
   ddf_windchill.longitude.attrs = longitudeAttrs
   ddf_windchill.latitude.attrs = latitudeAttrs
   ddf_windchill.windchill.attrs = windchillAttrs
   ddf_windchill.to_netcdf(outpath + datestampstr + 'windchill.nc')
   ###
   ddf_bgt.longitude.attrs = longitudeAttrs
   ddf_bgt.latitude.attrs = latitudeAttrs
   ddf_bgt.bgt.attrs = bgtAttrs
   ddf_bgt.to_netcdf(outpath + datestampstr + 'bgt.nc')
   ###
   ddf_utci.longitude.attrs = longitudeAttrs
   ddf_utci.latitude.attrs = latitudeAttrs
   ddf_utci.utci.attrs = utciAttrs
   ddf_utci.to_netcdf(outpath + datestampstr + 'utci.nc')
   ## ###
   ddf_heatindexadjusted.longitude.attrs = longitudeAttrs
   ddf_heatindexadjusted.latitude.attrs = latitudeAttrs
   ddf_heatindexadjusted.ddf_heatindexadjusted.attrs = ddf_heatindexadjustedAttrs
   ddf_heatindexadjusted.to_netcdf(outpath + datestampstr + 'ddf_heatindexadjusted.nc')
   ## ###
