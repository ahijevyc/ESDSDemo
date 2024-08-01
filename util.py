import cartopy
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import uxarray
import xarray

def mkcoord(ds):
    if "t_iso_levels" in ds:
        ds = ds.swap_dims(dict(nIsoLevelsT="t_iso_levels", nIsoLevelsZ="z_iso_levels"))
    return ds


def dec_ax(ax, extent):
    ax.add_feature(cartopy.feature.STATES)
    ax.set_extent(extent)

    gl = ax.gridlines(draw_labels=True, x_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def todBZ(Z):
    dBZ = np.log10(Z) * 10
    if hasattr(Z, "uxgrid"):
        return uxarray.UxDataArray(dBZ, uxgrid=Z.uxgrid)
    else:
        return dBZ

def dBZfunc(dBZ, func):
    """function of linearized Z, not logarithmic dbZ"""
    Z = 10 ** (dBZ / 10)
    fZ = func(Z)
    return todBZ(fZ)

def trim_ll(grid_path, data_paths, lon_bounds, lat_bounds):
    """
    trim grid file and data file to bounds
    SLOW: compare to uxarray.Grid.subset
    """

    # Open the Grid file
    grid_ds = xarray.open_dataset(grid_path)

    grid_ds["lonCell"] = np.degrees(grid_ds.lonCell)
    grid_ds["latCell"] = np.degrees(grid_ds.latCell)

    # before computing the triangulation
    grid_ds["lonCell"] = ((grid_ds["lonCell"] + 180) % 360) - 180

    # Open data files
    ds = xarray.open_mfdataset(
        data_paths, preprocess=mkcoord, concat_dim="Time", combine="nested"
    )
    lon0, lon1 = lon_bounds
    lat0, lat1 = lat_bounds
    ibox = (
        (grid_ds.lonCell >= lon0)
        & (grid_ds.lonCell < lon1)
        & (grid_ds.latCell >= lat0)
        & (grid_ds.latCell < lat1)
    )
    # Trim grid
    grid_ds = grid_ds[["latCell", "lonCell"]].where(ibox, drop=True)

    # Trim data
    ds = ds.where(ibox, drop=True)

    return grid_ds, ds
