import logging
from pathlib import Path

import cartopy
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import pandas as pd
import uxarray
import xarray


def dBZfunc(dBZ, func):
    """function of linearized Z, not logarithmic dbZ"""
    Z = 10 ** (dBZ / 10)
    fZ = func(Z)
    return z_to_dbz(fZ)


def dec_ax(ax, extent):
    ax.add_feature(cartopy.feature.STATES)
    ax.set_extent(extent)

    gl = ax.gridlines(draw_labels=True, x_inline=False)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def mkcoord(ds):
    if "t_iso_levels" in ds:
        ds = ds.swap_dims(dict(nIsoLevelsT="t_iso_levels"))
    if "z_iso_levels" in ds:
        ds = ds.swap_dims(dict(nIsoLevelsZ="z_iso_levels"))
    if "u_iso_levels" in ds:
        ds = ds.swap_dims(dict(nIsoLevelsU="u_iso_levels"))
    return ds


def z_to_dbz(Z):
    dBZ = np.log10(Z) * 10
    if hasattr(Z, "uxgrid"):
        return uxarray.UxDataArray(dBZ, uxgrid=Z.uxgrid)
    else:
        return dBZ


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


def xtime(ds: xarray.Dataset):
    """convert xtime variable to datetime and assign to coordinate"""

    # remove one-element-long Time dimension
    ds = ds.squeeze(dim="Time", drop=True)

    logging.info("decode initialization time variable")
    initial_time = pd.to_datetime(
        ds["initial_time"].load().item().decode("utf-8").strip(),
        format="%Y-%m-%d_%H:%M:%S",
    )

    # assign initialization time variable to its own coordinate
    ds = ds.assign_coords(
        initial_time=(
            ["initial_time"],
            [initial_time],
        ),
    )

    # extract member number from part of file path
    # assign to its own coordinate
    filename = Path(ds.encoding["source"])
    mem = [p for p in filename.parts if p.startswith("mem")]
    if mem:
        mem = mem[0].lstrip("mem_")
        mem = int(mem)
        ds = ds.assign_coords(mem=(["mem"], [mem]))

    logging.info("decode valid time and assign to variable")
    valid_time = pd.to_datetime(
        ds["xtime"].load().item().decode("utf-8").strip(),
        format="%Y-%m-%d_%H:%M:%S",
    )
    ds = ds.assign(valid_time=[valid_time])

    # calculate forecast hour and assign to variable
    forecastHour = (valid_time - initial_time) / pd.to_timedelta(1, unit="hour")
    ds = ds.assign(forecastHour=float(forecastHour))

    return ds