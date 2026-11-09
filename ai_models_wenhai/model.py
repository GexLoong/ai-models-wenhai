"""Native WenHai inputs and equations exposed through the ai-models lifecycle."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ._common import OceanModel, check_shape, check_wet_finite, tensor_dtype
from .bulk import bulk_flux

LOG = logging.getLogger(__name__)
DEPTHS = np.array([
    0.4940, 2.6457, 5.0782, 7.9296, 11.4050, 15.8101, 21.5988, 29.4447,
    40.3441, 55.7643, 77.8539, 92.3261, 109.7293, 130.6660, 155.8507,
    186.1256, 222.4752, 266.0403, 318.1274, 380.2130, 453.9377, 541.0889, 643.5668,
], dtype=np.float32)


class WenHai(OceanModel):
    model_name = "wenhai"
    netcdf_output_name = "wenhai-netcdf"
    expver = "wenh"
    hour_steps = 24
    download_files = ["WenHai.onnx", "min_GLORYS.npy", "max_GLORYS.npy", "min_flux.npy", "max_flux.npy", "mask_GLORYS.nc"]

    def parse_model_args(self, args):
        parser = argparse.ArgumentParser("ai-models wenhai", add_help=False)
        parser.add_argument("--help-wenhai", action="help")
        self.add_common_arguments(parser)
        parser.add_argument("--forcing-path", "--forcing_path", help="Daily ERA5/IFS forcing NetCDF")
        parser.add_argument("--flux-file", help="Prepared normalized flux NPY, shape (days,8,lat,lon)")
        parser.add_argument("--normalized-input", action="store_true", help="--file is normalized state NPY (1,93,lat,lon)")
        parser.add_argument("--coordinate-file", help="NetCDF supplying exact time/depth/latitude/longitude for NPY input")
        parser.add_argument("--init-path", "--init_path", help="Alias for the ocean initial NetCDF/NPY passed by --file")
        parser.add_argument("--bulk-workers", type=int, default=1)
        return parser.parse_args(args)

    @property
    def asset_files(self):
        root = Path(self.assets).expanduser()
        return [str(self.find_asset(name, optional=True) or root / name) for name in self.download_files]

    def print_fields(self):
        print("WenHai: daily ocean NetCDF uo,vo,thetao,so (23 depths), zos; one initial time.")
        print("Forcing: t2m,d2m,u10,v10,mtpr,ssr,strd,msl, on the same grid.")
        print("Alternatively use --normalized-input with --flux-file; see --help-wenhai.")

    def _stats(self, lower, upper, count):
        raw_low = np.load(self.find_asset(lower)).reshape(-1)
        raw_high = np.load(self.find_asset(upper)).reshape(-1)
        dtype = np.result_type(raw_low.dtype, raw_high.dtype, np.float32)
        low = raw_low.astype(dtype)
        # GLORYS statistics are float16: preserve its max-min rounding.
        # Flux statistics are float64 and must not lose precision here.
        span = (raw_high - raw_low).astype(dtype)
        if low.size != count or span.size != count or not np.isfinite(low).all() or not np.isfinite(span).all() or np.any(span <= 0):
            raise ValueError(f"{lower}/{upper} must have {count} finite min/max entries with max > min")
        return low, span

    @staticmethod
    def _rename_coordinates(dataset):
        aliases = {old: new for old, new in (("lat", "latitude"), ("lon", "longitude")) if old in dataset.dims and new not in dataset.dims}
        return dataset.rename(aliases) if aliases else dataset

    def _initial_state(self, minimum, span):
        path = self.init_path or self.file
        if not path:
            raise ValueError("Pass the ocean initial condition using --file or --init-path")
        path = Path(path).expanduser()
        if self.normalized_input:
            if path.suffix.lower() != ".npy":
                raise ValueError("--normalized-input expects a .npy state array")
            state = np.load(path, mmap_mode="r")
            if state.ndim != 4 or state.shape[:2] != (1, 93):
                raise ValueError(f"Normalized state must be (1,93,lat,lon), received {state.shape}")
            if not np.issubdtype(state.dtype, np.floating):
                raise ValueError("Normalized state must have a floating-point dtype")
            return state, None, None
        with xr.open_dataset(path) as opened:
            ds = self._rename_coordinates(opened)
            missing = [v for v in ("uo", "vo", "thetao", "so", "zos") if v not in ds]
            if missing:
                raise ValueError(f"Missing ocean initial variables: {missing}")
            if ds.sizes.get("time", 1) != 1 or ds.sizes.get("depth") != 23:
                raise ValueError("WenHai initial NetCDF requires one time and 23 depths")
            shape = (ds.sizes["latitude"], ds.sizes["longitude"])
            state = np.empty((1, 93) + shape, dtype=np.float32)
            for offset, var in enumerate(("uo", "vo", "thetao", "so")):
                da = ds[var].isel(time=0, drop=True) if "time" in ds[var].dims else ds[var]
                state[0, offset * 23:(offset + 1) * 23] = da.transpose("depth", "latitude", "longitude").values
            da = ds.zos.isel(time=0, drop=True) if "time" in ds.zos.dims else ds.zos
            state[0, 92] = da.transpose("latitude", "longitude").values
            coords = {name: ds[name].values.copy() for name in ("depth", "latitude", "longitude")}
            date = ds.time.values[0] if "time" in ds.coords else None
        state -= minimum[None, :, None, None]
        state /= span[None, :, None, None]
        np.nan_to_num(state, copy=False, nan=0, posinf=0, neginf=0)
        return state, coords, date

    def _mask_and_coordinates(self, shape, coords):
        with xr.open_dataset(self.find_asset("mask_GLORYS.nc")) as opened:
            ds = self._rename_coordinates(opened)
            if "mask" not in ds:
                raise ValueError("mask_GLORYS.nc must contain variable mask (1=ocean, 0=land)")
            wet = np.asarray(ds["mask"].values, dtype=bool)
            if wet.ndim == 4 and wet.shape[0] == 1:
                wet = wet[0]
            try:
                wet = np.broadcast_to(wet, (93,) + shape)
            except ValueError as exc:
                raise ValueError(f"Ocean mask {wet.shape} does not match state (93,{shape})") from exc
            if coords is None:
                coords = {n: ds[n].values.copy() for n in ("latitude", "longitude", "depth") if n in ds.coords}
        date = None
        if self.coordinate_file:
            with xr.open_dataset(self.coordinate_file) as opened:
                ds = self._rename_coordinates(opened)
                coords.update({n: ds[n].values.copy() for n in ("latitude", "longitude", "depth") if n in ds.coords})
                date = ds.time.values[0] if "time" in ds.coords else None
        if "latitude" not in coords or "longitude" not in coords:
            if shape != (2041, 4320):
                raise ValueError("NPY inputs need coordinate metadata; pass --coordinate-file")
            coords.update(latitude=np.linspace(-80, 90, 2041), longitude=np.arange(4320) / 12 - 180)
        coords.setdefault("depth", DEPTHS)
        if len(coords["depth"]) != 23 or len(coords["latitude"]) != shape[0] or len(coords["longitude"]) != shape[1]:
            raise ValueError("Coordinate lengths do not match the initial state")
        return wet, coords, date

    def _dataset(self, state, wet, coords, initial, valid, hours, minimum, span):
        data = state[0].astype(np.float32, copy=True)
        data *= span[:, None, None]
        data += minimum[:, None, None]
        for channel in range(93):
            np.copyto(data[channel], np.nan, where=~wet[channel])
        common = {"time": [valid.to_datetime64()], **coords}
        ds = xr.Dataset(
            {name: (("time", "depth", "latitude", "longitude"), data[None, start:start + 23])
             for name, start in (("uo", 0), ("vo", 23), ("thetao", 46), ("so", 69))},
            coords=common,
        )
        ds["zos"] = (("time", "latitude", "longitude"), data[None, 92])
        for name, unit in (("uo", "m s-1"), ("vo", "m s-1"), ("thetao", "degrees_C"), ("so", "1e-3"), ("zos", "m")):
            ds[name].attrs["units"] = unit
        ds.attrs.update(model="WenHai", forecast_reference_time=initial.isoformat(), lead_time_hours=int(hours))
        return ds

    def run(self):
        days = self.forecast_days(self.lead_time)
        if bool(self.forcing_path) == bool(self.flux_file):
            raise ValueError("Provide exactly one of --forcing-path or --flux-file")
        model_path = self.find_asset("WenHai.onnx")
        minimum, span = self._stats("min_GLORYS.npy", "max_GLORYS.npy", 93)
        state, coords, date = self._initial_state(minimum, span)
        wet, coords, coordinate_date = self._mask_and_coordinates(state.shape[-2:], coords)
        initial = self.initial_date(date if date is not None else coordinate_date)
        if self.normalized_input:
            check_wet_finite(state, np.zeros((1,) + state.shape[-2:], dtype=bool), "Normalized input")
        names = [f"fcst{(initial + pd.Timedelta(days=i)).strftime('%Y%m%d')}/fcst{(initial + pd.Timedelta(days=i)).strftime('%Y%m%d')}_lead{i}_byWenHai.nc" for i in range(1, days + 1)]
        if self.writes_netcdf:
            self.output.check_paths(names)
        flux_data = forcing = None
        try:
            if self.flux_file:
                flux_data = np.load(self.flux_file, mmap_mode="r")
                if flux_data.ndim != 4 or flux_data.shape[1:] != (8,) + state.shape[-2:] or len(flux_data) < days:
                    raise ValueError(f"Flux must be (days,8,lat,lon) with at least {days} days, received {flux_data.shape}")
            else:
                forcing = self._rename_coordinates(xr.open_dataset(self.forcing_path))
                if forcing.sizes.get("time", 0) < days:
                    raise ValueError(f"Forcing has fewer than {days} daily time steps")
                for name in ("latitude", "longitude"):
                    if name in forcing.coords and not np.allclose(forcing[name].values, coords[name]):
                        raise ValueError(f"Forcing {name} coordinates differ from ocean coordinates")
                flux_minimum, flux_span = self._stats("min_flux.npy", "max_flux.npy", 8)
            session = self.create_session(model_path)
            infos = session.get_inputs()
            if len(infos) != 2:
                raise ValueError("WenHai requires exactly two ONNX inputs (state, flux)")
            state_info = next((i for i in infos if len(i.shape) == 4 and i.shape[1] == 93), infos[0])
            flux_info = next((i for i in infos if len(i.shape) == 4 and i.shape[1] == 8), infos[1])
            if state_info.name == flux_info.name:
                raise ValueError("Cannot distinguish state and flux inputs")
            check_shape(state_info, state.shape)
            with self.stepper(24) as stepper:
                for i in range(days):
                    if flux_data is not None:
                        flux = np.asarray(flux_data[i:i + 1])
                    else:
                        flux = bulk_flux(state, forcing.isel(time=i), wet, minimum, span, flux_minimum, flux_span, self.bulk_workers)
                    if not np.isfinite(flux).all() or np.any(flux < 0) or np.any(flux > 1):
                        raise ValueError("Normalized flux must be finite and within [0,1]")
                    check_shape(flux_info, flux.shape)
                    feed = {state_info.name: np.asarray(state, dtype=tensor_dtype(state_info)), flux_info.name: np.asarray(flux, dtype=tensor_dtype(flux_info))}
                    prediction = session.run(None, feed)[0]
                    if prediction.shape != state.shape:
                        raise ValueError(f"Residual shape {prediction.shape} does not match state {state.shape}")
                    del feed
                    np.add(prediction, state, out=prediction)
                    prediction *= wet[None]
                    np.clip(prediction, 0, 1, out=prediction)
                    state = prediction
                    check_wet_finite(state, ~wet[None], "WenHai prediction")
                    hours = (i + 1) * 24
                    if self.writes_netcdf:
                        ds = self._dataset(state, wet, coords, initial, initial + pd.Timedelta(hours=hours), hours, minimum, span)
                        try:
                            self.output.write_dataset(ds, names[i])
                        finally:
                            ds.close()
                            del ds
                    stepper(i, hours)
        finally:
            if forcing is not None:
                forcing.close()
