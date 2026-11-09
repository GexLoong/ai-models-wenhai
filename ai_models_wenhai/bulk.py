"""Original WenHai bulk-flux equations, without inference-time global variables."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np


def _column_flux(arguments):
    from aerobulk.flux import noskin_np

    return noskin_np(*arguments, "ncar", 2, 10, 4, False)


def bulk_flux(state, forcing, wet, minimum, span, flux_minimum, flux_span, workers=1):
    try:
        from metpy.calc import specific_humidity_from_dewpoint
        from metpy.units import units
        from aerobulk.flux import noskin_np  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Raw forcing requires metpy and aerobulk-python (module aerobulk). "
            "Install the upstream bulk dependencies, or use --flux-file with prepared normalized flux."
        ) from exc
    if workers < 1:
        raise ValueError("--bulk-workers must be positive")
    required = ("t2m", "d2m", "u10", "v10", "mtpr", "ssr", "strd", "msl")
    missing = [name for name in required if name not in forcing]
    if missing:
        raise ValueError(f"Missing forcing variables: {missing}")
    # The upstream forcing file is float64. Preserve it for humidity and bulk
    # calculations; only the final normalized ONNX flux input becomes float16.
    fields = {name: np.asarray(forcing[name].values) for name in required}
    shape = state.shape[-2:]
    if any(array.shape != shape for array in fields.values()):
        raise ValueError(f"Forcing fields must match the ocean grid {shape}")
    u0 = (state[0, 0].astype(np.float32) * span[0] + minimum[0]) * wet[0]
    v0 = (state[0, 23].astype(np.float32) * span[23] + minimum[23]) * wet[23]
    t0 = ((state[0, 46].astype(np.float32) * span[46] + minimum[46]) * wet[46] + 273.15) * wet[46]
    humidity = specific_humidity_from_dewpoint(fields["msl"] * units.Pa, fields["d2m"] * units.K).to("kg/kg").magnitude
    humidity = np.nan_to_num(humidity)
    relative_u, relative_v = fields["u10"] - u0, fields["v10"] - v0
    columns = (
        tuple(np.ascontiguousarray(a[:, col]) for a in (t0, fields["t2m"], humidity, relative_u, relative_v, fields["msl"]))
        for col in range(shape[1])
    )
    if workers == 1:
        results = list(map(_column_flux, columns))
    else:
        # Never fork an already initialized GPU runtime.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_column_flux, columns, chunksize=max(1, shape[1] // (workers * 4))))
    qe, qh, taux, tauy, evap = (np.stack([result[i] for result in results], axis=1) for i in range(5))
    longwave = fields["strd"] / 3600 - 5.67e-8 * t0 ** 4
    flux = np.stack((longwave, fields["ssr"] / 3600, qh, qe, taux, tauy, evap / 1000, fields["mtpr"] / 1000))
    np.nan_to_num(flux, copy=False)
    flux -= flux_minimum[:, None, None]
    flux /= flux_span[:, None, None]
    flux *= wet[0]
    return np.clip(flux, 0, 1).astype(np.float16)[None]
