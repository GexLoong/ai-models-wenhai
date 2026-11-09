"""Optional native aerobulk regression against the upstream WenHai equations."""

import importlib.util
import unittest

import numpy as np
import xarray as xr

from ai_models_wenhai.bulk import bulk_flux


@unittest.skipUnless(importlib.util.find_spec("aerobulk") and importlib.util.find_spec("metpy"), "requires aerobulk-python and MetPy")
class BulkFluxTest(unittest.TestCase):
    def test_native_flux_and_spawn_workers_match_upstream_equations(self):
        from aerobulk.flux import noskin_np
        from metpy.calc import specific_humidity_from_dewpoint
        from metpy.units import units

        shape = (4, 3)
        wet = np.ones((93,) + shape, dtype=bool)
        wet[:, 0, 0] = False
        minimum, span = np.zeros(93, dtype=np.float32), np.full(93, 10, dtype=np.float32)
        minimum[[0, 23, 46]] = -2
        span[46] = 40
        state = np.full((1, 93) + shape, .55, dtype=np.float32)
        values = {"t2m": 290., "d2m": 285., "u10": 6., "v10": 2., "mtpr": .0001,
                  "ssr": 360000., "strd": 1100000., "msl": 101325.}
        forcing = xr.Dataset({name: (("latitude", "longitude"), np.full(shape, value, dtype=np.float64)) for name, value in values.items()})
        flux_min = np.array([-500., 0., -500., -500., -1., -1., -1e-5, 0.], dtype=np.float64)
        flux_span = np.array([1000., 500., 1000., 1000., 2., 2., 2e-5, 1e-6], dtype=np.float64)
        u0 = (state[0, 0] * span[0] + minimum[0]) * wet[0]
        v0 = (state[0, 23] * span[23] + minimum[23]) * wet[23]
        t0 = ((state[0, 46] * span[46] + minimum[46]) * wet[46] + 273.15) * wet[46]
        fields = {name: forcing[name].values.copy() for name in values}
        fields["ssr"] /= 3600
        fields["strd"] /= 3600
        fields["mtpr"] /= 1000
        h2m = np.nan_to_num(specific_humidity_from_dewpoint(fields["msl"] * units.Pa, fields["d2m"] * units.K).to("kg/kg").magnitude)
        columns = [noskin_np(t0[:, i], fields["t2m"][:, i], h2m[:, i], fields["u10"][:, i] - u0[:, i],
                            fields["v10"][:, i] - v0[:, i], fields["msl"][:, i], "ncar", 2, 10, 4, False) for i in range(shape[1])]
        qe, qh, taux, tauy, evap = (np.array([column[i] for column in columns]).T.reshape(shape) for i in range(5))
        expected = np.nan_to_num(np.stack((fields["strd"] - 5.67e-8 * t0**4, fields["ssr"], qh, qe, taux, tauy, evap / 1000, fields["mtpr"])))
        expected = (expected - flux_min[:, None, None]) / flux_span[:, None, None]
        expected *= wet[0]
        expected = expected.astype(np.float16).clip(0, 1)[None]
        for workers in (1, 2):
            actual = bulk_flux(state, forcing, wet, minimum, span, flux_min, flux_span, workers=workers)
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
