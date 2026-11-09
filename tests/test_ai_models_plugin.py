"""End-to-end CLI test using a small residual ONNX, without GPU or real weights.

Run after installing the plugin: python -m unittest discover -s tests -p test_ai_models_plugin.py
Requires onnx for fixture generation, in addition to inference dependencies.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import xarray as xr
from ai_models_wenhai.model import WenHai


class PluginTest(unittest.TestCase):
    def test_framework_thread_option_is_preserved(self):
        code = (
            "import sys; from ai_models_wenhai.model import WenHai; "
            "from ai_models.__main__ import main; "
            "WenHai.print_assets_list=lambda self: print(self.num_threads); "
            "sys.argv=['ai-models','--num-threads','8','--assets-list','wenhai']; main()"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "8")

    def test_statistics_keep_upstream_rounding_and_float64_flux(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner = SimpleNamespace(find_asset=lambda name: root / name)
            for dtype in (np.float16, np.float64):
                low = np.array([1.123456789, 7.123456789], dtype=dtype)
                high = np.array([3.987654321, 11.987654321], dtype=dtype)
                np.save(root / "low.npy", low)
                np.save(root / "high.npy", high)
                actual_low, actual_span = WenHai._stats(owner, "low.npy", "high.npy", 2)
                np.testing.assert_array_equal(actual_low, low)
                np.testing.assert_array_equal(actual_span, high - low)
                self.assertEqual(actual_span.dtype, np.dtype(np.float64 if dtype == np.float64 else np.float32))

    def test_residual_rollout_native_and_prepared_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h, w = 3, 4
            low = np.arange(93, dtype=np.float32)
            np.save(root / "min_GLORYS.npy", low)
            np.save(root / "max_GLORYS.npy", low + 10)
            for name, value in (("min_flux", 0), ("max_flux", 1)):
                np.save(root / (name + ".npy"), np.full(8, value, dtype=np.float32))
            wet = np.ones((93, h, w), dtype=bool)
            wet[:, 0, 0] = False
            coords = {"latitude": [-1., 0., 1.], "longitude": [10., 11., 12., 13.], "depth": np.arange(23)}
            xr.Dataset({"mask": (("channel", "latitude", "longitude"), wet)},
                       coords={n: coords[n] for n in ("latitude", "longitude")}).to_netcdf(root / "mask_GLORYS.nc")
            state = np.broadcast_to((low + 2)[None, :, None, None], (1, 93, h, w)).copy()
            state[:, :, 0, 0] = np.nan
            initial = xr.Dataset({name: (("time", "depth", "latitude", "longitude"), state[:, start:start + 23])
                                  for name, start in (("uo", 0), ("vo", 23), ("thetao", 46), ("so", 69))},
                                 coords={"time": [np.datetime64("2019-01-01")], **coords})
            initial["zos"] = (("time", "latitude", "longitude"), state[:, 92])
            initial.to_netcdf(root / "initial.nc")
            flux = np.stack([np.full((8, h, w), value, dtype=np.float32) for value in (.1, .2)])
            np.save(root / "flux.npy", flux)
            nodes = [helper.make_node("Mul", ["state", "zero"], ["base"]),
                     helper.make_node("ReduceMean", ["flux"], ["increment"], keepdims=0),
                     helper.make_node("Add", ["base", "increment"], ["residual"])]
            graph = helper.make_graph(nodes, "residual-fixture", [
                helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, 93, h, w]),
                helper.make_tensor_value_info("flux", TensorProto.FLOAT, [1, 8, h, w])],
                [helper.make_tensor_value_info("residual", TensorProto.FLOAT, [1, 93, h, w])],
                [numpy_helper.from_array(np.array(0, dtype=np.float32), "zero")])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
            onnx.checker.check_model(model)
            onnx.save(model, root / "WenHai.onnx")

            def run(*extra, success=True):
                command = [sys.executable, "-m", "ai_models", "--assets", str(root), *map(str, extra)]
                result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result.stdout + result.stderr

            args = ["--lead-time", 48, "--file", root / "initial.nc", "--path", root / "native", "wenhai", "--provider", "cpu", "--flux-file", root / "flux.npy"]
            run(*args)
            for day, normalized in ((1, .3), (2, .5)):
                path = root / "native" / f"fcst2019010{day + 1}" / f"fcst2019010{day + 1}_lead{day}_byWenHai.nc"
                with xr.open_dataset(path) as ds:
                    self.assertEqual(set(ds.data_vars), {"uo", "vo", "thetao", "so", "zos"})
                    self.assertEqual(ds.sizes["depth"], 23)
                    self.assertEqual(str(ds.time.values[0])[:10], f"2019-01-0{day + 1}")
                    self.assertEqual(ds.attrs["lead_time_hours"], day * 24)
                    for name, channel in (("uo", 0), ("vo", 23), ("thetao", 46), ("so", 69)):
                        np.testing.assert_allclose(ds[name].values[0, :, 1:, :], (low[channel:channel + 23] + normalized * 10)[:, None, None] * np.ones((23, 2, w)), atol=2e-5)
                        self.assertTrue(np.isnan(ds[name].values[0, :, 0, 0]).all())
            self.assertIn("Output exists", run(*args, success=False))
            np.save(root / "state.npy", np.nan_to_num((state - low[None, :, None, None]) / 10))
            run("--lead-time", 24, "--file", root / "state.npy", "--path", root / "prepared", "wenhai", "--provider", "cpu", "--normalized-input", "--coordinate-file", root / "initial.nc", "--flux-file", root / "flux.npy")
            with xr.open_dataset(root / "prepared/fcst20190102/fcst20190102_lead1_byWenHai.nc") as a, xr.open_dataset(root / "native/fcst20190102/fcst20190102_lead1_byWenHai.nc") as b:
                xr.testing.assert_equal(a, b)
            run("--lead-time", 24, "--file", root / "initial.nc", "--path", root / "none", "--output", "none", "wenhai", "--provider", "cpu", "--flux-file", root / "flux.npy")
            self.assertFalse((root / "none").exists())
            np.save(root / "one_flux.npy", flux[:1])
            self.assertIn("at least 2 days", run("--lead-time", 48, "--file", root / "initial.nc", "--output", "none", "wenhai", "--flux-file", root / "one_flux.npy", success=False))


if __name__ == "__main__":
    unittest.main()
