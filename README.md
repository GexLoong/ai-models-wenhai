# ai-models-wenhai

`ai-models-wenhai` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run [WenHai](https://github.com/Cuiyingzhe/WenHai).

## Installation

To install the package, run:

```bash
pip install ai-models-wenhai
```

Until a PyPI release is published, install from this repository instead:

```bash
python -m pip install .
```

ONNX Runtime is installed separately: `onnxruntime` for CPU,
`onnxruntime-gpu` for NVIDIA CUDA, or the vendor's MACA ONNX Runtime wheel for
MetaX. `ONNXRUNTIME` is not used by this package. In an existing environment
with the required dependencies, `pip install --no-deps .` retains them.


`--download-assets` is not implemented, so download the assets from the upstream project, then point
`AI_MODELS_ASSETS` or `--assets` to their directory.

Native NetCDF forcing requires MetPy and **aerobulk-python**. Install MetPy with
`python -m pip install '.[bulk]'` when installing from this repository (or
`python -m pip install 'ai-models-wenhai[bulk]'` after a PyPI release).
Install aerobulk-python in the same runtime environment following the
[upstream conda-forge instructions](https://github.com/xgcm/aerobulk-python#installation).

## RUN

```bash
AI_MODELS_ASSETS=/path/to/models/wenhai \
ai-models --lead-time 24 \
  --file /path/to/models/wenhai/sample_GLORYS_23lev.nc \
  --path ./wenhai_output wenhai \
  --forcing-path /path/to/models/wenhai/sample_ERA5_d0.083.nc
```

`--lead-time` is in **hours**, a positive multiple of 24. `--path` is an output
directory, default `wenhai_output`. Each day writes
`fcstYYYYMMDD/fcstYYYYMMDD_leadN_byWenHai.nc`, using the forecast valid date,
with `uo,vo,thetao,so,zos`, FP32 values and NaN on land. Output is NetCDF, not GRIB.
Use `--overwrite` to replace existing files, or `--output none` to skip writing.

```bash
ai-models wenhai --help-wenhai
```
