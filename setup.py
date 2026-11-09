from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="ai-models-wenhai",
    version="0.1.0",
    description="WenHai ocean forecasting adapter for ai-models and ONNX Runtime",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="European Centre for Medium-Range Weather Forecasts (ECMWF)",
    author_email="software.support@ecmwf.int",
    url="https://github.com/GexLoong/ai-models-wenhai",
    project_urls={
        "Source": "https://github.com/GexLoong/ai-models-wenhai",
        "Issues": "https://github.com/GexLoong/ai-models-wenhai/issues",
    },
    packages=find_packages(include=["ai_models_wenhai", "ai_models_wenhai.*"]),
    python_requires=">=3.9",
    # ai-models 0.7 imports the pre-1.0 earthkit FieldArray API.
    install_requires=["ai-models>=0.7,<0.8", "earthkit-data>=0.20,<1", "numpy>=1.21,<2", "xarray", "pandas", "netCDF4"],
    extras_require={"bulk": ["metpy"]},
    entry_points={
        "ai_models.model": ["wenhai = ai_models_wenhai.model:WenHai"],
        "ai_models.output": ["wenhai-netcdf = ai_models_wenhai._common:NetCDFOutput"],
    },
)
