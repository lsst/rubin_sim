#!/bin/bash
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --chdir=/sdf/scratch/users/n/neilsen/avgclouds
#SBATCH --output=/sdf/scratch/users/n/neilsen/avgclouds/lsst64-%j.out
#SBATCH --error=/sdf/scratch/users/n/neilsen/avgclouds/lsst64-%j.err

set -euo pipefail

# Keep the checkout and all job artifacts in this scratch directory.
workdir=/sdf/scratch/users/n/neilsen/avgclouds
cd "$workdir"
checkout="$workdir/rubin_sim"
historical_data="$workdir/clouds_1975-01-01_to_2022-09-04.txt"
test -f "$historical_data"
test -f "$checkout/tests/clouds/test_average_clouds.py"
export LSST64_HISTORICAL_DATA="$historical_data"
export TMPDIR="$workdir/tmp"
export UV_CACHE_DIR="$workdir/uv-cache"
export UV_PYTHON_INSTALL_DIR="$workdir/uv-python"
export XDG_CACHE_HOME="$workdir/cache"
export MPLCONFIGDIR="$workdir/matplotlib"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME"
venv="$workdir/.venv"
if [[ ! -f "$venv/bin/activate" ]]; then
    uv venv "$venv"
fi
uv pip install --python "$venv/bin/python" --editable "$checkout[test,dev,extras,vseqarchive]"
source "$venv/bin/activate"
export MPLBACKEND=Agg
report="$workdir/lsst64-${SLURM_JOB_ID:-local}.xml"
python -m pytest -v "$checkout/tests/clouds/test_average_clouds.py::test_historical_data_workflow" \
    -o junit_family=legacy --junitxml="$report" \
    -o cache_dir="$workdir/pytest-cache" \
    --basetemp="$workdir/lsst64-${SLURM_JOB_ID:-local}"
# Pytest exits successfully for skipped tests; require actual passing evidence.
python - "$report" <<'PY'
import sys
import xml.etree.ElementTree as ET

cases = ET.parse(sys.argv[1]).findall(".//testcase")
if len(cases) != 1 or cases[0].get("name") != "test_historical_data_workflow":
    raise SystemExit("Expected exactly the historical-data acceptance test")
if any(cases[0].find(tag) is not None for tag in ("skipped", "failure", "error")):
    raise SystemExit("Historical-data acceptance test did not pass")
properties = {item.get("name"): item.get("value") for item in cases[0].findall("./properties/property")}
if not 0 < float(properties.get("workflow_seconds", "inf")) < 3600:
    raise SystemExit("Missing or unacceptable workflow timing evidence")
print(f"20-year historical workflow passed in {properties['workflow_seconds']} seconds")
PY
