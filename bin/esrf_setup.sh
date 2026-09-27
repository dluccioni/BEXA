#!/usr/bin/env bash
# One-time setup of bexa on the ESRF cluster, from a JupyterLab terminal (jupyter-slurm.esrf.fr)
# or an ssh session:
#
#     bash ~/bexa/bin/esrf_setup.sh
#
# It creates a virtualenv next to the checkout that also sees the packages of the system Python,
# installs the requirements into it, registers the environment as a Jupyter kernel called
# "Python (bexa)", makes the checkout importable from that kernel (a .pth file, nothing is
# installed or copied), and writes the shell variables to ~/.bexa_env for terminal use.
# Nothing here needs root. Re-running is safe.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${BEXA_ENV:-$HOME/bexa-env}"
python_bin="${PYTHON:-python3}"

echo "bexa checkout:  $root"
echo "environment:    $env_dir"
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "error: $python_bin is older than 3.10; load a newer Python module or run PYTHON=/path/to/python3 $0"
    exit 1
fi

if [ ! -x "$env_dir/bin/python" ]; then
    "$python_bin" -m venv --system-site-packages "$env_dir"
fi
py="$env_dir/bin/python"
"$py" -m pip install --quiet --upgrade pip
"$py" -m pip install --quiet -r "$root/requirements.txt" ipykernel ipympl lmfit plotly
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "a GPU is visible; installing cupy (skip with BEXA_NO_CUPY=1)"
    [ -n "${BEXA_NO_CUPY:-}" ] || "$py" -m pip install --quiet cupy-cuda12x || echo "cupy did not install; bexa runs on the CPU"
fi

# the kernel finds the checkout through a .pth file: importable, not installed
site_dir="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$root" > "$site_dir/bexa.pth"
"$py" -m ipykernel install --user --name bexa --display-name "Python (bexa)" >/dev/null

cat > "$HOME/.bexa_env" <<ENV
# source this file in a terminal before using bexa: source ~/.bexa_env
export PATH="$env_dir/bin:$root/bin:\$PATH"
export PYTHONPATH="$root\${PYTHONPATH:+:\$PYTHONPATH}"
export BEXA_MEMORY_FRACTION=0.3
# export BEXA_CACHE_DIR=/data/visitor/<proposal>/id03/<date>/PROCESSED_DATA/bexa_cache
# export BEXA_PROFILE=<proposal>
ENV
chmod +x "$root/bin/bexa" 2>/dev/null || true

echo
echo "done."
echo "  terminal:   source ~/.bexa_env && bexa settings"
echo "  notebook:   choose the kernel 'Python (bexa)', then: import bexa; bexa.settings()"
echo "  smoke test: bexa.demo('mosa').info()   (writes a synthetic scan to /tmp and opens it)"
echo "  profile:    cp $root/configs/beamtimes/example_esrf_id03.yaml $root/configs/beamtimes/<proposal>.yaml"
