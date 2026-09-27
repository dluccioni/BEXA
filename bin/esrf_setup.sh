#!/usr/bin/env bash
# One-time setup of bexa on the ESRF cluster, from a JupyterLab terminal (jupyter-slurm.esrf.fr)
# or an ssh session. Clone the code first, then run this script:
#
#     git clone https://github.com/dluccioni/BEXA.git ~/bexa
#     bash ~/bexa/bin/esrf_setup.sh          # the script inside the checkout
#     bash ~/esrf_setup.sh ~/bexa            # or the script uploaded on its own, with the checkout
#
# Without an argument the checkout is the folder above this script when the script sits in a
# checkout's bin/, and ~/bexa otherwise.
#
# It creates a virtualenv (~/bexa-env) that also sees the packages of the system Python,
# installs the requirements into it, registers the environment as a Jupyter kernel called
# "Python (bexa)", makes the checkout importable from that kernel (a .pth file, nothing is
# installed or copied), and writes the shell variables to ~/.bexa_env for terminal use.
# Nothing here needs root. Re-running is safe.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${1:-}" ]; then
    if [ ! -d "$1" ]; then
        echo "error: $1 is not a folder"
        exit 1
    fi
    root="$(cd "$1" && pwd)"
elif [ -f "$here/../bexa/__init__.py" ]; then
    root="$(cd "$here/.." && pwd)"
else
    root="$HOME/bexa"
fi
if [ ! -f "$root/bexa/__init__.py" ]; then
    echo "error: no bexa checkout at $root"
    echo "clone it first (git clone https://github.com/dluccioni/BEXA.git ~/bexa) or pass its path:"
    echo "    bash $0 /path/to/bexa"
    exit 1
fi
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
if [ ! -f "$root/bexa/core/resources.py" ]; then
    echo
    echo "warning: this checkout sizes its batches from the node's free memory, not from the SLURM"
    echo "  job's allocation, so a large reduction can get the session killed. Update the checkout"
    echo "  (git pull) once the memory fix is on GitHub; until then keep reductions small (an ROI,"
    echo "  downsample) or set BEXA_MEMORY_FRACTION to about job memory / node free memory."
fi
