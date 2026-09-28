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
# The cluster has x86_64 nodes and IBM POWER (ppc64le) GPU nodes, and a Python environment only
# runs on the architecture it was built for. The script therefore builds one virtualenv per
# architecture (~/bexa-env-x86_64, ~/bexa-env-ppc64le) and one Jupyter kernel per architecture
# ("Python (bexa, x86_64)", "Python (bexa, ppc64le)"): run it once in a session of each kind you
# use, and pick the kernel that matches the session.
#
# Packages come from the cluster's Python environment when it has them (the virtualenv sees its
# packages) and otherwise from prebuilt packages only. PyPI has few prebuilt packages for POWER,
# and compiling numpy or pandas on a compute node takes long and often fails, so nothing is
# compiled. The checkout is made importable through a .pth file: nothing is installed or copied.
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
arch="$(uname -m)"
env_dir="${BEXA_ENV:-$HOME/bexa-env-$arch}"
kernel="bexa-$arch"
python_bin="${PYTHON:-python3}"

echo "bexa checkout:  $root"
echo "architecture:   $arch"
echo "environment:    $env_dir"
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "error: $python_bin is older than 3.10; load a newer Python module or run PYTHON=/path/to/python3 $0"
    exit 1
fi

venv_python() {  # bin/python on Linux, Scripts/python.exe on Windows
    if [ -x "$1/bin/python" ]; then echo "$1/bin/python"; else echo "$1/Scripts/python.exe"; fi
}
py="$(venv_python "$env_dir")"
if [ ! -x "$py" ]; then
    "$python_bin" -m venv --system-site-packages "$env_dir"
    py="$(venv_python "$env_dir")"
fi
pip_binary() {  # prebuilt packages only; already-present packages are kept
    "$py" -m pip install --quiet --disable-pip-version-check --only-binary=:all: "$@"
}
"$py" -m pip install --quiet --disable-pip-version-check --upgrade pip

# requirements.txt: everything is needed except tables (PAL-XFEL point files) and zarr (.zarr files)
required=()
optional=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    case "$line" in
        "") ;;
        tables* | zarr*) optional+=("$line") ;;
        *) required+=("$line") ;;
    esac
done < "$root/requirements.txt"

echo "installing what bexa needs (from the cluster's Python or prebuilt packages; nothing is compiled)"
if ! pip_binary "${required[@]}" ipykernel; then
    echo
    echo "error: a requirement is neither in the cluster's Python environment nor available as a"
    echo "  prebuilt package for $arch. Run the same install without --quiet to see which one:"
    echo "    $py -m pip install --only-binary=:all: -r $root/requirements.txt"
    exit 1
fi
for extra in "${optional[@]}" ipympl lmfit plotly; do
    case "$extra" in
        tables*) use="reading PAL-XFEL point files" ;;
        zarr*) use="writing .zarr files" ;;
        ipympl) use="interactive figures in notebooks" ;;
        lmfit) use="the rocking-curve fits" ;;
        *) use="interactive 3-D plots" ;;
    esac
    pip_binary "$extra" 2>/dev/null || echo "skipped $extra: no prebuilt package for $arch (only needed for $use)"
done

if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${BEXA_NO_CUPY:-}" ]; then
    if "$py" -c "import cupy" >/dev/null 2>&1; then
        echo "GPU: cupy is already available"
    elif [ "$arch" = x86_64 ] || [ "$arch" = aarch64 ]; then
        echo "a GPU is visible; installing cupy (skip with BEXA_NO_CUPY=1)"
        pip_binary cupy-cuda12x || echo "cupy did not install; bexa runs on the CPU"
    else
        echo "a GPU is visible, but PyPI has no prebuilt cupy for $arch, so bexa runs on the CPU here."
        echo "  Compiling cupy takes 30 minutes or more: load the CUDA toolkit, then $py -m pip install cupy"
    fi
fi

# the kernel finds the checkout through a .pth file: importable, not installed
site_dir="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
pth_root="$root"
if command -v cygpath >/dev/null 2>&1; then  # Git Bash on Windows: Python wants C:/... paths
    pth_root="$(cygpath -m "$root")"
fi
echo "$pth_root" > "$site_dir/bexa.pth"
"$py" -m ipykernel install --user --name "$kernel" --display-name "Python (bexa, $arch)" >/dev/null

if [ -n "${BEXA_ENV:-}" ]; then
    env_line="$env_dir"
else
    env_line="\$HOME/bexa-env-\$(uname -m)"  # resolved when the file is sourced, on that machine
fi
cat > "$HOME/.bexa_env" <<ENV
# source this file in a terminal before using bexa: source ~/.bexa_env
# it picks the environment built for the machine you are on
export PATH="$env_line/bin:$root/bin:\$PATH"
export PYTHONPATH="$root\${PYTHONPATH:+:\$PYTHONPATH}"
export BEXA_MEMORY_FRACTION=0.3
# export BEXA_CACHE_DIR=/data/visitor/<proposal>/id03/<date>/PROCESSED_DATA/bexa_cache
# export BEXA_PROFILE=<proposal>
ENV
chmod +x "$root/bin/bexa" 2>/dev/null || true

# JupyterLab loads front-end extensions from the user's data folder as well as the server's.
# When the server lacks the widgets, ipympl or plotly front ends ("Error displaying widget", a
# blank plotly figure), BEXA_LINK_LABEXTENSIONS=1 offers the ones this environment brought.
if [ -n "${BEXA_LINK_LABEXTENSIONS:-}" ]; then
    ext_src="$("$py" -c 'import sys; print(sys.prefix)')/share/jupyter/labextensions"
    ext_dst="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/labextensions"
    mkdir -p "$ext_dst"
    for ext in "$ext_src"/*/ "$ext_src"/@*/*/; do
        [ -f "$ext/package.json" ] || continue
        name="${ext#"$ext_src"/}"
        name="${name%/}"
        mkdir -p "$(dirname "$ext_dst/$name")"
        [ -e "$ext_dst/$name" ] || ln -s "${ext%/}" "$ext_dst/$name"
        echo "linked front-end extension $name"
    done
    echo "reload the JupyterLab page; to undo, remove the links in $ext_dst"
fi

echo
echo "checking the environment"
# from a neutral folder, so a checkout in the current folder cannot shadow the real package
if ! (cd / && "$py" -c "import bexa.core.scan, bexa.viz" && "$py" -m bexa settings); then
    echo "error: bexa does not import in $env_dir; send the lines above"
    exit 1
fi

echo
echo "done."
echo "  terminal:   source ~/.bexa_env && bexa settings"
echo "  notebook:   choose the kernel 'Python (bexa, $arch)', then: import bexa; bexa.settings()"
echo "  smoke test: bexa.demo('mosa').info()   (writes a synthetic scan to /tmp and opens it)"
echo "  profile:    cp $root/configs/beamtimes/example_esrf_id03.yaml $root/configs/beamtimes/<proposal>.yaml"
if [ -d "$HOME/bexa-env" ] && [ "$HOME/bexa-env" != "$env_dir" ]; then
    echo
    echo "an environment from an earlier setup is still at ~/bexa-env; it is not used any more:"
    echo "    rm -rf ~/bexa-env && $py -m jupyter kernelspec remove -y bexa"
fi
if [ ! -f "$root/bexa/core/resources.py" ]; then
    echo
    echo "warning: this checkout sizes its batches from the node's free memory, not from the SLURM"
    echo "  job's allocation, so a large reduction can get the session killed. Update the checkout"
    echo "  (git pull) once the memory fix is on GitHub; until then keep reductions small (an ROI,"
    echo "  downsample) or set BEXA_MEMORY_FRACTION to about job memory / node free memory."
fi
