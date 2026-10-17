# bexa: Beamline EXperiment Analysis

Tools for X-ray beamtime data: reading the raw files of several beamlines, streaming reductions
(sums, previews, centre-of-mass maps, per-point pump-probe cubes) that adapt to the machine's
memory and GPU, analysis functions ported from the lab's notebooks, and plots for terminal,
notebook and batch use.

- Formats: ESRF ID03 BLISS (2024 to 2026 layouts), PAL-XFEL point files and spec-style files,
  LCLS XCS cubes and smalldata, EuXFEL through `extra_data`, legacy and bexa cubes, plain arrays.
- Reductions: one pass per scan, batches sized from free memory, prefetching, cupy when a GPU is
  present, several scans at a time in worker processes, results as xarray with provenance,
  cached on disk.
- Analysis: dark and hot-pixel handling, alignment and stitching, rocking-curve statistics and
  lmfit fits, pump-probe binning and run combination, peak statistics, segmentation, strain
  fields, four-circle diffractometer solvers, zone axis and omega search, attenuation and Bragg
  magnifier optics.
- Interfaces: `bexa` command line (`scan`, `cube`, `auto`, `crystal`, `optics`, `bench`),
  Python API (`bexa.open`, `bexa.open_profile`, `scan.reduce`), matplotlib plots with optional
  plotly and napari front ends.

This README is also the reference. [Function reference](#function-reference) documents every
public function, class, method and property with its parameters, what it returns and an
example, and [Command-line reference](#command-line-reference) documents every command and
option. Search the page for a name (`motor_com`, `ROI.parse`, `bexa scan reduce`) or for a word
(`strain`, `attenuation`, `GIF`, `samz`); [Find a function by task](#find-a-function-by-task)
lists the everyday jobs with the function that does each.

## Contents

- [Getting started](#getting-started)
- [Examples](#examples)
- [Layout](#layout)
- [Walkthrough: ID03 data on the ESRF Jupyter-SLURM service](#walkthrough-id03-data-on-the-esrf-jupyter-slurm-service)
- [Find a function by task](#find-a-function-by-task)
- [Function reference](#function-reference)
  - Opening, reducing, regions of interest: [Top-level names](#top-level-names-bexa) · [bexa.core.scan](#bexacorescan-opening-scans-and-the-scan-handle) · [bexa.core.status](#bexacorestatus-the-state-of-a-scan) · [bexa.core.dataset](#bexacoredataset-the-scans-of-one-folder-or-several) · [bexa.core.scan_table](#bexacorescan_table-the-rows-of-a-datasets-table) · [bexa.core.stacks](#bexacorestacks-statistics-brightest-condition-and-colour-limits-of-stacked-volumes) · [bexa.core.roi](#bexacoreroi-regions-of-interest) · [bexa.core.reductions (bexa.acc)](#bexacorereductions-bexaacc-accumulators-and-the-streaming-engine) · [bexa.core.accessor](#bexacoreaccessor-bexa-on-xarray-objects)
  - Reading and writing files: [bexa.io.formats](#bexaioformats-format-specs-sniffing-and-drafting) · [bexa.io.survey](#bexaiosurvey-every-dataset-under-a-folder) · [bexa.io.base](#bexaiobase-the-source-protocol-and-basesource) · [bexa.io.scan_commands](#bexaioscan_commands-scan-command-lines) · [bexa.io.h5files](#bexaioh5files-opening-hdf5-files-another-program-is-writing) · [bexa.io.engines.hdf5_stack](#bexaioengineshdf5_stack-esrf-bliss-master-and-detector-files) · [bexa.io.engines.hdf5_master](#bexaioengineshdf5_master-what-the-bliss-master-says-about-a-scan) · [bexa.io.engines.hdf5_points](#bexaioengineshdf5_points-pal-xfel-point-files-september-2025) · [bexa.io.engines.spec_h5](#bexaioenginesspec_h5-pal-xfel-spec-style-scan-files-april-2025) · [bexa.io.engines.spec_tiff](#bexaioenginesspec_tiff-aps-7-id-spec-sessions-with-a-tiff-per-point) · [bexa.io.engines.array_cube](#bexaioenginesarray_cube-reduced-cubes-as-sources) · [bexa.io.engines.smalldata](#bexaioenginessmalldata-lcls-smalldata-run-files) · [bexa.io.engines.extra_data](#bexaioenginesextra_data-euxfel-runs-through-extra_data) · [bexa.io.generic](#bexaiogeneric-plain-arrays-tiff-stacks-and-fileh5path) · [bexa.io.multi](#bexaiomulti-several-scans-as-one-source) · [bexa.io.inspect](#bexaioinspect-hdf5-trees-and-path-summaries) · [bexa.io.cube](#bexaiocube-bexa-files-bexasave-and-bexaload) · [bexa.io.darfix](#bexaiodarfix-the-bridge-to-darfix)
  - Analysis: [bexa.analysis.preprocess](#bexaanalysispreprocess-darks-hot-pixels-normalisation-cropping-binning) · [bexa.analysis.rocking](#bexaanalysisrocking-per-pixel-rocking-curve-statistics) · [bexa.analysis.peaks](#bexaanalysispeaks-the-peak-on-the-detector-frame-by-frame) · [bexa.analysis.fitting](#bexaanalysisfitting-rocking-curve-and-edge-fits) · [bexa.analysis.registration](#bexaanalysisregistration-shifts-alignment-stitching-and-mosaics) · [bexa.analysis.projections](#bexaanalysisprojections-projections-integrated-maps-and-rlp-point-clouds) · [bexa.analysis.segmentation](#bexaanalysissegmentation-regions-from-maps) · [bexa.analysis.fields](#bexaanalysisfields-vector-fields-boundaries-and-strain-integration) · [bexa.analysis.pump_probe](#bexaanalysispump_probe-laser-onoff-analysis)
  - Plots and notebooks: [bexa.viz.style](#bexavizstyle-plot-conventions-themes-and-colour-limits) · [bexa.viz.auto](#bexavizauto-one-call-picks-the-figure) · [bexa.viz.images](#bexavizimages-single-images-tiles-roi-boxes-and-scalebars) · [bexa.viz.curves](#bexavizcurves-rocking-curves-pump-probe-panels-and-moments) · [bexa.viz.maps](#bexavizmaps-centre-of-mass-bivariate-hsv-and-vector-field-maps) · [bexa.viz.volume](#bexavizvolume-projections-slices-isosurfaces-and-3-d-renderings) · [bexa.viz.interactive](#bexavizinteractive-sliders-and-roi-pickers) · [bexa.viz.sliders](#bexavizsliders-what-the-slider-browsers-share) · [bexa.viz.animation](#bexavizanimation-gif-and-mp4-of-a-stack) · [bexa.viz.napari_app](#bexaviznapari_app-napari-viewers-and-the-apps-behind-them) · [bexa.notebook](#bexanotebook-jupyter-and-vs-code-helpers) · [bexa.notebook.picker](#bexanotebookpicker-choosing-scans-in-a-notebook) · [bexa.notebook.listing](#bexanotebooklisting-the-scans-listed-and-a-selection-across-folders) · [bexa.notebook.selection](#bexanotebookselection-scans-chosen-in-several-dataset-folders) · [bexa.notebook.picker_state](#bexanotebookpicker_state-what-a-picker-remembers) · [bexa.notebook.picker_widgets](#bexanotebookpicker_widgets-the-widget-views-of-the-picker) · [bexa.notebook.selection_panel](#bexanotebookselection_panel-the-selected-scans-of-every-folder)
  - Pipelines: [bexa.pipelines.dfxm_report](#bexapipelinesdfxm_report-the-standard-dfxm-figure-set) · [bexa.pipelines.xfel_cube](#bexapipelinesxfel_cube-pal-xfel-point-files-to-a-laser-onoff-cube) · [bexa.pipelines.cube_report](#bexapipelinescube_report-figures-and-gifs-of-one-cube) · [bexa.pipelines.auto_process](#bexapipelinesauto_process-the-beamtime-watcher) · [bexa.pipelines.benchmark](#bexapipelinesbenchmark-timings-and-baselines)
  - Synthetic data: [bexa.testing.demo](#bexatestingdemo-bexademo-synthetic-data-in-one-call) · [bexa.testing.synthetic](#bexatestingsynthetic-small-valid-files-for-every-layout-with-the-planted-truth)
  - Geometry and crystallography: [bexa.geometry.conventions](#bexageometryconventions-named-lab-frames-and-motor-stacks) · [bexa.geometry.detector](#bexageometrydetector-detector-and-beam-geometry) · [bexa.geometry.transforms](#bexageometrytransforms-pixels-to-angles-to-q-to-hkl) · [bexa.crystal.lattice](#bexacrystallattice-cells-d-spacings-and-reflections) · [bexa.crystal.cif](#bexacrystalcif-cell-and-space-group-from-cif-files) · [bexa.crystal.diffractometer](#bexacrystaldiffractometer-four-circle-motor-solutions) · [bexa.crystal.zone_axis](#bexacrystalzone_axis-the-zone-axis-from-one-reflection) · [bexa.crystal.omega_search](#bexacrystalomega_search-omega-where-two-peaks-share-a-mu-window) · [bexa.crystal.sample_frame](#bexacrystalsample_frame-sample-to-grain-matrix)
  - Units and optics: [bexa.core.units](#bexacoreunits-physical-constants-and-unit-conversions) · [bexa.optics.conversions](#bexaopticsconversions-energy-wavelength-and-monochromator-angles) · [bexa.optics.attenuation](#bexaopticsattenuation-transmission-and-attenuation-lengths) · [bexa.optics.nist](#bexaopticsnist-nist-mass-attenuation-tables) · [bexa.optics.bragg_magnifier](#bexaopticsbragg_magnifier-asymmetric-bragg-magnifier)
  - Machinery and configuration: [bexa.core.structure](#bexacorestructure-dims-coordinates-and-frame-to-grid-mapping-of-a-scan) · [bexa.core.backend](#bexacorebackend-numpy-or-cupy-devices-memory-budget-batch-sizes) · [bexa.core.resources](#bexacoreresources-memory-and-cpus-of-the-process-slurm-cgroups) · [bexa.core.settings](#bexacoresettings-what-bexa-uses-right-now-bexasettings) · [bexa.core.cache](#bexacorecache-the-result-cache) · [bexa.core.parallel](#bexacoreparallel-threads-processes-and-prefetching) · [bexa.core.provenance](#bexacoreprovenance-provenance-attrs-and-fingerprints) · [bexa.core.registry](#bexacoreregistry-plugin-registries-for-engines-accumulators-and-plots) · [bexa.config](#bexaconfig-profiles-models-and-paths-in-one-import) · [bexa.config.loader](#bexaconfigloader-finding-loading-and-writing-beamtime-profiles) · [bexa.config.models](#bexaconfigmodels-the-fields-of-a-beamtime-profile) · [bexa.config.paths](#bexaconfigpaths-repository-cache-folder-and-path-aliases) · [bexa._log](#bexa_log-logging-and-progress-bars) · [bexa._profiling](#bexa_profiling-timing-a-block-with-cprofile-bexaprofile) · [bexa._help](#bexa_help-the-api-listing-bexahelp) · [Environment variables](#environment-variables)
- [Command-line reference](#command-line-reference)
  - [Global options](#global-options) · [General commands](#general-commands) · [bexa profile](#bexa-profile-beamtime-profiles-and-format-specs) · [bexa cache](#bexa-cache-the-result-cache) · [bexa scan](#bexa-scan-frame-stack-scans) · [bexa cube](#bexa-cube-laser-onoff-cubes-of-xfel-runs) · [bexa auto](#bexa-auto-the-beamtime-watcher) · [bexa crystal](#bexa-crystal-crystallography) · [bexa optics](#bexa-optics-x-ray-optics-estimates) · [bexa bench](#bexa-bench-timings-and-regressions) · [Python helpers of the command line](#python-helpers-of-the-command-line)

## Getting started

```
python -m pip install -r requirements.txt
python -m pytest -q                      # about 300 tests on synthetic data (about 3 minutes)
python -m bexa --help
```

Python 3.10 or newer is required. See `docs/quickstart.md`, `docs/formats.md`,
`docs/cookbook.md`, `docs/memory_and_speed.md`, `docs/cli.md`, `docs/beamtime_checklist.md` and
the scripts in `examples/`.

## Examples

Every script in `examples/` runs on synthetic data written to a temporary folder, as a plain
script or as `# %%` cells in VS Code or Jupyter. Swap the generated dataset for a real one and
the rest stays the same.

| Script | Case |
|---|---|
| `01_esrf_scan_preview_and_report.py` | open a scan, preview it, centre-of-mass maps, the DFXM report |
| `02_pal_xfel_cube.py` | build and combine PAL-XFEL laser on/off cubes |
| `03_lcls_cube_binning.py` | bin LCLS per-shot data into a cube |
| `04_stitching_and_fitting.py` | stitch overlapping rocking scans, fit the rocking curve |
| `05_crystal_and_optics.py` | reflections, motor solutions, attenuation, Bragg magnifier |
| `06_strain_fields_and_segmentation.py` | vector fields, boundaries, k-means regions |
| `07_alignment_scans.py` | the first quick scans: centre, width, in-range and saturation checks, advice |
| `08_beamtime_scan_log.py` | a live table and contact sheet of every scan in a dataset |
| `09_zstack_maps.py` | a z-stack with a mosaicity scan and an energy series per height: maps, strain, sliders |
| `10_zstack_3d_conditions.py` | the z-stack as a 3-D volume stepped through (chi, mu) or (energy, mu); the Slices-to-Voxels model and GUI |
| `11_roi_to_3d_rlp.py` | from a pixel ROI on the projections to the 3-D reciprocal-space view |
| `12_esrf_dfxm_workflow.ipynb` | a notebook for an ID03 DFXM beamtime end to end: alignment scans, quick mosa scan, fine mosa scan, energy scan, z-stack, 3-D rendering with a slider per scanned motor |
| `13_esrf_dfxm_general.ipynb` | one notebook for any ID03 dataset (a single mosaicity scan, a z-stack, an energy series, both, or several folders): it reads what varies between the scans and stacks on that; one streaming pass per scan (several scans at a time with `WORKERS`) gives the binned previews and the full-resolution maps, stacked into one file in the cache that every later cell reads block by block; section 10 also renders the total scattered intensity per voxel, and section 11 compares two sets of scans from any folders, B minus A pair by pair (intensity and tilt changes) |

## Layout

```
bexa/            the package (core, io, analysis, geometry, crystal, optics, viz, pipelines, cli)
configs/formats  format specs (YAML) - one per file layout
configs/beamtimes  beamtime profiles (add yours; see bexa profile new)
tests/           pytest suite with synthetic generators for every format
examples/        runnable scripts mirroring the legacy workflows
docs/            guides
benchmarks/      run.py and per-machine baselines
```

Version 1 is a plain repository: run it from a checkout with `python -m bexa`, or the `bexa.cmd`
/ `bin/bexa` shims. Packaging is planned for a later release (PLAN.md, section 20).

## Walkthrough: ID03 data on the ESRF Jupyter-SLURM service

This section goes from a fresh clone in your ESRF home directory to finished maps, entirely
inside a JupyterLab session started from https://jupyter-slurm.esrf.fr. The same steps work in a
terminal on the cluster, on the beamline workstation, and later on the lab PC with a copy of
the data.

### 1. Get the code there and install once

The public branch, `main`, is on GitHub at https://github.com/dluccioni/BEXA. Open a terminal
in JupyterLab at https://jupyter-slurm.esrf.fr (File > New > Terminal):

```
git clone https://github.com/dluccioni/BEXA.git ~/bexa
bash ~/bexa/bin/esrf_setup.sh
```

If the clone has no `bin/esrf_setup.sh`, copy the script from the lab PC to your ESRF home
(the upload button of the JupyterLab file browser, or `scp`) and give it the checkout:
`bash ~/esrf_setup.sh ~/bexa`.

GitHub carries the library, the format specs and the example profiles. The examples,
notebooks and guides live on the private `development` branch: copy the ones you need into
your ESRF home the same way, for example `examples/12_esrf_dfxm_workflow.ipynb` (one beamtime
end to end) or `examples/13_esrf_dfxm_general.ipynb` (one notebook for any dataset). The bexa
kernel imports bexa wherever the notebook sits. To bring the whole `development` branch instead, run
`git bundle create ../bexa.bundle --all` on the lab PC, upload the file and
`git clone -b development ~/bexa.bundle ~/bexa`.

The setup script makes a virtualenv that also sees the cluster's Python packages, adds what
is missing from `requirements.txt` plus ipykernel, ipympl, lmfit and plotly (cupy too when a GPU
is visible), registers a Jupyter kernel, puts the checkout on that kernel's path through a
`.pth` file (importable, not installed), and writes `~/.bexa_env` for terminals. It only uses
prebuilt packages and never compiles, and it names what it skipped and what that is needed
for. Re-running it is safe. Updates during the beamtime: push `main` from the lab PC, then
`cd ~/bexa && git pull` on the cluster.

The cluster has x86_64 nodes and IBM POWER (ppc64le) GPU nodes, and an environment only runs on
the architecture it was built for. The script therefore builds one per architecture,
`~/bexa-env-x86_64` or `~/bexa-env-ppc64le`, with the kernel "Python (bexa, x86_64)" or
"Python (bexa, ppc64le)": run it once in a session of each kind you use, and pick the kernel
that matches the session. `~/.bexa_env` picks the right environment by itself.

Notes for the ESRF environment:

- `hdf5plugin` is in `requirements.txt` on purpose: the PCO detector files are written with
  compression filters that plain h5py cannot open. bexa registers the filters on import.
- pip needs internet access. If it times out on a compute node, run the setup script from the
  login node or a terminal with access, or ask the local contact for the proxy setting.
- GPU on the POWER nodes: PyPI has no prebuilt cupy for POWER, so bexa runs on the CPU there
  unless the cluster's Python already provides cupy; the setup script says which. Compiling
  cupy takes 30 minutes or more with the CUDA toolkit loaded. On the CPU, a 2 x 2 downsample
  makes a centre-of-mass pass four times faster.
- Memory: inside a SLURM job bexa reads the job's allocation from its cgroup and sizes every
  batch from that, not from the node's memory, so a Jupyter-SLURM session is not killed for
  going over its limit. `bexa settings` shows the limit and where it came from. The worker
  counts follow the cores the job owns: `bexa.stack(..., workers="auto")` reduces one scan per
  core but one at the same time, each worker with its share of the job's memory, and the
  Gaussian smoothing of the centre of mass runs in up to eight threads on the CPU.
- Previews of full scans are gigabytes, so keep the result cache out of your home quota: set
  `processed_root` in the beamtime profile (the cache goes to `<processed_root>/bexa_cache`), or
  `export BEXA_CACHE_DIR=/data/visitor/<proposal>/id03/<date>/PROCESSED_DATA/bexa_cache` in the
  terminal (in a notebook, `os.environ["BEXA_CACHE_DIR"] = ...` before `import bexa`). The cache
  holds ordinary bexa `.h5` files, so anything in it is also a valid product.
- Interactive figures and sliders need the ipympl and ipywidgets extensions of the JupyterLab
  server. If a figure shows "Error displaying widget", the server lacks them: rerun the setup
  with `BEXA_LINK_LABEXTENSIONS=1 bash ~/bexa/bin/esrf_setup.sh`, which offers the extensions
  the environment brought to JupyterLab, and reload the page. With static (inline) figures the
  slider browsers still work: they redraw the figure when a slider is released.
- 3-D renderings (`render`, `browse_render`, the notebook's stack cells) are plotly figures
  shown as inline frames, which work in any JupyterLab without the plotly extension; the
  browser loads plotly.js from the web unless `include_plotlyjs=True` is passed.

### 2. Start a notebook

Pick the kernel "Python (bexa, x86_64)" or "Python (bexa, ppc64le)", whichever matches the
session (Kernel > Change Kernel). The checkout is on its path already:

```python
import bexa
bexa.notebook.setup()   # widget backend when ipympl is installed (inline otherwise), wide cells, logs
bexa.settings()         # device, memory limit of the job, cache folder, profile
```

Another kernel works too, after `import sys; sys.path.append("/path/to/bexa")`.

`bexa.help()` lists the everyday names with a one-line summary each; `bexa.help("all")` every
public function, `bexa.help("plots")` one group. `bexa.settings()` prints the device, the memory
budget, the cache folder and the profile in use, and where each of them comes from. Without data
at hand, `bexa.demo("mosa")` or `bexa.demo("zstack")` writes a synthetic dataset to try things on.

### 3. Where the data is and what bexa sees

BLISS writes one folder per dataset with a master HDF5 file (motors, scan parameters, energy)
and one sub-folder of detector files per scan:

```
/data/visitor/hc6293/id03/20260119/RAW_DATA/
└── WS2G_100/                              the sample
    └── WS2G_100_DFXM/                     the dataset
        ├── WS2G_100_DFXM.h5               master file: 7.1/measurement, 7.1/instrument/...
        ├── scan0007/pco_ff_0000.h5        frames of scan 7 (large scans are split over several files)
        └── scan0012/pco_ff_0000.h5 ...
```

Ask bexa what it recognises before analysing anything:

```
cd ~/bexa
python -m bexa info /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM --scan 7
python -m bexa info /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM/scan0007
```

The first lines say which format spec matched (`esrf_id03_bliss_2026` for the current layout)
and then describe the scan: type (`fscan1d`, `fscan2d`, ...), dims, motor ranges, energy, fixed
positioners such as `samz`, frame shape, stored dtype and the size of the stack. If no spec
matches (a new layout), `python -m bexa profile sniff PATH -o configs/formats/<name>.yaml`
drafts one; see `docs/formats.md`. With `--recursive` and the `RAW_DATA` folder as PATH it
walks the whole beamtime instead, says which spec recognises every dataset and drafts one
spec per layout it does not know.

In Python the same information comes from a `Scan` handle, which reads only metadata until you
ask for frames:

```python
raw = "/data/visitor/hc6293/id03/20260119/RAW_DATA"
scan = bexa.open(f"{raw}/WS2G_100/WS2G_100_DFXM", scan=7)
print(scan.info())                                   # in a notebook, `scan` alone shows a table
scan.dims, scan.shape, scan.coords["mu"], scan.energy_keV

ds = bexa.open_dataset(f"{raw}/WS2G_100/WS2G_100_DFXM")   # every scan of the folder; a list of folders when a measurement continued in another
ds.table()                                           # id, type, title, motors, ranges, frames, energy, samz, ...
ds.table(compact=True)                               # without the positioner columns that never change
ds[7]                                                # the same Scan as above (for one folder the id is the scan number)
ds.select(scans=[7, 12, 13])                         # by id; or by content: type="fscan2d", samz=0.5
ds.varying(type="fscan2d")                           # what differs between those scans: {"samz": [...]} for a z-stack, "energy" for a series
```

A beamtime profile saves typing the path in every notebook and records the geometry. Create it
once from the terminal and edit the YAML:

```
python -m bexa profile new hc6293 --format esrf_id03_bliss_2026 --root /data/visitor/hc6293/id03/20260119/RAW_DATA
```

```yaml
# ~/bexa/configs/beamtimes/hc6293.yaml
name: hc6293
format: esrf_id03_bliss_2026
root: /data/visitor/hc6293/id03/20260119/RAW_DATA
processed_root: /data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa   # results and the cache go here
path_aliases:
  /data/visitor/hc6293/id03/20260119: X:/Beamtimes/hc6293   # the same profile works on the lab PC later
samples:
  WS2G_100: {dataset: WS2G_100/WS2G_100_DFXM}
  D2: {dataset: D2_dfxm/D2_dfxm_mosa_test, cif: cifs/WS2.cif}
detector: pco_ff
geometry: {effective_pixel_nm: 235}    # sample-plane pixel size: scalebars of bexa scan report
```

From then on `bexa.open(profile="hc6293", sample="WS2G_100", scan=7)` and
`bexa.open_dataset(profile="hc6293", sample="WS2G_100")` in Python, and `-p hc6293 -s WS2G_100`
on the command line, replace the path. With `export BEXA_PROFILE=hc6293` the profile argument can
be left out in Python (`bexa.open(sample="WS2G_100", scan=7)`); the command line still needs
`-p`. The example profiles in `configs/beamtimes/` show every field.

Reference entries: `bexa.open`, `Scan.info`, `Scan.dims` and `Dataset.table`, `Dataset.select`,
`Dataset.varying` in [bexa.core.scan](#bexacorescan-opening-scans-and-the-scan-handle) and [bexa.core.dataset](#bexacoredataset-the-scans-of-one-folder-or-several); the profile fields in [bexa.config.models](#bexaconfigmodels-the-fields-of-a-beamtime-profile),
`load_profile` in [bexa.config.loader](#bexaconfigloader-finding-loading-and-writing-beamtime-profiles), `path_aliases` in [bexa.config.paths](#bexaconfigpaths-repository-cache-folder-and-path-aliases); `draft_spec` in
[bexa.io.formats](#bexaioformats-format-specs-sniffing-and-drafting); the commands in [bexa scan](#bexa-scan-frame-stack-scans) and [bexa profile](#bexa-profile-beamtime-profiles-and-format-specs).

### 4. Case A: a 1D rocking curve

A rocking curve at ID03 is an `fscan` on `mu` with the far-field camera: bexa reports
`scan type fscan1d: dims ('mu', 'y', 'x')`. One streaming pass over the frames gives the summed
image, the rocking curve of the whole detector and of a pixel region, and the per-pixel centre
of mass along `mu`:

```python
scan = bexa.open(profile="hc6293", sample="WS2G_100", scan=7)
print(scan.info())

peak = bexa.ROI(y=(900, 1300), x=(1000, 1500))        # pixel window of the reflection, full resolution
curve = scan.rocking_curve(peak)                      # one verb, one pass: intensity against mu
maps = scan.com(axes=("mu",), sigma=3.0)              # com_mu, width_mu and total, per pixel

res = scan.reduce(                                    # or several accumulators in one pass
    [
        bexa.acc.Sum(),                                # sum(y, x)
        bexa.acc.FrameStats(),                         # frame_sum, frame_max, frame_com_y, frame_com_x vs mu
        bexa.acc.RoiIntegral({"peak": peak}),          # roi_peak(mu): the rocking curve of the window
        bexa.acc.MotorCOM(axes=("mu",), sigma=3.0),    # com_mu, width_mu, total per pixel
    ],
    show_progress=True,
)
```

The results form one `xarray.Dataset` whose variables carry the motor values as coordinates and
the source files, parameters, timing and git hash in `attrs`. Plot them with `bexa.plot`, with
the `.bexa` accessor, or with the `bexa.viz` modules (every function returns its figure and never
calls `plt.show()`):

```python
from bexa.viz import curves, images, maps

res["sum"].bexa.plot(log=True, roi={"peak": peak}, title="scan 7, summed")
curves.rocking_curves({"whole detector": res["frame_sum"], "peak ROI": res["roi_peak"]}, normalize=True)
```

Fit the curve and read off the centre and width (needs lmfit):

```python
from bexa.analysis.fitting import fit_multi_gaussian, fit_report

x = res["roi_peak"].coords["mu"].values
fit = fit_multi_gaussian(x, res["roi_peak"].values, n=1, background="constant")
print(fit_report(fit))                                 # center_1, fwhm_1, amplitude_1, r2, ...
curves.rocking_curves({"peak ROI": res["roi_peak"]}, fits={"peak ROI": (x, fit.best_fit)})
```

The per-pixel maps show where the crystal tilts and how sharp each pixel's curve is:

```python
maps.com_map(res["com_mu"], title="centre of mass in mu (deg)")           # diverging colours around the mean
maps.com_map(res["width_mu"], center="median", cmap="viridis", title="rocking-curve width (deg)")
bexa.plot(res["width_mu"])                                                 # the same, chosen from the name
```

Choosing the window interactively: `picker = bexa.viz.interactive.pick_roi(res["sum"])`, draw
a rectangle, then `peak = picker.roi`; or give it in the pixels of a preview with
`peak = prev.bexa.roi(y=(225, 325), x=(250, 375))`, which comes back in full resolution. Keep the
numbers: `bexa.save(res, f"{out}/scan7.h5")`
writes every array with coordinates and provenance, and `bexa.load(f"{out}/scan7.h5")` reads
them back as a dataset.

The terminal does the same without a notebook. A `--roi` restricts the whole pass to that
pixel window, so `stats` then gives the rocking curve of the window:

```
python -m bexa scan info   -p hc6293 -s WS2G_100 --scan 7
python -m bexa scan reduce -p hc6293 -s WS2G_100 --scan 7 --acc sum,stats,com --roi y=900:1300,x=1000:1500 -o scan7.h5
```

Reference entries: `Scan.rocking_curve`, `Scan.com`, `Scan.reduce` in [bexa.core.scan](#bexacorescan-opening-scans-and-the-scan-handle); `ROI` in [bexa.core.roi](#bexacoreroi-regions-of-interest); `Sum`,
`FrameStats`, `RoiIntegral`, `MotorCOM` in [bexa.core.reductions](#bexacorereductions-bexaacc-accumulators-and-the-streaming-engine); `.bexa.plot` in [bexa.core.accessor](#bexacoreaccessor-bexa-on-xarray-objects);
`rocking_curves` in [bexa.viz.curves](#bexavizcurves-rocking-curves-pump-probe-panels-and-moments); `fit_multi_gaussian`, `fit_report` in [bexa.analysis.fitting](#bexaanalysisfitting-rocking-curve-and-edge-fits);
`com_map` in [bexa.viz.maps](#bexavizmaps-centre-of-mass-bivariate-hsv-and-vector-field-maps); `plot` in [bexa.viz.auto](#bexavizauto-one-call-picks-the-figure); `pick_roi` in [bexa.viz.interactive](#bexavizinteractive-sliders-and-roi-pickers);
`save`, `load` in [bexa.io.cube](#bexaiocube-bexa-files-bexasave-and-bexaload); `bexa scan reduce` in [bexa scan](#bexa-scan-frame-stack-scans).

### 5. Case B: DFXM mosaicity scan (rocking and rolling)

A mosaicity scan is an `fscan2d` with `chi` (rolling) as the slow motor and `mu` (rocking) as
the fast one: `dims ('chi', 'mu', 'y', 'x')`, for example `shape (8, 172, 2160, 2560)`. Start
with a preview, a block-averaged copy of the whole scan that is cached on disk, and look at it
from every side:

```python
scan = bexa.open(profile="hc6293", sample="WS2G_100", scan=12)
print(scan.info())

prev = scan.preview(downsample=(1, 1, 4, 4))          # one factor per dim: (chi, mu, 540, 640)

from bexa.viz import interactive, volume
volume.projections_panel(prev, log=True)               # xy, chi_y, chi_x, mu_y, mu_x and the chi_mu map
interactive.browse(prev)                               # one slider per motor
volume.projections_panel(prev, log=True, backend="plotly")   # interactive version, if plotly is installed
```

The maps come from one full-resolution pass. Pixel ranges select the part of the detector
that matters; motor ranges are given in motor units and select grid points:

```python
window = bexa.ROI(y=(700, 1500), x=(800, 1800))
res = scan.reduce(
    [
        bexa.acc.Sum(),
        bexa.acc.Max(),
        bexa.acc.Projections(),                              # xy, chi_y, chi_x, mu_y, mu_x, chi_mu, grid
        bexa.acc.RoiIntegral({"grain": bexa.ROI(y=(1000, 1100), x=(1200, 1300))}),   # roi_grain(chi, mu)
        bexa.acc.MotorCOM(axes=("chi", "mu"), sigma=3.0),    # com_chi, width_chi, com_mu, width_mu, total
    ],
    roi=window,                                              # add chi=(-0.2, 0.2) to use part of the rocking range
    show_progress=True,
)

res.bexa.plot()                                                       # one panel per map
maps.com_map(res["com_mu"], title="COM mu")
maps.com_map(res["com_chi"], title="COM chi")
maps.bivariate(res["com_chi"], res["com_mu"], labels=("chi", "mu"))   # both tilts in one RGB image with a key
maps.hsv_mosaicity(res["com_chi"], res["com_mu"])                     # hue from com_chi, saturation from com_mu
images.show(res["roi_grain"], title="rocking curve of one grain (chi, mu)")
```

The same on the GPU or CPU depends only on what is installed; pass `device="cpu"` to force the
CPU. On the lab workstation (16 cores, RTX 4090) Sum and MotorCOM with smoothing over 160
frames of 1024 x 1024 take 1.6 s on the CPU, where the smoothing runs in eight threads, and
0.6 s on the GPU (`docs/memory_and_speed.md`).

The standard figure set (sum, max, projections, tiles, COM and width maps, bivariate map) and
an HDF5 file with every array behind it come from one call or one command:

```python
from bexa.pipelines.dfxm_report import report
written = report(scan, f"{out}/scan12", roi=window, downsample=(1, 1, 4, 4))
```

```
python -m bexa scan preview -p hc6293 -s WS2G_100 --scan 12 --downsample 1,1,4,4 --plot
python -m bexa scan report  -p hc6293 -s WS2G_100 --scan 12 --roi y=700:1500,x=800:1800 -o /data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa/scan12
```

Reference entries: `Scan.preview` in [bexa.core.scan](#bexacorescan-opening-scans-and-the-scan-handle); `projections_panel` in [bexa.viz.volume](#bexavizvolume-projections-slices-isosurfaces-and-3-d-renderings);
`browse` in [bexa.viz.interactive](#bexavizinteractive-sliders-and-roi-pickers); `Max`, `Projections` in [bexa.core.reductions](#bexacorereductions-bexaacc-accumulators-and-the-streaming-engine); `res.bexa.plot` in
[bexa.core.accessor](#bexacoreaccessor-bexa-on-xarray-objects); `bivariate`, `hsv_mosaicity` in [bexa.viz.maps](#bexavizmaps-centre-of-mass-bivariate-hsv-and-vector-field-maps); `show` in [bexa.viz.images](#bexavizimages-single-images-tiles-roi-boxes-and-scalebars);
`report` in [bexa.pipelines.dfxm_report](#bexapipelinesdfxm_report-the-standard-dfxm-figure-set); `bexa scan preview`, `bexa scan report` in [bexa scan](#bexa-scan-frame-stack-scans).

### 6. Case C: energy scan

An energy scan at ID03 is a series of scans of one dataset, each at a different monochromator
angle `ccmth`. Open the range of scan numbers (or `scan=None` for every scan of the dataset):
bexa converts the angles to keV and stacks the scans on a leading `energy` dim, sorted by
energy. `EnergyCOM` is the centre of mass along that dim; the other motors keep their own maps:

```python
series = bexa.open(profile="hc6293", sample="WS2G_100", scan=(20, 40))   # or ds.series(scans=(20, 40))
print(series.info())                                   # dims ('energy', 'chi', 'mu', 'y', 'x') or ('energy', 'mu', 'y', 'x')
series.coords["energy"]                                # keV, one value per scan

res = series.energy_com(roi=window)                    # com_energy, width_energy, total
res = series.reduce(                                   # or together with other accumulators
    [bexa.acc.Sum(), bexa.acc.EnergyCOM(), bexa.acc.MotorCOM(axes=("mu",), sigma=3.0)],
    roi=window,
)

e_com = res["com_energy"]                              # keV per pixel
e0 = float(series.coords["energy"].mean())
strain = e0 / e_com - 1                                # Bragg's law at fixed angles: d is proportional to 1/E
maps.com_map(strain, center=0.0, title="lattice strain from the energy centre of mass")
maps.com_map(res["width_energy"], center="median", cmap="viridis", title="energy width (keV)")
```

The same without opening the scans as a series: `bexa.stack` reduces every scan on its own and
puts the summed images on the dims that differ between the scans (`dim="auto"`, here
`energy`), and `energy_com_from_stack` takes the centre of mass from those images. One pass per
scan, cached, and it extends to a z-stack repeated at every energy, where the grid becomes
`(samz, energy, y, x)` and the maps keep the `samz` dim:

```python
from bexa.analysis.rocking import energy_com_from_stack

sums = bexa.stack(ds.select(scans=(20, 40)), [bexa.acc.Sum()], dim="auto", roi=window)["sum"]   # (energy, y, x)
e = energy_com_from_stack(sums)                        # com_energy, width_energy, total per pixel
```

Previews and browsing work as before (`series.preview(downsample=(1, 1, 1, 4, 4))` for a chi by
mu series). If instead the energy was scanned inside one scan (an energy mosa: BLISS `fscan3d`
with `ccmth` as the outer loop), that scan itself has an `energy` dim in keV next to `chi` and
`mu`, so `bexa.acc.MotorCOM(axes=("energy",))` gives `com_energy` and `width_energy` straight
from it (the angle stays in `scan.structure.per_frame["ccmth"]`); a 1-D mono scan likewise
has an `energy` dim.

Reference entries: `open` with a scan range and `Scan.energy_com` in [bexa.core.scan](#bexacorescan-opening-scans-and-the-scan-handle);
`Dataset.series`, `stack` in [bexa.core.dataset](#bexacoredataset-the-scans-of-one-folder-or-several); `EnergyCOM` in [bexa.core.reductions](#bexacorereductions-bexaacc-accumulators-and-the-streaming-engine);
`energy_com_from_stack` in [bexa.analysis.rocking](#bexaanalysisrocking-per-pixel-rocking-curve-statistics); `com_map` in [bexa.viz.maps](#bexavizmaps-centre-of-mass-bivariate-hsv-and-vector-field-maps).

### 7. Case D: z-scan (layers on samz)

Topography layers are recorded as one rocking scan per sample height, with `samz` fixed
during each scan and reported among the scan's positioners. One call reduces every layer with
the same window and stacks the results on a `samz` dim; `dim="auto"` finds that dim by itself
from what differs between the scans (`ds.varying`), `dim="samz"` names it by hand. Give it
everything the rest of the session needs, in one pass per layer: the maps at full resolution
and, next to them, a binned preview of every layer for the 3-D cells (`Preview(downsample=8)`
bins this accumulator alone, so the maps keep their pixels; the binned pixel dims are `yb`,
`xb`). `store=True` writes the results scan by scan into one HDF5 file in the cache and returns
it opened lazily, so the kernel holds the results of a layer only until they are written and
the next run of the same request reads the file instead of the data. `workers="auto"` reduces
several layers at the same time, one per core of the job but one (three on a GPU), each in its
own process, under one progress bar over the scans:

```python
ds = bexa.open_dataset(profile="hc6293", sample="WS2G_100")
layers = ds.select(scans=(13, 24))                     # scans 13 to 24, one per height; or by content:
layers = ds.select(type="fscan1d", motors="mu", samz=(-0.1, 0.1))
ds.varying(type="fscan1d", motors="mu")                # {"samz": [...]}: what differs between the layers
stacked = bexa.stack(                                  # one pass per layer, several layers at a time
    layers,
    [
        bexa.acc.Preview(downsample=8),                # the binned volume of every layer: (mu, yb, xb)
        bexa.acc.Sum(),                                # the summed image at full resolution
        bexa.acc.FrameStats(),                         # the total of every frame: the rocking curves
        bexa.acc.MotorCOM(axes=("mu",), sigma=3.0),    # com_mu, width_mu, total at full resolution
    ],
    dim="auto", roi=window, store=True, workers="auto",
)
list(stacked.data_vars)                                # preview, sum, frame_sum, ..., total, com_mu, width_mu
stacked["com_mu"]                                      # (samz, y, x), samz read from every scan; read block by block
stacked["preview"]                                     # (samz, mu, yb, xb): the layers binned 8 x 8
```

Everything below reads from that stack and never touches the data again. One layer's results
come out with `isel`, the maps of every height with a list of names:

```python
layer = stacked.isel(samz=5)                           # the sixth height: sum, frame_sum, com_mu, ... of that scan
layer["sum"].bexa.plot(log=True)
curves.rocking_curves({"height 5": layer["frame_sum"]})   # the rocking curve of the whole window
zmaps = stacked[["sum", "com_mu", "width_mu", "total"]]   # the maps of every height
images.tiles(zmaps["com_mu"], axis="samz", n=12, per_row=4)   # one COM map per height
zmaps["sum"].bexa.browse()                             # slider over samz
zmaps["com_mu"].bexa.browse_volume()                   # the three projections of the (samz, y, x) volume
bexa.save(zmaps.load(), f"{out}/WS2G_100_zstack_maps.h5")   # the maps, loaded, in one file of their own
```

A z-stack repeated at several energies gives a `(samz, energy, y, x)` grid from the same call,
NaN where a combination was not measured. The binned previews of the layers are the 3-D
volume, which rarely fits in memory: the browsers read one block per slider move from the
stack file, and `interactive.brightest` finds the condition that lights the volume up from
totals computed while stacking:

```python
vol = stacked["preview"].transpose("mu", "samz", "yb", "xb")   # a slider for mu in front of the (samz, yb, xb) volume
peak = interactive.brightest(vol, over=("mu",))               # {"mu": index}: the mu that lights up the whole stack
view = vol.bexa.browse_volume(log=True)                       # the three projections, one block read per move
view.update(**peak)
vol.bexa.browse_render(mode="translucent", log=True, spacing=(1.0, 1.9, 1.9), units="um")   # plotly rendering, redrawn on release
```

Without a cache, leave `store` out: the stack is then built in memory, allocated once and
filled as the layers finish (a `MemoryError` names the size when it would not fit), and
`workers` works the same way. Each worker reopens its scan from `scan.recipe`, gets the job's
memory budget divided by the number of workers, and uses the device the notebook would use, so
the workers share the GPU when there is one; `bexa.settings()` in the notebook shows the
budget they divide.

Layers that drift laterally between heights can be shifted before stacking with
`bexa.analysis.registration.align_stack` (FFT correlation) or placed by hand with
`stack_layers(layers, offsets)`. On a machine with a display, `bexa.viz.napari_app.view_volume`
shows the stack in 3-D and `slices_to_voxels_app` reproduces the slices-to-voxels workflow.

If the layers were instead recorded as one `fscan2d` with `samz` as the slow motor, the scan
is a two-motor scan like the mosaicity case (`dims ('samz', 'mu', 'y', 'x')`) and the stack
above becomes a loop over `scan.coords["samz"]` with `roi=bexa.ROI(samz=(z, z))` selecting one
layer of the same scan. `ds.series(dim="samz", ...)` goes the other way: it opens separate
layers as one scan with a leading `samz` dim.

Reference entries: `Dataset.select`, `Dataset.varying`, `stack` in [bexa.core.dataset](#bexacoredataset-the-scans-of-one-folder-or-several);
`Preview` in [bexa.core.reductions](#bexacorereductions-bexaacc-accumulators-and-the-streaming-engine); `tiles` in [bexa.viz.images](#bexavizimages-single-images-tiles-roi-boxes-and-scalebars); `.bexa.browse`, `.bexa.browse_volume`,
`.bexa.browse_render` in [bexa.core.accessor](#bexacoreaccessor-bexa-on-xarray-objects); `brightest` in [bexa.core.stacks](#bexacorestacks-statistics-brightest-condition-and-colour-limits-of-stacked-volumes); `settings` in
[bexa.core.settings](#bexacoresettings-what-bexa-uses-right-now-bexasettings); `align_stack`, `stack_layers` in [bexa.analysis.registration](#bexaanalysisregistration-shifts-alignment-stitching-and-mosaics); `view_volume`,
`slices_to_voxels_app` in [bexa.viz.napari_app](#bexaviznapari_app-napari-viewers-and-the-apps-behind-them); the picker that chooses the layers in a notebook in
[bexa.notebook.picker](#bexanotebookpicker-choosing-scans-in-a-notebook).

### 8. Longer jobs: batch on the cluster

Anything the notebook does, the command line does headless, so a long reduction can be an
`sbatch` script instead of a notebook that must stay open. The job's `--mem` and
`--cpus-per-task` are picked up by bexa by themselves:

```
#!/bin/bash
#SBATCH --job-name=bexa-scan12 --cpus-per-task=16 --mem=64G --time=01:00:00
source ~/.bexa_env
export BEXA_CACHE_DIR=/data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa_cache
bexa settings
bexa --headless scan report -p hc6293 -s WS2G_100 --scan 12 --roi y=700:1500,x=800:1800 \
    -o /data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa/scan12
```

`python -m bexa settings` at the top of a job script prints the device, memory budget, cache
folder and profile the job will use, with the source of each value.

The report, the reduce command and `bexa.save` all write files whose attributes record the
source files, the parameters, the timings and the git hash, so a figure can always be traced
back to the scan and the settings that produced it. The HDF5 files also carry NeXus groups, so
the h5web viewer inside JupyterLab plots them directly.

### 9. During the beamtime

- A dataset folder holds scans in every state: `ds.table()` lists them all with a `status`
  column (`complete`, `running`, `aborted`, `empty`, `unreadable`), and a scan that cannot be
  read yet never hides the others (`ds.problems` says why). `ds.select(...)` keeps the finished
  scans (complete and aborted) unless `status="any"` asks for more; a running scan opens with
  `ds[id]` and reads as far as its files go. `ds.refresh()` (or `ds.table(refresh=True)`) is the
  poll: new scans appear, running ones are read again, finished ones keep their handles.
  `python -m bexa scan list FOLDER` prints the same table from the shell.
- A scan that is still being written can be opened; `scan.refresh()` (or re-opening) picks up
  the new frames and `scan.info()` says its status and how many planned frames are missing.
- `python -m bexa scan preview ... --plot` right after each scan is the quickest check that the
  reflection sits inside the rocking range.
- Keep paths in the profile, not in notebooks: with `path_aliases` the same notebook runs on
  the lab PC once the data is copied.
- `docs/beamtime_checklist.md` has the full list, including how to draft a format spec when the
  beamline changes its file layout.

Reference entries: `Dataset.table`, `Dataset.problems`, `Dataset.refresh` in [bexa.core.dataset](#bexacoredataset-the-scans-of-one-folder-or-several);
`ScanStatus` in [bexa.core.status](#bexacorestatus-the-state-of-a-scan); `Scan.refresh`, `Scan.info` in [bexa.core.scan](#bexacorescan-opening-scans-and-the-scan-handle); `bexa scan list`
in [bexa scan](#bexa-scan-frame-stack-scans).

## Find a function by task

The everyday jobs and the functions that do them; each name has an entry in the
[Function reference](#function-reference) or the [Command-line reference](#command-line-reference).

**Finding and opening data**

| I want to | Use | notebook 13 |
|---|---|---|
| see what a folder or file holds | `bexa info PATH`; `bexa.io.inspect.describe`, `bexa.io.inspect.h5_tree` |  |
| list the scans of a dataset with their motors and positioners | `bexa.open_dataset(path).table()`, `Dataset.table(compact=True)`, `bexa.list_scans` |  |
| a measurement that continued in a second dataset folder | `bexa.open_dataset([folder1, folder2])`: one table, ids running over both folders; `Dataset.table`, `Dataset.series` and `bexa.stack` work across them | §5, §11 |
| the datasets under a RAW_DATA folder | `bexa.notebook.list_datasets` |  |
| choose the scans, the motors to extract and the stack axis with widgets | `bexa.notebook.ScanPicker`, `bexa.notebook.pick_scans`, `Dataset.motor_candidates` | §3, §11 |
| select scans from several dataset folders and see them all in one panel | `ScanPicker.set_folders`, `ScanPicker.selection`, `ScanPicker.selection_view` | §3 |
| open an APS 7-ID session (a SPEC file with a TIFF per point) | `bexa.open(folder, scan=4)`, `bexa.open_dataset(folder)`; the profile `example_aps_7id` |  |
| open one scan | `bexa.open(path, scan=7)` |  |
| open through a beamtime profile | `bexa.open(profile=..., sample=..., scan=...)`, `bexa profile new`, `bexa.config.load_profile` |  |
| pick scans by type, scanned motor or positioner (`samz`, energy) | `Dataset.select`, `Dataset.select_numbers`, `Dataset.groups` | §7 |
| see what differs between scans (heights, energies) | `Dataset.varying`, `bexa.core.dataset.varying` | §4 |
| treat an energy series or a z-stack of scans as one scan | `bexa.open(path, scan=(20, 24))`, `Dataset.series` |  |
| read a plain `.npy`, tiff or HDF5 stack | `bexa.open("file.h5::/entry/data")`, `bexa.io.generic.load_array` |  |
| read a PAL-XFEL run, an LCLS cube or smalldata, EuXFEL | `bexa.open(path)`, `bexa.io.cube.load_lcls_cube`, `SmalldataSource`, `ExtraDataSource` |  |
| support a new file layout | `bexa profile sniff`, `bexa.io.formats.draft_spec`, `bexa.io.formats.write_draft` |  |
| check every dataset of a beamtime at once, and draft specs for the layouts bexa does not know | `bexa profile sniff --recursive`, `bexa.io.survey.survey`, `bexa.io.survey.write_drafts` |  |
| try the API without beamtime data | `bexa.demo`, `bexa.testing.synthetic.make_esrf_scan` | §3 |

**Reducing scans**

| I want to | Use | notebook 13 |
|---|---|---|
| the summed or maximum image | `Scan.sum`, `Scan.max` | §4, §5, §6, §7 |
| the rocking curve of a pixel region | `Scan.rocking_curve`, `RoiIntegral` | §7 |
| centre-of-mass (COM) and width maps, mosaicity | `Scan.com`, `MotorCOM`, `bexa.analysis.rocking.motor_com`, `bexa.analysis.rocking.moments` | §5, §11 |
| the energy centre of mass and strain of an energy scan | `Scan.energy_com`, `EnergyCOM`; from stacked sums `bexa.analysis.rocking.energy_com_from_stack` | §9 |
| a downsampled preview of a whole scan | `Scan.preview` |  |
| full-resolution maps and a binned volume from one pass (one pass, two resolutions) | `Preview(downsample=8)` next to `Sum`, `MotorCOM`, ... in `Scan.reduce` or `bexa.stack` | §5, §11 |
| the frames in memory | `Scan.read` (with cache=True the volume is kept for the next session), `Scan.to_dask`, `Scan.batches` | §5 |
| several results from one pass over the files | `Scan.reduce`, `bexa.reduce` |  |
| maps of every scan on the positioners that vary (a z-stack, an energy series, both) | `bexa.stack(..., dim="auto")` | §5, §11 |
| parallel scan loading: many scans reduced at once, each in its own process | `bexa.stack(..., workers="auto")` (every scan reopened from `Scan.recipe`), `bexa.core.parallel.process_pool` | §5, §11 |
| a stack too large for memory, built on disk and read block by block | `bexa.stack(..., store=True)`, `bexa.io.cube.open_lazy`, `bexa.io.cube.StoreWriter` | §5, §11 |
| the condition (chi, mu, energy) that lights up a stacked volume, shared colour limits | `bexa.viz.interactive.brightest`, `bexa.viz.interactive.shared_limits` | §10 |
| a region of interest | `bexa.ROI`, `ROI.parse`, `da.bexa.roi(...)`, `bexa.viz.interactive.pick_roi` | §2, §3 |
| keep results and reload them | `bexa.save`, `bexa.load`, `da.bexa.save(...)` | §6, §8, §9, §11 |

**Analysis**

| I want to | Use | notebook 13 |
|---|---|---|
| subtract darks, remove hot pixels | `bexa.analysis.preprocess.subtract_dark`, `bexa.analysis.preprocess.remove_hot_pixels` |  |
| normalise by I0 or by a background box | `bexa.analysis.preprocess.normalize_i0`, `bexa.analysis.preprocess.normalize_box` |  |
| bin or crop pixels | `bexa.analysis.preprocess.bin_spatial`, `bexa.analysis.preprocess.crop` |  |
| fit a rocking curve (Gaussians, Voigt) | `bexa.analysis.fitting.fit_multi_gaussian`, `bexa.analysis.fitting.fit_voigt`, `bexa.analysis.fitting.fit_report` | §7 |
| fit a knife-edge or sample-height scan | `bexa.analysis.fitting.fit_edge` | §7 |
| per-frame COM, variance, skew and kurtosis on the detector | `bexa.analysis.peaks.frame_moments`, `bexa.analysis.peaks.frame_stats` |  |
| fit a 2-D Gaussian to a peak on the detector | `bexa.analysis.peaks.gaussian2d_fit`, `bexa.analysis.peaks.fit_stack` |  |
| align drifting frames or layers | `bexa.analysis.registration.align_stack`, `bexa.analysis.registration.phase_correlation_shift` |  |
| stitch overlapping scans, build a mosaic | `bexa.analysis.registration.stitch_along_axis`, `bexa.analysis.registration.mosaic_from_grid` |  |
| segment grains and domains | `bexa.analysis.segmentation.kmeans_com`, `bexa.analysis.segmentation.dbscan_map`, `bexa.analysis.segmentation.smoothness_mask` |  |
| boundaries, vector fields, strain integration | `bexa.analysis.fields.edge_mask`, `bexa.analysis.fields.vector_field`, `bexa.analysis.fields.compatibility` |  |
| points in reciprocal space (angles, Q, hkl) | `bexa.analysis.projections.rlp_points`, `bexa.geometry.transforms.build_coordinate_grids` |  |
| laser on/off split and differential signal | `bexa.analysis.pump_probe.split_on_off`, `bexa.analysis.pump_probe.differential`, `OnOffSplit` |  |
| combine runs, bin LCLS shots | `bexa.analysis.pump_probe.combine_runs`, `bexa.analysis.pump_probe.bin_lcls` |  |

**Plots**

| I want to | Use | notebook 13 |
|---|---|---|
| plot whatever bexa returned | `bexa.plot`, `da.bexa.plot()` | §2, §6, §7, §8 |
| an image with ROI boxes and a scalebar | `bexa.viz.images.show`, `bexa.viz.images.tiles` | §6, §7, §8, §9 |
| COM maps, both tilts in one image | `bexa.viz.maps.com_map`, `bexa.viz.maps.bivariate`, `bexa.viz.maps.hsv_mosaicity` | §8 |
| the projections of a scan | `bexa.viz.volume.projections_panel`, `Projections` | §6 |
| rocking curves with their fits | `bexa.viz.curves.rocking_curves` |  |
| sliders through a scan | `da.bexa.browse()`, `bexa.viz.interactive.browse`, `bexa.viz.interactive.compare` | §9, §11 |
| a 3-D view of a z-stack | `da.bexa.browse_volume()`, `bexa.viz.volume.render`, `da.bexa.browse_render()`, `bexa.viz.napari_app.view_volume` | §10 |
| a GIF or MP4 | `bexa.viz.animation.animate`, `bexa.viz.animation.save_gif`, `bexa.viz.animation.save_mp4` |  |
| pump-probe figures | `bexa.viz.curves.on_off_diff`, `bexa.viz.curves.moments_vs_axis` |  |
| figures for a paper | `bexa.viz.style.use("paper")` |  |
| the colour limits, colormap, colourbar or size of one plot | the `clim`, `cmap`, `colorbar`, `figsize` keywords of every plot function (`bexa.viz.images.show`) | §6, §7, §8, §9 |
| the same for every plot of a notebook, or a figure already drawn | `bexa.viz.style.set`, `bexa.viz.style.context`, `bexa.viz.style.apply` | §2 |

**Pipelines and the beamtime**

| I want to | Use | notebook 13 |
|---|---|---|
| the standard DFXM figure set of a scan | `bexa.pipelines.dfxm_report.report`, `bexa scan report` |  |
| a PAL-XFEL laser on/off cube | `bexa.pipelines.xfel_cube.reduce_run`, `bexa cube build` |  |
| the figures and GIFs of a cube | `bexa.pipelines.cube_report.figure_set`, `bexa.pipelines.cube_report.animations`, `bexa cube plot` |  |
| process every new run while the beamtime goes on | `bexa.pipelines.auto_process.watch`, `bexa auto --watch` |  |
| time this machine | `bexa.pipelines.benchmark.run_suite`, `bexa bench` |  |

**Crystallography and optics**

| I want to | Use | notebook 13 |
|---|---|---|
| d-spacing and Bragg angle of a reflection | `Crystal.d_spacing`, `Crystal.bragg_angle_deg`, `bexa crystal dspacing` |  |
| motor angles that bring a reflection onto the detector | `bexa.crystal.diffractometer.create_diffractometer`, `FourCircleDiffractometer.solve_for_peak_family`, `bexa crystal solve` |  |
| the zone axis from one indexed reflection | `bexa.crystal.zone_axis.solve_zone_axis`, `bexa crystal zone-axis` |  |
| omega settings where two peaks are close in mu | `bexa.crystal.omega_search.find_omega_offsets`, `bexa crystal omega-search` |  |
| a crystal from a CIF file | `bexa.crystal.cif.crystal_from_cif` |  |
| energy, wavelength and monochromator angle | `bexa.core.units.energy_to_wavelength`, `bexa.core.units.mono_angle_to_energy`, `bexa.optics.conversions.summary`, `bexa optics convert` |  |
| transmission and attenuation length of a filter or sample | `bexa.optics.attenuation.transmission`, `bexa.optics.attenuation.attenuation_length_um`, `bexa optics transmission` |  |
| photons per pulse | `bexa.core.units.photons_from_pulse_energy`, `bexa optics photons` |  |
| the efficiency of a Bragg magnifier | `bexa.optics.bragg_magnifier.efficiency`, `bexa.optics.bragg_magnifier.optimal_split`, `bexa optics magnifier` |  |

**The machine and the settings**

| I want to | Use | notebook 13 |
|---|---|---|
| know the device, memory budget, cache and profile in use | `bexa.settings()`, `bexa settings` |  |
| force the CPU or the GPU | `device="cpu"` or `device="cuda"` on any reduction, verb, preview or report; `bexa.core.backend.resolve_device` |  |
| use less memory per batch | `BEXA_MEMORY_FRACTION`, `bexa.core.backend.memory_budget`, `bexa.core.backend.choose_batch_frames` (batches are capped at 256 MiB, `MAX_BATCH_BYTES`) |  |
| threads for the CPU work of one process (the smoothing of the centre of mass) | `BEXA_THREADS`, `bexa.core.parallel.compute_threads`, `bexa.core.backend.gaussian_frames` |  |
| know whether a result fits before computing it | `bexa.core.backend.check_fits` (what `Scan.read`, previews, `bexa.stack` and `bexa.load` refuse with) | §5, §6, §8, §9 |
| see or clear the result cache | `bexa cache ls`, `bexa cache clear`, `bexa.core.cache.Cache`, `Dataset.clear_cache` | §3 |
| find where a result came from | `bexa.core.provenance.parameters_of`, the `attrs` of every result |  |
| more or fewer log messages | `BEXA_LOG_LEVEL` (set before `import bexa`), `bexa._log.set_level` |  |
| find what is slow | `bexa.profile()`, `bexa --profile`, `bexa bench` |  |

## Function reference

Every public function, class, method and property of bexa, module by module, in the order data
flows through the package: opening and reducing scans, reading and writing files, analysis,
plots, pipelines, synthetic data, geometry and crystallography, units and optics, and the
machinery underneath. Each entry has:

- a heading with the name, which GitHub's heading filter and the VS Code outline list;
- the full import path with every parameter and its default;
- what it does, with which inputs and units, and what it returns;
- an example.

Conventions used throughout:

- The examples of a section continue from the setup block at its top. Most run as they are on
  the synthetic data of `bexa.demo` and `bexa.testing.synthetic`; paths such as
  `/data/visitor/hc6293/...` and the profile `hc6293` stand for your own beamtime.
- `scan` is a `bexa.Scan`, `ds` a `bexa.Dataset` (the scans of one folder or several, not an
  xarray Dataset), `res` the `xarray.Dataset` a reduction returns, `prev` a preview, `vol` a
  volume in memory or in a stack file, `frames` an `(n, y, x)` array and `cube` a pump-probe
  cube.
- Pixel coordinates `y`, `x` are full-resolution detector indices; motor coordinates are in the
  units the file records (deg for the ID03 rotations `mu`, `chi`, `phi`); energies are in keV.
- Functions in `bexa.analysis` accept numpy or cupy arrays and return the same kind unless
  their entry says otherwise; the results of reductions come back as numpy.
- Plot functions return their figure and artists, never call `plt.show()` and use percentile
  colour limits by default; the single-panel ones take `ax=None` to draw into existing axes.
- Optional packages are named where a function needs them; `requirements-optional.txt` lists
  them all.
- Names that start with `_` are private and may change without notice.

### Top-level names: bexa.*

`import bexa` takes about 0.04 s: the names below are imported on first use, and heavy optional
packages (cupy, napari, lmfit, plotly, dask, pymatgen, extra_data) load only inside the
functions that need them. Each name is documented in full under the module that defines it.

| Name | What it does | Documented under |
|---|---|---|
| `bexa.open` | open one scan, or several as a series, from a path or a beamtime profile | `bexa.core.scan.open` |
| `bexa.open_profile` | open a scan from a beamtime profile and a sample name | `bexa.core.scan.open_profile` |
| `bexa.list_scans` | scan numbers of a dataset folder | `bexa.core.scan.list_scans` |
| `bexa.open_dataset` | every scan of a dataset folder (or of several folders), as a table | `bexa.core.dataset.open_dataset` |
| `bexa.stack` | reduce several scans (several at a time with `workers=`) and stack the results on the positioners that vary between them (`samz`, `energy`), in memory or in one HDF5 file | `bexa.core.dataset.stack` |
| `bexa.common_grid` | one motor grid spanning scans measured with different ranges or steps | `bexa.core.dataset.common_grid` |
| `bexa.regrid` | interpolate a pass's results onto such a grid | `bexa.core.dataset.regrid` |
| `bexa.reduce` | run accumulators over a scan in one streaming pass | `bexa.core.reductions.reduce` |
| `bexa.acc` | the accumulators: `bexa.acc.Sum()`, `bexa.acc.MotorCOM(...)`, ... (the module `bexa.core.reductions`) | `bexa.core.reductions` |
| `bexa.ROI` | region of interest over pixels, motors and energy | `bexa.core.roi.ROI` |
| `bexa.Scan`, `bexa.Dataset` | the classes behind `open` and `open_dataset` | `bexa.core.scan.Scan`, `bexa.core.dataset.Dataset` |
| `bexa.ScanStatus`, `bexa.ScanNotReadyError` | the state of a scan (`complete`, `running`, `aborted`, `empty`, `unreadable`) and the error for a scan with nothing to read yet | `bexa.core.status.ScanStatus`, `bexa.core.status.ScanNotReadyError` |
| `bexa.save`, `bexa.load` | write and read results (HDF5 or zarr with provenance), read legacy and LCLS cubes | `bexa.io.cube.save`, `bexa.io.cube.load` |
| `bexa.plot` | the figure that fits an array or a result | `bexa.viz.auto.plot` |
| `bexa.demo` | write a synthetic dataset and open it | `bexa.testing.demo.demo` |
| `bexa.settings` | print the device, memory budget, cache folder and profile in use | `bexa.core.settings.show` |
| `bexa.help` | list the API: the everyday names, one group, or everything | `bexa._help.help` |
| `bexa.profile` | profile a block of code with cProfile | `bexa._profiling.profile` |
| `bexa.to_host` | bring cupy arrays (also inside dicts, lists and xarray objects) back to numpy | `bexa.core.backend.to_host` |
| `bexa.__version__` | the version string, `"1.0.0.dev0"` | |

The sub-packages are imported on first use too: `bexa.analysis`, `bexa.crystal`,
`bexa.geometry`, `bexa.io`, `bexa.notebook`, `bexa.optics`, `bexa.pipelines`, `bexa.testing`,
`bexa.viz`.

```python
import bexa
bexa.__version__                 # '1.0.0.dev0'
bexa.help("plots")               # every plotting function with a one-line summary
scan = bexa.demo("mosa")         # a synthetic scan to try the API on
```

### bexa.core.scan: opening scans and the Scan handle

`bexa.open` matches the files against the format specs in `configs/formats/` (or starts from a
beamtime profile) and returns a `Scan`: a lazy handle that has read only metadata. Frames are
read when a verb (`sum`, `max`, `rocking_curve`, `com`, `energy_com`, `stats`), `preview`,
`read` or `reduce` asks for them, in one streaming pass per call. The examples of this section
run on the synthetic mosaicity scan of `bexa.demo`; `folder` stands for a dataset folder.

```python
import numpy as np
import bexa

scan = bexa.demo("mosa")     # chi x mu scan: dims ('chi', 'mu', 'y', 'x'), shape (6, 16, 64, 80)
folder = "/data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM"  # a dataset folder (the synthetic z-stack stands in for it)
```

#### `open`
`bexa.core.scan.open(path=None, *, profile=None, sample=None, dataset=None, format=None, scan=None, detector=None, overrides=None, energy_keV=None, geometry=None, crystal=None, cache=True, cache_reductions=None, motors=None, **kwargs)`

Open a measurement and return a `Scan` (top-level name `bexa.open`). `path` is a dataset
folder, its master file, a `scanNNNN` folder, a PAL-XFEL run folder, a cube file, or
`"file.h5::/path/to/stack"` for a plain frame stack in any HDF5 file (the `generic_stack`
spec). The layout is sniffed unless `format` names a spec; `overrides` is a dict merged into
the spec. Instead of a path, `profile` (a name, a YAML path or a `BeamtimeProfile`; the
`BEXA_PROFILE` variable when left out) with `sample` or `dataset` opens through
`open_profile`.

- `scan`: one number, an inclusive `(start, end)` tuple or a list. Several scans are opened as
  one `Scan` with a new leading dim named by `stack_dim=` (passed through `**kwargs`):
  `"energy"` (the default when every scan has its own energy; sorted by energy), `"scan"` (the
  scan numbers) or a positioner such as `"samz"` read from every scan. `scan=None` opens the
  only scan, or every scan of a dataset as a series.
- `detector`: the detector to read when the layout records several (`pco_ff` at ID03).
- `energy_keV`: overrides the energy derived from the monochromator angle.
- `geometry`, `crystal`: a `DetectorGeometry` and a `Crystal` to attach; the geometry is
  built from the spec's detector table otherwise.
- `cache`: `True` (a cache in the folder of `bexa.config.paths.cache_root`, or the profile's),
  a folder path (a cache there), a `Cache`, or `False`. Previews are cached by default;
  `cache_reductions=True` caches the results of `reduce` and of the verbs as well.
- `motors`: per-frame channels to use as the dims instead of the ones the scan declares
  (`["ccmth", "chi", "mu"]` for the monochromator angle in degrees instead of the energy; see
  `Scan.with_dims`); layouts with a master file only.
- Other keywords go to the engine, for example `coords=` for a generic stack.
- The scan remembers how it was opened in `Scan.recipe` (`path`, `scan`, the name of the
  matched spec, `detector`, `overrides`, `motors` and the extra keywords), so
  `bexa.open(**scan.recipe)` opens it again without sniffing, in another process too.
- A scan with nothing to read yet (still running without a frame, empty, unreadable) raises
  `bexa.ScanNotReadyError`, which names its status; one that is running with frames opens and
  reads as far as its files go (`scan.status`).

```python
# folder: a dataset folder on disk; one scan of it, then a series of scans as one handle
scan7 = bexa.open(folder, scan=7)                              # one scan of a BLISS dataset
series = bexa.open(folder, scan=(20, 24))                      # five energies: dims ('energy', 'mu', 'y', 'x')
layers = bexa.open(folder, scan=[1, 7, 13], stack_dim="samz")  # dims ('samz', 'chi', 'mu', 'y', 'x')
scan7 = bexa.open(profile="hc6293", sample="WS2G_100", scan=7)
frames = bexa.open("stack.h5::/entry/data")                    # a plain (n, y, x) dataset
cached = bexa.open(folder, scan=7, cache="bexa_cache", cache_reductions=True)   # previews and results kept in ./bexa_cache
angles = bexa.open(scan.recipe["path"], scan=1, format=scan.spec.name, motors=["chi", "mu"], cache=False)   # the dims chosen by hand
```

#### `open_profile`
`bexa.core.scan.open_profile(profile, sample=None, scan=None, detector=None, dataset=None, cache_reductions=None, cache=None, **kwargs)`

Open a scan described by a beamtime profile (`configs/beamtimes/<name>.yaml`): the format spec,
the data root, the sample's dataset folder, its detector, CIF file and hkl centre, and the
profile's geometry section all come from the YAML (top-level name `bexa.open_profile`;
`bexa.open(profile=..., sample=...)` calls it). `profile=None` reads `BEXA_PROFILE`. The cache
lives under the profile's `processed_root` unless `cache` names another folder (or a `Cache`, or
`False` for none), and reductions are cached when the profile has a `processed_root` (or
`BEXA_CACHE_DIR` is set) unless `cache_reductions=False`. The CIF of the sample, when given,
becomes `scan.crystal`. The scan's `recipe` holds the profile (its YAML path, or the object
when it was built in memory), `sample`, `dataset`, `scan`, `detector` and the extra keywords,
so `bexa.open(**scan.recipe)` opens it again.

```python
# a beamtime profile names the format, the root folder, the samples and the geometry
scan7 = bexa.open_profile("hc6293", sample="WS2G_100", scan=7)
scan7.profile.processed_root, scan7.geometry.effective_pixel_nm  # (Path('.../processed'), 235.0)
own = bexa.open_profile("hc6293", sample="WS2G_100", scan=7, cache="bexa_cache")   # this folder instead of the profile's
```

#### `list_scans`
`bexa.core.scan.list_scans(path, *, format=None, detector=None, require_files=True)`

Scan numbers of a dataset, sorted (top-level name `bexa.list_scans`): the `scanNNNN` folders
that hold detector files, or every `scanNNNN` folder with `require_files=False` (a scan that has
just started has none yet); a folder named otherwise (`scan0007_old`) is not a scan. `path` is a
dataset folder, its master file or one of its `scanNNNN` folders; layouts without numbered scans
give `[]`.

```python
# folder: a dataset folder; the scan numbers it holds, without opening them
bexa.list_scans(folder)                        # [1, 2, 3, ...]: the scans with detector files
bexa.list_scans(folder, require_files=False)   # ... plus a scan whose folder is still empty
```

#### `open_source`
`bexa.core.scan.open_source(spec, path, scan=None, detector=None, stack_dim=None, **kwargs)`

The engine step of `open`: build the `Source` (the reader) of a format spec for `path`
without wrapping it in a `Scan`. Several scans become a `bexa.io.multi.MultiScanSource` stacked
on `stack_dim` as described under `open`. Useful when writing an engine or a test.

```python
# the engine step of bexa.open: a format spec and a folder give a Source
from bexa.core.scan import open_source
from bexa.io.formats import load_spec

source = open_source(load_spec("esrf_id03_bliss_2026"), folder, scan=[1, 7, 13], stack_dim="samz")
source.structure().dims          # ('samz', 'chi', 'mu', 'y', 'x')
```

#### `combine_sources`
`bexa.core.scan.combine_sources(parts, scans, stack_dim=None)`

The stacking step of `open_source`: already-open sources become one
`bexa.io.multi.MultiScanSource` with a new leading dim, sorted along it. `scans` labels the
parts (their scan numbers, or the ids of a `Dataset`); `stack_dim` is `"energy"` (the default
when every part has its own energy), `"scan"` (the labels, in the given order) or a positioner
such as `"samz"` read from every part (`ValueError` when a part lacks it). `Dataset.series`
uses it to stack scans that live in different folders.

```python
# several sources (one per scan) stacked along a new dim, as a series of scans is
from bexa.core.scan import combine_sources

parts = [bexa.open(folder, scan=n).source for n in (1, 7, 13)]
source = combine_sources(parts, [1, 7, 13], stack_dim="samz")
source.dim, source.coords        # 'samz', array([-0.002, -0.001, 0.]): one height per part, sorted
```

#### `Scan`
`bexa.core.scan.Scan(source, spec=None, geometry=None, crystal=None, name="", cache=None, profile=None, cache_reductions=False, hkl_center=None)`

The lazy handle that `bexa.open` returns (top-level name `bexa.Scan`); build one yourself only
around a custom `Source`. In a notebook a `Scan` shows itself as a table (format, scan type,
dims, motor ranges, energy, frames, files), and it is a context manager:
`with bexa.open(...) as scan:` closes the files at the end.

Properties (metadata only, no frames are read):

- `structure`: the `bexa.core.structure.Structure` of the scan: dims, coordinates, which frame
  sits where, per-frame motor readbacks (`structure.per_frame`) and fixed positioners
  (`structure.scalars`).
- `dims`, `shape`: the motor dims followed by `("y", "x")`, and their sizes.
- `coords`: `{motor: values}` for the motor dims.
- `n_frames`, `frame_shape`: the number of frames and `(height, width)` of one frame.
- `energy_keV`: photon energy in keV (from the monochromator), or `None`.
- `files`: every file the scan reads, as a list of `Path`.
- `is_multi`: True for a series opened from several scans.

Attributes: `source` (the engine, for example `scan.source.shot_table()` for PAL point files),
`spec` (the `FormatSpec`), `geometry` (a `DetectorGeometry`), `crystal`, `hkl_center`, `name`,
`cache`, `cache_reductions`, `profile` (the `BeamtimeProfile` or `None`), `roi`, the scan's
default `ROI`, which every call without `roi=` uses, and `recipe`, how the scan was opened
(see `Scan.recipe`).

```python
# scan: the demo chi x mu scan (6 x 16 frames of 64 x 80 pixels); its grid and coordinates
scan.dims, scan.shape            # ('chi', 'mu', 'y', 'x'), (6, 16, 64, 80)
scan.coords["mu"]                # 16 mu values in deg
scan.energy_keV                  # 17.0
scan.structure.scalars["samz"]   # a positioner that stayed fixed during the scan
```

#### `Scan.info`
`bexa.core.scan.Scan.info()`

A text summary: engine and format spec, scan type and title, dims and shape, the range and
units of every motor, the energy, the fixed positioners, the status (and the planned frames
still missing for a running or aborted scan), frame size and stored dtype, the memory the
stack takes as float32, the files, the ROI and the geometry. `bexa info PATH` and `bexa scan
info` print the same.

```python
# scan: the demo chi x mu scan; a text summary of its files, grid, energy and status
print(scan.info())               # ... "  status: complete" ...
```

#### `Scan.sum`
`bexa.core.scan.Scan.sum(roi=None, downsample=None, **kwargs)`

The summed image over all frames (inside the motor ranges of the ROI), from one streaming
pass: a `DataArray` `(y, x)` named `sum`, whose `y` and `x` coordinates are full-resolution
pixel indices, with provenance attrs. `roi`, `downsample` and the keywords of `reduce`
(`device`, `batch_frames`, `show_progress`, ...) apply.

```python
# scan: the demo chi x mu scan; the image summed over every frame, from one pass
total = scan.sum()                                      # DataArray (y, x)
total = scan.sum(roi=bexa.ROI(mu=(-0.2, 0.2)), device="cpu")
```

#### `Scan.max`
`bexa.core.scan.Scan.max(roi=None, downsample=None, **kwargs)`

The pixel-wise maximum over the frames: a `DataArray` `(y, x)` named `max`. Keywords as for
`Scan.sum`.

```python
# scan: the demo chi x mu scan; the brightest value of every pixel over the whole scan
brightest = scan.max()
scan.max(roi=bexa.ROI(mu=(-0.2, 0.2)), downsample=(1, 1, 2, 2)).shape  # (32, 40)
```

#### `Scan.rocking_curve`
`bexa.core.scan.Scan.rocking_curve(roi=None, method="sum", downsample=None, **kwargs)`

The integrated intensity of a pixel window at every motor point: 1-D against `mu` for a
rocking scan, `(chi, mu)` for a mosaicity scan. `roi` gives the window (the scan's `roi` when
`None`, the whole frame when it has no pixel ranges; motor ranges limit the points); `method`
is `"sum"`, `"mean"` or `"max"` over the window. Returns a `DataArray` named `rocking_curve`.
Replaces `compute_integrated_map(roi)` of the v9 module.

```python
# scan: the demo chi x mu scan; the intensity of a pixel window at every grid point
curve = scan.rocking_curve(bexa.ROI(y=(20, 40), x=(30, 50)))   # dims ('chi', 'mu')
bexa.demo("rocking").rocking_curve().dims                       # ('mu',)
```

#### `Scan.com`
`bexa.core.scan.Scan.com(axes=None, sigma=3.0, roi=None, downsample=None, moments=2, **kwargs)`

Per-pixel centre-of-mass maps along the motor axes (the DFXM COM maps) from one pass:
`com_<axis>` in motor units (deg for `mu` and `chi`), `width_<axis>` (the standard deviation of
each pixel's rocking curve) and `total` (the summed weights). `axes` defaults to every motor
dim; `sigma` smooths every frame with a Gaussian of that many pixels before weighting (0
disables it); `moments=3` adds `skew_<axis>` and `moments=4` adds `kurtosis_<axis>`. Returns an
`xarray.Dataset`; `MotorCOM` has the formulas. Replaces `plotter.compute_center_of_mass`.

```python
# scan: the demo chi x mu scan; the centre of mass and width of every pixel along chi and mu
maps = scan.com(axes=("chi", "mu"), sigma=3.0)   # total, com_chi, width_chi, com_mu, width_mu
maps["com_mu"].bexa.plot()                      # (Figure, Axes, AxesImage)
```

#### `Scan.energy_com`
`bexa.core.scan.Scan.energy_com(sigma=0.0, roi=None, downsample=None, **kwargs)`

The centre of mass along the `energy` dim of an energy series (a `Scan` opened from several
scans at different monochromator energies): `com_energy` in keV per pixel, `width_energy` and
`total`. No smoothing by default.

```python
# series: an energy series (one scan per energy); the energy centre of mass of every pixel
series = bexa.demo("energy")                     # dims ('energy', 'chi', 'mu', 'y', 'x')
res = series.energy_com()
strain = float(series.coords["energy"].mean()) / res["com_energy"] - 1
```

#### `Scan.stats`
`bexa.core.scan.Scan.stats(roi=None, downsample=None, **kwargs)`

Per-frame statistics on the motor grid: `frame_sum`, `frame_mean`, `frame_max` and the detector
centre of mass `frame_com_y`, `frame_com_x` in full-resolution pixels. The quickest check that
the reflection stays on the detector and inside the rocking range.

```python
# scan: the demo chi x mu scan; the total, maximum and centre of every frame, on the grid
st = scan.stats()                    # every variable has dims ('chi', 'mu')
st["frame_sum"].sel(chi=0, method="nearest").plot()  # [Line2D]
```

#### `Scan.preview`
`bexa.core.scan.Scan.preview(downsample=8, roi=None, apply_log=False, method="mean", device="auto", cache=True, **kwargs)`

The whole scan as a block-averaged volume `(motors..., y, x)` in memory, cached on disk in
`scan.cache` so that the next call returns at once. `downsample` is an int for both pixel axes,
`(ds_y, ds_x)`, one factor per dim such as `(1, 1, 4, 4)`, or a mapping by name such as
`{"mu": 2, "y": 4, "x": 4}` (a motor factor keeps every n-th grid point; names the scan lacks
are ignored); `method` combines the pixel blocks (`"mean"`, `"sum"` or `"max"`); `apply_log`
stores `log10(1 + I)`. Returns a `DataArray` named `preview` whose `y`, `x` coordinates are
full-resolution pixel indices. The volume is checked against the memory budget before it is
allocated (`MemoryError` with the size, the budget and the way out: a larger downsample, an
ROI, or `bexa.stack(..., store=True)`); `cache=False` computes it without reading or writing
the cache. Replaces `dataset.load_preview` of the v9 module (which used `scipy.ndimage.zoom`;
bexa takes block means).

```python
# scan: the demo chi x mu scan; the whole scan binned 4 x 4 in the pixels, in memory
prev = scan.preview(downsample=(1, 1, 4, 4))     # (6, 16, 16, 20)
prev.bexa.browse()                               # one slider per motor
fresh = scan.preview(downsample=(1, 1, 4, 4), cache=False)   # computed again, the cache untouched
scan.preview(downsample={"mu": 2, "y": 4, "x": 4}, cache=False).shape   # (6, 8, 16, 20): every second mu point
```

#### `Scan.read`
`bexa.core.scan.Scan.read(roi=None, downsample=None, device="cpu", dtype=np.float32, cache=False, **kwargs)`

Load frames into memory as a `DataArray` `(motors..., y, x)` with coordinates, for analysis
functions that need the volume. The size is checked against the memory budget first
(`bexa.core.backend.check_fits`); a `MemoryError` names the size and the budget and suggests an
ROI, downsampling or `reduce`. The result is a numpy array whatever `device` did the reading.
`cache=True` keeps the volume in `scan.cache` (when the scan has one), so the next session reads
it back instead of the raw frames.

```python
# scan: the demo chi x mu scan; the frames themselves (a pixel window) as one array
vol = scan.read(roi=bexa.ROI(y=(10, 50)))        # (6, 16, 40, 80) float32
vol = scan.read(roi=bexa.ROI(y=(10, 50)), cache=True)   # the same, kept in the scan's cache for next time
```

#### `Scan.reduce`
`bexa.core.scan.Scan.reduce(accumulators, **kwargs)`

Run several accumulators in one streaming pass. The keywords are those of
`bexa.core.reductions.reduce`: `roi`, `downsample`, `device`, `batch_frames`, `dtype`,
`method`, `cache`, `prefetch`, `show_progress`. Returns one `xarray.Dataset` with every result
(see "Names of the results" under `bexa.core.reductions`).

```python
# scan: the demo chi x mu scan; several accumulators computed in one streaming pass
res = scan.reduce(
    [bexa.acc.Sum(), bexa.acc.Projections(), bexa.acc.MotorCOM(axes=("chi", "mu"), sigma=3.0)],
    roi=bexa.ROI(y=(8, 56), x=(8, 72)),
    show_progress=True,
)
sorted(res.data_vars)                           # ['chi_mu', 'chi_x', 'chi_y', ..., 'xy' (13 items)]
```

#### `Scan.batches`
`bexa.core.scan.Scan.batches(roi=None, downsample=None, **kwargs)`

Iterate over the scan as `bexa.io.base.FrameBatch` objects (`frames` of shape `(n, y, x)`,
`frame_ids`, and the pixel window `y`, `x`) for a reduction of your own. The keywords are those
of `iter_batches` (`batch_frames`, `dtype`, `device`, `method`, `prefetch`, `live_copies`);
batches are sized from the memory budget and the next one is read in a background thread.

```python
# scan: the demo chi x mu scan; its frames in batches, for a loop of your own
peak = 0.0
for batch in scan.batches(batch_frames=32):
    peak = max(peak, float(batch.frames.max()))
round(float(peak), 1)                           # 3786.0
```

#### `Scan.to_dask`
`bexa.core.scan.Scan.to_dask(chunks="auto")`

A lazy dask-backed `DataArray` `(frame, y, x)` of the full-resolution stack in float32 (needs
dask). `chunks` is the number of frames per chunk; `"auto"` makes chunks of about 200 MB.

```python
# scan: the demo chi x mu scan as a lazy dask array (needs dask)
lazy = scan.to_dask()
lazy.sum("frame").compute()                     # DataArray frames ('y', 'x') (64, 80) float32
```

#### `Scan.refresh`
`bexa.core.scan.Scan.refresh()`

Re-read the metadata of a scan that is still being written (new frames, new files) and return
the scan. `scan.info()` then says how many grid points are still missing.

```python
# scan: a scan whose files may have grown since it was opened (a scan still running)
scan.refresh().structure.n_missing              # 0
```

#### `Scan.status`
`bexa.core.scan.Scan.status`

Property: the state of the scan as a `bexa.core.status.ScanStatus`, read by the engine from the
files. For a BLISS dataset it comes from the master entry (`complete`, `running` while the
scan has started and not ended, `aborted` when it ended with fewer frames than planned); an
engine that only knows the frames reports `complete` or `aborted`. A running scan reads as far
as its files go.

```python
# scan: the demo chi x mu scan, complete; the state of a scan on disk
scan.status, str(scan.status)   # (<ScanStatus.COMPLETE: 'complete'>, 'complete')
```

#### `Scan.with_dims`
`bexa.core.scan.Scan.with_dims(motors)`

The same scan read with `motors` (names of per-frame channels) as the dims of its grid:
`["ccmth", "chi", "mu"]` reads an energy mosa with the monochromator angle in degrees instead
of the energy, `["chi", "mu_enc"]` a scan with a motor's encoder instead of the motor; channels
that step together (a motor and its encoder) collapse to the first named; `"energy"` names the
energy of an energy mosa (in keV; `["ccmth", "energy", ...]` names the angle first and reads
degrees). The choice goes into
the recipe, so the scan reopens the same way in a worker, and into the cache keys, so results
on the two grids never mix. The grid is built at once: a name that is not a channel of the
scan raises `ValueError` listing the motors the scan recorded. Layouts with a master file only
(`bexa.open(..., motors=...)` does the same at opening).

```python
# scan: the demo chi x mu scan; the same scan read with dims chosen by hand
angles = scan.with_dims(["chi", "mu"])          # the same dims, chosen by hand
angles.dims, angles.recipe["motors"]            # (('chi', 'mu', 'y', 'x'), ['chi', 'mu'])
sorted(scan.structure.per_frame)                # the channels to choose from: motors and counters
```

#### `Scan.close`
`bexa.core.scan.Scan.close()`

Close the files the source holds open. `with bexa.open(...) as scan:` calls it at the end of
the block.

```python
# scan: an opened scan; its files are closed (they reopen when the scan is used again)
scan.close()
scan.n_frames                                   # 96
```

#### `Scan.recipe`
`scan.recipe`

Attribute: how the scan was opened, as the keywords of `bexa.open`, so that
`bexa.open(**scan.recipe)` opens the same scan again, in another process too; `bexa.stack`
reopens every scan this way in its worker processes. `bexa.open` records `path`, `scan`,
`format` (the name of the matched spec, so the recipe never sniffs), `detector`, `overrides`
and its extra keywords (`stack_dim`, `energy_keV`, ...); `open_profile` records `profile`
(the YAML path, or the object when the profile was built in memory), `sample`, `dataset`,
`scan`, `detector` and the extra keywords. The cache is not part of it: a worker takes the
cache from the scan. `None` for a `Scan` built by hand around a source and for a series that
spans several dataset folders (`Dataset.series` across folders), which `stack(..., workers=)`
refuses with a `ValueError`; a series within one folder, from `bexa.open(path, scan=(20, 24))`
or `Dataset.series`, has one.

```python
# scan7: a scan opened from a folder; what reopens it the same way in a worker process
scan7 = bexa.open(folder, scan=7)
scan7.recipe                     # {'path': '...WS2G_100_DFXM', 'scan': 7, 'format': 'esrf_id03_bliss_2026', 'detector': None, 'overrides': None}
again = bexa.open(**scan7.recipe, cache=False)   # the same scan, no sniffing
bexa.Scan(scan7.source, scan7.spec).recipe       # None: built by hand
```

### bexa.core.status: the state of a scan

During a beamtime a dataset folder holds scans in every state at once: the control system
appends a scan to the master file when it starts and writes its end only when it finishes,
while the detector files fill up in between. Every engine reports one `ScanStatus` per scan
(`scan.status`), the dataset table lists it, `Dataset.select_numbers` filters on it, and a scan
with nothing to read yet raises `ScanNotReadyError`. The examples use the synthetic beamtime
of `bexa.testing.make_esrf_beamtime`: three complete scans, one aborted, one running with
frames, one that has just started.

```python
import tempfile
import bexa
from bexa.core.status import FINISHED, ScanNotReadyError, ScanStatus, status_filter, worst
from bexa.testing import make_esrf_beamtime

beamtime = make_esrf_beamtime(tempfile.mkdtemp())  # a synthetic beamtime: complete, aborted, running and empty scans
ds = bexa.open_dataset(beamtime.dataset_dir, cache=False)  # its scans as a Dataset
```

#### `ScanStatus`
`bexa.core.status.ScanStatus(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)`

A `str` enum (top-level name `bexa.ScanStatus`): `COMPLETE` (every planned frame is there),
`RUNNING` (started and not ended: the files are still growing), `ABORTED` (ended early, with
fewer frames than planned, readable as far as it goes), `EMPTY` (ended, or never started, with
no frame) and `UNREADABLE` (the files could not be read: locked, truncated, missing). `str()`
gives the lower-case word, which the table shows. How an engine decides is under
`bexa.io.engines.hdf5_master.decide`.

```python
# ds: a synthetic beamtime with complete, aborted, running and empty scans; the state of scan 4
ds[4].status, str(ds[4].status)              # (<ScanStatus.ABORTED: 'aborted'>, 'aborted')
ScanStatus("running") is ScanStatus.RUNNING  # True
[str(s) for s in ScanStatus]                 # ['complete', 'running', 'aborted', 'empty', 'unreadable']
```

#### `status_filter`
`bexa.core.status.status_filter(value)`

The statuses a selection keeps, as a frozenset, or `None` for every status: `"finished"` gives
the complete and the aborted scans (`FINISHED`; their files do not change any more),
`"complete"` the complete ones, `"any"` or `None` all of them; a status name or a tuple of
names gives exactly those. An unknown word raises `ValueError`. This is what the `status`
keyword of `Dataset.select_numbers` and the `--status` option of `bexa scan list` go through.

```python
# the statuses a selection keeps, from a word, a tuple of names or "any"
status_filter("finished") == FINISHED        # True: {COMPLETE, ABORTED}
status_filter(("running", "empty")), status_filter("any")   # (frozenset({RUNNING, EMPTY}), None)
```

#### `worst`
`bexa.core.status.worst(statuses)`

The status of a series of scans: the one that lets the least be read (`unreadable` over
`empty` over `running` over `aborted` over `complete`); `EMPTY` for no scans. A
`MultiScanSource` reports it for its parts.

```python
# the status of several scans read as one: the worst of them
worst([ScanStatus.COMPLETE, ScanStatus.RUNNING])   # <ScanStatus.RUNNING: 'running'>
```

#### `ScanNotReadyError`
`bexa.core.status.ScanNotReadyError(message, status, scan=None)`

A `ValueError` (top-level name `bexa.ScanNotReadyError`) raised by `bexa.open`, `Dataset.scan`
and `ds[id]` for a scan with nothing to read yet: `status` is its `ScanStatus`, `scan` its id
when known. `Dataset.table` catches it and keeps a placeholder row.

```python
# ds[6]: a scan that has just started and holds no frame yet
try:
    ds[6]                                    # just started: no frame on disk
except ScanNotReadyError as error:
    str(error), error.status, error.scan     # ('scan 6 of beamtime is running: no frame to read yet', RUNNING, 6)
```

Constants:

- `FINISHED`: `frozenset({ScanStatus.COMPLETE, ScanStatus.ABORTED})`, the statuses whose files do not change any more.

### bexa.core.dataset: the scans of one folder or several

A BLISS dataset folder holds many numbered scans: alignment scans, a mosaicity scan per sample
height, energy series; a measurement sometimes continues in a second folder. A `Dataset` reads
the metadata of every scan once, shows it as a table, and opens scans by id or by what they
contain. The id is the first column of the table: for one folder it is the scan number, so
`ds[7]` is scan 7; for several folders it is a running number over the combined table, next to
a `dataset` column naming the folder of each scan. `ds[id]`, `select`, `series` and `groups`
all take ids. `varying` says which positioners (and whether the energy) differ between chosen
scans, and `stack` reduces them with the same accumulators onto that grid, in memory or in one
HDF5 file. During a beamtime the folder also holds the scan being written: every scan has a
`status` in the table, a scan that cannot be read yet never hides the others, and the
selection keeps the finished scans unless asked otherwise. A `Dataset` is not an
`xarray.Dataset`.

```python
import bexa
from bexa.testing import make_esrf_beamtime, make_esrf_zstack

ds = bexa.demo("zstack")   # 30 scans: at each of five samz heights, a chi x mu scan and five mu scans at 16.98 to 17.02 keV
part2 = make_esrf_zstack(ds.path.parent, dataset="demo_zstack_2", z_values=(0.003, 0.004), energies=(17.0,))   # two more heights, in a second folder
live = bexa.open_dataset(make_esrf_beamtime(ds.path.parent).dataset_dir, cache=False)   # scans 1-3 complete, 4 aborted, 5 running, 6 just started
folder = "/data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM"  # a dataset folder (the synthetic z-stack stands in for it)
```

#### `open_dataset`
`bexa.core.dataset.open_dataset(path=None, **kwargs)`

Open a dataset folder (or its master file, or one of its `scanNNNN` folders) as a `Dataset`
(top-level name `bexa.open_dataset`); a list of folders when a measurement continued in another
dataset (one table, ids running over all of them); `profile=` with `sample=` (or `dataset=`)
instead of a path, as for `bexa.open`. The keywords are those of `Dataset`.

```python
# folder: a dataset folder (the synthetic z-stack); every scan of it as one Dataset
ds = bexa.open_dataset(folder)
ds = bexa.open_dataset(profile="hc6293", sample="WS2G_100")
both = bexa.open_dataset([ds.path, part2.dataset_dir], cache="bexa_cache")   # a measurement continued in a second folder; the cache in ./bexa_cache
```

#### `Dataset`
`bexa.core.dataset.Dataset(path=None, *, profile=None, sample=None, dataset=None, format=None, detector=None, cache=True, cache_reductions=None, **open_kwargs)`

The class behind `open_dataset` (top-level name `bexa.Dataset`). `format`, `detector`,
`cache_reductions` and `open_kwargs` are passed to `bexa.open` for every scan; `cache` is
`True` (the folder of `bexa.config.paths.cache_root`, or the profile's), a folder path (one
cache for the dataset there), a `Cache` or `False`, and every scan shares it. The layout of a
folder is sniffed once, when the dataset is opened (a folder that cannot be read fails there:
a BLISS dataset copied without its master file says "no master file ... is missing"), and the
scans are then opened by that spec's name. Scans are opened once and their handles kept until
`refresh` or `close`; a `Dataset` is a context manager, and in a notebook it shows its table.

- `scans`: property, every scan id, sorted: the scan numbers with detector files for one
  folder, `0, 1, 2, ...` over the combined table for several.
- `len(ds)`, `for number in ds`, `7 in ds`: work on the ids.
- `ds[7]` is `ds.scan(7)`; `ds[[1, 7, 13]]` and `ds[(1, 5)]` (an inclusive range) return
  lists of `Scan`.
- `folders`: property, the dataset folders in the order their scans are numbered; `paths` is
  the same list, `path` the first folder and `name` the folder names joined with `" + "`.
- `cache`: the shared `bexa.core.cache.Cache`, or `None`; `clear_cache()` empties it.
- `problems`: property, `{id: reason}` for the scans the table could not read (see `table`).

```python
# ds: the z-stack dataset; how many scans, their ids, one scan or several by id
len(ds), ds.scans[:3]            # 30, [1, 2, 3]
first, layers = ds[1], ds[[1, 7, 13]]
len(both), both.scans[-1]        # 34 scans over two folders: ids 0 to 33
both[30].source.scan, both[30].name   # (1, 'demo_zstack_2'): id 30 is scan 1 of the second folder
```

#### `Dataset.table`
`bexa.core.dataset.Dataset.table(refresh=False, compact=False)`

A pandas DataFrame with one row per scan: `id` (`dataset` next, for several folders), `scan`,
`status` (`complete`, `running`, `aborted`, `empty` or `unreadable`: `bexa.core.status`),
`type` (`fscan2d`, `amesh`, `ascan`, ...), `title`, `motors` (`"chi x mu"`), `shape`, `ranges`,
`frames`, `missing` (grid points without a frame, or planned frames not taken), `energy_keV`,
then one column per positioner recorded in the master file (`samz`, `ccmth`, `phi`, ...). A
scan that cannot be read yet (running without a frame, empty, unreadable) keeps a row with its
status and nothing else, its reason goes to `problems` and one warning is logged: one such
scan never hides the others. Built once and kept; `refresh=True` looks at the folders again
first (`refresh`). `compact=True` drops the positioner columns that never change or change by
less than the spec's `motors.min_step` (drift), a motor also listed under an alias, the
followers of a motor that steps (`samz` when `uz` does; the spec's `motors.coupled`), the
`missing` column when no scan has missing frames and the `status` column when every scan is
complete: what a notebook shows.

```python
# ds: the z-stack dataset; one row per scan with its type, motors, energy and positioners
ds.table()[["id", "type", "motors", "energy_keV", "samz"]]  # DataFrame: 30 rows x 5 columns (id, type, motors, energy_keV, samz)
both.table(compact=True)         # id, dataset, scan, type, ..., energy_keV, ccmth, chi, samz, z1
live.table()[["id", "status", "frames", "missing"]]   # 1-3 complete, 4 aborted (19 of 24), 5 running (15), 6 running (0)
live.problems                    # {6: 'scan 6 of beamtime is running: no frame to read yet'}
```

#### `Dataset.info`
`bexa.core.dataset.Dataset.info()`

The table as text, with the folder and the number of scans on top.

```python
# ds: the z-stack dataset; the table as text with the folder on top
print(ds.info())
ds.info().splitlines()[0]                       # 'hc6293/WS2G_100/WS2G_100_DFXM: 30 scans in C:\\Users\\sport...'
```

#### `Dataset.select_numbers`
`bexa.core.dataset.Dataset.select_numbers(scans=None, *, type=None, motors=None, where=None, status="finished", **positioners)`

Scan ids chosen by hand and/or by content. `scans` is an id, an inclusive `(start, end)` tuple
or a list (ids that are not on disk raise); `type` the scan type; `motors` the scanned motors
exactly (`"mu"` or `("chi", "mu")`); `where` a function of a table row that returns True to
keep the scan; `status` the scan statuses to keep: `"finished"` (complete and aborted, whose
files do not change any more; the default), `"complete"`, `"any"`, a status name or a tuple of
names, a scan named in `scans` but left out for its status being reported once; any other
keyword names a table column with a value (matched within 1e-9) or an inclusive `(low, high)`
range, `energy_keV` included, and `dataset` (a folder name) for several folders.

```python
# ds: the z-stack dataset; the ids of the scans that match a type, a positioner, a test
ds.select_numbers(type="fscan2d")                   # [1, 7, 13, 19, 25]
ds.select_numbers(type="fscan1d", samz=(-0.0015, 0.0015), where=lambda row: row["energy_keV"] > 17.005)  # [11, 12, 17, 18, 23, 24]
both.select_numbers(dataset="demo_zstack_2")        # [30, 31, 32, 33]: the scans of the second folder
live.select_numbers()                               # [1, 2, 3, 4]: the finished scans; 5 and 6 are still running
live.select_numbers(status="complete"), live.select_numbers(status="any")   # ([1, 2, 3], [1, 2, 3, 4, 5, 6])
```

#### `Dataset.select`
`bexa.core.dataset.Dataset.select(scans=None, **criteria)`

The scans of `select_numbers`, opened: a list of `Scan`.

```python
# ds: the z-stack dataset; the chi x mu scans as opened Scan handles
layers = ds.select(type="fscan2d")              # one mosaicity scan per height
[s.source.scan for s in ds.select(type="fscan2d", samz=(-0.0015, 0.0015))]  # [7, 13, 19]
```

#### `Dataset.series`
`bexa.core.dataset.Dataset.series(scans=None, *, dim=None, **criteria)`

Several scans opened as one `Scan` with a new leading dim: `"energy"` (the default when every
scan has its own energy; sorted by energy), `"scan"` (the ids) or a positioner such as
`"samz"` read from every scan. The scans must share one motor grid; the selection works as in
`select_numbers`, and may span the folders of the dataset (`combine_sources` stacks the parts).

```python
# ds: the z-stack dataset; several scans read as one along a new dim (energy, height)
energy = ds.series(type="fscan1d", samz=0.0)    # dims ('energy', 'mu', 'y', 'x')
zstack = ds.series(type="fscan2d", dim="samz")  # dims ('samz', 'chi', 'mu', 'y', 'x')
heights = both.series(type="fscan2d", dim="samz")   # the seven heights of both folders as one scan
```

#### `Dataset.groups`
`bexa.core.dataset.Dataset.groups(by, scans=None, **criteria)`

`{value: [ids]}` grouped by a table column, sorted by value; `samz` gives the layers of a
z-stack. The selection works as in `select_numbers`.

```python
# ds: the z-stack dataset; the chi x mu scans grouped by the height they were taken at
ds.groups("samz", type="fscan2d")   # {-0.002: [1], -0.001: [7], 0.0: [13], 0.001: [19], 0.002: [25]}
ds.groups("energy_keV", type="fscan1d", samz=0.0)  # {16.98: [14], 16.99: [15], 17.0: [16], 17.01: [17], 17.02: [18]}
```

#### `Dataset.varying`
`bexa.core.dataset.Dataset.varying(scans=None, axes=None, ignore=None, **criteria)`

What differs between the chosen scans (selection as in `select_numbers`): `{dim: sorted
values}` of the positioners, and of the energy as `"energy"` in keV, that are not the same in
every scan, outer dim first; see `varying`. These are the dims `stack(..., dim="auto")` builds
its grid on. `axes` names the dims by hand instead, `ignore` names positioners that are never
axes (see `varying`).

```python
# ds: the z-stack dataset; what differs between the chosen scans (the stack axes)
ds.varying(type="fscan2d")                       # {'samz': array([-0.002, -0.001, 0., 0.001, 0.002])}
ds.varying(type="fscan1d", samz=0.0)             # {'energy': array([16.98, 16.99, 17., 17.01, 17.02])}
ds.varying(type="fscan1d")                       # {'samz': ..., 'energy': ...}: a z-stack at every energy
ds.varying(scans=[1])                            # {}: one scan, nothing varies
ds.varying(type="fscan1d", axes=["energy"])      # {'energy': ...}: the energy alone, whatever else moved
```

#### `Dataset.scan`
`bexa.core.dataset.Dataset.scan(number)`

Open one scan by id (the same as `ds[number]`); the handle is kept for the next call. A scan
that is still running opens and reads as far as its files go; one with nothing to read yet
(running without a frame, empty, unreadable) raises `bexa.core.status.ScanNotReadyError`,
which names its status.

```python
# ds: the z-stack dataset; one scan by id; live: a beamtime with a scan still running
scan = ds.scan(13)
live[5].status, live[5].n_frames             # (<ScanStatus.RUNNING: 'running'>, 15): readable so far
try:
    live[6]
except bexa.ScanNotReadyError as error:
    error.status, error.scan                 # (<ScanStatus.RUNNING: 'running'>, 6)
```

#### `Dataset.origin`
`bexa.core.dataset.Dataset.origin(scan_id)`

The dataset folder and the scan number behind an id, `(Path, int)`: for one folder the number
is the id itself, for several it says which folder a running id comes from. An id that is not
in the dataset raises `ValueError`. The scan picker keeps its selection across folders as these
pairs.

```python
# both: the z-stack continued in a second folder; where a running id comes from
folder_of, number = both.origin(30)
folder_of.name, number                       # ('demo_zstack_2', 1)
```

#### `Dataset.refresh`
`bexa.core.dataset.Dataset.refresh(everything=False)`

Look at the folders again, for a dataset that is still being written, and return the dataset:
new scans appear, the scans that were running, empty or unreadable are read anew by the next
`table`, and the complete and aborted ones keep their handles and rows (their files do not
change any more). `everything=True` closes and forgets every scan.

```python
# ds, live: datasets whose folders may hold new or finished scans since they were opened
ds.refresh().scans                              # [1, 2, 3, ..., 30 (30 items)]
live.refresh().table()["status"].tolist()    # the running scans read again; the others as before
```

#### `Dataset.clear_cache`
`bexa.core.dataset.Dataset.clear_cache()`

Delete every file of the dataset's cache (previews, cached results and the stacks `store=True`
built) and return their number; 0 without a cache. The notebook's "clear the cache and
recompute" switch.

```python
# both: a dataset with a cache folder; every cached result of it removed
both.clear_cache()               # the number of cached files deleted
```

#### `Dataset.close`
`bexa.core.dataset.Dataset.close()`

Close every scan the dataset opened. `with bexa.open_dataset(...) as ds:` calls it at the end.

```python
# ds: an opened dataset; the files of every opened scan closed
ds.close()
len(ds)                                         # 30
```

#### `Dataset.problems`
`bexa.core.dataset.Dataset.problems`

Property: `{id: reason}` for the scans whose table row is a placeholder because they could not
be read (still running without a frame, empty, unreadable), the reason being the error met;
empty when every scan could be read. Rebuilt with the table.

```python
# live: a beamtime with a scan too young to read; why its row is a placeholder
live.problems                    # {6: 'scan 6 of beamtime is running: no frame to read yet'}
```

#### `Dataset.motor_candidates`
`bexa.core.dataset.Dataset.motor_candidates(scans=None, **criteria)`

Every motor the chosen scans (selection as in `select_numbers`) could be read along:
`{name: MotorCandidate(n_values, where, default)}`. The motors that move within the first
chosen scan come first (`where="scan"`: its scanned motors, pressed by default, then every
other recorded motor that moves by more than the spec's `min_step`, released: `ccmth` next to
`energy`), then the positioners that differ between the chosen scans (`where="stack"`, pressed
when `varying` would take them as axes, released for followers, aliases and drift: `samz` next
to `uz`). The scan picker's motor buttons are built from it; see
`bexa.core.scan_table.motor_candidates`.

```python
# ds: the z-stack dataset; the motors its energy series could be read along
{n: tuple(c) for n, c in ds.motor_candidates(type="fscan1d").items()}  # {'mu': (8, 'scan', True), 'samz': (5, 'stack', True), 'energy': (5, 'stack', True), 'cc...
# {'mu': (8, 'scan', True), 'samz': (5, 'stack', True), 'energy': (5, 'stack', True), 'ccmth': (5, 'stack', False)}
ds.motor_candidates(type="fscan2d")["samz"]      # MotorCandidate(n_values=5, where='stack', default=True)
ds.motor_candidates([])                          # {}
```

#### `varying`
`bexa.core.dataset.varying(scans, tolerance=None, axes=None, ignore=None)`

The positioners, and the energy, that differ between the given `Scan`s: `{dim: sorted values}`,
the energy reported as `"energy"` in keV. Scanned motors are left out, and so is the
monochromator motor the energy derives from (`ccmth`) when the energy itself varies, and a
motor the spec also records under an alias (`z1` for `samz`), so nothing counts twice. A motor
that another one drives (the spec's `motors.coupled`: at ID03 `ux`, `uy` and `uz` each move the
stage readbacks `samx`, `samy`, `samz` and `shexatx`, `shexaty`, `shexatz`) is not an axis of
its own when its leader steps, and neither is any positioner that steps in lockstep with
another one, whatever the spec says: of those, the leader, the energy, then the first in
`motors.known` order is kept. A change smaller than the spec's `motors.min_step` of a motor
(0.5 um for the ID03 stage motors) is drift, not a step: a temperature series moves the stage
readbacks by a tenth of a micron between its datasets, a z-stack by a micron per height. `ignore` names positioners that are never axes however they vary
(a readback that drifts between scans; a follower of an ignored leader stays dropped). Values
closer than `tolerance` (default 1e-3 of their spread) are one grid point. The dims come slowest first: the one that changes least often from
scan to scan is the outer one, as the height is for a z-stack repeated at every energy. `axes`
overrides all of this: these names, in this order, are the dims (positioners, `"energy"` or
`"scan"`; `ValueError` for one a scan does not record), for when something else drifts between
the scans and the automatic choice is not what was measured. `Dataset.varying` calls it on a
selection.

```python
# the function behind Dataset.varying, on a list of opened scans
from bexa.core.dataset import varying

varying(ds.select(type="fscan2d"))               # {'samz': array([-0.002, -0.001, 0., 0.001, 0.002])}
varying(ds[[2, 3, 4]])                           # {'energy': array([16.98, 16.99, 17.])}
varying(ds[[2, 3, 4]], tolerance=0.05)           # {}: energies within 0.05 keV count as one
varying(ds.select(type="fscan2d"), axes=["samz"])   # the same, named by hand (any positioner, varying or not)
varying(ds.select(type="fscan1d"), ignore=["samz"])   # {'energy': ...}: the heights are not an axis here
```

#### `common_grid`
`bexa.core.dataset.common_grid(scans, dims=None, roi=None, downsample=None)`

One coordinate array per scanned motor spanning every scan (top-level name `bexa.common_grid`),
for scans measured on different grids: another range or step at some heights, two datasets
scanned differently. For each motor: from the lowest to the highest value over the scans, in a
step no coarser than the finest median step any scan used (the grid ends on the highest value);
a motor whose grid is the same in every scan, up to the readback jitter (a quarter of a step),
keeps it. `dims` limits the answer to some motors
(default: every scanned motor, which every scan must have, else `ValueError`); `roi` and
`downsample`, as in `stack`, restrict the ranges the way the pass does. `regrid` puts results
onto this grid, and `stack` does so by itself.

```python
# two scans of the same motors on different grids (a coarse and a fine one)
from bexa.testing import make_esrf_scan

coarse = make_esrf_scan(ds.path.parent, dataset="grids", scan=1, motors=(("chi", 3), ("mu", 6)), ranges={"mu": (-1.0, 1.0)}, n_files=1)
fine = make_esrf_scan(ds.path.parent, dataset="grids", scan=2, motors=(("chi", 3), ("mu", 11)), ranges={"mu": (-0.5, 1.5)}, n_files=1)
two = bexa.open_dataset(coarse.dataset_dir, cache=False).select([1, 2])
grid = bexa.common_grid(two)                     # {'chi': the shared grid, 'mu': 14 points from -1 to 1.5}
len(grid["mu"]), float(grid["mu"][-1])           # (14, 1.5)
```

#### `regrid`
`bexa.core.dataset.regrid(result, coords, method="linear")`

`result` (a Dataset or DataArray from a pass, or any xarray object; top-level name
`bexa.regrid`) with every variable on a dim of `coords` interpolated onto those values along
them (`method` as in `DataArray.interp`, `"nearest"` for a dim with a single point), NaN outside
the variable's own range; variables without those dims, the maps, come back as they are, and the
attrs are kept.

```python
# curves: the per-frame totals of the coarse scan; grid: the common grid of both scans
curves = two[0].reduce([bexa.acc.FrameStats()], device="cpu")   # (chi, mu) on the coarse grid
on_grid = bexa.regrid(curves, grid)                             # (chi, mu) on the common grid
on_grid["frame_sum"].sizes["mu"], bool(on_grid["frame_sum"].isnull().values[:, -1].all())   # (14, True): nothing beyond mu = 1
bexa.regrid(curves["frame_sum"], {"mu": grid["mu"]}).dims       # ('chi', 'mu'): one array, one dim
```

#### `stack`
`bexa.core.dataset.stack(scans, accumulators, dim="auto", *, coords=None, store=None, dtype=None, tolerance=None, workers=None, grid="auto", ignore=None, **reduce_kwargs)`

Reduce every scan with the same accumulators and stack the results on a grid of scans
(top-level name `bexa.stack`). `dim` names the new dims: `"auto"` takes what `varying` finds (a
z-stack becomes `(samz, y, x)` maps, a z-stack at several energies `(samz, energy, y, x)`; one
scan gets no new dim, scans that differ in nothing are stacked on `"scan"`; `ignore` names
positioners it must never take as axes however they vary); one name
(`"samz"`) or a list (`["uz", "energy"]`) names them by hand, each read from every scan
(`"energy"` is the energy in keV, `"scan"` the scan numbers), unless `coords` gives the values
of a single dim. The grid is dense: NaN where no scan sits at a combination (a warning says how
many), and two scans at the same position are a `ValueError`. `roi`, `downsample`, `device` and
the other keywords of `reduce` apply to every scan, so the maps line up pixel by pixel. Scans
measured on different motor grids (another range or step at some heights) are handled by
`grid`: `"auto"` (default) interpolates every result that has motor dims (previews, per-frame
statistics, curves) onto `common_grid` of the scans, the full range in the finest step, and
says so in the log; a `{dim: values}` mapping names the grid (the same for two stacks that
will be compared); `False` requires one grid and raises otherwise. Maps without motor dims are
never touched. Replaces the layer loop of `betterCOM_stitchingZ`.

- In memory (the default): each output variable is allocated once (`dtype`, say `np.float32`,
  halves the maps) and filled scan by scan, so peak memory is the result plus one scan; the
  result is checked against the memory budget first (`MemoryError` suggests `store=True`).
- `store=True` writes the grid scan by scan into one HDF5 file in the scans' cache
  (`stacks/<fingerprint>.h5`; a path names the file instead), compressed with lzf and byte
  shuffle (`STORE_COMPRESSION`; a tenth of gzip's writing time), and returns it opened with
  `bexa.io.cube.open_lazy`: nothing is loaded until a block is indexed, which is how the
  previews of a dataset that does not fit in memory are browsed. A file built from the same
  scans with the same request is reused.
- `workers` reduces several scans at the same time, each in its own spawned process: this is
  how the reading and decompression of the frames run in parallel, since h5py lets one thread
  at a time into HDF5 and threads cannot do it. `None` or `1` reduces the scans one after the
  other in this process; an int is the number of processes; `"auto"` takes the usable cores
  of the job but one, or `GPU_WORKERS` (3) when the device is the GPU. A worker reopens its
  scan from `Scan.recipe` (a `Scan` built by hand has none: `ValueError`), receives the
  accumulators as built (`Accumulator` pickles its parameters, never its totals) and uses the
  device the parent would use (`device="auto"` is resolved here, so the workers share one GPU
  when there is one), with its share of the parent's memory budget as `BEXA_MEMORY_BYTES` and
  `BEXA_THREADS` set to `min(8, cores // workers)`. Results arrive as the scans finish and are
  written straight into the stack, in memory or on disk, under one progress bar over the
  scans; an error in a worker reaches the caller and stops the pool. In a script, call it under
  `if __name__ == "__main__":` as for any spawned pool.
- Every result with a `(y, x)` image carries per-frame statistics in its attrs
  (`block_stats_dims`, `block_total`, `block_p1`, `block_p99`) for `bexa.core.stacks.brightest`
  and the browsers' colour limits: the totals are exact, the 1st and 99th percentiles come
  from a strided sample of at most `STATS_SAMPLE` (262144) pixels per frame. The dataset
  attrs record `stack_dims`, `scan_ids`, `scan_names`, `accumulators`, `stack_key` and, for a
  store, `lazy_file`. An `energy` dim has the unit `keV`.
- One pass, two resolutions: a `Preview(downsample=8)` next to `Sum`, `FrameStats` and
  `MotorCOM` gives the binned volume of every scan (pixel dims `yb`, `xb`) and the
  full-resolution maps from the same pass, in one stack; `examples/13_esrf_dfxm_general.ipynb`
  reads everything it shows from such a stack.

```python
# ds: the z-stack dataset; the chi x mu scans reduced and stacked on the height axis
stack = bexa.stack(ds.select(type="fscan2d"), [bexa.acc.Sum(), bexa.acc.MotorCOM(axes=("mu",))], dim="auto")
stack["com_mu"].dims, stack.attrs["stack_dims"]            # ('samz', 'y', 'x'), ['samz']
grid = bexa.stack(ds.select(type="fscan1d"), [bexa.acc.Sum()], dim="auto")   # 25 mu scans: five heights at five energies
grid["sum"].dims, grid["sum"].attrs["block_total"].shape    # ('samz', 'energy', 'y', 'x'), (5, 5)
stored = bexa.stack(both.select(type="fscan2d"), [bexa.acc.Preview()], downsample=2, store=True)   # one file in ./bexa_cache/stacks
stored["preview"].dims, stored["preview"].isel(samz=3).values.shape    # ('samz', 'chi', 'mu', 'y', 'x'), (6, 15, 20, 24): one block read
two = bexa.stack(ds.select(scans=[1, 7]), [bexa.acc.Sum(), bexa.acc.MotorCOM(axes=("mu",))], dim="auto", workers=2)   # two heights, one process each
two["sum"].dims, two.attrs["scan_ids"]                     # ('samz', 'y', 'x'), [1, 7]: the same result, the scans read side by side
one_pass = bexa.stack(ds.select(type="fscan2d"), [bexa.acc.Preview(downsample=2), bexa.acc.Sum()], dim="auto")
one_pass["preview"].dims, one_pass["sum"].dims              # ('samz', 'chi', 'mu', 'yb', 'xb'), ('samz', 'y', 'x'): binned volume, full maps
```

Constants:

- `STORE_COMPRESSION`: `"lzf"`, the HDF5 compression (with byte shuffle) of stacks built with
  `store=`. `GPU_WORKERS`: `3`, the processes `workers="auto"` starts when the device is the
  GPU. `STATS_SAMPLE`: `262144`, the most pixels per frame the percentiles of the per-frame
  statistics look at.

### bexa.core.scan_table: the rows of a dataset's table

`Dataset.table` is assembled from these: one row per opened scan, a placeholder row for a scan
that could not be read, the column order, and the columns a compact table leaves out. Useful
when building a table of scans from your own list of `Scan` objects.

```python
import tempfile
import bexa
from bexa.core.scan_table import (FIXED_COLUMNS, LEAD_COLUMNS, MotorCandidate, arrange, compact_columns,
                                  followers, min_steps, motor_candidates, placeholder_row, scan_row)
from bexa.core.status import ScanStatus
from bexa.testing import make_esrf_beamtime

beamtime = make_esrf_beamtime(tempfile.mkdtemp())  # a synthetic beamtime: complete, aborted, running and empty scans
ds = bexa.open_dataset(beamtime.dataset_dir, cache=False)  # its scans as a Dataset
scan = ds[1]                                    # a complete chi x mu scan of it
```

#### `scan_row`
`bexa.core.scan_table.scan_row(scan, scan_id, number, dataset=None)`

The table row of an opened `Scan` as a dict: `id`, `dataset` (when given, for several
folders), `scan` (its number), `status`, `type`, `title`, `motors`, `shape`, `ranges`, `frames`,
`missing`, `energy_keV`, then every scalar positioner of its structure.

```python
# scan: scan 1 of the synthetic beamtime; its table row
row = scan_row(scan, 1, 1)
row["status"], row["motors"], row["frames"]      # ('complete', 'chi x mu', 24)
```

#### `placeholder_row`
`bexa.core.scan_table.placeholder_row(scan_id, number, status, dataset=None)`

The row of a scan that cannot be read yet: its `status`, empty text columns, 0 frames and no
energy, so that it sits in the table next to the others.

```python
# the row of a scan that cannot be read yet: its status and nothing else
placeholder_row(6, 6, ScanStatus.RUNNING)["status"]   # 'running'
placeholder_row(6, 6, ScanStatus.EMPTY, dataset="beamtime")["dataset"]  # 'beamtime'
```

#### `arrange`
`bexa.core.scan_table.arrange(rows)`

The rows as a DataFrame in the table's column order: `LEAD_COLUMNS` (`id`, `dataset`), then
`FIXED_COLUMNS` (`scan`, `status`, `type`, `title`, `motors`, `shape`, `ranges`, `frames`,
`missing`), `energy_keV`, then the positioners sorted by name.

```python
# rows of three scans put in the table's column order
table = arrange([scan_row(ds[n], n, n) for n in (1, 2, 3)])
list(table.columns[:5])                          # ['id', 'scan', 'status', 'type', 'title']
```

#### `motor_candidates`
`bexa.core.scan_table.motor_candidates(scan, table, defaults)`

Every motor that moves within `scan` or differs between the rows of `table` (the full table
rows of the scans in question), as `{name: MotorCandidate}`: the scanned motors of `scan`
(pressed by default), the other per-frame channels that are motors and move by more than the
spec's `min_step` (released), then the positioner columns that differ between the rows (the
energy column as `energy`; aliases such as `z1` beside `samz` counted once), pressed when
`defaults` names them (what `bexa.core.dataset.varying` found, in its order) and released
otherwise. `Dataset.motor_candidates` calls it.

```python
# scan: scan 1 of the synthetic beamtime; the motors it and the table could be read along
motor_candidates(scan, ds.table(), [])           # {'chi': MotorCandidate(n_values=3, where='scan', default=True), 'mu': MotorCandidate(n_values=8, where='scan', default=True)}
```

#### `MotorCandidate`
`bexa.core.scan_table.MotorCandidate(n_values, where, default)`

A named tuple: the distinct values a motor takes, `where` it moves (`"scan"`: within the scan;
`"stack"`: between the scans) and whether a reader takes it by default.

```python
# one candidate: five values, between the scans, pressed by default
MotorCandidate(5, "stack", True).default        # True
```

#### `compact_columns`
`bexa.core.scan_table.compact_columns(table, spec)`

The columns a compact table drops, judged with the format `spec`: positioners that never
change, a motor kept under a legacy name too (`z1` next to `samz`), the readbacks a stepping
leader drives (`motors.coupled`), a drift below `motors.min_step`, `missing` when no scan has
missing frames and `status` when every scan is complete.

```python
# ds: the synthetic beamtime; the columns a compact table leaves out
dropped = compact_columns(ds.table(), scan.spec)
"ccmth" in dropped, "status" in dropped          # (True, False): one scan is running
```

#### `min_steps`
`bexa.core.scan_table.min_steps(spec)`

`{motor: step}` from the spec's `motors.min_step` (empty for `None`): a change smaller than the
step between two scans is the motor's drift, not a move.

```python
# the smallest move of a motor that counts as a step (the spec's motors.min_step)
min_steps(scan.spec)["uz"]                       # 0.0005: half a micron for the ID03 stage
```

#### `followers`
`bexa.core.scan_table.followers(spec)`

`{leader: [followers]}` from the spec's `motors.coupled` (empty for `None`): the readbacks a
motor drives (`uz` moves `samz`), counted once by `varying` and the compact table.

```python
# the motors a stepping leader drives (the spec's motors.coupled)
followers(scan.spec)["uz"][:3]                   # ['samx', 'samy', 'samz']
```

Constants:

- `LEAD_COLUMNS`: `("id", "dataset")`. `FIXED_COLUMNS`: `("scan", "status", "type", "title", "motors", "shape", "ranges", "frames", "missing")`.

### bexa.core.stacks: statistics, brightest condition and colour limits of stacked volumes

A volume stacked by `bexa.stack` is `(outer..., motors..., y, x)` and may live on disk
(`store=`, read through `bexa.io.cube.open_lazy`). These functions never load more than a sample
of its frames: they use the per-frame statistics the stack carries in its attrs
(`block_stats_dims`, `block_total`, `block_p1`, `block_p99`, computed scan by scan while
stacking; a `transpose` keeps them), or stream over the frames one at a time when the attrs are
missing or no longer match the dims. `brightest`, `shared_limits` and `total_volume` are also
reachable from `bexa.viz.interactive`, and the browsers use the first two.

```python
import tempfile
import numpy as np
import bexa
from bexa.core import stacks

ds = bexa.open_dataset(bexa.demo("zstack").path, cache=tempfile.mkdtemp())   # the demo z-stack, with a disk cache
stacked = bexa.stack(ds.select(type="fscan2d"), [bexa.acc.Preview()], dim="auto", downsample=2, store=True)  # the previews of the chi x mu scans, stacked on disk
vol = stacked["preview"]                 # (samz, chi, mu, y, x) = (5, 6, 15, 20, 24), in a cache file, read block by block
```

#### `brightest`
`bexa.core.stacks.brightest(volume, over=None, name=None)`

The position of the frame with the largest total intensity, `{dim: index}`, from the per-frame
totals (a plain array is summed frame by frame). `volume` is a stacked DataArray or the Dataset
a stack returned, `name` picking the variable (default: the first). `over` lists the dims to
report, default every dim but the last two; the totals are summed over the others first, so
`over=vol.dims[:-3]` is the condition that lights up the whole `(z, y, x)` block, ready for
`browser.update(**peak)`. Dims not in front of the image raise `ValueError`.

```python
# vol: the previews of a z-stack, on disk; the condition that lights the volume up
stacks.brightest(vol)                                   # {'samz': 1, 'chi': 2, 'mu': 6}: the brightest frame
peak = stacks.brightest(vol, over=("chi", "mu"))        # {'chi': 2, 'mu': 6}: the condition that lights up every height
vol.transpose("chi", "mu", "samz", "y", "x").bexa.browse_volume().update(**peak)
stacks.brightest(stacked)                               # the Dataset: its first variable
```

#### `shared_limits`
`bexa.core.stacks.shared_limits(volume, low=1.0, high=99.0, log=False)`

Colour limits `(vmin, vmax)` shared by every frame of a volume, without loading all of it: for
the default `low=1, high=99` the minimum of the per-frame 1st percentiles and the maximum of
the 99th from the stack's statistics; for other percentiles, or a volume without statistics,
the pooled percentiles of the array itself when it is in memory under 256 MiB, else of a
sample of 64 frames spread evenly over the leading dims. `log=True` returns the limits of
`log10(1 + I)`, as the browsers show the data. `(0, 1)` when nothing is finite.

```python
# vol: the previews of a z-stack, on disk; one colour scale for every frame of it
stacks.shared_limits(vol)                        # (10.0, 2091.95): from the stored percentiles, no frame read
stacks.shared_limits(vol, 5, 99.5, log=True)     # (1.0414, 3.3208): limits of log10(1 + I)
```

#### `total_volume`
`bexa.core.stacks.total_volume(volume, keep=None, downsample=None, method="mean")`

The intensity summed over every leading dim but `keep` (a name or a list; default the first
one), read one kept position at a time, so a lazy stack on disk costs the memory of one such
block: `stacked["sum"]` of a z-stack with `keep="samz"` is the total scattered intensity per
voxel `(z, y, x)`, whatever was scanned (energies included), and `stacked["preview"]` summed over
the scan motors gives the same from the binned frames. `downsample` (an int or `(by, bx)`) bins
the pixels of the result with block `method` (`"mean"`, `"sum"` or `"max"`), the coordinates
being the full-resolution index of each block's first pixel, so a volume of summed images takes
the voxel size of the previews. NaN frames (missing scans) count as zero. The result is in
memory, named `total_<name>`, with attrs `summed_over`, `binning` and `method`; a dim that is
not in front of the image raises `ValueError`.

```python
# maps: the summed images of a z-stack; the intensity per voxel of the whole sample
maps = bexa.stack(ds.select(type="fscan2d"), [bexa.acc.Sum()], dim="auto", store=True)   # (samz, y, x) summed images
total = stacks.total_volume(maps["sum"], keep="samz", downsample=2)   # (5, 20, 24): the total intensity per voxel, binned by 2
total.attrs["summed_over"], float(total.sel(samz=0.0).sum())           # ([], 982299.2): nothing else to sum here; its total at samz = 0
over_motors = stacks.total_volume(vol, keep="samz")                    # (samz, y, x) from the previews, summed over chi and mu
over_motors.dims, over_motors.attrs["summed_over"]                     # (('samz', 'y', 'x'), ['chi', 'mu'])
```

#### `block_stats`
`bexa.core.stacks.block_stats(volume, max_frames=None)`

Per-frame `total`, `p1` and `p99` (1st and 99th percentiles) over the last two dims, as a dict
of arrays shaped like the leading dims: from the attrs when the stack carries them, else
computed (in one go for a small in-memory array, frame by frame for a lazy one). `max_frames`
visits only that many frames, spread evenly, and leaves the others NaN: enough for colour
limits, not for `brightest`.

```python
# vol: a stack with per-frame statistics in its attrs; plain: the same file without them
st = stacks.block_stats(vol)                     # from the attrs: total, p1, p99 of shape (5, 6, 15)
plain = bexa.io.cube.open_lazy(stacked.attrs["lazy_file"])["preview"]   # opened again: no statistics in its attrs
np.isfinite(stacks.block_stats(plain, max_frames=16)["p99"]).sum()       # 16: that many frames read, the rest NaN
```

#### `is_lazy`
`bexa.core.stacks.is_lazy(array)`

True for a DataArray that reads from a file on demand (`bexa.io.cube.open_lazy`, a stack built
with `store=`); an `isel` of it is still lazy (`.values` reads that block), `.load()` turns it
into an in-memory array.

```python
# vol: a volume read from the cache file block by block; a block of it; the block loaded
stacks.is_lazy(vol), stacks.is_lazy(vol.isel(samz=0)), stacks.is_lazy(vol.isel(samz=0).load())   # (True, True, False)
```

Constants:

- `IN_MEMORY_LIMIT`: `256 * 1024**2`, the size up to which an in-memory array is simply loaded for exact statistics.

### bexa.core.roi: regions of interest

An `ROI` selects part of a scan along any of its dims. Pixel ranges `y` and `x` are half-open
index ranges `[low, high)` in full-resolution pixels. Motor and energy ranges are inclusive
values in motor units (`chi=(0, 8)` keeps every grid point with 0 <= chi <= 8), or half-open
grid indices for dims listed in `index_dims`. A `None` bound means "to the edge". Pass an ROI
as `roi=` to the verbs, `preview`, `read`, `reduce` and `stack`, or set `scan.roi` once.

```python
import bexa

scan = bexa.demo("mosa")                        # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
prev = scan.preview(downsample=4)               # the scan binned 4 x 4 in the pixels
```

#### `ROI`
`bexa.core.roi.ROI(ranges=None, index_dims=(), **kwargs)`

Build an ROI from keywords, from a `{dim: (low, high)}` dict, or both (top-level name
`bexa.ROI`). `index_dims` lists motor dims whose ranges are grid indices; `y` and `x` are
always indices. An empty ROI selects everything and is falsy.

Editing (each returns the ROI, so calls chain):

- `set(dim, rng, index=None)`: set the range of `dim`, or remove it with `rng=None`;
  `index=True` makes it a grid-index range, `index=False` a value range.
- `update(**kwargs)`: `set` for several dims at once.
- `clear(*dims)`: remove the given dims, or every range when called without arguments.
- `copy()`: an independent copy.
- `get(dim)`: the `(low, high)` of `dim`, or `None`.
- `dim in roi`, `bool(roi)` and `roi == other` work as expected.

Resolving against a scan:

- `pixel_slices(frame_shape)`: `(slice_y, slice_x)` clipped to the frame; an empty window
  raises.
- `pixel_window(frame_shape)`: the same as `(y0, y1, x0, x1)`.
- `motor_region(structure)`: one slice per motor dim of a `Structure`, selecting the grid
  points inside the ranges; raises when a range holds no grid point.
- `describe(structure=None)`: text such as `ROI: mu -0.1 to 0.1 (values) -> grid (6, 2), 12 frames`.

Converting:

- `from_downsampled(factors)`: scale index ranges given in downsampled pixels back to full
  resolution; `factors` is `{dim: factor}` or `(ds_y, ds_x)`.
- `to_dict()` and `ROI.from_dict(data)`: a plain dict for YAML and JSON, and back.

```python
# a pixel window plus a motor range, and the ROI changed, range by range
roi = bexa.ROI(y=(700, 1100), x=(800, 1200), chi=(0, 8))
roi.set("mu", (-0.2, 0.2)).clear("chi")         # ROI(y=(700, 1100), x=(800, 1200), mu=(-0.2, 0.2))
roi.pixel_slices((2160, 2560))                    # (slice(700, 1100), slice(800, 1200))
bexa.ROI(mu=(-0.1, 0.1)).describe(scan.structure)  # 'ROI: mu -0.1 to 0.1 (values) -> grid (6, 2), 12 frames'
bexa.ROI(mu=(2, 10), index_dims=["mu"]).motor_region(scan.structure)   # (slice(None), slice(2, 10))
bexa.ROI(y=(10, 20), x=(5, 15)).from_downsampled((4, 4))               # ROI(y=(40, 80), x=(20, 60))
bexa.ROI.from_dict(roi.to_dict()) == roi          # True
```

#### `ROI.parse`
`bexa.core.roi.ROI.parse(text)`

Read the command-line form `"y=700:1100,x=800:1200,chi=-0.2:0.2"`: pixel bounds are integers,
motor bounds numbers, a missing bound (`x=800:`) means "to the edge", and empty text or `None`
gives an empty ROI. The `--roi` option of the `bexa scan` commands uses it.

```python
# the same ROI from the text the command line takes
roi = bexa.ROI.parse("y=700:1100,x=800:,chi=-0.2:0.2")
roi                                             # ROI(y=(700, 1100), x=(800, None), chi=(-0.2, 0.2))
```

#### `ROI.from_indices`
`bexa.core.roi.ROI.from_indices(image, y=None, x=None, **motors)`

An ROI drawn in the pixel indices of `image` (a preview or a windowed result), expressed in
full-resolution pixels: the first `y`/`x` coordinate of the image and its step map the indices
back. Plain arrays are taken as full resolution; motor ranges pass through unchanged.
`image.bexa.roi(y=..., x=...)` calls it.

```python
# prev: a binned preview; a window given in its own pixels, kept as full-resolution pixels
peak = bexa.ROI.from_indices(prev, y=(5, 10), x=(2, 8))   # ROI(y=(20, 40), x=(8, 32)) at 4 x 4
bexa.ROI.from_indices(prev, mu=(2, 6))          # ROI(mu=(2, 6))
```

#### `ROI.to_indices`
`bexa.core.roi.ROI.to_indices(image)`

The inverse of `from_indices`: `(slice_y, slice_x)` of this ROI in the pixel indices of
`image`, for cutting a preview or overlaying the ROI on it.

```python
# peak: an ROI in full-resolution pixels; the slices that cut it out of the binned preview
ys, xs = peak.to_indices(prev)
cut = prev[..., ys, xs]
cut.shape                                       # (6, 16, 5, 6)
```

`PIXEL_DIMS = ("y", "x")` names the pixel dims everywhere in bexa.

### bexa.core.reductions (bexa.acc): accumulators and the streaming engine

A reduction is an accumulator: an object that sees the frames of a scan once, in batches, and
keeps only running totals. `reduce` streams the frames through every accumulator in one pass:
batches are sized from the memory budget, the next batch is read in a background thread, the
work runs on the GPU when cupy finds one, an out-of-memory error halves the batch, and two GPU
failures fall back to the CPU. No array the size of the full stack is ever allocated:
motor-weighted sums use `tensordot` over the frames of each batch.

`bexa.acc` is this module, so `bexa.acc.Sum()` and `bexa.core.reductions.Sum()` are the same
class; accumulators may be passed as instances or as classes (`bexa.acc.Sum`). Every result is
an `xarray.Dataset` with coordinates; the dataset and each variable carry provenance attrs
(`source_files`, `parameters`, `bexa_version`, `git_hash`, `created`, `python`, `device`,
`elapsed_s`, `frames`, `frames_per_s`, `peak_rss_mb`), and each variable also `accumulator`
and `params`. `bexa.core.provenance.parameters_of(res.attrs)` decodes the parameters.

Names of the results:

| Accumulator or verb | Variables |
|---|---|
| `Sum`, `Mean`, `Max`, `Min`, `scan.sum`, `scan.max` | `sum`, `mean`, `max`, `min` (y, x) |
| `Preview`, `scan.preview`, `scan.read` | `preview` (motors..., y, x); (motors..., yb, xb) for a `Preview(downsample=...)` binned on its own |
| `Projections` | `xy` (y, x), `grid` (motors), `<motor>_y`, `<motor>_x`, `<motor_i>_<motor_j>` |
| `RoiIntegral({"peak": roi})` | `roi_peak` on the motor grid; `scan.rocking_curve` gives `rocking_curve` |
| `MotorCOM`, `scan.com` | `total`, `com_<axis>`, `width_<axis>`, `skew_<axis>`, `kurtosis_<axis>` (y, x) |
| `EnergyCOM`, `scan.energy_com` | `total`, `com_energy`, `width_energy` (y, x) |
| `ArgmaxMotor` | `argmax_<axis>`, `peak` (y, x) |
| `FrameStats`, `scan.stats` | `frame_sum`, `frame_mean`, `frame_max`, `frame_com_y`, `frame_com_x` (motors) |
| `Histogram` | `histogram` (value) |
| `OnOffSplit(Sum)` | `on_sum`, `off_sum` |
| `PerRoi({"grainA": roi}, [Sum, MotorCOM])` | `grainA_sum`, `grainA_com_<axis>`, ...: every inner result per region; a dim the region restricts becomes `grainA_<dim>` |
| `bexa.stack(..., dim="auto")` | every variable of the accumulators with the dims that vary between the scans in front (`samz`, `energy`, ...), NaN where no scan sits; per-frame statistics (`block_total`, `block_p1`, `block_p99`) in the attrs of every image variable |

```python
import numpy as np
import bexa

scan = bexa.demo("mosa")                      # dims ('chi', 'mu', 'y', 'x'), shape (6, 16, 64, 80)
window = bexa.ROI(y=(8, 56), x=(8, 72))         # a pixel window inside the frame
```

#### `reduce`
`bexa.core.reductions.reduce(target, accumulators, roi=None, downsample=None, device="auto", batch_frames=None, dtype=np.float32, method="mean", cache=None, prefetch=True, show_progress=None)`

Run accumulators over a scan in one streaming pass (top-level name `bexa.reduce`; `scan.reduce`
calls it) and return one `xarray.Dataset` with every result.

- `target`: a `Scan`, or a `(source, structure)` pair for a bare engine.
- `accumulators`: accumulator instances or classes.
- `roi`, `downsample`: the region and the downsampling, as for `Scan.preview`; a `Scan`
  supplies its own `scan.roi` when `roi` is `None`.
- `device`: `"auto"` (the GPU when cupy finds one), `"cpu"` or `"cuda"`; `None` reads the
  `BEXA_DEVICE` variable. Results always come back as numpy arrays.
- `batch_frames`: frames per batch; by default sized from the memory budget and rounded to
  the storage chunk.
- `dtype`: the precision frames are converted to; `method`: how pixel blocks are combined when
  downsampling (`"mean"`, `"sum"` or `"max"`).
- `cache`: `True` (the scan's cache), a `Cache`, or `None`/`False`. With `None`, a scan opened
  with `cache_reductions=True` uses its cache. Cached results are keyed on the files, the
  accumulator, the plan, `method` and `dtype`.
- `prefetch`: read the next batches in a background thread (`prefetch_depth` of them).
- `show_progress`: a progress bar (tqdm); `None` shows one for passes of 200 frames or more
  in an interactive session.

The frames are read as stored and converted to `dtype` on the GPU or over threads on the CPU
(`cast_frames`), so the reading thread only reads. On the CPU the accumulators fold each batch
in at the same time, one thread each (numpy and scipy release the GIL), so a `Sum`, a
`FrameStats` and a `MotorCOM` together cost about what the `MotorCOM` alone costs; on the GPU
one stream runs them in turn.

```python
# scan: the demo chi x mu scan; window: a pixel ROI; two accumulators in one pass
res = bexa.reduce(scan, [bexa.acc.Sum, bexa.acc.MotorCOM(axes=("mu",))], roi=window, device="cpu")
list(res)                                     # ['sum', 'total', 'com_mu', 'width_mu']
```

#### `Sum`
`bexa.core.reductions.Sum(**params)`

The summed image `(y, x)` over the frames of the plan, named `sum`, with the number of frames in
its attrs (`n_frames`).

```python
# scan: the demo chi x mu scan; the summed image
res = scan.reduce([bexa.acc.Sum()])
res["sum"].dims, float(res["sum"].max())        # (('y', 'x'), 28060.0)
```

#### `Mean`
`bexa.core.reductions.Mean(**params)`

The mean frame `(y, x)`, named `mean`.

```python
# scan: the demo chi x mu scan; the mean frame
res = scan.reduce([bexa.acc.Mean()])
res["mean"].dims, float(res["mean"].max())      # (('y', 'x'), 292.29)
```

#### `Max`
`bexa.core.reductions.Max(**params)`

The pixel-wise maximum over the frames `(y, x)`, named `max`.

```python
# scan: the demo chi x mu scan; the brightest value of every pixel
res = scan.reduce([bexa.acc.Max()])
res["max"].dims, float(res["max"].max())        # (('y', 'x'), 3786.0)
```

#### `Min`
`bexa.core.reductions.Min(**params)`

The pixel-wise minimum over the frames `(y, x)`, named `min`.

```python
# scan: the demo chi x mu scan; the darkest value of every pixel
res = scan.reduce([bexa.acc.Min()])
res["min"].dims, float(res["min"].min())        # (('y', 'x'), 10.0)
```

#### `Preview`
`bexa.core.reductions.Preview(apply_log=False, fill=float("nan"), downsample=None, method="mean")`

The (downsampled) volume itself, `(motors..., y, x)` with coordinates, named `preview`.
`apply_log` stores `log10(1 + I)` (the v9 preview did); `fill` is the value of grid points
without a frame in a partial scan. `downsample` (an int or `(by, bx)`) bins the pixels of this
preview alone, on top of the pass's `downsample`, so the other accumulators of the pass keep
the full frames: one pass gives full-resolution maps and a binned volume. Such a preview names
its pixel dims `yb`, `xb` (`bexa.core.structure.BINNED_PIXEL_DIMS`: one xarray result holds
one size per dim name), with the full-resolution index of each block's first pixel as
coordinates, and records the factors in `attrs["binning"]`; `method` combines each block by
`"mean"`, `"sum"` or `"max"`. The browsers, `bexa.viz.volume.projections_panel`,
`bexa.viz.images.tiles` and `show` take the last two dims as the pixels whatever their names.
`describe()` includes `downsample` and `method` only when binning, so an unbinned preview keeps
its cache key. The volume is checked against the memory budget before it is allocated
(`bexa.core.backend.check_fits`: a `MemoryError` names the size and suggests a larger
downsample, an ROI or `bexa.stack(..., store=True)`). `scan.preview` and `scan.read` use it.

```python
# scan: the demo chi x mu scan; the whole scan binned in the pixels, from the pass
vol = scan.reduce([bexa.acc.Preview(apply_log=True)], downsample=(1, 1, 4, 4))["preview"]
res = scan.reduce([bexa.acc.Preview(downsample=4), bexa.acc.Sum(), bexa.acc.MotorCOM(axes=("mu",))])   # one pass, two resolutions
res["preview"].dims, res["preview"].shape, res["sum"].shape    # ('chi', 'mu', 'yb', 'xb'), (6, 16, 16, 20), (64, 80)
res["preview"].coords["yb"].values[:3], res["preview"].attrs["binning"]   # [0 4 8], [4, 4]
```

#### `Projections`
`bexa.core.reductions.Projections(**params)`

The integrated intensity along every pair of axes: `xy` (the sum image), `grid` (the total
intensity at every grid point), `<motor>_y` and `<motor>_x` (intensity against one motor and
one pixel axis) and, for two or more motors, `<motor_i>_<motor_j>` (for example `chi_mu`).
Replaces `plotter.plot_projections`; `bexa.viz.volume.projections_panel` draws them.

```python
# scan: the demo chi x mu scan; its projections onto every pair of axes
proj = scan.reduce([bexa.acc.Projections()])
proj["mu_x"].dims, proj["chi_mu"].dims        # ('mu', 'x'), ('chi', 'mu')
```

#### `RoiIntegral`
`bexa.core.reductions.RoiIntegral(rois=None, method="sum")`

The integrated intensity of one or more pixel ROIs at every grid point: one variable
`roi_<name>` on the motor grid per entry of `rois` (`{name: ROI}` in full-resolution pixels,
or a single ROI, named `roi`). `method` is `"sum"`, `"mean"` or `"max"` over the ROI pixels.
`scan.rocking_curve` uses it.

```python
# scan: the demo chi x mu scan; the intensity of two pixel windows at every grid point
res = scan.reduce([bexa.acc.RoiIntegral({"peak": bexa.ROI(y=(20, 40), x=(30, 50)), "bg": bexa.ROI(y=(0, 8))})])
res["roi_peak"].dims                          # ('chi', 'mu')
```

#### `MotorCOM`
`bexa.core.reductions.MotorCOM(axes=None, sigma=3.0, weights="linear", clip=1e-10, moments=2)`

Per-pixel centre of mass and higher moments of the rocking curve along motor axes. With
weight `w_f` of frame `f` at motor value `m_f`, per pixel:
`com = sum(w m) / sum(w)` and `width = sqrt(sum(w m^2) / sum(w) - com^2)`.

- `axes`: motor dims to analyse (default all).
- `sigma`: Gaussian smoothing of every frame, in pixels, before weighting (as in the lab's
  notebooks); 0 disables it.
- `weights`: `"linear"` (the intensity) or `"log"` (`log10(1 + I)`, as v9's preview COM).
- `clip`: weights below it are raised to it, so empty pixels do not divide by zero.
- `moments`: 1 gives `com_<axis>`; 2 adds `width_<axis>`; 3 adds `skew_<axis>`; 4 adds
  `kurtosis_<axis>` (excess kurtosis). `total`, the sum of the weights, is always returned.

The moments of a batch are formed about the middle of each motor axis, in the frames' dtype
(float32 BLAS through `tensordot`, no float64 copy of the batch), and accumulated in float64
across batches; the centre of mass gets the middle added back and the width, skew and kurtosis
are unchanged by the shift, so the width of an energy series at 17 keV (a millionth of the
value squared) keeps its precision. The smoothing goes through
`bexa.core.backend.gaussian_frames`, in `BEXA_THREADS` threads on the CPU. On the lab
workstation (16 cores, RTX 4090), Sum and MotorCOM(sigma=3) over 160 frames of 1024 x 1024
take 1.6 s on the CPU (4.9 s before the threads and the dropped copy) and 0.6 s on the GPU
(1.4 s before). `scan.com` uses it. Replaces the per-pixel loops of the `betterCOM*` notebooks.

```python
# scan: the demo chi x mu scan; window: a pixel ROI; moments of every pixel's rocking curve
res = scan.reduce([bexa.acc.MotorCOM(axes=("chi", "mu"), sigma=3.0, moments=4)], roi=window)
res["com_mu"].attrs["units"]                  # 'deg'
```

#### `EnergyCOM`
`bexa.core.reductions.EnergyCOM(**kwargs)`

`MotorCOM` along the `energy` dim of an energy series, with no smoothing unless `sigma` is
given: `com_energy` (keV), `width_energy` and `total`. The keywords are those of `MotorCOM`
except `axes`. `scan.energy_com` uses it.

```python
# series: an energy series (one scan per energy); the energy centre of mass of every pixel
series = bexa.demo("energy")
res = series.reduce([bexa.acc.Sum(), bexa.acc.EnergyCOM(), bexa.acc.MotorCOM(axes=("mu",))])
sorted(res.data_vars)                           # ['com_energy', 'com_mu', 'sum', 'total', 'width_energy', 'width_mu']
```

#### `ArgmaxMotor`
`bexa.core.reductions.ArgmaxMotor(axis=None)`

For every pixel, the motor value of its brightest frame, `argmax_<axis>`, and that brightest
value, `peak`: a robust alternative to the centre of mass when the rocking curve has side
peaks. `axis` defaults to the last (fastest) motor dim.

```python
# scan: the demo chi x mu scan; the mu at which every pixel is brightest
res = scan.reduce([bexa.acc.ArgmaxMotor(axis="mu")])      # argmax_mu, peak
scan.reduce([bexa.acc.ArgmaxMotor(axis="chi")])["argmax_chi"].dims  # ('y', 'x')
```

#### `FrameStats`
`bexa.core.reductions.FrameStats(**params)`

Per-frame sum, mean, maximum and detector centre of mass on the motor grid: `frame_sum`,
`frame_mean`, `frame_max`, `frame_com_y`, `frame_com_x` (full-resolution pixels).
`scan.stats` uses it.

```python
# scan: the demo chi x mu scan; window: a pixel ROI; the total and centre of every frame
res = scan.reduce([bexa.acc.FrameStats()], roi=window)
sorted(res.data_vars), res["frame_sum"].dims    # (['frame_com_x', 'frame_com_y', 'frame_max', 'frame_mean', 'frame_sum'], ('chi', 'mu'))
```

#### `Histogram`
`bexa.core.reductions.Histogram(bins=256, range=None)`

A histogram of the pixel values over all frames, for colour limits and thresholds: `histogram`
with a `value` coordinate at the bin centres. `range=(low, high)` fixes the bins; otherwise
they span the values of the first batch.

```python
# scan: the demo chi x mu scan; the histogram of every pixel value of the scan
hist = scan.reduce([bexa.acc.Histogram(bins=64, range=(0, 5000))])["histogram"]
scan.reduce([bexa.acc.Histogram()])["histogram"].size  # 256
```

#### `OnOffSplit`
`bexa.core.reductions.OnOffSplit(factory, flag="laser")`

Run an accumulator separately on laser-on and laser-off frames. `factory` makes a fresh inner
accumulator (a class such as `bexa.acc.Sum`, or a function returning an instance); `flag` names
the per-frame channel in `scan.structure.per_frame` that holds 1 (on) or 0 (off), for example
`laser_flag` for PAL-XFEL point files. The results are prefixed `on_` and `off_`.

```python
# pal: a PAL-XFEL run with a laser flag per shot; the laser-on and laser-off sums
from bexa.testing.synthetic import make_pal_run

pal = bexa.open(make_pal_run("pal_data", run=42).measurement_dir)
res = pal.reduce([bexa.acc.OnOffSplit(bexa.acc.Sum, flag="laser_flag")])   # on_sum, off_sum
```

#### `PerRoi`
`bexa.core.reductions.PerRoi(rois, accumulators)`

Run accumulators on several regions of one scan in the same pass, one set of results per
region. `rois` maps a name to an `ROI` that may restrict any motor dims and the pixel window;
`accumulators` (classes or instances, copied as built for every region) run on each. Every
result is named `<roi>_<result>` (`grainA_sum`, `grainA_com_mu`). A dim a region restricts is
renamed `<roi>_<dim>` (`grainA_chi`), so regions of different sizes live in one Dataset, while a
dim it leaves whole keeps its name, so the maps of a motor ROI stay `(y, x)`. Pixel ranges snap
to the pixel grid of the pass's own window and downsampling, motor ranges to its grid; a region
that selects no frame or no pixel raises. One pass over the frames serves every region, which is
how `examples/13_esrf_dfxm_general.ipynb` gets the maps and curves of its `ROIS`. `describe()`
records the regions and the inner accumulators, so cached results tell them apart.

```python
# scan: the demo chi x mu scan; the same accumulators inside two regions, one pass
rois = {"edge": bexa.ROI(x=(0, 32)), "tight": bexa.ROI(chi=(-0.2, 0.2), mu=(-0.5, 0.5))}
res = scan.reduce([bexa.acc.PerRoi(rois, [bexa.acc.Sum(), bexa.acc.FrameStats(), bexa.acc.MotorCOM(axes=("mu",), sigma=0)])])
res["edge_sum"].dims, res["tight_frame_sum"].dims, res["tight_com_mu"].dims   # (('y', 'edge_x'), ('tight_chi', 'tight_mu'), ('y', 'x'))
res["tight_frame_sum"].coords["tight_chi"].values           # the chi points inside the region
```

#### `Accumulator`
`bexa.core.reductions.Accumulator(**params)`

The base class of every accumulator; subclass it for a reduction of your own. The engine calls
`start(plan, xp, dtype)` once (it stores the `Plan`, the array module `numpy` or `cupy`, and
the dtype, then calls the `_allocate` hook), `update(frames, frame_ids)` for every batch of
shape `(n, ny, nx)` (already windowed and downsampled, on the device), and `result()` at the
end, which returns a `DataArray` or a dict of them on the host. `describe()` identifies the
computation for cache keys and provenance (the class name and `params`).

A subclass sets `name` (the key of its result) and `live_copies` (batch-sized temporaries it
allocates, so the engine can size batches), and builds results with the helpers `_image`
(`(y, x)` maps), `_grid` (values on the motor grid), `_volume` and `_zeros`. Register it with
`bexa.core.registry.register_accumulator` so that pipelines can find it by name.

An accumulator pickles as built (`__getstate__`, `__setstate__`): the attributes `__init__`
set, never the totals, the plan or the array module of a pass, whether or not it has run. The
worker processes of `bexa.stack(..., workers=)` receive accumulators this way, so a subclass
keeps everything a fresh pass needs in attributes set by `__init__` (what `_allocate` makes is
left out, and `xp` is numpy again after unpickling).

```python
# an accumulator of your own, registered under a name and used like the built-in ones
from bexa.core.registry import register_accumulator

@register_accumulator("squared_sum")
class SquaredSum(bexa.acc.Accumulator):
    name = "squared_sum"

    def _allocate(self):
        self.total = self._zeros(self.plan.window.shape)

    def update(self, frames, frame_ids):
        self.total += (frames.astype(self.total.dtype) ** 2).sum(axis=0)
        self.n_frames_seen += len(frame_ids)

    def result(self):
        return self._image(self.total, self.name, n_frames=self.n_frames_seen)

res = scan.reduce([bexa.acc.Sum(), SquaredSum()])            # sum, squared_sum
```

#### `make_plan`
`bexa.core.reductions.make_plan(structure, roi=None, downsample=None)`

Resolve an ROI and a downsample request against a `Structure`: which frames to read, the pixel
window, and the downsampled grid. Returns a `Plan`. Motor downsampling keeps every n-th grid
point; the request takes every form `parse_downsample` does, a mapping by name included.

```python
# scan: the demo chi x mu scan; what one pass will read: the frames, the window, the binning
from bexa.core.reductions import make_plan

plan = make_plan(scan.structure, bexa.ROI(y=(0, 32), mu=(-0.5, 0.5)), (1, 1, 2, 2))
plan.dims, plan.shape                         # ('chi', 'mu', 'y', 'x'), (6, 8, 16, 40)
```

#### `Plan`
`bexa.core.reductions.Plan(structure, window, motor_factors, frame_ids, grid_shape, coords={}, grid=None)`

What `make_plan` returns: `structure` (after the motor ROI), `window` (a `Window`),
`motor_factors`, `frame_ids` (sorted ids of the frames to read), `grid_shape` (the motor grid
after downsampling), `coords` (the downsampled motor coordinates) and `grid`, the downsampled
grid itself as a `Structure`, for plans within this one (`PerRoi`).

- `motor_dims`, `dims`, `shape`: properties; `dims` and `shape` include `y` and `x`.
- `grid_positions(frame_ids)`: the position of each frame in the downsampled grid.
- `motor_values(name, frame_ids)`: the value of motor `name` for each frame.
- `describe()`: window, downsampling, grid, frame count, dims and `motors` (the first and last
  value and the count of every motor coordinate, so two motor windows of the same size get
  different cache keys) as a dict (stored in the provenance `parameters`).
- `all_coords()`: motor and pixel coordinates together.

```python
# plan: the pass over a motor and pixel window, binned 2 x 2; its frames and grid
plan.grid_positions(plan.frame_ids[:3]), plan.motor_values("mu", plan.frame_ids[:3])  # ((array(list of 3), array(list of 3)), array(list of 3))
plan.describe()                                 # {'window': [0, 32, 0, 80], 'downsample': [1, 1, 2, 2], 'grid': [6, 8], 'n_frames': 48, ...
```

#### `Window`
`bexa.core.reductions.Window(y, x, ds_y=1, ds_x=1)`

The pixel window of a plan (`y`, `x` slices in full-resolution pixels) and its spatial
downsampling. `full_shape` and `shape` (properties) are its size before and after downsampling;
`coords()` gives the full-resolution index of the first pixel of every downsampled pixel, which
is what the `y` and `x` coordinates of every result hold.

```python
# plan: the pass of make_plan; the pixel window it reads, before and after binning
plan.window.full_shape, plan.window.shape, plan.window.coords()["x"][:3]   # (32, 80), (16, 40), [0 2 4]
```

#### `parse_downsample`
`bexa.core.reductions.parse_downsample(downsample, structure)`

Split a downsample request into motor factors and `(ds_y, ds_x)`: an int or a 2-tuple applies
to the pixel axes only, one factor per dim `(motor0, ..., ds_y, ds_x)` sets everything (the
forms the v9 module accepted), and a mapping by name, `{"mu": 2, "y": 4, "x": 4}`, sets what it
names (motors of the structure, `y`, `x`; names the structure lacks are ignored, so one setting
serves scans of different motors; a factor below 1 raises). A motor factor keeps every k-th grid
point, so k times fewer frames are read; a pixel factor bins blocks of pixels. Anything else
raises with the expected forms.

```python
# scan: the demo chi x mu scan; a downsample given as one factor, a tuple or by name
from bexa.core.reductions import parse_downsample

parse_downsample(4, scan.structure)             # ((1, 1), (4, 4))
parse_downsample((1, 2, 4, 4), scan.structure)  # ((1, 2), (4, 4))
parse_downsample({"mu": 2, "y": 4, "x": 4}, scan.structure)   # ((1, 2), (4, 4)): by name
parse_downsample({"phi": 2}, scan.structure)    # ((1, 1), (1, 1)): phi is not scanned here
```

#### `block_reduce`
`bexa.core.reductions.block_reduce(frames, dy, dx, method="mean")`

Downsample the last two axes of a numpy or cupy array by integer factors, combining each
`dy x dx` block by `"mean"`, `"sum"` or `"max"`; edges that do not fill a block are dropped.
`bexa.analysis.preprocess.bin_spatial` wraps it.

```python
# a (10, 64, 80) array binned 4 x 4 in the pixels
from bexa.core.reductions import block_reduce

small = block_reduce(np.ones((10, 64, 80), np.float32), 4, 4)   # (10, 16, 20)
```

#### `iter_batches`
`bexa.core.reductions.iter_batches(source, plan, batch_frames=None, dtype=np.float32, device="cpu", method="mean", prefetch=True, live_copies=4)`

Stream the frames of a plan as `FrameBatch` objects: read through the source's
`read_frames`, cut to the window, block-reduced, and moved to `device` (`"cpu"` or `"cuda"`).
The batch size comes from the memory budget, keeping `live_copies` batch-sized arrays inside
it, unless `batch_frames` is given; with `prefetch` the next batches are read in a background
thread, `prefetch_depth` of them ahead. The frames are read as stored (uint16, half the bytes
of float32) and converted to `dtype` on the GPU, or over threads on the CPU (`cast_frames`), so
the reading thread only reads: on a share delivering 1 GB/s the conversion in the reader had
halved the reading speed. `Scan.batches` calls it.

```python
# scan: the demo chi x mu scan; plan: its pass; the frames in batches of 16
from bexa.core.reductions import iter_batches

for batch in iter_batches(scan.source, plan, batch_frames=16):
    print(batch.frames.shape, batch.frame_ids[:3])   # (16, 16, 40) [4 5 6]
    break
```

#### `prefetch_depth`
`bexa.core.reductions.prefetch_depth(batch_bytes)`

How many batches `iter_batches` reads ahead of the computation: between 2 and `MAX_PREFETCH`
(8), as many as fit in `PREFETCH_SHARE` (a tenth) of the memory budget. A deeper queue rides
out the pauses of a network file system while the frames keep being processed, at the cost of
that many batches of host memory.

```python
# how many batches are read ahead of the accumulators, for a batch of a given size
from bexa.core.reductions import prefetch_depth

prefetch_depth(10**6), prefetch_depth(10**15)    # (8, 2): tiny batches go deep, huge ones stay at two
```

Constants: `PIXEL_DIMS = ("y", "x")` and `BINNED_PIXEL_DIMS = ("yb", "xb")`, from
`bexa.core.structure` (the pixel dims of a `Preview` binned on its own);
`AUTO_PROGRESS_FRAMES = 200`, the pass length from which interactive sessions show a progress
bar; `PREFETCH_SHARE = 0.1` and `MAX_PREFETCH = 8`, the bounds of `prefetch_depth`.

### bexa.core.accessor: .bexa on xarray objects

Every `DataArray` and `Dataset` gets a `.bexa` attribute with plotting, browsing, ROI and
saving one call away. The accessor is registered when `bexa.core.scan`,
`bexa.core.reductions` or `bexa.viz` is imported, which `bexa.open`, `bexa.demo`,
`bexa.reduce` and `bexa.plot` do; after only `bexa.load` in a fresh session, run
`import bexa.viz` first.

```python
import bexa

scan = bexa.demo("mosa")                        # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
res = scan.com(axes=("chi", "mu"))              # com_chi, com_mu, width_chi, width_mu, total per pixel
prev = scan.preview(downsample=(1, 1, 4, 4))    # DataArray (chi, mu, y, x) = (6, 16, 16, 20)
```

#### `ArrayAccessor`
`bexa.core.accessor.ArrayAccessor(obj)`

`DataArray.bexa`. xarray creates it on first access; its methods follow.

```python
# res: the COM maps of the demo scan; the .bexa accessor of one map
res["com_mu"].bexa             # the accessor of one map
```

#### `ArrayAccessor.plot`
`bexa.core.accessor.ArrayAccessor.plot(kind=None, **kwargs)`

`da.bexa.plot()`: the figure that fits the array, chosen from its dims and attrs by
`bexa.viz.auto.plot` (a COM map with diverging colours around the mean, an image with
percentile limits, a curve, projections of a volume); `kind` forces one, and the keywords go to
the plotting function. Returns what that function returns.

```python
# res: the COM maps of the demo scan; the figure that fits the map
res["com_mu"].bexa.plot()                       # (Figure, Axes, AxesImage)
scan.sum().bexa.plot(log=True, title="summed")  # (Figure, Axes, AxesImage)
```

#### `ArrayAccessor.browse`
`bexa.core.accessor.ArrayAccessor.browse(**kwargs)`

`da.bexa.browse()`: one image at a time with a slider per motor dim
(`bexa.viz.interactive.browse`).

```python
# prev: a preview (chi, mu, y, x); sliders over chi and mu
prev.bexa.browse()                              # Browser
```

#### `ArrayAccessor.browse_volume`
`bexa.core.accessor.ArrayAccessor.browse_volume(**kwargs)`

`da.bexa.browse_volume()`: the last three dims shown as a `(z, y, x)` volume (projections or an
isosurface), with a slider for every dim in front of them (`bexa.viz.interactive.browse_volume`);
a stack built with `store=True` is read one block per move.

```python
# stack: the summed images of a z-stack (samz, y, x); the volume with its projections
stack = bexa.stack(bexa.demo("zstack").select(type="fscan2d"), [bexa.acc.Sum()], dim="samz")
stack["sum"].bexa.browse_volume()               # VolumeBrowser
```

#### `ArrayAccessor.roi`
`bexa.core.accessor.ArrayAccessor.roi(y=None, x=None)`

`da.bexa.roi(y=..., x=...)`: an ROI given in this array's pixel indices (for example those of
a preview), returned in full-resolution pixels (`ROI.from_indices`).

```python
# prev: a binned preview; an ROI in its own pixels, as ROI.from_indices
peak = prev.bexa.roi(y=(5, 10), x=(2, 8))        # ROI(y=(20, 40), x=(8, 32))
prev.bexa.roi(x=(2, 8))                         # ROI(x=(8, 32))
```

#### `ArrayAccessor.render`
`bexa.core.accessor.ArrayAccessor.render(**kwargs)`

`da.bexa.render()`: a plotly volume rendering of a `(z, y, x)` stack whose colour and opacity
follow the intensity (`bexa.viz.volume.render`; needs plotly). Show it in a notebook with
`bexa.notebook.show_plotly`.

```python
# stack: the summed images of a z-stack; a plotly rendering in micrometres
fig = stack["sum"].bexa.render(spacing=(10, 0.235, 0.235), units="um")
fig.layout.scene.xaxis.title.text               # 'x (um)'
```

#### `ArrayAccessor.browse_render`
`bexa.core.accessor.ArrayAccessor.browse_render(**kwargs)`

`da.bexa.browse_render()`: the last three dims are rendered as the `(z, y, x)` volume, and every
dim in front of them gets a slider; the block is rendered again when a slider is released
(`bexa.viz.interactive.browse_render`).

```python
# layers: the previews of a z-stack; sliders over chi and mu, one rendering per move
layers = bexa.demo("zstack").series(type="fscan2d", dim="samz").preview(downsample=2)
layers.transpose("chi", "mu", "samz", "y", "x").bexa.browse_render(mode="translucent")  # RenderBrowser
```

#### `ArrayAccessor.save`
`bexa.core.accessor.ArrayAccessor.save(path, **kwargs)`

`da.bexa.save(path)`: write the array with `bexa.save` (HDF5 or zarr, coordinates and
provenance kept) and return the path.

```python
# res: the COM maps of the demo scan; one map to a bexa HDF5 file
res["com_mu"].bexa.save("com_mu.h5")            # Path('.../com_mu.h5')
```

#### `DatasetAccessor`
`bexa.core.accessor.DatasetAccessor(obj)`

`Dataset.bexa`, with two methods:

- `plot(kind=None, **kwargs)`: a panel of every map of a reduce result, or the laser on/off
  figure for a pump-probe cube (`bexa.viz.auto.plot`).
- `save(path, **kwargs)`: `bexa.save` of the whole dataset.

```python
# res: the COM maps of the demo scan; the panel of every map, and the file of all of them
res.bexa.plot()                  # one panel per map
res.bexa.save("scan_maps.h5")                   # Path('.../scan_maps.h5')
```

### bexa.io.formats: format specs, sniffing and drafting

A format spec is a YAML document in `configs/formats/` describing one file layout: path templates,
motor and key aliases, the energy source, the detectors, the engine that walks the files, and a
`fingerprint` by which `bexa.open(path)` recognises the layout (sniffing). Specs inherit with
`extends` (`docs/formats.md`). `bexa.io` re-exports `FormatSpec`, `list_specs`, `load_spec` and
`match_spec`. The shipped specs:

| Spec | Engine | Layout |
|---|---|---|
| `esrf_id03_bliss_F2026` | `hdf5_stack` | ESRF ID03 BLISS, autumn 2026 (ma7352): the 2026 layout (`extends`) plus the frames of a scan reachable through the master as one virtual dataset, `N.1/instrument/pco_ff/image`; `pco_ff` read out as 2048 x 2048 |
| `esrf_id03_bliss_2026` | `hdf5_stack` | ESRF ID03 BLISS, January 2026 (hc6293): master HDF5 per dataset, `scanNNNN/pco_ff_*.h5`, `fscan_parameters` |
| `esrf_id03_bliss_2025` | `hdf5_stack` | ESRF ID03 BLISS, July 2025 (hc6043), the same layout |
| `esrf_id03_bliss_2024` | `hdf5_stack` | ESRF ID03 BLISS, 2024 (WTe2): no `fscan_parameters`, grid detected from the motor readbacks |
| `esrf_bliss_base` | `hdf5_stack` | abstract base of the ID03 specs (never sniffed) |
| `pal_xfel_points_2025_09` | `hdf5_points` | PAL-XFEL, September 2025 (ue_250913_FXS): `type=measurement/.../pNNNN.h5` shot tables and `type=raw/...` Jungfrau frames |
| `pal_xfel_spec_2025_04` | `spec_h5` | PAL-XFEL, April 2025 (ue_250419_DFXM): one HDF5 per scan with a spec-like `scanHistory` |
| `lcls_xcs_cube` | `array_cube` | LCLS XCS cube triple `RunNNNN_onStk.npy`, `_offStk.npy`, `_stats.csv` |
| `lcls_smalldata` | `smalldata` | LCLS smalldata_tools run file: per-event scalars and ROI images, `Sums/` |
| `bexa_cube_legacy` | `array_cube` | legacy PAL-XFEL `runN.h5` cubes (`Aaron_allscan_cube_parallel.py`, `Scan_Combiner.py`) |
| `bexa_cube` | `array_cube` | files written by `bexa.save` |
| `euxfel_extra_data` | `extra_data` | EuXFEL MID runs through `extra_data` (needs it; not sniffed, pass `format=`) |
| `generic_stack` | `generic` | `.npy`, `.npz`, tiff stacks, `file.h5::/path` (not sniffed) |

```python
import tempfile
from pathlib import Path
from bexa.io import formats
from bexa.testing import make_esrf_scan, make_pal_run

work = Path(tempfile.mkdtemp())                  # synthetic files go here
esrf = make_esrf_scan(work / "esrf")             # an ID03 chi x mu scan, 2026 layout
pal = make_pal_run(work / "pal", run=1)          # a PAL-XFEL run, September 2025 layout
spec = formats.load_spec("esrf_id03_bliss_2026")  # the ESRF ID03 spec of 2026, by name
```

#### `load_spec`
`bexa.io.formats.load_spec(name_or_path, overrides=None, _chain=())`

Loads a spec by name (a file stem from `list_specs`) or from a `.yaml`/`.yml` path and returns a
`FormatSpec`. The `extends` parent is deep-merged under the child (`name`, `extends` and `abstract`
are not inherited), then `overrides` on top (see `deep_merge`); `_chain` is internal. An unknown
name raises `FileNotFoundError` listing the available ones. `bexa.open(..., format=...,
overrides=...)` and a profile's `format_overrides` go through it.

```python
# spec: the ESRF 2026 spec loaded by name; its engine, its layout, a copy with an override
spec.engine, spec.layout["frames"]               # ('hdf5_stack', 'entry_0000/ESRF-ID03/{detector}/data')
renamed = formats.load_spec("esrf_id03_bliss_2026", overrides={"motors": {"aliases": {"mu": ["mu_new1"]}}})
renamed.motors["aliases"]                        # {'z1': ['samz'], 'mu': ['mu_new1']}
```

#### `list_specs`
`bexa.io.formats.list_specs()`

Dict of spec name (file stem) to YAML path over the folders of `format_dirs`; a name in an earlier
folder hides the same name later, so a spec in `BEXA_FORMAT_DIR` replaces a shipped one. `bexa
profile list` prints it.

```python
# every format spec bexa ships
sorted(formats.list_specs())[:3]                 # ['bexa_cube', 'bexa_cube_legacy', 'esrf_bliss_base']
```

#### `all_specs`
`bexa.io.formats.all_specs()`

Every spec of `list_specs`, loaded, as a list of `FormatSpec`; a spec that fails to load is logged
as a warning and left out, so one broken file never stops sniffing. `match_spec` and
`bexa.io.survey.survey` use it when no specs are given.

```python
# every loadable spec, by name and engine
{s.name: s.engine for s in formats.all_specs()}["aps_7id_spec_2025"]  # 'spec_tiff'
```

#### `match_spec`
`bexa.io.formats.match_spec(path, specs=None, quick=False, listing=None)`

Scores specs (default: all of `list_specs`) against a file or folder and returns `(score,
FormatSpec)` pairs, best first. The score is the fraction of fingerprint checks passed: master
files and their groups (`master_glob`, `master_paths`), `scanNNNN` folders (`scan_dir_glob`),
folder and file names (`dir_globs`, `file_globs`), a text file and its opening lines (`text_glob`,
`text_tags`), HDF5 paths and attributes (`h5_paths`, `h5_attrs`). Ties go to the longer
fingerprint, then the later name (the newest dated spec). Abstract specs, specs without a
fingerprint and specs that fail to load (warning) are skipped. `quick=True` opens no HDF5 file:
only the file-system part of each fingerprint counts, and a spec with nothing else to check is
left out. `listing`, a `FolderListing`, keeps the folder listings between calls, as a survey of a
tree does.

```python
# esrf: a synthetic ESRF dataset; every spec scored against it
[(score, s.name) for score, s in formats.match_spec(esrf.dataset_dir)[:3]]  # [(1.0, 'esrf_id03_bliss_2026'), (1.0, 'esrf_id03_bliss_2025'), (1.0, 'esrf_id03_bliss_2...
formats.match_spec(esrf.dataset_dir, [spec])[0][0]  # 1.0
formats.match_spec(esrf.dataset_dir, quick=True)[0][0]   # 1.0: the master and the scan folders are there; no HDF5 file opened
```

#### `best_spec`
`bexa.io.formats.best_spec(path)`

The first spec of `match_spec(path)` if its score is 1, else `None`. `bexa.open(path)` uses it and,
without a full match, raises `ValueError` naming the three closest specs.

```python
# esrf, pal: synthetic ESRF and PAL-XFEL data; the spec whose fingerprint matches
formats.best_spec(esrf.scan_folder).name         # 'esrf_id03_bliss_2026' (through its dataset folder)
formats.best_spec(pal.measurement_dir).name      # 'pal_xfel_points_2025_09'
```

#### `FolderListing`
`bexa.io.formats.FolderListing()`

The entries of folders, each folder listed once and kept. `entries(folder)` gives `{name: (is a
folder, is a link)}`, empty for a folder that cannot be read; `glob(folder, pattern,
folders=None)` the sorted paths whose names match `pattern`, `folders=True` or `False` keeping
one kind. Matching a spec against a folder looks at its names, and at its parents', several
times, and on a network share every listing costs tens of milliseconds: `match_spec` makes one
per call, and a survey of a tree shares one for every folder it visits.

```python
# esrf: a synthetic ESRF dataset; its scan folders from one listing, reused by match_spec
listing = formats.FolderListing()
[p.name for p in listing.glob(esrf.dataset_dir, "scan*", folders=True)]  # ['scan0001']
formats.match_spec(esrf.dataset_dir, listing=listing)[0][0]  # 1.0
```

#### `FormatSpec`
`bexa.io.formats.FormatSpec(*, name, engine, extends=None, description="", abstract=False, fingerprint={}, layout={}, motors={}, keys={}, energy={}, detectors={}, convention=None, axis_priority=[], options={}, source_path=None, **extra_data)`

A parsed spec: a pydantic model, built by `load_spec`, with one attribute per YAML entry
(`abstract` marks a base for `extends` only; `motors` holds the `per_frame`, `scalar` and
`structure` templates, `known`, `aliases`, `units`, `scalars_from`, `{name: path template}` for
values recorded elsewhere in the master that count as positioners of the scan (the mean of the
channel: the furnace temperature `nanodac1_temp` that BLISS samples in the `<scan>.2`
monitoring entry, in the autumn 2026 spec), `min_step`, `{motor: step}` below which a change
between scans is drift rather than a step (0.5 um for the ID03 stage motors), and `coupled`,
`{leader: [followers]}` for a
motor that drives others, at ID03 `ux`, `uy` and `uz` each moving `samx`, `samy`, `samz` and
`shexatx`, `shexaty`, `shexatz`, so that `varying` and the compact table count a stepping
leader once; `keys` maps a logical key to its alias list; `source_path` is the YAML file).
Unknown entries are kept. Methods:

- `aliases(key)`: every name of the logical key, preferred first; `[key]` for a key the spec does not list.
- `resolve_key(key, available)`: the first alias of `key` present in `available` (columns, HDF5 paths), or `None`.
- `path(item, **fields)`: the `layout` template `item` filled with `fields` (see `safe_format`); `KeyError` if absent.
- `motor_path(kind, **fields)`: the same for the motor templates `per_frame`, `scalar` and `structure`.
- `known_motors()`: the `motors.known` list.
- `min_steps()`: `{motor: step}` from `motors.min_step`: a smaller change is drift, not a move.
- `motor_aliases(name)`: the names to try in the file for the logical motor `name`, `[name, *aliases]`.
- `detector_info(detector)`: a copy of the `detectors` entry of `detector` (`{}` if unknown).

```python
# pal_spec: the PAL-XFEL points spec; what a FormatSpec knows about its layout
pal_spec = formats.load_spec("pal_xfel_points_2025_09")
pal_spec.aliases("laser_flag")                   # ['event_info.THIRTY_HERTZ', 'event_info.FIFTEEN_HERTZ']
pal_spec.resolve_key("laser_flag", ["delay_input", "event_info.FIFTEEN_HERTZ"])   # 'event_info.FIFTEEN_HERTZ'
spec.path("scan_folder", dataset="D", scan=7), spec.motor_path("per_frame", scan=7, motor="mu")  # ('D/scan0007', '7.1/measurement/mu')
spec.known_motors()[:3], spec.motor_aliases("z1"), spec.detector_info("pco_ff")["pixel_size_um"]  # (['mu', 'chi', 'phi'], ['z1', 'samz'], 6.5)
```

#### `draft_spec`
`bexa.io.formats.draft_spec(path, name=None)`

A starting spec (a dict) for an unknown layout, from up to 20 HDF5 files at `path` (a file, or a
folder and its sub-folders): 3-D datasets of frames at least 16 x 16 pixels are frame candidates,
1-D datasets as long as a frame count motor candidates, and the top-level keys pick the engine.
`TODO` marks what a person must fill in. `FileNotFoundError` without HDF5 files.

```python
# esrf: a synthetic ESRF dataset; a spec drafted from what its files hold
draft = formats.draft_spec(esrf.dataset_dir, name="id03_new")
draft["engine"], draft["layout"]["frames"]       # ('hdf5_stack', 'entry_0000/ESRF-ID03/pco_ff/data')
```

#### `write_draft`
`bexa.io.formats.write_draft(path, out, name=None)`

Writes `draft_spec(path, name)` as YAML to `out` (folders created, a file replaced) and returns the
path; `bexa profile sniff PATH -o OUT` runs it. Fill in the `TODO` entries, then keep the file in
`configs/formats/` or a `BEXA_FORMAT_DIR` folder so that `load_spec` finds it.

```python
# esrf: a synthetic ESRF dataset; the drafted spec written as YAML and loaded back
out = formats.write_draft(esrf.dataset_dir, work / "id03_new.yaml", name="id03_new")
formats.load_spec(out).engine                    # 'hdf5_stack'
```

#### `format_dirs`
`bexa.io.formats.format_dirs()`

The existing folders searched for specs, in order: those of `BEXA_FORMAT_DIR` (joined with
`os.pathsep`), `configs/formats` under the current directory, the repository's `configs/formats`.

```python
# where format specs are looked for, in order
formats.format_dirs()                            # [the repository's configs/formats] by default
```

#### `deep_merge`
`bexa.io.formats.deep_merge(base, override)`

The merge behind `extends` and `overrides`: dicts merge key by key, other values of `override`
(lists included) replace those of `base`. Returns a new dict.

```python
# two spec dicts merged the way extends does: lists joined, dicts merged
formats.deep_merge({"known": ["mu"], "units": {"mu": "deg"}}, {"known": ["chi"], "units": {"chi": "deg"}})  # {'known': ['chi'], 'units': {'mu': 'deg', 'chi': 'deg'}}
# {'known': ['chi'], 'units': {'mu': 'deg', 'chi': 'deg'}}
```

#### `safe_format`
`bexa.io.formats.safe_format(template, **fields)`

`str.format` that leaves placeholders without a field untouched, so path templates fill in steps.

```python
# a path pattern of a spec filled in (missing names stay as they are)
formats.safe_format("{dataset}/scan{scan:04d}", scan=7)      # '{dataset}/scan0007'
```

Constants:

- `FORMAT_DIR_ENV`: `"BEXA_FORMAT_DIR"`, the environment variable naming extra spec folders.
- `SPEC_SUFFIXES`: `(".yaml", ".yml")`, the suffixes of spec files.

### bexa.io.survey: every dataset under a folder

One walk over a data tree (a `RAW_DATA` folder, a copy on the lab share, a folder of beamline
examples) that says what recognises each dataset in it: the folders a format spec recognises as
a whole (a BLISS dataset, a SPEC session, a PAL-XFEL run) and the data files a spec recognises
one by one (one HDF5 file per scan, saved cubes). A recognised folder is not entered, the scan
folders of a known layout are skipped, and of the files that share a naming pattern only the
first is opened, so a tree of hundreds of datasets takes seconds. Data no spec recognises is
reported with the spec that came closest, and `write_drafts` drafts one spec per unknown HDF5
layout, not one per dataset. `bexa profile sniff PATH -o DRAFTS --recursive` runs all of it from
the shell, and the scan picker lists its dataset folders with the same walk.

```python
import tempfile
from pathlib import Path
import h5py
import numpy as np
from bexa.io.survey import describe_survey, survey, survey_table, write_drafts
from bexa.testing import make_aps_session, make_esrf_scan, make_pal_run

tree = Path(tempfile.mkdtemp())                  # a data tree holding several beamlines' data
esrf = make_esrf_scan(tree / "RAW_DATA" / "sample5", "beamtime", n_files=1)   # an ESRF dataset, three levels down
aps = make_aps_session(tree / "aps")             # an APS 7-ID SPEC session with its TIFFs
pal = make_pal_run(tree / "pal_run", run=1)      # a PAL-XFEL run, September 2025 layout
unknown = tree / "custom" / "scan_data.h5"       # an HDF5 layout no spec knows
unknown.parent.mkdir()
with h5py.File(unknown, "w") as f:               # frames and a motor under entry/
    f["entry/data/frames"] = np.zeros((5, 32, 32), np.uint16)
    f["entry/motors/theta"] = np.linspace(0, 1, 5)
```

#### `survey`
`bexa.io.survey.survey(root, depth=4, quick=False, specs=None)`

Every dataset under `root`, up to `depth` folder levels down, as a list of `SurveyEntry` in the
order of the walk (folders sorted by name). At every folder the specs are matched against the
folder as a whole first (`bexa.io.formats.match_spec`); a folder recognised is reported and not
entered. Otherwise its data files are grouped by naming pattern (`p0012.h5` and `p0013.h5` are
`p#.h5`) and the first file of each HDF5 group is matched on its own, and the walk goes on into
its subfolders. Hidden folders, `bexa_cache`, `__pycache__`, the scan folders of a known layout
(`scanNNNN`) and linked folders are skipped. `quick=True` opens no HDF5 file: a folder is
recognised from the file-system part of the fingerprints alone (a master file beside scan
folders, the opening lines of a SPEC file) and data files are not checked; the scan picker lists
folders this way. `specs` limits the specs tried. A file given as `root` is checked alone; a
missing root raises `FileNotFoundError`.

```python
# tree: the data tree of the setup; what recognises each dataset in it
entries = survey(tree)
[e.spec for e in entries if e.recognised]        # ['esrf_id03_bliss_2026', 'aps_7id_spec_2025', 'pal_xfel_points_2025_09']
[(e.path.name, e.closest) for e in entries if not e.recognised]  # [('scan_data.h5', 'esrf_id03_bliss_F2026')]
[e.path.name for e in survey(tree, quick=True)]  # ['beamtime', 'aps', 'pal_run']
[e.path.name for e in survey(tree, depth=1)]    # ['aps', 'scan_data.h5', 'pal_run']
```

#### `SurveyEntry`
`bexa.io.survey.SurveyEntry(path, kind, recognised, spec=None, engine=None, score=0.0, closest=None, count=1, pattern=None, opens_as_dataset=False)`

A dataset `survey` found, or data no spec recognises. `path` is a dataset folder
(`kind="folder"`) or the first of the data files sharing one naming pattern (`kind="files"`,
`count` of them, `pattern` their name with digits as `#`). `spec` and `engine` say what
recognised it; a quick survey names the spec only when one alone fits, so the dated ESRF specs
give the engine only. `closest` names the spec that came nearest for unrecognised data, with its
`score`, the fraction of the fingerprint that fitted. `opens_as_dataset` is True for a folder of
numbered scans, the kind `bexa.open_dataset` and the scan picker open.

```python
# entries: the survey of the tree; the ESRF dataset and the unknown file
first = entries[0]
first.kind, first.spec, first.engine, first.opens_as_dataset  # ('folder', 'esrf_id03_bliss_2026', 'hdf5_stack', True)
custom = [e for e in entries if not e.recognised][0]
custom.kind, custom.count, custom.pattern, custom.closest, custom.score  # ('files', 1, 'scan_data.h5', 'esrf_id03_bliss_F2026', 0.33333)
```

#### `describe_survey`
`bexa.io.survey.describe_survey(entries, root=None)`

The entries as text, paths relative to `root` when given: what is recognised and by which spec
(or which engine, for a quick survey), which folders open with `open_dataset` and the scan picker,
and what is not recognised with the closest spec. A group of files shows as
`folder/pattern (N files)`. `bexa profile sniff --recursive` prints it.

```python
# entries: the survey of the tree; printed as the command line shows it
text = describe_survey(entries, tree)
print(text)                                      # '3 recognised under ...', a line per dataset and its spec, then '1 not recognised:'
```

#### `survey_table`
`bexa.io.survey.survey_table(entries, root=None)`

The entries as a `pandas.DataFrame`, one row per entry and one column per field of
`SurveyEntry`, the paths relative to `root` when given: handy for sorting and filtering a large
tree in a notebook.

```python
# entries: the survey of the tree; as a table
table = survey_table(entries, tree)
table["path"].tolist()                           # ['RAW_DATA/sample5/beamtime', 'aps', 'custom/scan_data.h5', 'pal_run']
```

#### `write_drafts`
`bexa.io.survey.write_drafts(entries, out_dir)`

Drafts one format spec per HDF5 layout that no spec recognises and returns `{draft path: [data
paths it covers]}`. The unrecognised HDF5 data is grouped by layout (the names of its top-level
groups and of its 3-D datasets, digits ignored); each group gets one draft, made by
`bexa.io.formats.draft_spec` from the folder of its first entry and written to
`<out_dir>/TODO_<folder>.yaml` with the covered paths under `seen_in`. A draft of that name from
an earlier run is replaced. Data in other formats (TIFF, NumPy, SPEC) gets no draft: drafting
reads HDF5 only. Fill in the `TODO` entries as for `bexa.io.formats.write_draft`.

```python
# entries: the survey of the tree; one draft for the unknown layout
drafts = write_drafts(entries, tree / "drafts")
[(path.name, len(covered)) for path, covered in drafts.items()]  # [('TODO_custom.yaml', 1)]
```

Constants:

- `HDF5_SUFFIXES`: `(".h5", ".hdf5", ".nxs", ".nx")`, the files matched one by one.
- `DATA_SUFFIXES`: those plus `.spec`, `.tif`, `.tiff`, `.npy`, `.npz`, `.edf`, `.cbf`: the files
  a survey reports when nothing recognises them.
- `SKIPPED_FOLDERS`: `bexa_cache` and `__pycache__`.

### bexa.io.base: the `Source` protocol and `BaseSource`

An engine is a source: it knows the number, shape and dtype of a scan's frames, reads any of them
cut to a pixel window, and describes the scan's structure. `Scan`, the reductions and the plots use
nothing else. `scan.source` is the source of an open scan; `bexa.io` re-exports `Source`,
`BaseSource` and `FrameBatch`.

```python
import numpy as np
import bexa
from bexa.io.base import BaseSource, FrameBatch, Source

scan = bexa.demo("mosa")          # synthetic chi x mu scan, 6 x 16 frames of 64 x 80 pixels
src = scan.source                 # its engine, an Hdf5StackSource
```

#### `BaseSource`
`bexa.io.base.BaseSource(spec=None)`

Abstract base class of the engines, keeping the `FormatSpec` as `spec` and the engine name as the
class attribute `name`. An engine implements `files`, `frame_shape`, `n_frames`, `dtype`,
`read_frames` and `structure`, and registers with `@bexa.core.registry.register_engine("name")`. A
source is a context manager (`with` closes it). The `Source` members, which every engine has, and
the helpers:

- `files()`: list of `Path`, every file read; used for provenance and cache keys (`scan.files`).
- `frame_shape`, `n_frames`, `dtype`: properties; `(height, width)` in pixels, frames available now, stored numpy dtype.
- `read_frames(index, y, x, dtype, out)`: frames as a numpy array, see `BaseSource.read_frames`.
- `structure()`: the scan's `bexa.core.structure.Structure`: motor dims (slow first) and coordinates, `frame_index` (the frame id at each grid position, -1 where a partial scan has none), `per_frame` channels, `scalars`, `units`, `scan_type`, `energy_keV`, `title`. Built once, kept until `refresh()`, and the same object as `scan.structure`.
- `scalars()`: dict of the values fixed during the scan (positioners); default `structure().scalars`.
- `aux_tables()`: dict of extra pandas tables (the shot tables of `Hdf5PointsSource`); default `{}`.
- `energy_keV`: property, photon energy in keV or `None`; default `structure().energy_keV`.
- `status`: property, the `bexa.core.status.ScanStatus` of the scan; the default knows the frames alone (`complete` when every grid point has one, else `aborted`), the BLISS engine reads the master entry.
- `with_motors(motors)`: the same scan with these per-frame channels as its dims, where the layout allows (`NotImplementedError` by default; `Hdf5StackSource` implements it).
- `chunk_frames`: property, frames per HDF5 chunk; reduction batches are multiples of it; default 1.
- `refresh()`: re-reads the metadata of a scan still being written, returns the source (`scan.refresh()` calls it).
- `close()`: closes the open file handles.
- `cache_records()`: what identifies the data in cache keys, one dict per file (`path`, `size`, `mtime` by default: `bexa.core.provenance.file_records`); an engine whose files change while the data they serve does not (`Hdf5StackSource`) overrides it with records of the part it reads.
- `frame_bytes(y=slice(None), x=slice(None), dtype=None)`: bytes of one frame cut to the window, at `dtype` (default: stored).
- `contiguous_runs(index)`: static; yields `(start, stop)` for each run of consecutive ids of a sorted id array.
- `normalise_index(index, n_frames)`: static; a slice or ids as sorted unique `int64` ids; `IndexError` outside `[0, n_frames)`.

```python
# src: the source behind the demo scan; what every engine answers
src.frame_shape, src.n_frames, src.dtype, src.chunk_frames      # ((64, 80), 96, dtype('uint16'), 1)
src.energy_keV, src.scalars()["samz"], [f.name for f in src.files()]  # (17.0, 0.0, ['demo_mosa.h5', 'pco_ff_0000.h5'])
src.cache_records()[0]                                          # {'path': '.../demo_mosa.h5', 'entry': '1.1', 'n_frames': 96}
s = src.structure()
s.dims, s.motor_shape, s.scan_type, s.units["mu"]              # (('chi', 'mu', 'y', 'x'), (6, 16), 'fscan2d', 'deg')
src.frame_bytes(), list(BaseSource.contiguous_runs(np.array([0, 1, 2, 7, 8])))   # (10240, [(0, 3), (7, 9)])
BaseSource.normalise_index(slice(0, 10, 3), src.n_frames)      # array([0, 3, 6, 9])
```

#### `BaseSource.read_frames`
`bexa.io.base.BaseSource.read_frames(index, y=slice(None, None, None), x=slice(None, None, None), dtype=None, out=None)`

Reads the frames `index` (a slice, or flat ids in acquisition order, sorted and made unique first)
cut to the pixel window `y`, `x`, as an `(n, height, width)` numpy array in `dtype` (default:
stored), written into `out` if given. Consecutive ids are read as one block. It is the read behind
`scan.read`, previews and reductions; `scan.structure.frame_index` maps a grid position to its id.

```python
# src: the source behind the demo scan; four frames, a pixel window, as float32
frames = src.read_frames(np.array([0, 1, 2, 50]), y=slice(10, 40), x=slice(20, 60), dtype=np.float32)
frames.shape, frames.dtype                       # ((4, 30, 40), dtype('float32'))
fid = scan.structure.frame_index[2, 5]           # the frame at chi index 2, mu index 5
src.read_frames([fid])[0].shape                  # (64, 80)
```

#### `Source`
`bexa.io.base.Source(*args, **kwargs)`

The `typing.Protocol` of a reader: `name`, `files`, `frame_shape`, `n_frames`, `dtype`,
`read_frames`, `structure`, `status`, `with_motors`, `scalars`, `aux_tables`, `energy_keV`,
`chunk_frames`, `refresh` and `close`, as listed under `BaseSource`. Never instantiated: use it
in type hints and, being runtime-checkable, in `isinstance` tests.

```python
# the protocol: a source is anything with the members of BaseSource
isinstance(src, Source), isinstance(np.zeros(3), Source)      # (True, False)
```

#### `FrameBatch`
`bexa.io.base.FrameBatch(frames, frame_ids, y, x)`

A dataclass for one block of frames streamed by the reductions: `frames` `(n, height, width)`,
windowed and downsampled (numpy, or cupy on the GPU), their `frame_ids`, and the window `y`, `x` as
slices; `len(batch)` is `n`. `scan.batches(roi=..., downsample=...)` yields them.

```python
# scan: the demo chi x mu scan; the first batch of 16 frames and what it carries
batch = next(iter(scan.batches(batch_frames=16)))
len(batch), batch.frames.shape, batch.frame_ids[:3]     # (16, (16, 64, 80), array([0, 1, 2]))
```

### bexa.io.scan_commands: scan command lines

A BLISS or SPEC scan writes its command line as the title of the scan: `ascan chi 6.7 6.9 10
0.1`, `amesh mu 9.56 10.56 20 phi -1.08 -0.48 12 0.4`, `fscan2d chi 4.14 0.023 31 mu 11.92
0.04 200 2 2.08`. The parser reads the command word and the motor blocks, so a scan of any
kind is read with its motors even when the file declares none (the BLISS engine uses it after
`fscan_parameters` and before the readbacks). Two grammars: the step scans of SPEC and BLISS,
`<command> (<motor> <start> <stop> <intervals>)... <count time>`, whose point count is the
intervals plus one (`STEP_COMMANDS`), and the continuous scans of ID03, `<command> (<motor>
<start> <step> <npoints>)...` (`CONTINUOUS_COMMANDS`). A command outside both families (a
beamline macro such as `dfmesh`) is read with the same block structure but its counts are not
trusted; one in `STATIC_COMMANDS` (`loopscan`, `timescan`, `ct`) moves no motor.

```python
from bexa.io.scan_commands import ScanCommand, ScanMotor, parse_scan_command
```

#### `parse_scan_command`
`bexa.io.scan_commands.parse_scan_command(text, motor_names=None, intervals=None)`

A `ScanCommand` from the command line `text`, which starts with the command word. `motor_names`
limits the motor blocks to those names (the positioners the file knows): another word ends the
blocks. `intervals` says whether the third number of a block counts intervals (points are
intervals plus one) or points; by default the command word decides, and an unknown command is
read as a step scan with `exact=False`. An `a2scan`-style command with one count after the
last block shares it between the motors; a command without motor blocks gives an empty motor
list; an empty `text` raises `ValueError`.

```python
# the command line of an amesh scan, as the master's title records it
mesh = parse_scan_command("amesh mu 9.5558 10.5558 20 phi -1.0827 -0.4827 12 0.4")   # a step mesh of ID03
mesh.command, mesh.names, mesh.shape, mesh.n_points   # ('amesh', ['mu', 'phi'], (21, 13), 273)
parse_scan_command("fscan2d chi 4.14 0.0233 31 mu 11.92 0.04 200 2 2.08").shape      # (31, 200): points, as fscans count
macro = parse_scan_command("dfmesh difftz 0 1 10 samy -1 1 4 0.2")                    # a beamline macro: motors known, counts not trusted
macro.names, macro.exact, macro.n_points           # (['difftz', 'samy'], False, None)
parse_scan_command("loopscan 100 0.1").motors       # []: a sequence of frames
```

#### `ScanCommand`
`bexa.io.scan_commands.ScanCommand(command, motors, text, exact=True, lockstep=False)`

A dataclass: the command word, the list of `ScanMotor` (outer first), the original `text`, and
`exact`, whether the point counts are to be trusted (true for the step and continuous families,
false for an unknown command). Properties: `shape` (points per motor), `names` (the motor
names) and `n_points` (the frames the command plans, `None` when unknown).

```python
# mesh: the parsed amesh command; its motors, the planned grid, whether it was understood
mesh.motors[0].name, mesh.shape, mesh.exact      # ('mu', (21, 13), True)
```

#### `ScanMotor`
`bexa.io.scan_commands.ScanMotor(name, start, stop, npoints)`

A dataclass for one motor of a command: its name, the range (`start`, `stop`; the step for a
continuous scan) and the number of points.

```python
# mesh: the parsed amesh command; its second motor
mesh.motors[1]                                   # ScanMotor(name='phi', start=-1.0827, stop=-0.4827, npoints=13)
```

Constants:

- `STEP_COMMANDS`: `ascan`, `dscan`, `a2scan`, `d2scan`, `a3scan`, `d3scan`, `a4scan`, `d4scan`, `amesh`, `dmesh`, `pointscan`, `anscan`, `dnscan` (intervals in the blocks). `CONTINUOUS_COMMANDS`: `fscan`, `fscan2d`, `fscan3d`, `finterlaced`, `fscan2d_cyclic`, `fsweep` (points). `STATIC_COMMANDS`: `loopscan`, `timescan`, `ct`, `sct` (no motor).

### bexa.io.h5files: opening HDF5 files another program is writing

At a beamline the master file of a dataset stays open in the control system for the whole
beamtime, and the detector writes its files while a scan runs. HDF5 refuses to open a file
another process holds open unless file locking is off, and a file caught in the middle of a
flush fails once and opens a moment later. `open_h5` is `h5py.File` with both taken care of;
every master and detector file bexa reads goes through it (bexa's own result files do not).

```python
import tempfile
import h5py
from bexa.io.h5files import RETRY_DELAY, locking_kwargs, open_h5
from bexa.testing import make_esrf_scan

made = make_esrf_scan(tempfile.mkdtemp(), "scan", n_files=1)  # a synthetic BLISS scan: a master and one detector file
```

#### `open_h5`
`bexa.io.h5files.open_h5(path, mode="r", retry=True, **kwargs)`

`h5py.File(path, mode)` for a file another program may be writing: the file lock is not taken
(`locking_kwargs`), and when a read-only open fails with `OSError` and `retry` is on, it is
tried once more after `RETRY_DELAY` seconds, then in SWMR mode (single writer, multiple
readers, which a file written that way allows); the first error is raised when all three fail.
Other keywords go to `h5py.File` (the chunk cache settings of the frame reader).

```python
# made: a synthetic scan; its master opened read-only without an HDF5 lock
with open_h5(made.master) as f:                  # the master, even while BLISS holds it open
    list(f)                                      # ['1.1']
```

#### `locking_kwargs`
`bexa.io.h5files.locking_kwargs()`

`{"locking": False}` when this h5py and HDF5 take the keyword (HDF5 1.12.1 and later), else
`{}`; computed once. The `HDF5_USE_FILE_LOCKING` variable, when the user sets it, is read by
the library itself and wins.

```python
# what open_h5 passes to h5py.File on this machine's HDF5
locking_kwargs()                                 # {'locking': False} with h5py 3.12 on HDF5 1.14
```

Constants:

- `RETRY_DELAY`: `0.5`, the seconds before the one retry of an open that failed.

### bexa.io.engines.hdf5_stack: ESRF BLISS master and detector files

The engine `hdf5_stack` reads the ESRF ID03 layout (specs `esrf_id03_bliss_2024`, `_2025`,
`_2026`, `_F2026`): a master HDF5 per dataset with the motors of every scan (`N.1/measurement`,
`N.1/instrument/positioners`, `fscan_parameters`) and a `scanNNNN/` folder of detector files per
scan (the autumn 2026 layout also links the frames into the master as a virtual dataset; the
engine still reads the detector files). `bexa.open(dataset_folder, scan=7)` creates one (`scan.source`). Importing `bexa.io.engines`
registers every engine in `bexa.core.registry.engines`; `bexa.open` does it by itself.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.io.engines.hdf5_stack import Hdf5StackSource
from bexa.io.formats import load_spec
from bexa.testing import make_esrf_scan

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
made = make_esrf_scan(work, dataset="synth_dfxm")    # chi x mu, 48 frames of 48 x 64 in 2 files
spec = load_spec("esrf_id03_bliss_2026")        # the ESRF ID03 spec of 2026
src = bexa.open(made.dataset_dir, scan=1).source    # an Hdf5StackSource
```

#### `Hdf5StackSource`
`bexa.io.engines.hdf5_stack.Hdf5StackSource(spec, root, dataset, scan, detector=None, energy_keV=None, master=None, motors=None)`

Scan `scan` of the dataset folder `root/dataset`: frames from the files of `detector` (default: the
spec's first, `pco_ff`) in `scanNNNN/`, metadata from the master (`root/dataset/dataset.h5` unless
`master` is given). It implements the `Source` members `files`, `frame_shape`, `n_frames`,
`dtype`, `chunk_frames`, `energy_keV`, `read_frames`, `structure`, `status`, `with_motors`,
`refresh` and `close` (see `BaseSource`). Which channels are the dims is settled in this order
(`bexa.io.engines.hdf5_master.discover_motors`): the motors declared in `fscan_parameters`
(`motor`, `slow_motor`/`fast_motor`, `slow1_motor`/`slow2_motor`/`fast_motor` as BLISS writes
them for an fscan, fscan2d and fscan3d, or `outer`/`middle`/`inner`; `MOTOR_KEYS`), else the
motors the command line in the title names (`amesh mu ... phi ...`, `ascan chi ...`, a beamline
macro: `bexa.io.scan_commands`), else every recorded channel that is a motor of the file and
moved (a `loopscan` or `ct` moves none); channels that step together (a motor and its encoder,
an `a2scan`) collapse to the first named, `motors` passed by hand replace the choice
(`with_motors`, `Scan.with_dims`), the readbacks decide which loop is the outer one, and frames
are placed on the grid by their values, so snake and partial scans land in place. The scan type
is the declared one (`fscan2d`), else the command word (`amesh`, `dmesh`, `dfmesh`), else `list`.
When the monochromator angle (`energy.from`, `ccmth`) is one of the scanned
motors, an energy mosa, that dim is named `energy` and holds keV (`ENERGY_DIM`; the angle stays a
per-frame channel), so `MotorCOM(axes=("energy",))` gives `com_energy` straight from the scan.
`measurement` channels are per-frame channels, positioners scalars (plus the mean of every
channel `motors.scalars_from` names, the furnace temperature of the `<scan>.2` monitoring entry
at ID03), renamed motors take their logical names from `motors.aliases`, and the energy is
`energy_keV` or `ccmth` converted with the Si 111 d-spacing (the mean angle of an energy
mosa). How the frames are split over the detector
files comes from the master when it says (`layout.frames_total`, `acq_nb_frames`, and
`layout.frames_per_file`, `saving_frame_per_file`, as LIMA records them; the frame shape and
dtype from the virtual dataset `layout.master_frames` in the autumn 2026 layout), so opening a
scan opens one detector file, the last, whose count is the one a partial scan changes, instead
of all sixty-four; a scan still being written has fewer files than the master announces and a
last file that cannot be read yet, both read as far as they go, and when the files do not add
up every file is opened. A scan without a readable frame builds (0 frames, the spec's frame
shape) and raises `bexa.core.status.ScanNotReadyError` from `structure()`; the master entry
says whether it is `running`, `empty` or, when the files cannot be read, `unreadable`
(`status`). `read_frames` without `dtype` returns the frames as stored, read straight into the
output array. Files are opened through `bexa.io.h5files.open_h5` (no file lock, one retry),
once per thread with a chunk cache; `refresh()` re-lists the files of a scan still being
written and reads its status again. Also:

- `list_scans(spec, root, dataset, detector=None, require_files=True)`: classmethod, the sorted numbers of the `scanNNNN` folders (a folder named otherwise, `scan0007_old`, is not a scan), those holding files of `detector` unless `require_files=False`; `bexa.list_scans(dataset_folder)` does the same from a path.
- `structure(motors=None)`: the scan's `Structure`, built once; with `motors` (per-frame channel names) a structure on those dims instead, not kept.
- `status`: property, the `ScanStatus` from the master entry (`hdf5_master.decide`), settled when the structure is built; `problem` is the error text when it is `unreadable`.
- `scan_status(spec, root, dataset, scan, detector=None)`: classmethod, the status of one scan, opening and closing its files, for a scan whose source cannot be built.
- `with_motors(motors)`: a new source on the same files with `motors` as its dims (`Scan.with_dims`).
- `cache_records()`: the detector files by path, size and modification time, but the master by its path, the entry this scan reads and the frame count: BLISS appends every new scan to the master, so keying on the file would discard every cached result of the dataset at each new scan.
- attributes: `root`, `dataset`, `scan`, `detector`, `master`, `scan_folder`, `frames_path`.

```python
# src: the engine behind a synthetic ESRF scan; its files and cache records
[f.name for f in src.files()]                    # ['synth_dfxm.h5', 'pco_ff_0000.h5', 'pco_ff_0001.h5']
src.cache_records()[0]                           # {'path': '.../synth_dfxm.h5', 'entry': '1.1', 'n_frames': 48}
src.structure().dims, src.energy_keV, src.structure().scalars["ffz"]  # (('chi', 'mu', 'y', 'x'), 17.0, 100.0)
src.status, src.structure().planned_frames       # (<ScanStatus.COMPLETE: 'complete'>, 48)
Hdf5StackSource(spec, work, "synth_dfxm", 1, energy_keV=17.1).energy_keV      # 17.1
Hdf5StackSource.list_scans(spec, work, "synth_dfxm"), bexa.list_scans(made.dataset_dir)   # ([1], [1])
running = make_esrf_scan(work, dataset="going", partial=20, state="running")   # 28 of 48 frames so far
going = Hdf5StackSource.from_path(spec, running.dataset_dir)
going.status, going.n_frames, going.structure().n_missing   # (<ScanStatus.RUNNING: 'running'>, 28, 20)
Hdf5StackSource.scan_status(spec, work, "going", 1)         # <ScanStatus.RUNNING: 'running'>
src.with_motors(["mu", "chi"]).structure().motor_dims       # ('chi', 'mu'): the readbacks still order the loops
```

#### `Hdf5StackSource.from_path`
`bexa.io.engines.hdf5_stack.Hdf5StackSource.from_path(spec, path, scan=None, **kwargs)`

Classmethod: the source from a dataset folder, its master file or a `scanNNNN` folder (which gives
the scan). Without `scan`, a one-scan dataset opens it and one with several raises `ValueError`
(`bexa.open(..., scan=None)` then opens all as a series, see `MultiScanSource`). `kwargs` go to the
constructor.

```python
# made: a synthetic ESRF scan; the engine from its master file, or from a scan folder
Hdf5StackSource.from_path(spec, made.master, scan=1).n_frames      # 48
Hdf5StackSource.from_path(spec, made.scan_folder).scan              # 1
```

Constants:

- `MOTOR_KEYS`, `FALLBACK_MOTORS`: re-exported from `bexa.io.engines.hdf5_master`.
- `MIN_CACHE_BYTES`, `MAX_CACHE_BYTES`: bounds of the per-file chunk cache (four chunks), 8 MiB and 512 MiB.
- `SCAN_FOLDER`: the pattern of a scan folder's name, `scan(\d+)`.

### bexa.io.engines.hdf5_master: what the BLISS master says about a scan

The master holds, per scan entry `N.1`, the readback of every channel in `measurement` (one
value per frame), the fixed positioners, the description of a continuous scan in
`fscan_parameters`, the command line as the title, and when and why the scan started and
ended. These functions read those and turn them into the motors a scan was made of, the grid
it was meant to cover and the state it is in; `Hdf5StackSource` calls them with the master
open and reads the frames itself. They are useful on their own to look at a master file.

```python
import tempfile
import h5py
import bexa
from bexa.io.engines.hdf5_master import (EntryState, MotorChoice, MotorTable, collapse_lockstep, decide,
                                         declared_motors, decode, discover_motors, entry_state,
                                         moving_motors, read_motor_table, read_title)
from bexa.io.formats import load_spec
from bexa.io.scan_commands import parse_scan_command
from bexa.testing import make_esrf_scan

made = make_esrf_scan(tempfile.mkdtemp(), "mesh", motors=(("samy", 3), ("samz", 5)), n_files=1, command="dmesh")   # a step mesh: no fscan_parameters
spec = load_spec("esrf_id03_bliss_2026")        # the ESRF ID03 spec of 2026
f = h5py.File(made.master, "r")                 # the master file, open for reading
```

#### `read_motor_table`
`bexa.io.engines.hdf5_master.read_motor_table(f, spec, scan, detector)`

The `MotorTable` of scan `scan` in the open master `f`: every 1-D numeric channel of
`measurement` at full length (`channels`), the 0-d positioners and the means of the channels
`motors.scalars_from` names (`scalars`), and every name the file calls a motor
(`motor_names`: the keys of `positioners`, `positioners_start` and `positioners_end`, or the
`NXpositioner` groups of an older layout).

```python
# f: the master of a synthetic scan, open; every per-frame channel and positioner of scan 1
table = read_motor_table(f, spec, 1, "pco_ff")
sorted(table.channels), table.scalars["ccmth"], "samy" in table.motor_names   # (['elapsed_time', 'samy', 'samz'], 6.678..., True)
```

#### `MotorTable`
`bexa.io.engines.hdf5_master.MotorTable(channels={}, scalars={}, motor_names=frozenset())`

A dataclass: `channels` (`{name: array}` per frame, at the length the master has, which lags or
leads the detector files while a scan runs), `scalars` (`{name: value}`) and `motor_names`
(a frozenset), as `read_motor_table` fills them.

```python
# table: the motor table of scan 1; one per-frame channel of it
len(table.channels["samy"]), table.channels["samy"][:4]   # (15, array([0., 0., 0., 0., ...]))
```

#### `declared_motors`
`bexa.io.engines.hdf5_master.declared_motors(f, spec, scan)`

`(scan_type, motor names slow first, points per motor)` from the `fscan_parameters` group the
spec's `motors.structure` template names (`MOTOR_KEYS`: `motor`, `slow_motor`/`fast_motor`,
`slow1_motor`/`slow2_motor`/`fast_motor`, or `outer`/`middle`/`inner`), or `None` when the scan
has none (a step scan, the 2024 layout).

```python
# f: an open master; what the scan declares as its motors (fscan_parameters)
declared_motors(f, spec, 1)                      # None: a dmesh declares nothing
fscan = make_esrf_scan(made.root, "fscan", n_files=1)
with h5py.File(fscan.master, "r") as g:
    declared_motors(g, spec, 1)                  # ('fscan2d', ['chi', 'mu'], [4, 12])
```

#### `read_title`
`bexa.io.engines.hdf5_master.read_title(f, entry)`

The command line of the scan entry (`"1.1"`), or an empty string.

```python
# f: an open master; the command line of scan 1
read_title(f, "1.1")                             # 'dmesh samy 0 1 2 samz 0 1 4 0.1'
```

#### `entry_state`
`bexa.io.engines.hdf5_master.entry_state(f, entry, spec, scan, detector)`

The `EntryState` of a scan entry: whether it is in the master at all, whether it started
(`start_time` or a `writer` group) and ended (`end_time`), the `end_reason` and the writer
status words when written, and the planned frame count from `layout.frames_total`
(`acq_nb_frames`).

```python
# f: an open master; whether scan 1 started, ended, how it ended, how many frames it planned
state = entry_state(f, "1.1", spec, 1, "pco_ff")
state.started, state.ended, state.end_reason, state.planned_frames   # (True, True, 'SUCCESS', 15)
entry_state(f, "9.1", spec, 9, "pco_ff").present                      # False: not in the master yet
```

#### `EntryState`
`bexa.io.engines.hdf5_master.EntryState(present=True, started=False, ended=False, end_reason=None, writer_status=None, planned_frames=None)`

A dataclass with the fields `entry_state` reads.

```python
# the state of a scan that has started and not ended
EntryState(started=True)                         # a scan that started and has not ended
EntryState(started=True, ended=True, end_reason="SUCCESS").ended  # True
```

#### `decide`
`bexa.io.engines.hdf5_master.decide(state, n_frames, planned=None)`

The `ScanStatus` of a scan from its entry state and the frames found, first match wins: an
entry the master does not hold yet, or one that started and has not ended, is `running`; no
frame at all is `empty`; an end reason other than success, or fewer frames than planned
(`state.planned_frames`, else `planned`), is `aborted`; otherwise the scan is `complete`.
Masters without the time fields (older files) are judged by the frames alone.

```python
# state: the entry state of a finished scan; the status for 15 frames, for 10, for a scan just started
decide(state, 15), decide(state, 10)             # (<ScanStatus.COMPLETE: 'complete'>, <ScanStatus.ABORTED: 'aborted'>)
decide(EntryState(started=True), 0)              # <ScanStatus.RUNNING: 'running'>
```

#### `discover_motors`
`bexa.io.engines.hdf5_master.discover_motors(channels, declared, command, motor_names, known, min_steps, logical_name=None)`

Which channels are the scanned motors, as a `MotorChoice`, in this order of trust: the
declared motors of a continuous scan when the file has them and they are recorded, else the
motors the command line names, else every recorded channel that is a motor and moved
(`moving_motors`; a command that moves no motor gives none). Channels that step together
collapse to the first named (`collapse_lockstep`). The scan type is the declared one, else the
command word, else `fscan<n>d`. `known` is the spec's `motors.known` list (it orders the
candidates), `min_steps` its `motors.min_step`, `logical_name` the `{physical: logical}`
renames of `motors.aliases`.

```python
# table, command: the channels and the parsed title of scan 1; the motors that move
command = parse_scan_command(read_title(f, "1.1"), motor_names=table.motor_names)
choice = discover_motors(table.channels, None, command, table.motor_names, spec.known_motors(), spec.min_steps())
choice.scan_type, choice.order, choice.planned_shape, choice.candidates   # ('dmesh', ['samy', 'samz'], (3, 5), ['samy', 'samz'])
```

#### `MotorChoice`
`bexa.io.engines.hdf5_master.MotorChoice(scan_type, order, planned_shape, lockstep, candidates)`

A dataclass: the scan type, the dims slowest first as the file or the command names them
(the readbacks settle the loop order later), the planned points per dim (or `None`), the
channels that step with a dim (`{dim: [channels]}`, kept as coordinates) and every channel
that is a motor and moved, the choices a reader has.

```python
# choice: the motors found for scan 1; the channels stepping together
choice.lockstep                                  # {}: nothing steps with another motor here
```

#### `moving_motors`
`bexa.io.engines.hdf5_master.moving_motors(channels, motor_names, known, min_steps)`

The per-frame channels that are motors and moved, in the spec's known order first: a channel is
a motor when the file lists it as a positioner or the spec knows it (`motor_names`, `known`,
`FALLBACK_MOTORS`), never a time stamp or counter (`NOT_MOTORS`); it moved when its range
exceeds the spec's `min_step` for it (else `MOVE_THRESHOLD`).

```python
# table: the motor table of scan 1; every motor that moves within the scan
moving_motors(table.channels, table.motor_names, spec.known_motors(), spec.min_steps())   # ['samz', 'samy']
```

#### `collapse_lockstep`
`bexa.io.engines.hdf5_master.collapse_lockstep(order, channels)`

Keep one of several channels that step together (`a2scan`, a motor and its encoder): the dims
that remain, in order, and `{kept: [dropped]}`.

```python
# three channels, two of them stepping together: one dim for the pair
twins = {"mu": table.channels["samz"], "mu_enc": table.channels["samz"] + 1e-7, "chi": table.channels["samy"]}
collapse_lockstep(["mu", "mu_enc", "chi"], twins)   # (['mu', 'chi'], {'mu': ['mu_enc']})
```

#### `decode`
`bexa.io.engines.hdf5_master.decode(value)`

Bytes and 0-d string arrays as `str`; anything else as it is (the way HDF5 string scalars come
back).

```python
# f: an open master; a text dataset as str, whatever HDF5 stored it as
decode(f["1.1/title"][()])                       # 'dmesh samy 0 1 2 samz 0 1 4 0.1'
f.close()
```

Constants:

- `MOTOR_KEYS`: the datasets of `fscan_parameters` that name the scanned motors, slowest loop first. `FALLBACK_MOTORS`: `mu`, `chi`, `phi`, `theta`, `omega`, `delta`, `gamma`, `ux`, `uy`, `uz`, the motors tried when nothing names any. `NOT_MOTORS`: `elapsed_time`, `epoch`, `epoch_trig`, `frame`, `timestamp`, channels that are never motors. `MOVE_THRESHOLD`: `1e-6`, below which a readback did not move.

### bexa.io.engines.hdf5_points: PAL-XFEL point files (September 2025)

The engine `hdf5_points` reads the PAL-XFEL layout of September 2025 (spec
`pal_xfel_points_2025_09`, ue_250913_FXS): per motor point, a pandas table with one row per shot
(`type=measurement/run=NNN/scan=NNN/pNNNN.h5`) and the Jungfrau frames (`type=raw/.../pNNNN.h5`).
All shots form one stack (dim `frame`, scan type `points`) whose per-frame channels are the table
columns found through the spec's `keys`. `bexa.open(run_folder)` creates one (`scan.source`);
`bexa.pipelines.xfel_cube.build_cube` reduces it to a laser on/off cube.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.io.engines.hdf5_points import Hdf5PointsSource, point_number, split_run_path
from bexa.io.formats import load_spec
from bexa.testing import make_pal_run

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
pal = make_pal_run(work, run=1, n_points=5, shots_per_point=16)   # 5 delays, 80 shots of 24 x 32
spec = load_spec("pal_xfel_points_2025_09")     # the PAL-XFEL points spec
scan = bexa.open(pal.measurement_dir)            # .../type=measurement/run=001/scan=001
src = scan.source                                # an Hdf5PointsSource
```

#### `Hdf5PointsSource`
`bexa.io.engines.hdf5_points.Hdf5PointsSource(spec, raw_root, run, scan=1, detector=None, energy_keV=None)`

The shots of run `run`, scan `scan` under `raw_root` (the folder holding `type=measurement` and
`type=raw`) in point order; a point without its raw file yet is skipped with a warning. `detector`
picks a `frames_path` from the spec's `detectors` (default `jungfrau2`); `energy_keV` sets the
energy, which the files lack. It implements the `Source` members `files`, `frame_shape`,
`n_frames`, `dtype`, `chunk_frames`, `energy_keV`, `read_frames`, `structure`, `aux_tables`,
`refresh` and `close` (see `BaseSource`): per-frame channels `point`, `shot` and every key found in
the tables (`laser_flag`, `i0`, `roi_stat`, `delay`, `th`, ...), scalars the medians of the
`axis_priority` keys, and `aux_tables()` is `{"shots": shot_table(), "points": point_summary()}`.
Also:

- `n_points`: property, the number of points; `points` lists their `PointFiles`.
- `read_point(i, y=slice(None), x=slice(None))`: every shot of the `i`-th point, cut to the window.
- `table(i)`: the per-shot pandas table of the `i`-th point, with the file's column names (cached).
- `columns()`: dict of logical key to the column present in the tables.
- `scan_axis()`: the first `axis_priority` key (`delay`, `phi`, `chi`, `th`, ...) whose median varies between points, else `"index"`.
- `list_runs(spec, raw_root)`: classmethod, the sorted run numbers with a folder in `raw_root/type=measurement`.
- `refresh()` re-lists the points of a run still being written; attributes `raw_root`, `run`, `scan`, `points_dir`, `frames_dir`, `frames_path`.

```python
# src: the engine behind a synthetic PAL-XFEL run; its points, shots and per-shot columns
src.n_points, src.n_frames, src.read_point(2).shape        # (5, 80, (16, 24, 32))
src.columns()["laser_flag"], src.columns()["i0"]            # ('event_info.THIRTY_HERTZ', 'qbpm:eh1:qbpm1:sum')
src.scan_axis(), Hdf5PointsSource.list_runs(spec, work)     # ('delay', [1])
```

#### `Hdf5PointsSource.from_path`
`bexa.io.engines.hdf5_points.Hdf5PointsSource.from_path(spec, path, scan=None, **kwargs)`

Classmethod: the source from a `scan=NNN` folder (under `type=measurement` or `type=raw`), a
`run=NNN` folder (scan 1 unless `scan=` is given) or the raw root with `run=<n>` in `kwargs`;
`ValueError` when the path names no run. Other `kwargs` go to the constructor.

```python
# pal: a synthetic PAL-XFEL run; the engine from its raw folder, or from a run number
Hdf5PointsSource.from_path(spec, pal.raw_dir).n_points, Hdf5PointsSource.from_path(spec, work, run=1).run   # (5, 1)
```

#### `Hdf5PointsSource.shot_table`
`bexa.io.engines.hdf5_points.Hdf5PointsSource.shot_table()`

Every per-shot table in one pandas DataFrame in frame order (rows beyond a point's frames
dropped): the columns `point` (the number of `pNNNN.h5`) and `shot` (the index within the point),
then the files' columns. It replaces the tables built by hand in
`PAL-XFEL_FXS_Data plot_200918.ipynb`.

```python
# src: a PAL-XFEL run; one row per shot with its point, delay and event flags
src.shot_table()[["point", "shot", "delay_input", "event_info.THIRTY_HERTZ"]].head(3)  # DataFrame: 3 rows x 4 columns (point, shot, delay_input, event_info.THIRTY_HERTZ)
```

#### `Hdf5PointsSource.point_summary`
`bexa.io.engines.hdf5_points.Hdf5PointsSource.point_summary()`

One pandas row per point: `point`, `n_shots` (frames), `n_rows` (table rows), `n_on` and `n_off`
(laser flag 1 and 0; a dropped shot with a NaN flag counts in neither) and the median of every
numeric key (`delay`, `i0`, `th`, ...), the value the legacy cube builder stored for a point.

```python
# src: a PAL-XFEL run; one row per point: shots, laser on and off, delay
src.point_summary()[["point", "n_shots", "n_on", "n_off", "delay"]]  # DataFrame: 5 rows x 5 columns (point, n_shots, n_on, n_off, delay)
```

#### `PointFiles`
`bexa.io.engines.hdf5_points.PointFiles(number, table, raw, n_shots, offset)`

A dataclass for one motor point: its `number`, the `table` and `raw` file paths, `n_shots` (frames
in the raw file) and `offset` (the flat id of its first frame).

```python
# src: a PAL-XFEL run; the files and shots of its third point
p = src.points[2]
p.number, p.n_shots, p.offset, p.raw.name        # (3, 16, 32, 'p0003.h5')
```

#### `split_run_path`
`bexa.io.engines.hdf5_points.split_run_path(path)`

`(raw_root, run, scan)` from a path inside a `type=...` tree: the part before the first `type=`
folder and the numbers of the first `run=NNN` and `scan=NNN` folders after it (`None` if absent);
a path without `type=` gives `(path, None, None)`.

```python
# the run folder of a PAL-XFEL measurement split into root, type and run number
split_run_path(pal.measurement_dir)[1:], split_run_path(work / "type=raw" / "run=007")[1:]   # ((1, 1), (7, None))
```

#### `point_number`
`bexa.io.engines.hdf5_points.point_number(path)`

The number of a `pNNNN.h5` file name, -1 for other names; the point files are sorted by it.

```python
# the point number in a PAL-XFEL file name
point_number(Path("p0012.h5")), point_number(Path("notes.h5"))     # (12, -1)
```

Constants:

- `FLAG_KEYS`: `("laser_flag", "raw_laser_flag")`, the keys left out of the medians of `point_summary`.
- `AXIS_TOLERANCE`: `1e-9`, the spread above which `scan_axis` counts a key as varying.

### bexa.io.engines.spec_h5: PAL-XFEL spec-style scan files (April 2025)

The engine `spec_h5` reads the PAL-XFEL layout of April 2025 (spec `pal_xfel_spec_2025_04`,
ue_250419_DFXM): one HDF5 per scan whose group `run/scanNNNN` carries `scanMode` and a spec-like
`scanHistory` (`a1scan`/`a2scan`), with motor positions in `motor/mN` and one summed image per
point in `det/<name>/data`. The points an aborted scan lacks are marked missing, not padded with
NaN. `bexa.open(scan_file)` replaces the reading part of `genCubepalApril2025.py`.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.io.engines.spec_h5 import SpecH5Source, parse_scan_command
from bexa.io.formats import load_spec
from bexa.testing import make_pal_spec_scan

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
one = make_pal_spec_scan(work, run=1)            # a1scan delay -1 6 8: 8 images of 20 x 24
two = make_pal_spec_scan(work, run=2, motors=(("th", 3, 0.0, 1.0), ("delay", 4, -1.0, 2.0)), partial=2)  # a SPEC-style file holding two scans
spec = load_spec("pal_xfel_spec_2025_04")       # the PAL-XFEL spec-style spec
scan = bexa.open(one.path)                       # sniffed from the scanHistory attribute
```

#### `SpecH5Source`
`bexa.io.engines.spec_h5.SpecH5Source(spec, path, scan=1, detector=None, energy_keV=None)`

The images of the group `run/scanNNNN` of `path` as frames on the command's grid (first motor
outer), coordinates from `motor/mN` (or the command's start, stop, npoints). `detector` is the
`det` subgroup; by default the first with 3-D or 4-D `data` other than the I0
(`ohqbpm2_totalsum`), which becomes the per-frame channel `i0`. `energy_keV` sets the energy (not
in the file); `KeyError` for a missing group or command. It implements the `Source` members
`files`, `frame_shape`, `n_frames`, `dtype`, `energy_keV`, `read_frames`, `structure`, `refresh`
and `close` (see `BaseSource`; the scan type is the command word, `refresh()` re-reads a growing
file). Also:

- `is_partial`: property, True when fewer images were recorded than the command asked for.
- `from_path(spec, path, scan=None, **kwargs)`: classmethod, from the file or a folder (its first `.h5`); default scan: the lowest.
- `list_scans(spec, path)`: classmethod, the sorted numbers of the `scanNNNN` groups of the file.
- attributes: `command` (a `ScanCommand`), `scan_mode`, `coords`, `detector`, `path`, `scan`.

```python
# scan: a synthetic SPEC-style scan; its command and detector; one scan of a file of two
scan.source.command.text, scan.source.detector   # ('a1scan delay -1 6 8 0.1', 'jungfrau')
part = SpecH5Source.from_path(spec, two.path)    # a2scan th 0 1 3 delay -1 2 4, 2 of 3 rows recorded
part.command.shape, part.n_frames, part.is_partial, part.structure().n_missing   # ((3, 4), 8, True, 4)
SpecH5Source.list_scans(spec, one.path)          # [1]
```

#### `parse_scan_command`
`bexa.io.engines.spec_h5.parse_scan_command(text, one_motor=("a1scan", "sscan"), two_motor=("a2scan",))`

A `ScanCommand` from a spec-like command: the first word of `one_motor` or `two_motor` counts
(earlier tokens such as a scan number are skipped), followed by one or two `name start stop
npoints` blocks, the third number counting points as the PAL files write it. `ValueError`
without a known word or for an incomplete block. The spec's `options.one_motor_commands` and
`two_motor_commands` set the word lists of the engine. The general parser, for any BLISS or
SPEC command line, is `bexa.io.scan_commands.parse_scan_command`, which also defines
`ScanCommand` and `ScanMotor` (re-exported here).

```python
# a SPEC a2scan command line, parsed
cmd = parse_scan_command("12 a2scan th 0 1 3 delay -1 2 4 0.1")
cmd.command, cmd.shape                           # ('a2scan', (3, 4))
[m.name for m in cmd.motors], cmd.motors[1]      # (['th', 'delay'], ScanMotor(name='delay', start=-1.0, stop=2.0, npoints=4))
```

### bexa.io.engines.spec_tiff: APS 7-ID SPEC sessions with a TIFF per point

The engine `spec_tiff` reads the APS 7-ID-C layout of 2025 (spec `aps_7id_spec_2025`): one SPEC
text file per session, holding every scan (`#S 4 ascan phi 8.8 9.8 40 1`, the `#P` positions,
the `#U_` beamline state, the `#L` columns and one data row per point), and one EPICS
areaDetector TIFF per point next to it, `<session>_S004_00012.tif`. The rows are the per-frame
motor and counter values, the TIFFs the frames; a point whose frame was not saved (or not copied
along) is a missing frame, a scan SPEC aborted (`#C ... Scan aborted after N points.`) reports
the points it planned, and the file is parsed once per process. `bexa.open(folder, scan=4)`,
`bexa.open(frame.tif)` and `bexa.open_dataset(folder)` all work; a session has many scans of
many kinds, so `bexa.open(folder)` without `scan=` says which to choose.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.io.engines.spec_tiff import SpecTiffSource, frame_point, frames_by_scan, read_spec_file, spec_file_of
from bexa.io.formats import load_spec
from bexa.testing import make_aps_session

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
session = make_aps_session(work)                 # a loopscan, ascans (one aborted), an a2scan, a scan without frames
spec = load_spec("aps_7id_spec_2025")           # the APS 7-ID spec
scan = bexa.open(session.path, scan=2, cache=False)   # ascan phi 8.8 9.8 8 1: 9 frames of 24 x 32
```

#### `SpecTiffSource`
`bexa.io.engines.spec_tiff.SpecTiffSource(spec, path, scan, detector=None, energy_keV=None, motors=None)`

One scan of a session as a frame stack: `path` is the SPEC file, the folder holding it or one of
its frames; `scan` the `#S` number. The dims come from the command (`ascan phi` gives `phi`;
the motors of an `a2scan` step together and give one dim named after the first, the other
staying a per-frame channel; `loopscan` gives frames only), the coordinates from the data rows
(motors by their `#o` mnemonic, `"Nano SY"` as `nsy`, counters as labelled: `r1`, `IC_1`), the
positioners of `scalars` from `#P`, the energy from the `#U_` block the spec names
(`energy: {header: mono, field: Energy}`) and `detector` names the spec's camera (its `shape`
is the frame shape of a scan without a frame yet). `motors` chooses other columns as dims
(`Scan.with_dims`). `KeyError` for a scan the file does not hold. It implements the `Source`
members `files`, `frame_shape`, `n_frames`, `dtype`, `energy_keV`, `read_frames`, `structure`,
`status`, `with_motors`, `scalars`, `aux_tables` (`header`: one row of the `#U_` state,
`<block>_<field>` columns), `cache_records` (the scan's rows of the growing file plus the
frames), `refresh` and `close` (see `BaseSource`). Also:

- `from_path(spec, path, scan=None, **kwargs)`: classmethod, from the file, its folder or a frame
  (whose name gives the scan); `LookupError` when the session holds several scans and none is named.
- `list_scans(spec, root, dataset, detector=None, require_files=True)`: classmethod, the scan
  numbers of `root/dataset` (a folder, or the file's stem), those with a frame by default.
- attributes: `spec_file`, `session` (the `SpecFile`), `record` (the `SpecScan`), `command` (a
  `ScanCommand`), `points` (the point numbers that have a frame), `frames` (their files), `scan`,
  `detector`.

```python
# scan: scan 2 of the synthetic session; its grid, status and the counters per frame
scan.dims, scan.shape, scan.structure.scan_type  # (('phi', 'y', 'x'), (9, 24, 32), 'ascan')
scan.status, scan.energy_keV, scan.structure.scalars["tth"]   # (<ScanStatus.COMPLETE: 'complete'>, 11.5, 5.85)
scan.structure.per_frame["r1"][:3]               # the r1 counter: the total of each frame
aborted = SpecTiffSource(spec, session.root, 3)  # ascan nsy ... 4 1, aborted after 3 points
aborted.status, aborted.structure().n_missing, aborted.structure().planned_frames   # (<ScanStatus.ABORTED: 'aborted'>, 2, 5)
pair = bexa.open(session.path, scan=4, cache=False)   # a2scan phi ... yaw ...: one dim, yaw per frame
pair.dims, pair.structure.per_frame["yaw"][:2]   # (('phi', 'y', 'x'), array([23.7, 24.1]))
pair.with_dims(["yaw"]).dims                      # ('yaw', 'y', 'x')
bexa.open(session.frames[2][0], cache=False).source.scan   # 2: a frame names its scan
scan.source.aux_tables()["header"][["mono_Energy", "ring_Current(mA)"]]   # DataFrame: 1 rows x 2 columns
```

#### `read_spec_file`
`bexa.io.engines.spec_tiff.read_spec_file(path)`

Parse a SPEC session file into a `SpecFile`, once per process: the parse is kept and reused
until the file's size or modification time changes (a session grows while it runs).

```python
# session.path: the synthetic SPEC file; its motors, counters and scans
parsed = read_spec_file(session.path)
parsed.motor_names[:4], parsed.label_to_name["Nano SY"]   # (['tth', 'th', 'chi', 'phi'], 'nsy')
sorted(parsed.scans), parsed.scans[3].aborted    # ([1, 2, 3, 4, 5], True)
read_spec_file(session.path) is parsed           # True: cached
```

#### `SpecFile`
`bexa.io.engines.spec_tiff.SpecFile(path, title="", motor_labels=<factory>, motor_names=<factory>, counter_labels=<factory>, counter_names=<factory>, header_fields=<factory>, scans=<factory>)`

A parsed session: `title` (`#F`), the motors' `#O` labels and `#o` mnemonics, the counters'
`#J`/`#j`, `header_fields` (the field names of every `#U0_<block>` line) and `scans`
(`{number: SpecScan}`). `label_to_name` maps a label to its mnemonic and `channel_names(scan)`
gives the per-frame channel of every data column of a scan.

```python
# parsed: the synthetic session; the column names of scan 2 as channels
parsed.channel_names(parsed.scans[2])[:5]        # ['phi', 'H', 'K', 'L', 'Epoch']
parsed.counter_labels, parsed.header_fields["mono"][:2]   # (['Seconds', 'IC_1', 'IC_2', 'Btot', 'r1'], ['Energy', 'Lambda'])
```

#### `SpecScan`
`bexa.io.engines.spec_tiff.SpecScan(number, command, date="", labels=<factory>, rows=<factory>, positions=<factory>, header=<factory>, comments=<factory>, aborted=False, lines=(0, 0))`

One scan of the file: its `number` and `command` (the `#S` line, whitespace normalised), `date`,
the `#L` `labels` and the data `rows` (`(n_rows, n_columns)`), the `positions` (`#P`, by
mnemonic), the `header` (`{block: {field: value}}` from the `#U_` lines), the `#C` `comments`,
`aborted` (SPEC's "aborted" comment) and the `lines` of the file it spans. `n_rows` is the number
of points recorded and `column(label)` one data column.

```python
# parsed.scans[2]: ascan phi 8.8 9.8 8 1 of the synthetic session
record = parsed.scans[2]
record.n_rows, record.column("Phi")[:3]          # (9, array([8.8, 8.925, 9.05]))
record.positions["phi"], record.header["mono"]["Energy"]   # (8.8, 11.5)
```

#### `spec_file_of`
`bexa.io.engines.spec_tiff.spec_file_of(path, glob="*.spec")`

The SPEC file behind a path: the file itself, the one found in a folder, or the session of a
frame (the one whose name starts the frame's). `FileNotFoundError` without one, `ValueError` when
a folder holds several.

```python
# session.root: the folder of the synthetic session, and one of its frames
spec_file_of(session.root).name, spec_file_of(session.frames[2][0]).name   # ('wse2_1_1.spec', 'wse2_1_1.spec')
```

#### `frames_by_scan`
`bexa.io.engines.spec_tiff.frames_by_scan(folder, prefix)`

`{scan: {point: path}}` of every `<prefix>_S<scan>_<point>.tif` in a folder, from one listing of
it; `frame_point(path)` gives the point number of one frame (-1 for another name).

```python
# session.root, session.name: the frames folder and the session prefix
found = frames_by_scan(session.root, session.name)
sorted(found), sorted(found[2])[:3]              # ([1, 2, 3, 4], [0, 1, 2])
```

#### `frame_point`
`bexa.io.engines.spec_tiff.frame_point(path)`

The point number in a frame's name: `<prefix>_S004_00012.tif` gives 12, another name -1.

```python
# session.frames[2][3]: the fourth frame of scan 2
frame_point(session.frames[2][3]), frame_point("notes.tif")   # (3, -1)
```

Constants:

- `FRAME_NAME`: the compiled pattern of a frame name, with the groups `prefix`, `scan`, `point`.
- `SPEC_SUFFIX`: `".spec"`.

### bexa.io.engines.array_cube: reduced cubes as sources

The engine `array_cube` (specs `bexa_cube_legacy`, `lcls_xcs_cube`, `bexa_cube`) serves reduced
data as a scan: a legacy PAL-XFEL `runN.h5`, an LCLS XCS cube triple, a `bexa.save` file or a plain
`.npy` stack. The leading dims (`laser`, `delay`, `shot`, ...) are the grid, so previews,
reductions and plots work as on raw scans. `bexa.open(cube_file)` creates one (`scan.source`).

```python
import tempfile
from pathlib import Path
import numpy as np
import bexa
from bexa.io.engines.array_cube import ArrayCubeSource
from bexa.io.formats import load_spec
from bexa.testing import make_lcls_cube, make_legacy_cube

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
legacy = make_legacy_cube(work / "run42.h5")     # laser off/on x 6 delays, 20 x 24 frames
lcls = make_lcls_cube(work / "lcls", run=296)    # Run0296_onStk.npy, _offStk.npy, _stats.csv
scan = bexa.open(legacy)                         # sniffed as bexa_cube_legacy
```

#### `ArrayCubeSource`
`bexa.io.engines.array_cube.ArrayCubeSource(spec, path, run=None, energy_keV=None)`

Serves the cube's `frames` variable (or the first with 3 or more dims) as frames, leading dims in
C order. `path` is a bexa `.h5`/`.zarr` or legacy `runN.h5` (read into memory by
`bexa.io.cube.load`), an LCLS folder or `RunNNNN_*` file (`load_lcls_cube`; `run` picks the run),
or another `.npy` stack (memory-mapped, dims `dim_0, ..., y, x`). It implements the `Source`
members `files`, `frame_shape`, `n_frames`, `dtype`, `energy_keV`, `read_frames` and `structure`
(see `BaseSource`; `spec` may be `None`): scan type `cube`, `laser` labels `off`/`on` as 0 and 1
(other labels as their index), numeric per-point series (`signal`, `i0`, `scanvar`, `phi`, ...)
spread to per-frame channels, energy from `energy_keV` or the `energy_keV` attr. Also:

- `from_path(spec, path, scan=None, **kwargs)`: classmethod used by `bexa.open`; drops `detector`, ignores `scan`, passes `run` and `energy_keV`.
- attributes: `frames_name`, `lead_dims`, `pixel_dims`, `path`, `run`.

```python
# scan: a reduced LCLS cube opened as a scan; its dims and the laser axis
scan.dims, scan.coords["laser"], "signal" in scan.structure.per_frame   # (('laser', 'delay', 'y', 'x'), array([0., 1.]), True)
ArrayCubeSource.from_path(load_spec("lcls_xcs_cube"), lcls.on_path).structure().dims   # ('laser', 'shot', 'y', 'x')
np.save(work / "stack.npy", np.ones((7, 4, 5), dtype=np.uint16))
bexa.open(work / "stack.npy", format="bexa_cube").dims                           # ('dim_0', 'y', 'x')
```

#### `ArrayCubeSource.dataset`
`bexa.io.engines.array_cube.ArrayCubeSource.dataset()`

The cube as loaded, an `xarray.Dataset` (frames, per-point series, coordinates, provenance attrs),
for cube-level analysis such as `bexa.analysis.pump_probe`.

```python
# scan: a reduced cube opened as a scan; the cube as an xarray Dataset again
cube = scan.source.dataset()
(cube["frames"].sel(laser="on") - cube["frames"].sel(laser="off")).dims     # ('delay', 'y', 'x')
```

Constants:

- `LASER_LEVELS`: `{"off": 0.0, "on": 1.0}`, the grid values of the `laser` labels.

### bexa.io.engines.smalldata: LCLS smalldata run files

The engine `smalldata` (spec `lcls_smalldata`) reads LCLS smalldata_tools files, one per run with
one row per event: scalars (`lightStatus/laser`, `ipm4/sum`, `scan/lxt`, `ebeam/photon_energy`,
...), per-event images or ROIs when the producer stored them (`<detector>/ROI_0_area`) and run sums
under `Sums/`. The images are the frames (dim `frame`, scan type `events`), the scalars their
per-frame channels. `bexa.open(run_file)` creates one (`scan.source`).

```python
import tempfile
from pathlib import Path
import h5py
import numpy as np
import bexa
from bexa.io.engines.smalldata import SmalldataSource
from bexa.io.formats import load_spec

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
path = work / "xcsl1004621_Run0043.h5"           # a tiny smalldata file: 30 events, 8 x 10 ROI
rng = np.random.default_rng(0)                  # the random generator of the synthetic file
with h5py.File(path, "w") as f:
    f["event_time"] = np.arange(30)
    f["lightStatus/laser"] = np.arange(30) % 2
    f["ipm4/sum"] = rng.uniform(0.9, 1.1, 30)
    f["scan/lxt"] = np.repeat(np.linspace(-1, 1, 5), 6)
    f["ebeam/photon_energy"] = np.full(30, 9500.0)                 # eV
    f["jungfrau1M/ROI_0_area"] = rng.random((30, 8, 10)).astype(np.float32)
    f["Sums/jungfrau1M_calib"] = f["jungfrau1M/ROI_0_area"][()].sum(axis=0)
scan = bexa.open(path)                          # the smalldata run opened as a scan
src = scan.source                                # a SmalldataSource
```

#### `SmalldataSource`
`bexa.io.engines.smalldata.SmalldataSource(spec, path, detector=None, energy_keV=None)`

The per-event images of one smalldata file as frames. `detector` names an entry of the spec's
`detectors` (`jungfrau1M`, `jungfrau4M`, `epix10k2M`, `zyla`; an unlisted name reads
`<detector>/ROI_0_area`), by default the first present; `KeyError` if none is, `ValueError` unless
the images are `(event, y, x)`. The spec's `keys` are matched against the 1-D datasets with one
value per event (the attribute `columns`, logical key to HDF5 path). It implements the `Source`
members `files`, `frame_shape`, `n_frames`, `dtype`, `chunk_frames`, `energy_keV`, `read_frames`,
`structure` and `close` (see `BaseSource`): numeric keys are per-frame channels, the varying one of
`delay`, `scanvar` is recorded in `structure().units["scan_axis"]`, and the energy is `energy_keV`
or the median `photon_energy` (eV in the file) in keV. Also:

- `from_path(spec, path, scan=None, **kwargs)`: classmethod, from the file or a folder (its first `.h5`); `scan` is ignored.

```python
# src: the engine behind a synthetic LCLS smalldata run; detector, frames, energy, columns
src.detector, src.n_frames, src.frame_shape, scan.energy_keV      # ('jungfrau1M', 30, (8, 10), 9.5)
src.columns["laser_flag"], src.columns["delay"], scan.structure.units["scan_axis"]   # ('lightStatus/laser', 'scan/lxt', 'delay')
SmalldataSource.from_path(load_spec("lcls_smalldata"), work, detector="jungfrau1M").n_frames   # 30
```

#### `SmalldataSource.sums`
`bexa.io.engines.smalldata.SmalldataSource.sums()`

Dict of the run-level summed images under `Sums/` (name to numpy array; `{}` without `Sums/`), read
at each call. It replaces the `Sums/` cell of the `Usefulthings` notebook.

```python
# src: an LCLS smalldata run; the summed images the file carries
{name: image.shape for name, image in src.sums().items()}      # {'jungfrau1M_calib': (8, 10)}
```

#### `SmalldataSource.scalars_table`
`bexa.io.engines.smalldata.SmalldataSource.scalars_table()`

Every resolved per-event scalar in a pandas DataFrame, one row per event, columns named by the
logical keys (`laser_flag`, `i0`, `scanvar`, `delay`, `photon_energy`, `event_time`, ...).

```python
# src: an LCLS smalldata run; one row per shot of its scalar channels
src.scalars_table().head(3)                     # DataFrame: 3 rows x 6 columns (laser_flag, i0, scanvar, delay, photon_energy, event_time)
```

### bexa.io.engines.extra_data: EuXFEL runs through extra_data

The engine `extra_data` (spec `euxfel_extra_data`) reads European XFEL runs with the `extra_data`
package, as the `Usefulthings` notebook did: per-train images of an area detector are the frames,
a scanned motor alias (`ssry`) gives their coordinate. It needs `extra_data` (available on the
EuXFEL computing systems): without it the module raises `ImportError` and the engine is not
registered. The spec has no fingerprint; pass `format="euxfel_extra_data"`.

```python
import numpy as np
import bexa
from bexa.io.engines.extra_data import ExtraDataSource, parse_run_id
from bexa.io.formats import load_spec

spec = load_spec("euxfel_extra_data")            # detector key, image key, scan alias ssry
```

#### `ExtraDataSource`
`bexa.io.engines.extra_data.ExtraDataSource(spec, proposal, run, detector_key=None, image_key=None, scan_alias=None, resolution=None, energy_keV=None)`

Opens the run with `extra_data.open_run(proposal=..., run=..., data="all")` and serves `image_key`
of the source `detector_key` (spec defaults `MID_EXP_IMG/CAM/ZYLA_2:daqOutput`,
`data.image.pixels`), one train per frame. `scan_alias` (default `ssry`) is read per train and
grouped into steps within `resolution` (default `8e-4`, motor units); `energy_keV` sets the energy.
It implements the `Source` members `files`, `frame_shape`, `n_frames`, `dtype`, `energy_keV`,
`read_frames` and `structure` (see `BaseSource`): the structure is the grid of alias values (scan
type `scan`, one frame per step: the first train at that value), or a `frame` sequence without an
alias or when its length differs from the images (with a warning). Also:

- `motor_values()`: the per-train alias values as a float array, or `None` (no alias, or a length mismatch).
- `from_path(spec, path, scan=None, **kwargs)`: classmethod used by `bexa.open`; `path` names the run as text (see `parse_run_id`), `detector` is dropped and `scan` ignored.
- attributes: `run_data` (the `extra_data` run), `images` (the key data), `proposal`, `run`.

```python
# a EuXFEL run opened through extra_data (not installed here: shown, not run)
src = ExtraDataSource(spec, 6832, 259)           # needs extra_data and access to the run
src.n_frames, src.frame_shape, src.motor_values()[:5]
scan = bexa.open("p6832_r0259", format="euxfel_extra_data")      # the same through from_path
```

#### `ExtraDataSource.binned_steps`
`bexa.io.engines.extra_data.ExtraDataSource.binned_steps(dark=None)`

One image per scan step minus `dark` (an image, such as the mean of a dark run), as a float64
DataArray `(<alias>, y, x)` with the step values as coordinate; `ValueError` without a scan alias.
It stands for `obtain_data` of the `Usefulthings` notebook, but each step's image is the grid's
single frame for that step (its first train), not the mean over the step's trains that the
docstring promises.

```python
# dark: the dark run averaged; the run binned per scan step after dark subtraction
dark = ExtraDataSource(spec, 6832, 258).read_frames(slice(0, 200), dtype=np.float64).mean(axis=0)
steps = src.binned_steps(dark=dark)              # DataArray (ssry, y, x)
```

#### `parse_run_id`
`bexa.io.engines.extra_data.parse_run_id(text)`

`(proposal, run)` from text such as `"6832/259"`, `"p6832:r259"` or `"p6832_r0259"`; `ValueError`
otherwise. A full EuXFEL folder path, with `raw/` between the proposal and the run
(`.../p006832/raw/r0259`), is not understood, nor a Windows path with a backslash.

```python
# a EuXFEL run id split into proposal and run
parse_run_id("p6832_r0259")                      # (6832, 259)
```

### bexa.io.generic: plain arrays, tiff stacks and `file.h5::/path`

For data with no beamline layout: `.npy`, `.npz`, a (multi-page) tiff, a folder of tiffs, or one
HDF5 dataset addressed as `"file.h5::/entry/data"`. The engine `generic` (spec `generic_stack`,
never sniffed) serves the stack as frames; a coordinates file turns it into a motor grid.
`bexa.open("file.h5::/path")` picks it by itself; for other files pass `format="generic_stack"`.

```python
import json
import tempfile
from pathlib import Path
import h5py
import numpy as np
import bexa
from bexa.io import generic

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
stack = np.random.default_rng(0).random((6, 4, 5)).astype(np.float32)    # 6 frames of 4 x 5
np.save(work / "stack.npy", stack)
with h5py.File(work / "stack.h5", "w") as f:
    f["entry/data"] = stack
(work / "coords.json").write_text(json.dumps({"mu": [0, 0, 0, 1, 1, 1], "chi": [0, 1, 2, 0, 1, 2]}))
```

#### `GenericStackSource`
`bexa.io.generic.GenericStackSource(spec, path, coords=None, energy_keV=None)`

Serves the array of `load_array(path)` as frames (2-D is one frame, more than 3 dims are flattened
to frames), in memory or, for `.npy`, memory-mapped. `coords`, a coordinates file (see
`read_coords`) or a dict of per-frame values, gives the grid: the motor that changes least often is
the slow dim, frames are placed by value, a repeated position keeps its first frame. Without it the
dims are `(frame, y, x)`. It implements the `Source` members `files`, `frame_shape`, `n_frames`,
`dtype`, `energy_keV`, `read_frames` and `structure` (see `BaseSource`; `spec` may be `None`).
Also:

- `from_path(spec, path, scan=None, **kwargs)`: classmethod used by `bexa.open`; drops `detector`, ignores `scan`, passes `coords` and `energy_keV`.

```python
# a plain .npy stack with a coords.json beside it, opened as a scan
scan = bexa.open(work / "stack.npy", format="generic_stack", coords=work / "coords.json")
scan.dims, scan.shape                            # (('mu', 'chi', 'y', 'x'), (2, 3, 4, 5))
src = generic.GenericStackSource(None, f"{work / 'stack.h5'}::/entry/data", coords={"z": np.arange(6.0)})
src.structure().dims, src.files()[0].name        # (('z', 'y', 'x'), 'stack.h5')
generic.GenericStackSource.from_path(None, work / "stack.npy").n_frames      # 6
```

#### `load_array`
`bexa.io.generic.load_array(source, mmap=True)`

Loads one array: `"file.h5::/path"` (into memory), an `.h5`/`.hdf5`/`.nxs` without a path (its
largest dataset of 2 or more dims), `.npy` (memory-mapped read-only when `mmap`), `.npz` (its first
array), a tiff or multi-page tiff (tifffile, or Pillow without it), or a folder of tiffs (stacked in
name order). `ValueError` for other suffixes.

```python
# an array from an HDF5 dataset (file::path) or a .npy file
generic.load_array(f"{work / 'stack.h5'}::/entry/data").shape     # (6, 4, 5)
type(generic.load_array(work / "stack.npy")).__name__             # 'memmap'
```

#### `read_coords`
`bexa.io.generic.read_coords(path)`

Per-frame motor values from a `.json` (name to list), a `.npy` (1-D: motor `axis`; 2-D `(n, k)`:
`axis0` to `axis{k-1}`) or else a CSV with a header row, as a dict of float arrays.

```python
# the motor coordinates of a generic stack, from a JSON file
generic.read_coords(work / "coords.json")["chi"]      # array([0., 1., 2., 0., 1., 2.])
```

#### `split_h5_url`
`bexa.io.generic.split_h5_url(text)`

Splits `"file.h5::/entry/data"` into `(Path("file.h5"), "/entry/data")`, dropping a silx-style
`path=` prefix and turning backslashes into slashes; without `::` the dataset path is `None`.

```python
# a file::path URL split into the file and the dataset path
generic.split_h5_url("scan.h5::/entry/data")[1], generic.split_h5_url("stack.npy")[1]   # ('/entry/data', None)
```

Constants:

- `TIFF_SUFFIXES`: `(".tif", ".tiff")` and `H5_SUFFIXES`: `(".h5", ".hdf5", ".nxs")`, what `load_array` reads as tiff and HDF5.
- `URL_SEPARATOR`: `"::"`, between the file and the dataset path.

### bexa.io.multi: several scans as one source

`MultiScanSource` concatenates sources with identical grids along a new leading dim: an energy
series, layers on `samz`, or scans by number. `bexa.open(dataset, scan=(20, 40))`, a list of scans,
or `scan=None` on a dataset with several scans creates one (`bexa.core.scan.open_source`): the dim
is `energy` when every scan has its own energy, else `scan`, or the positioner named by
`stack_dim=`. `ds.series(...)` does the same for a `Dataset`.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.io.multi import MultiScanSource
from bexa.testing import make_esrf_energy_series

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
made = make_esrf_energy_series(work, motors=(("chi", 3), ("mu", 8)), n_files=1)   # scans 1-3, 17.00-17.10 keV
series = bexa.open(made[0].dataset_dir, scan=(1, 3))  # three scans of one dataset opened as one series
```

#### `MultiScanSource`
`bexa.io.multi.MultiScanSource(parts, dim, coords)`

Stacks the sources `parts` on a new leading dim `dim`, with one value of `coords` per part
(`ValueError` for no parts or a length mismatch). Frame ids are global: the frames of part `i`
follow those of part `i - 1`. Grids and frame shapes must match (`structure()` raises `ValueError`
otherwise); per-frame channels present in every part are concatenated, and the scalars, units,
energy and title are those of the first part. It implements the `Source` members
`files`, `frame_shape`, `n_frames`, `dtype`, `chunk_frames`, `read_frames`, `structure`,
`status`, `scalars`, `cache_records`, `refresh` and `close` (see `BaseSource`): `read_frames`
reads the parts in turn, `refresh()` and `close()` go to every part, `files()` lists each file
once, `cache_records()` gives every part's records once plus one for the dim and its
coordinates, `status` is the worst of the parts' (`bexa.core.status.worst`), and `frame_shape`,
`dtype`, `chunk_frames` and `scalars()` are the first part's. Attributes: `parts`, `dim`,
`coords`.

```python
# series: several scans opened as one; the source that stacks them
src = series.source
src.dim, src.coords, series.dims                 # ('energy', array([17.  , 17.05, 17.1 ]), ('energy', 'chi', 'mu', 'y', 'x'))
src.status                                       # <ScanStatus.COMPLETE: 'complete'>: every part is
src.n_frames, [p.scan for p in src.parts], len(src.files())      # (72, [1, 2, 3], 4)
len(src.cache_records()), src.cache_records()[-1]   # (7, {'path': '', 'dim': 'energy', 'coords': [17.0, 17.05, 17.1]})
by_scan = MultiScanSource([bexa.open(made[0].dataset_dir, scan=s).source for s in (1, 2)], "scan", [1, 2])
by_scan.structure().motor_dims, by_scan.structure().motor_shape  # (('scan', 'chi', 'mu'), (2, 3, 8))
```

### bexa.io.inspect: HDF5 trees and path summaries

Quick looks inside files, as `bexa info PATH` prints them.

```python
import tempfile
from pathlib import Path
from bexa.io.inspect import describe, h5_tree, print_h5_structure
from bexa.testing import make_esrf_scan

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
made = make_esrf_scan(work, dataset="synth_dfxm")      # a master file and 2 compressed detector files
```

#### `h5_tree`
`bexa.io.inspect.h5_tree(path, max_depth=None, attrs=False, max_items=2000)`

One line per group (ending in `/`) and dataset of an HDF5 file, indented by depth, after a line
with the file name; datasets show shape, dtype, chunks, compression filters (`gzip`, `bitshuffle`,
`lz4`, `blosc`, `zfp`, `none`) and compression ratio. `max_depth` leaves out deeper levels (0 is
the top), `attrs=True` adds `@name = value` lines, and the list stops at `max_items` lines.

```python
# made: a synthetic scan; the tree of one of its detector files, as text lines
h5_tree(made.detector_files[0])[4].strip()       # 'data  (24, 48, 64) uint16 chunks (1, 48, 64) [bitshuffle] ratio 3.4x'
len(h5_tree(made.master, max_depth=1))          # 9
```

#### `print_h5_structure`
`bexa.io.inspect.print_h5_structure(path, **kwargs)`

Prints `h5_tree(path, **kwargs)`: the `print_h5_structure` helper of the lab's `scratch.md`, kept
under its name, now with shapes, chunks and compression.

```python
# made: a synthetic scan; the groups of its master, one level deep
print_h5_structure(made.master, max_depth=1)     # prints the groups of the master, one level deep
```

#### `describe`
`bexa.io.inspect.describe(path, max_depth=3)`

A text summary of a path, the first part of `bexa info PATH`: the matching spec and engine (or the
closest scores), then a folder's first 40 entries with sizes in MB, an HDF5 file's `h5_tree` down
to `max_depth`, a `.npy`'s shape, dtype and size in GB (header only), or the file size.
`FileNotFoundError` for a missing path.

```python
# made: a synthetic scan; what bexa sees in its folder, as text
print(describe(made.dataset_dir))                # format: esrf_id03_bliss_2026 (engine hdf5_stack), ...
describe(made.master, max_depth=1).splitlines()[0]  # 'path: C:\\Users\\sport\\AppData\\Local\\Temp\\claude\\x--Dressel...'
```

### bexa.io.cube: bexa files, `bexa.save` and `bexa.load`

bexa's format for reduced data (HDF5 or zarr with coordinates and provenance) and the readers of
legacy reduced cubes. `bexa.save` and `bexa.load` are this module's `save` and `load`;
`res.bexa.save(path)` is the accessor form. `open_lazy` opens a bexa HDF5 file without loading
it, and `StoreWriter` builds one piece by piece (what `bexa.stack(..., store=True)` does). Old
and new pump-probe cubes look the same in memory: `frames(laser, <axis>, y, x)` with `laser` =
`["off", "on"]`, and per-point series.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.io import cube as cube_io
from bexa.testing import make_lcls_cube, make_legacy_cube

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
scan = bexa.demo("rocking", root=work)           # a synthetic mu scan
res = scan.com(axes=("mu",))                     # Dataset: com_mu, width_mu, total
legacy = make_legacy_cube(work / "run42.h5")     # a legacy PAL-XFEL cube
lcls = make_lcls_cube(work / "lcls", run=296)    # an LCLS XCS cube triple
```

#### `save`
`bexa.io.cube.save(obj, path, compression="gzip", overwrite=True)`

Writes a DataArray, a Dataset or a dict of DataArrays to `path`: zarr for `.zarr`, HDF5 otherwise
(use `.h5` so that `load` recognises it). Variables keep dims, coordinates and attrs (dicts and
lists as JSON); the file attrs get the provenance of `bexa.core.provenance.build_attrs` (bexa
version, git hash, date, Python, source files, parameters) unless `bexa_version` is set. In HDF5,
arrays of 2 or more dims over 1024 values are compressed in one chunk per frame: `compression` is
`"gzip"` (level 4; every HDF5 reader has it, so `save` keeps it as its default), `"lzf"` (with
byte shuffle, which floats need to compress with lzf at all; what the cache and
`bexa.stack(..., store=)` write: one scan's results, a 116 MB preview and 101 MB of maps, take
0.7 s to write against 6.9 s with gzip and 0.3 s to read against 0.9 s) or `None` for none.
NeXus `NXdata` groups let h5web and silx plot the file. An attr over 60 KB (`MAX_ATTR_BYTES`; the per-frame statistics of a large stack) is dropped, as
HDF5 keeps attributes in the object header. GPU arrays are copied to the host;
`overwrite=False` raises `FileExistsError` for an existing file. Returns the path. `bexa.save`
is this function.

```python
# res: the COM maps of the demo scan; written to one bexa HDF5 file, then as zarr
path = bexa.save(res, work / "scan_com.h5")
bexa.save({"com": res["com_mu"], "sum": res["total"]}, work / "maps.zarr")  # Path('.../maps.zarr')
```

#### `load`
`bexa.io.cube.load(path, squeeze_single=True)`

Reads back what `save` wrote (`.h5`, `.hdf5`, `.zarr`) with dims, coordinates, attrs and
provenance, a legacy PAL-XFEL `runN.h5` (`load_legacy_cube`), or an LCLS cube triple given as its
folder, a `.npy` or its `_stats.csv` (`load_lcls_cube`), all into memory. Returns an
`xarray.Dataset`, or a DataArray (the file attrs merged into its own) when the file holds one data
variable and `squeeze_single` is true. A bexa file larger than the memory budget is refused with
a `MemoryError` that points to `open_lazy`. `bexa.load` is this function; it replaces reading
`runN.h5` with h5py by hand.

```python
# path: the file save wrote; the maps read back with their provenance
back = bexa.load(path)
sorted(back.data_vars), back["com_mu"].dims       # (['com_mu', 'total', 'width_mu'], ('y', 'x'))
bexa.load(work / "maps.zarr")["com"].dims          # ('y', 'x')
bexa.load(legacy)["frames"].dims                   # ('laser', 'delay', 'y', 'x')
```

#### `open_lazy`
`bexa.io.cube.open_lazy(path)`

Open a bexa HDF5 file (written by `save` or a `StoreWriter`) as an `xarray.Dataset` whose
variables are read on demand: coordinates and attrs are loaded, but `ds[name].isel(...)`, `.sel`
and `.transpose` read nothing and `.values` reads only the selected block (the file is opened
per read, so nothing stays locked). Anything that needs the whole array (a sum over everything,
`save`) loads it, so keep to blocks. The attrs get `lazy_file`, the path. This is how a volume
stacked with `store=` is browsed without fitting in memory; `bexa.core.stacks.is_lazy` tells a
lazy array from a loaded one. `ValueError` for a file that is not a bexa file.

```python
# path: a bexa file; its arrays read from disk only when sliced
lazy = cube_io.open_lazy(path)                     # nothing read yet
lazy["com_mu"].isel(y=slice(0, 8)).values.shape    # (8, 80): only that block is read
lazy.attrs["lazy_file"]                            # the path
```

#### `StoreWriter`
`bexa.io.cube.StoreWriter(path, attrs=None, compression="gzip")`

Write a bexa HDF5 file piece by piece, for results built scan by scan. Declare every variable
with its full shape, write the coordinates, then put blocks into position and close; the file
is built under a temporary name (`<stem>.partial.h5`) and appears under `path` only when it is
complete. `attrs` are the file attrs (provenance is added unless `bexa_version` is in them);
`compression` applies to variables of two or more dims, as in `save` (`bexa.stack` passes
`"lzf"`, `bexa.core.dataset.STORE_COMPRESSION`, which writes about ten times faster than gzip).
The file reads back with `load` or `open_lazy`. Members:

- `add_variable(name, dims, shape, dtype, attrs=None, compression=True)`: declare a variable; a float one starts as NaN (`compression=False` or another name overrides the file's).
- `add_coord(name, values, dims=None, attrs=None)`: write a coordinate, 1-D along its own dim unless `dims` says otherwise (text becomes bytes).
- `write(name, index, block)`: put `block` at `index` (one int or slice per leading dim) of a declared variable.
- `close()`: write the variable list and the NeXus links, close, rename to `path`; returns the path.
- `abort()`: close and delete the partial file after a failure.
- attribute `path`.

```python
# a file built layer by layer, one array at a time, for stacks too large for memory
import numpy as np
writer = cube_io.StoreWriter(work / "layers.h5", attrs={"note": "built layer by layer"})
writer.add_coord("samz", [0.0, 0.001, 0.002], attrs={"units": "mm"})
writer.add_variable("sum", ("samz", "y", "x"), (3, 64, 80), np.float32)
for i in range(3):
    writer.write("sum", (i,), np.full((64, 80), float(i)))   # one block at a time; unwritten blocks stay NaN
writer.close()                                             # the file appears under its final name
cube_io.open_lazy(work / "layers.h5")["sum"].isel(samz=2).values.mean()   # 2.0
```

#### `load_legacy_cube`
`bexa.io.cube.load_legacy_cube(path, axis=None)`

Reads a `runN.h5` of the legacy PAL-XFEL cube builder (`Aaron_allscan_cube_parallel.py`,
`Scan_Combiner.py`): `jungfrau_on`/`_off` become `frames(laser, <axis>, y, x)`, `signals_on`/`_off`
(mean ROI/I0 per point) `signal(laser, <axis>)`. The axis is `axis` or the first of `delays`,
`phi`, `chi`, `th`, `tth`, `laser_v`, `laser_h`, `index` that varies (`delays` gives the dim
`delay`); the other axes stay as coordinates. Attrs: provenance and `scan_axis`. `ValueError`
without `jungfrau_on`.

```python
# legacy: a pump-probe cube written by the old XFEL scripts; read as a Dataset
cube = cube_io.load_legacy_cube(legacy)
cube["frames"].dims, cube.attrs["scan_axis"]      # (('laser', 'delay', 'y', 'x'), 'delay')
cube_io.load_legacy_cube(legacy, axis="phi")["signal"].dims                  # ('laser', 'phi')
```

#### `load_lcls_cube`
`bexa.io.cube.load_lcls_cube(path, run=None)`

Reads an LCLS XCS triple, `RunNNNN_onStk.npy`, `_offStk.npy`, `_stats.csv` (shot, scan variable,
on I0, off I0 after a header row), from its folder or one of its files: `frames(laser, shot, y, x)`
with the per-shot coordinate `scanvar`, and `i0(laser, shot)`. `run` defaults to that of the file
given or the first of the folder. Attrs: provenance and `scan_axis = "scanvar"`.
`bexa.analysis.pump_probe.bin_lcls` bins it.

```python
# lcls: a synthetic LCLS cube folder; the shots read as a Dataset
shots = cube_io.load_lcls_cube(lcls.root)
shots["frames"].dims, shots["i0"].dims           # (('laser', 'shot', 'y', 'x'), ('laser', 'shot'))
```

#### `is_bexa_file`
`bexa.io.cube.is_bexa_file(path)`

True for an `.h5` or `.hdf5` file whose root has the `bexa_format` attribute that `save` writes,
False otherwise (unreadable files included).

```python
# path: a bexa file; legacy: an old cube file
cube_io.is_bexa_file(path), cube_io.is_bexa_file(legacy)      # (True, False)
```

Constants:

- `FORMAT_MARKER`: `"bexa_format"`, the root attribute of a bexa HDF5 file; `FORMAT_VERSION`: `"cube-v1"`, its value.
- `MAX_ATTR_BYTES`: `60_000`, the largest attr `save` and `StoreWriter` write.
- `LEGACY_AXES`: the axes looked for in a legacy cube, in priority order.
- `LASER_COORD`: `np.array(["off", "on"])`, the `laser` coordinate of loaded cubes.

### bexa.io.darfix: the bridge to darfix

darfix (ESRF's DFXM package) reads a scan through silx data URLs (`file.h5::/path`). These functions
build them for a bexa scan of the ESRF engine (`hdf5_stack`), create the darfix `Dataset`, and
convert moment maps, replacing the `Converting darfix` notebook. Only `to_darfix_dataset` needs
darfix.

```python
import numpy as np
import bexa
from bexa.io import darfix

scan = bexa.demo("rocking")                      # a mu scan in the ESRF BLISS layout
```

#### `data_urls`
`bexa.io.darfix.data_urls(scan)`

`"file.h5::<frames path>"` for every detector file of the scan (master left out), in order; darfix
is seeded with the first. `ValueError` for a source without `frames_path` (meant for `hdf5_stack`).

```python
# scan: the demo mu scan; the HDF5 URLs darfix reads its frames from
darfix.data_urls(scan)[0].split("::")[1]         # 'entry_0000/ESRF-ID03/pco_ff/data'
```

#### `metadata_url`
`bexa.io.darfix.metadata_url(scan)`

`"master.h5::/N.1/instrument/positioners"` of the scan, where darfix reads the motor positions.
`ValueError` for a source without a master file.

```python
# scan: the demo mu scan; the HDF5 URL darfix reads the motor positions from
darfix.metadata_url(scan).split("::")[1]         # '/1.1/instrument/positioners'
```

#### `to_darfix_dataset`
`bexa.io.darfix.to_darfix_dataset(scan, work_dir, in_memory=True)`

A darfix `Dataset` (`darfix.core.dataset.Dataset`) seeded with `data_urls(scan)[0]` and
`metadata_url(scan)` (`isH5=True`), darfix's object for its own processing; darfix writes its
working files to `work_dir`. Needs darfix (`pip install darfix`); `ImportError` without it.

```python
# scan: the demo mu scan; a darfix Dataset built from it (needs darfix: shown, not run)
dfx = darfix.to_darfix_dataset(scan, "darfix_work")      # needs darfix
```

#### `compute_moments`
`bexa.io.darfix.compute_moments(values, frames)`

`(com, std, skew, kurtosis)` maps of a stack `frames` `(n, y, x)` along the motor `values`, as
darfix's `compute_moments` gives them, computed without darfix by `bexa.analysis.rocking.moments`.
Four `(y, x)` float arrays; com and std in motor units, kurtosis the excess kurtosis.

```python
# scan: the demo mu scan; the darfix moments of every pixel's rocking curve
com, std, skew, kurt = darfix.compute_moments(scan.coords["mu"], scan.read().values)
com.shape                                        # (64, 80)
```

#### `moments_dataset`
`bexa.io.darfix.moments_dataset(moments, dims=None)`

darfix's `apply_moments` output, a dict of motor to a `(4, y, x)` array (com, std, skew, kurtosis),
as an `xarray.Dataset` of `com_<motor>`, `std_<motor>`, `skew_<motor>`, `kurtosis_<motor>` on
`(y, x)`, bexa's own names, with provenance attrs. A list of arrays is labelled with `dims`
(default `dim_0`, ...). `ValueError` for a block that is not `(4, y, x)`.

```python
# com, std, skew, kurt: the darfix moments; as a bexa Dataset of maps
maps = darfix.moments_dataset({"mu": np.stack([com, std, skew, kurt])})
sorted(maps.data_vars)                           # ['com_mu', 'kurtosis_mu', 'skew_mu', 'std_mu']
```

Constants:

- `MOMENT_NAMES`: `("com", "std", "skew", "kurtosis")`, the order of the four planes per motor.

### bexa.analysis.preprocess: darks, hot pixels, normalisation, cropping, binning

Every function in `bexa.analysis` takes numpy or cupy arrays and returns the same kind
(`bexa.core.backend.array_module(x)` picks the module), so a stack moved to the GPU with
`bexa.core.backend.to_device(frames, "cuda")` stays there until `to_host`. The entries say where a
function differs: it returns numpy, or needs scipy, lmfit or scikit-learn. For scans too large for
memory, `scan.reduce` and its accumulators do the same jobs while streaming.

This module cleans a frame stack `(n, y, x)` before any statistics; it reads no files.

```python
import numpy as np
import bexa
from bexa.analysis import preprocess

scan = bexa.demo("rocking")                  # synthetic mu scan, 25 frames of 64 x 80 pixels
frames = scan.read().values                  # numpy (25, 64, 80), float32
dark = bexa.demo("rocking", amplitude=0.0, noise=2.0).read().values   # a dark scan, no peak
```

#### `subtract_dark`
`bexa.analysis.preprocess.subtract_dark(frames, dark, mode="mean")`

Returns `frames - dark`. `dark` is one image `(y, x)` or a dark stack with as many dims as
`frames`, first reduced to one image by its mean (`mode="mean"`) or median (`"median"`, or any
other value). The dark step of the `basicDFXM` notebook.

```python
# frames: the demo mu scan in memory; dark: a dark frame; the dark taken off every frame
clean = preprocess.subtract_dark(frames, dark)          # or mode="median"
```

#### `remove_hot_pixels`
`bexa.analysis.preprocess.remove_hot_pixels(frames, size=(1, 3, 3))`

Median filter of `size` pixels per axis (`scipy.ndimage`, `cupyx.scipy.ndimage` on the GPU). The
default filters each frame separately over 3 x 3 pixels, so an isolated hot or dead pixel takes
the median of its neighbours; a single image needs `size=3`. From `basicDFXM`.

```python
# hot: a copy of the frames with one pixel saturated; the hot pixel replaced by its neighbours
hot = frames.copy()
hot[5, 20, 30] = 65535                                  # one saturated pixel
float(preprocess.remove_hot_pixels(hot)[5, 20, 30])     # 681.0
```

#### `normalize_i0`
`bexa.analysis.preprocess.normalize_i0(frames, i0)`

Divides frame `k` of an `(n, y, x)` stack by `i0[k]`, one monitor value per frame; frames whose
I0 is 0 become NaN instead of infinite.

```python
# frames: the demo mu scan; every frame divided by its monitor reading
normed = preprocess.normalize_i0(frames, np.linspace(1.0, 0.9, 25))   # e.g. a decaying beam
```

#### `box_mean`
`bexa.analysis.preprocess.box_mean(frames, box, robust=False)`

Mean of a pixel box in every frame, float64 `(n,)`: the corner background of the legacy scripts.
`box` is a `bexa.ROI`, `((y0, y1), (x0, x1))` or `{"y": (y0, y1), "x": (x0, x1)}`, ranges as in
Python slices. `robust=True` first replaces the box values above its 90th percentile by the
median, then those below the 10th, as the legacy laser-off animation did.

```python
# frames: the demo mu scan; the mean of a background box in every frame
preprocess.box_mean(frames, ((0, 8), (0, 8)))[:3]      # [258.86, 457.22, 724.3]
preprocess.box_mean(frames, bexa.ROI(y=(0, 8), x=(0, 8)), robust=True)[:3]  # array([256.79, 454.74, 721.9])
```

#### `normalize_box`
`bexa.analysis.preprocess.normalize_box(frames, box, robust=False)`

Divides every frame by its `box_mean` (gain normalisation), so the box averages 1 in every frame.

```python
# frames: the demo mu scan; every frame divided by its background box
normed = preprocess.normalize_box(frames, ((0, 8), (0, 8)))
preprocess.normalize_box(frames, bexa.ROI(y=(0, 8), x=(0, 8)), robust=True).shape  # (25, 64, 80)
```

#### `crop`
`bexa.analysis.preprocess.crop(frames, roi)`

Cuts the last two axes to a pixel window (any form `box_mean` accepts; motor ranges of a
`bexa.ROI` are ignored) and returns a view.

```python
# frames: the demo mu scan; a pixel window cut out of every frame
preprocess.crop(frames, bexa.ROI(y=(16, 48), x=(20, 60))).shape         # (25, 32, 40)
```

#### `bin_spatial`
`bexa.analysis.preprocess.bin_spatial(frames, factors, method="mean")`

Block-reduces the last two axes by integer factors (one int, or `(fy, fx)`) with `"mean"`,
`"sum"` or `"max"`, dropping rows and columns that do not fill a block: the `downsample` of
`scan.preview` (`bexa.core.reductions.block_reduce`).

```python
# frames: the demo mu scan; the pixels binned 2 x 2, then 4 x 2 by sum
preprocess.bin_spatial(frames, 2).shape                        # (25, 32, 40)
preprocess.bin_spatial(frames, (4, 2), method="sum").shape     # (25, 16, 40)
```

#### `clip_nonpositive`
`bexa.analysis.preprocess.clip_nonpositive(frames, eps=0.0001, copy=True)`

Replaces every value `<= 0` by `eps`, the `raw[raw <= 0] = 1e-4` of the legacy PAL-XFEL cube
builder (dead Jungfrau pixels read negative), so logarithms and ratios stay finite. Works on a
copy unless `copy=False`; in an integer array `eps` rounds to 0.

```python
# clean: dark-subtracted frames with negative pixels; made strictly positive for a log
positive = preprocess.clip_nonpositive(clean)
round(float(clean.min()), 2), round(float(positive.min()), 6)    # (-1.52, 0.0001)
```

#### `threshold_floor`
`bexa.analysis.preprocess.threshold_floor(frames, percentile=1.0)`

Raises every value below the `percentile`-th percentile of the whole stack to that level.

```python
# frames: the demo mu scan; everything below the 5th percentile set to it
floored = preprocess.threshold_floor(frames, percentile=5)
float(preprocess.threshold_floor(frames).min())  # 10.0
```

#### `log_scale`
`bexa.analysis.preprocess.log_scale(frames)`

`log10(1 + I)` with negative values clipped to 0: the scaling of the v9 previews and of
`scan.preview(apply_log=True)`.

```python
# frames: the demo mu scan; log10(1 + I)
round(float(preprocess.log_scale(frames).max()), 2)    # 3.6
```

### bexa.analysis.rocking: per-pixel rocking-curve statistics

Maps from a volume in memory whose first axis (or `axis`) is a motor, plus the motor values:
centre of mass, widths and moments, argmax, weighted median. The in-memory counterparts of
`scan.com` and `bexa.acc.MotorCOM`; they replace the per-pixel loops of the `betterCOM*` notebooks.

```python
import numpy as np
import bexa
from bexa.analysis import rocking

scan = bexa.demo("rocking")          # synthetic mu scan, 25 frames of 64 x 80 pixels
vol = scan.read().values             # numpy (mu, y, x) = (25, 64, 80)
mu = scan.coords["mu"]               # motor values in deg, one per frame
```

#### `motor_com`
`bexa.analysis.rocking.motor_com(volume, values, axis=0, sigma=0.0, clip=1e-10)`

Intensity-weighted mean motor position per pixel, `sum(I m) / sum(I)` along `axis`, in the units
of `values`: the DFXM centre-of-mass map. `sigma` first smooths the frames with a Gaussian of that
many pixels (v9 used 3) and `clip` floors the weights so dark pixels do not divide by zero (`None`
leaves them). On `(mu, y, x)` it equals `com_mu` of `scan.com(axes=("mu",), sigma=...)`; on a
volume with two motor axes `sigma` smooths the pixels only, never along the other motor.

```python
# vol: the demo mu scan in memory; mu: its motor values; the centre of mass of every pixel
com = rocking.motor_com(vol, mu, sigma=1.0)       # (64, 80) map in deg
round(float(com[32, 40]), 4)                      # 0.0075
```

#### `energy_com`
`bexa.analysis.rocking.energy_com(volume, values, axis=0, sigma=0.0, clip=1e-10)`

`motor_com` under a second name, for the per-pixel energy centre of mass (keV) of an energy series.

```python
# series: an energy series; the energy centre of mass of every pixel from its summed images
series = bexa.demo("energy")                      # dims (energy, chi, mu, y, x)
e_com = rocking.energy_com(series.read().sum(("chi", "mu")).values, series.coords["energy"])
```

#### `energy_com_from_stack`
`bexa.analysis.rocking.energy_com_from_stack(sums, dim="energy", clip=1e-10)`

The energy centre of mass per pixel from summed images stacked on an energy dim: `sums` is a
DataArray with an `energy` coordinate in keV, the `sum` of `bexa.stack(scans, [bexa.acc.Sum()],
dim="auto")` over an energy series. `com_energy = sum(I E) / sum(I)` and `width_energy` (the
standard deviation) are what `bexa.acc.EnergyCOM` computes with `sigma=0`, without opening the
scans again as a series; other dims (a height) are kept. Returns an `xarray.Dataset` with
`com_energy`, `width_energy` (keV) and `total`; pixels whose total is below `clip` are NaN.
A stack larger than `ENERGY_COM_BLOCK_BYTES` (256 MiB) is processed one slice of its first
other dim at a time, so a lazy stack on disk (`bexa.stack(..., store=True)`) is read block by
block and the float64 temporaries stay the size of one slice. `ValueError` when `dim` is not a
dim of `sums`.

```python
# sums: the summed images of an energy series stacked on energy; the same from a stack
ds = bexa.demo("zstack")                                   # five mu scans at 16.98 to 17.02 keV per height
sums = bexa.stack(ds.select(type="fscan1d"), [bexa.acc.Sum()], dim="auto")["sum"]   # (samz, energy, y, x)
e = rocking.energy_com_from_stack(sums)                    # com_energy, width_energy, total: dims (samz, y, x)
strain = float(sums.coords["energy"].mean()) / e["com_energy"] - 1
```

#### `moments`
`bexa.analysis.rocking.moments(volume, values, axis=0, sigma=0.0, clip=1e-10)`

A dict of maps as darfix computes them: `com`, `std` (equal to `width_mu` of `scan.com`), `fwhm`
(`2.3548 std`, `bexa.core.units.FWHM_PER_SIGMA`), `skew`, `kurtosis` (excess, 0 for a Gaussian)
and `total` (sum of the weights). `sigma` and `clip` as for `motor_com`.

```python
# vol, mu: the demo mu scan and its motor; width, skew and kurtosis of every pixel
stats = rocking.moments(vol, mu, sigma=1.0)
round(float(stats["fwhm"][32, 40]), 3), round(float(stats["kurtosis"][32, 40]), 2)   # (0.579, 0.27)
```

#### `argmax_motor`
`bexa.analysis.rocking.argmax_motor(volume, values, axis=0)`

Motor value of the brightest frame per pixel: the peak position on the scan grid, insensitive to
tails.

```python
# vol, mu: the demo mu scan and its motor; the mu at which every pixel is brightest
peak_mu = rocking.argmax_motor(vol, mu)
peak_mu.shape, float(peak_mu[32, 40])           # ((64, 80), 0.0)
```

#### `weighted_median`
`bexa.analysis.rocking.weighted_median(values, weights, axis=0)`

Per-pixel weighted median of the motor values (`weighted_quantile` with `q=0.5`), less sensitive
than `motor_com` to a second peak or a sloping background. Motor values come first here.

```python
# vol, mu: the demo mu scan and its motor; the median mu of every pixel's curve
med = rocking.weighted_median(mu, vol)
med.shape, round(float(med[32, 40]), 4)         # ((64, 80), 0.0)
```

#### `weighted_quantile`
`bexa.analysis.rocking.weighted_quantile(values, weights, q=0.5, axis=0)`

Per-pixel weighted quantile: the first motor value, in increasing order, at which the cumulative
weight reaches `q` of the total. Vectorises the `Yue idea` notebook's `weighted_median` loop, whose
cutoff `total / 1.5` is `q=2/3`.

```python
# vol, mu: the demo mu scan and its motor; the upper tercile of every pixel's curve
upper = rocking.weighted_quantile(mu, vol, q=2 / 3)     # the notebook's "weighted median"
float(rocking.weighted_quantile(mu, vol, q=0.5)[32, 40]) == float(med[32, 40])  # True
```

#### `stat_images`
`bexa.analysis.rocking.stat_images(stack, values)`

The `Topography` notebook's `stat_images`: a dict with the `mean` and `std` images over the first
axis and `com`, per pixel the motor value of the frame whose intensity is closest to that pixel's
mean (not a centre of mass).

```python
# vol, mu: the demo mu scan and its motor; every statistic map at once
images = rocking.stat_images(vol, mu)                   # keys com, mean, std
```

#### `rocking_curve_stats`
`bexa.analysis.rocking.rocking_curve_stats(stack, qmin=0.0, qmax=100.0)`

The `Topography` notebook's `rocking_curves`, one value per frame: `sum`, `min` and `max` (the
`qmin`-th and `qmax`-th percentiles of the frame, the true extremes by default) and `mean`, the
mean of the pixels between those percentiles.

```python
# vol: the demo mu scan in memory; the rocking curve of the whole frame and of its brightest pixels
curves = rocking.rocking_curve_stats(vol, qmin=5, qmax=95)
float(mu[np.argmax(curves["sum"])])                     # 0.0: the brightest frame
```

#### `combine_axes`
`bexa.analysis.rocking.combine_axes(mu, phi, sign=-1.0)`

The combined tilt `mu + sign * phi`, which puts mu scans taken at different phi on one axis before
stitching. Always a numpy float array.

```python
# mu: the motor values of the demo scan; combined with a second axis into one
rocking.combine_axes(mu, 0.2)[:3]                       # [-1.2, -1.1167, -1.0333]
rocking.combine_axes(mu, 0.2, sign=1.0)[:3]     # array([-0.8, -0.71667, -0.63333])
```

Constants:

- `ENERGY_COM_BLOCK_BYTES`: `256 * 2**20`, the size of a stack above which
  `energy_com_from_stack` works one outer slice at a time.

### bexa.analysis.peaks: the peak on the detector, frame by frame

Where the Bragg peak sits on the detector in every frame of a stack `(n, y, x)`, and its width and
skew: centre of mass, raw central moments, 2-D Gaussian fits. Used on pump-probe cubes (one frame
per delay) and on scans; replaces the per-frame loops of the LCLS notebooks.

```python
import numpy as np
import bexa
from bexa.analysis import peaks

scan = bexa.demo("rocking")                  # synthetic mu scan: the bright band moves along x with mu
frames = scan.read().values                  # (25, 64, 80)
roi = bexa.ROI(y=(16, 48), x=(10, 70))       # pixel window around the peak
```

#### `frame_com`
`bexa.analysis.peaks.frame_com(frames, roi=None)`

Detector centre of mass `(com_y, com_x)` of every frame, two `(n,)` arrays in pixels: the numbers
of `scipy.ndimage.center_of_mass` without the loop. With `roi` (pixel ranges only) they count from
the window's corner. Takes an image, a stack or more leading axes; empty frames give NaN.

```python
# frames: the demo mu scan in memory; the centre of the spot in every frame
com_y, com_x = peaks.frame_com(frames)
np.round(com_x[::6], 1)                      # [12.1, 17.2, 39.6, 61.9, 67.0]: moves with mu
```

#### `frame_moments`
`bexa.analysis.peaks.frame_moments(frames, orders=(2, 3), roi=None, centre=None)`

Raw central moments about the frame COM, per axis: `com_y`, `com_x` and per order `var_y`/`var_x`
(2), `skew_y`/`skew_x` (3), `kurt_y`/`kurt_x` (4) or `moment{n}_y`/`_x`, each
`sum((coord - com)^n I) / sum(I)` in pixels^n, not normalised by the width, as in the legacy
plots. `centre=(com_y, com_x)` gives the centres. Replaces the LCLS notebooks' per-frame loops.

```python
# frames, roi: the demo mu scan and a pixel window; variance, skew and kurtosis of every frame
m = peaks.frame_moments(frames, orders=(2, 3, 4), roi=roi)
round(float(np.sqrt(m["var_x"][12])), 2)     # 12.96: r.m.s. width along x of frame 12, pixels
```

#### `frame_stats`
`bexa.analysis.peaks.frame_stats(frames, roi=None, orders=(2, 3, 4))`

`frame_moments` plus the `total` and `max` of every frame, in one call.

```python
# frames, roi: the demo mu scan and a pixel window; total, maximum and centre of every frame
stats = peaks.frame_stats(frames, roi=roi)   # com_*, var_*, skew_*, kurt_*, total, max
sorted(stats)                                   # ['com_x', 'com_y', 'kurt_x', ..., 'var_y' (10 items)]
```

#### `gaussian2d_fit`
`bexa.analysis.peaks.gaussian2d_fit(image, p0=None, sigma0=(20.0, 10.0), **kwargs)`

Fits `gaussian2d` to one image with `scipy.optimize.curve_fit`, on the host. Start values as in the
legacy notebook: amplitude = max, centre = COM, `(sigma_x, sigma_y)` = `sigma0`, offset = min;
`p0` replaces them and `**kwargs` go to `curve_fit`. Returns floats `amp`, `x0`, `y0`, `sigma_x`,
`sigma_y`, `offset` (pixels) and `success` (0.0 with NaN parameters when the fit fails). A sigma
may come out negative.

```python
# frames[12]: the frame at the peak of the demo mu scan; a 2-D Gaussian fitted to its spot
fit = peaks.gaussian2d_fit(frames[12])
round(fit["x0"], 2), round(fit["y0"], 2), fit["success"]      # (39.58, 32.0, 1.0)
```

#### `fit_stack`
`bexa.analysis.peaks.fit_stack(frames, roi=None, **kwargs)`

`gaussian2d_fit` for every frame (inside `roi`), one numpy array per parameter. Replaces the 2-D
Gaussian fit per delay of the LCLS notebooks.

```python
# frames: the demo mu scan; the spot fitted in a few frames
np.round(peaks.fit_stack(frames[6:19:6], roi=roi)["x0"], 1)   # [2.0, 29.6, 57.2] in the window
peaks.fit_stack(frames[6:19:6])["success"]      # array([1.0, 1.0, 1.0])
```

#### `find_peaks_1d`
`bexa.analysis.peaks.find_peaks_1d(curve, n_peaks=1, **kwargs)`

Indices of the `n_peaks` most prominent peaks of a curve, most prominent first, from
`scipy.signal.find_peaks` (keywords go to it), as a numpy array, empty when there is none.

```python
# the rocking curve of the demo mu scan; the mu of its peaks
scan.coords["mu"][peaks.find_peaks_1d(frames.sum(axis=(1, 2)))]   # [0.0]
peaks.find_peaks_1d(frames.sum(axis=(1, 2)), n_peaks=2)  # array([12])
```

#### `gaussian2d`
`bexa.analysis.peaks.gaussian2d(xy, amp, x0, y0, sigma_x, sigma_y, offset)`

The model of `gaussian2d_fit`, `amp exp(-(x - x0)^2 / (2 sigma_x^2) - (y - y0)^2 / (2 sigma_y^2))
+ offset` on grids `xy = (x, y)`, flattened as `curve_fit` expects.

```python
# the 2-D Gaussian the fits use, evaluated on a pixel grid
x, y = np.meshgrid(np.arange(80), np.arange(64))
model = peaks.gaussian2d((x, y), 4000, 40, 32, 14, 29, 10).reshape(64, 80)
model.shape, round(float(model.max()), 1)       # ((64, 80), 4010.0)
```

Constants:

- `MOMENT_NAMES`: dict of moment order to key prefix, `{2: "var", 3: "skew", 4: "kurt"}`.

### bexa.analysis.fitting: rocking-curve and edge fits

Fits of one curve, each returning a `FitResult`: sums of Gaussians and a Voigt profile (lmfit, the
`Topography` notebook fits) and an error function for edge scans. `fit_multi_gaussian`,
`fit_rocking_curves`, `fit_voigt` and `multi_gaussian_model` need lmfit; `fit_edge` and `edge`
need scipy only. Fits run on the host and return numpy.

```python
import numpy as np
import bexa
from bexa.analysis import fitting

scan = bexa.demo("rocking")                                    # synthetic mu scan
curve = scan.rocking_curve(bexa.ROI(y=(24, 40), x=(30, 50)))  # DataArray: ROI intensity against mu
mu, y = curve.coords["mu"].values, curve.values  # the motor values and the rocking curve
```

#### `fit_multi_gaussian`
`bexa.analysis.fitting.fit_multi_gaussian(x, y, n=None, centers=None, sigmas=0.2, amplitudes=None, background="none", sigma_bounds=(0.05, 5.0), center_bounds=None, scale=True, prefix="g")`

Fits a sum of Gaussians plus an optional background to one curve and returns a `FitResult`.
Replaces the lmfit cells of the `Topography` notebook.

- `n`, `centers`: `n` peaks start at the `n` most prominent maxima (`peaks.find_peaks_1d`), or
  give their start `centers`; with neither, one peak.
- `sigmas`, `amplitudes`: start widths in x units (one or one per peak) and heights (80 % of the
  maximum by default); `sigma_bounds` and `center_bounds` (default: the x range) limit them.
- `background`: `"none"`, `"constant"`, `"linear"` or `"gaussian"` (a broad one, as the notebook).
- `scale`: fit `y / max|y|` for stability; amplitudes and background come back in units of `y`.

```python
# mu, y: the rocking curve of the demo mu scan; one Gaussian plus a constant fitted to it
fit = fitting.fit_multi_gaussian(mu, y, n=1, background="constant")
round(float(fit.centers[0]), 4), round(float(fit.fwhm[0]), 3), round(fit.r2, 5)   # (0.0002, 0.603, 1.0)
two = fitting.fit_multi_gaussian(mu, y, centers=[-0.1, 0.1], background="gaussian")
```

#### `FitResult`
`bexa.analysis.fitting.FitResult(centers, sigmas, amplitudes, fwhm, r2, reduced_chi2, best_fit, background={}, success=True, result=None)`

What the fits return (a dataclass): numpy arrays `centers`, `sigmas`, `amplitudes` (peak heights)
and `fwhm`, one entry per peak; `r2` and `reduced_chi2` (residuals weighted by `1 / sqrt(|y|)`, as
for counts); `best_fit`, the model on `x`; `background`, a dict (`c`; `slope`, `intercept`; or
`amplitude`, `center`, `sigma`); `success`; and `result`, the lmfit `ModelResult` (None for
`fit_edge`).

- `to_dict()`: the fields as plain lists and numbers, without `best_fit` and `result` (for JSON).

```python
# fit: the Gaussian fit of the rocking curve; the fitted curve, the background, the report
fit.best_fit.shape, fit.background["c"] > 0, fit.to_dict()["success"]   # ((25,), True, True)
```

#### `fit_report`
`bexa.analysis.fitting.fit_report(fit)`

A `FitResult` as one flat dict: `center_1`, `sigma_1`, `amplitude_1`, `fwhm_1`, ... per peak,
`background_<name>`, `r2`, `reduced_chi2` and `success`; one row of a results table.

```python
# fit: the Gaussian fit of the rocking curve; its parameters as text lines
list(fitting.fit_report(fit))    # ['center_1', 'sigma_1', 'amplitude_1', 'fwhm_1', 'background_c', 'r2', ...]
```

#### `fit_rocking_curves`
`bexa.analysis.fitting.fit_rocking_curves(x, curves, workers=None, **kwargs)`

`fit_multi_gaussian` (`**kwargs`) for every row of `curves` `(n_curves, n_points)`, such as the
curves of several ROIs, in `workers` threads. Returns a list of `FitResult` in row order.

```python
# curves: the rocking curves of three pixel windows of the demo scan; all fitted at once
vol = scan.read().values
curves = np.stack([vol[:, 24:40, x0:x0 + 10].sum(axis=(1, 2)) for x0 in (10, 35, 60)])
[round(float(f.centers[0]), 2) for f in fitting.fit_rocking_curves(mu, curves, n=1)]   # [-0.38, 0.0, 0.38]
```

#### `fit_voigt`
`bexa.analysis.fitting.fit_voigt(x, y, center=None)`

One Voigt profile (lmfit's `VoigtModel`, gamma tied to sigma) on a constant, started from lmfit's
guess or at `center`. `amplitudes` holds the peak height, `fwhm` the Voigt FWHM and
`background["c"]` the constant.

```python
# mu, y: the rocking curve of the demo mu scan; a Voigt profile fitted to it
round(float(fitting.fit_voigt(mu, y).fwhm[0]), 3)       # 0.618
fitting.fit_voigt(mu, y, center=float(mu[np.argmax(y)])).r2 > 0.9  # True
```

#### `fit_edge`
`bexa.analysis.fitting.fit_edge(x, y, center=None)`

Fits a blurred step (`edge`) to a knife-edge or sample-height scan with `scipy.optimize.curve_fit`:
`centers` is the half-way position, `sigmas` the Gaussian blur, `fwhm` its full width (a beam
size), `amplitudes` the step height (negative for a falling edge), `background["c"]` the level
before the step. `center` sets the start. A fit that does not converge raises scipy's
RuntimeError, so `success` is always True.

```python
# edge_scan: a height scan of the sample edge through the beam; the edge position fitted
edge_scan = bexa.demo("alignment")[3].rocking_curve()    # scan 3 of the demo is a samz edge scan
step = fitting.fit_edge(edge_scan.coords["samz"].values, edge_scan.values)
round(float(step.centers[0]), 4), round(float(step.fwhm[0]), 4)   # (0.0002, 0.0514) mm
```

#### `multi_gaussian_model`
`bexa.analysis.fitting.multi_gaussian_model(n, background="none", prefix="g")`

The lmfit `Model` of `fit_multi_gaussian`, for constraints of your own: `n` Gaussians prefixed
`g1_`, `g2_`, ... (`prefix`) with `amplitude`, `center`, `sigma`, plus a background prefixed `bg_`
(`bg_c`; `bg_slope`, `bg_intercept`; or a further Gaussian). ValueError for `n < 1` or an unknown
background.

```python
# the lmfit model behind the fits: two Gaussians on a linear background
fitting.multi_gaussian_model(2, background="linear").param_names   # ['g1_amplitude', ..., 'bg_intercept']
fitting.multi_gaussian_model(1).param_names     # ['g1_amplitude', 'g1_center', 'g1_sigma']
```

#### `gaussian`
`bexa.analysis.fitting.gaussian(x, amplitude, center, sigma)`

`amplitude exp(-(x - center)^2 / (2 sigma^2))`, with the peak height (not the area) as amplitude.

```python
# the Gaussian the models use, evaluated at two points
fitting.gaussian(np.array([0.0, 0.25]), 1.0, 0.0, 0.25)      # [1.0, 0.6065]
```

#### `edge`
`bexa.analysis.fitting.edge(x, amplitude, center, sigma, offset)`

The model of `fit_edge`, `offset + amplitude (1 + erf((x - center) / (sqrt(2) sigma))) / 2`: a
step of height `amplitude` at `center` blurred by a Gaussian of width `sigma`. Needs scipy and a
numpy `x`.

```python
# the error-function edge the models use, evaluated at three points
fitting.edge(np.array([-1.0, 0.0, 1.0]), 100.0, 0.0, 0.1, 10.0)   # [10, 60, 110]
```

### bexa.analysis.registration: shifts, alignment, stitching and mosaics

Ports of the DFXM notebooks: shift detection (`basicDFXM`, here an FFT cross-correlation), the
overlap search and stitch of `betterCOM_stitchingMu` and `betterCOM_stitchingZ`, the
sample-position mosaic of `RLP_code_SpatialOverlap` and the layer stacking of `Slices_to_Voxels`.

```python
import numpy as np
import bexa
from bexa.analysis import registration

scan = bexa.demo("rocking")                   # synthetic mu scan, 25 frames of 64 x 80 pixels
frames, mu = scan.read().values, scan.coords["mu"]  # the frames in memory and the motor values
image = frames.sum(axis=0)                    # summed image (64, 80)
drifted = np.stack([np.roll(image, (k, -k), axis=(0, 1)) for k in range(5)])   # drifts 1 pixel per frame
parts, axes = [frames[:18], frames[8:]], [mu[:18], mu[8:]]   # as two mu scans overlapping by 10 frames
```

#### `phase_correlation_shift`
`bexa.analysis.registration.phase_correlation_shift(reference, image, max_shift=None)`

Integer shift `(dy, dx)` that moves `image` onto `reference` (apply it with
`scipy.ndimage.shift(image, (dy, dx))`), from the FFT cross-correlation of the mean-subtracted
images: the `correlate2d` search of `basicDFXM` with periodic boundaries, thousands of times
faster. `max_shift` limits the search to `|dy|, |dx| <= max_shift`. Returns Python ints.

```python
# image, drifted: the demo image and a drifted copy; the shift between them
registration.phase_correlation_shift(image, drifted[3])      # (-3, 3)
registration.phase_correlation_shift(image, drifted[3], max_shift=4)  # (-3, 3)
```

#### `align_stack`
`bexa.analysis.registration.align_stack(frames, reference=0, max_shift=5, expected=None, order=1)`

Shifts every frame onto frame `reference` (`ndimage.shift` with spline `order`) and returns
`(aligned, shifts)`, the stack and an `(n, 2)` numpy array of the `(dy, dx)` applied: the
`phase_correlation_shift` within `max_shift`, plus `expected[i]` from an `(n, 2)` array of
motor-driven shifts. The correlation is measured on the unshifted frame, so the two add up. Port
of `detect_and_apply_shifts` (`basicDFXM`).

```python
# drifted: frames that drifted over the scan; aligned, with the shift found for each
aligned, shifts = registration.align_stack(drifted, max_shift=6)
shifts[:, 0]                                            # [0, -1, -2, -3, -4]
```

#### `motor_expected_shift`
`bexa.analysis.registration.motor_expected_shift(motor_values, pixels_per_unit=1.0)`

Pixel shift of every frame relative to the first, `(values - values[0]) * pixels_per_unit`, from a
motor readback such as a sample translation. Two side by side make the `expected` of `align_stack`.

```python
# three motor positions and the pixels per motor unit; the shift expected in pixels
registration.motor_expected_shift([0.0, 0.47, 0.94], pixels_per_unit=1 / 0.235)   # um, 235 nm pixels: [0, 2, 4]
```

#### `find_overlap_by_axis`
`bexa.analysis.registration.find_overlap_by_axis(axes, tolerance=None)`

Start index of each scan on a common grid, from the motor values: each axis is placed where its
first value falls on the grid of the first axis (nearest sample), then all are shifted so the
earliest starts at 0. With `tolerance` (motor units) a start between grid points is a ValueError.

```python
# axes: the motor values of two overlapping parts; where the parts overlap
offsets = registration.find_overlap_by_axis(axes)          # [0, 8]
registration.find_overlap_by_axis(axes, tolerance=0.01)  # [0, 8]
```

#### `find_overlap_1d`
`bexa.analysis.registration.find_overlap_1d(a, b, max_offset=None, min_overlap=2)`

Offset `k` of curve `b` relative to curve `a` that makes them agree best (`b[j]` matches
`a[j + k]`; negative when `a` starts inside `b`), scored by the mean absolute difference over at
least `min_overlap` shared samples, as the stitching notebooks did. Returns
`(best_offset, scores)`, a dict of the score per offset. For scans whose motor values are wrong.

```python
# parts: two overlapping parts of the demo scan; the offset that lines up their curves
best, scores = registration.find_overlap_1d(parts[0].sum(axis=(1, 2)), parts[1].sum(axis=(1, 2)))
best                                                    # 8
```

#### `stitch_along_axis`
`bexa.analysis.registration.stitch_along_axis(volumes, offsets, mode="max", fill=0.0)`

Combines arrays that overlap along their first axis, each starting at its offset: `mode` `"max"`
(element-wise), `"brightest"` (per index, the slice of the array with the larger total: the rule
of the mu-stitching notebook), `"sum"` or `"mean"`; indices no array covers take `fill`. Returns
float64.

```python
# parts, offsets: two overlapping parts and their offsets; stitched into one scan
stitched = registration.stitch_along_axis(parts, offsets, mode="brightest")
stitched.shape, np.array_equal(stitched, frames)        # ((25, 64, 80), True)
```

#### `mosaic_from_grid`
`bexa.analysis.registration.mosaic_from_grid(tiles, step_px, mode="max", fill=0.0)`

Mosaic of tiles `(ny, nx, H, W)` from a regular grid of sample positions: tile `[i, j]` goes to
pixel `(i * step_y, j * step_x)` (`step_px` one int or a pair); overlaps keep the maximum, or the
`"sum"` or `"mean"`. Replaces the `RLP_code_SpatialOverlap` mosaic.

```python
# tiles: a 2 x 2 grid of overlapping windows of the demo image; the mosaic they form
tiles = np.stack([[image[i:i + 40, j:j + 50] for j in (0, 30)] for i in (0, 24)])   # (2, 2, 40, 50)
mosaic = registration.mosaic_from_grid(tiles, step_px=(24, 30))
mosaic.shape, np.allclose(mosaic, image)                # ((64, 80), True)
```

#### `mosaic_from_positions`
`bexa.analysis.registration.mosaic_from_positions(tiles, positions, mode="max", fill=0.0)`

Places 2-D tiles of one size with their top-left corners at pixel positions `(y, x)`, non-negative
integers; `mode` and `fill` as for `mosaic_from_grid`, which calls it.

```python
# flat: the same tiles as a list with their positions; the same mosaic
flat = [tiles[0, 0], tiles[0, 1], tiles[1, 0], tiles[1, 1]]
registration.mosaic_from_positions(flat, [(0, 0), (0, 30), (24, 0), (24, 30)], mode="mean").shape  # (64, 80)
```

#### `stack_layers`
`bexa.analysis.registration.stack_layers(layers, offsets=None, fill=0.0)`

Stacks 2-D layers into a `(z, y, x)` float64 volume, moving layer `k` by `offsets[k] = (dy, dx)`
pixels (down and right when positive) inside a canvas of the same size, the uncovered border
taking `fill`: the `phi_off`/`mu_off` sliders of `Slices_to_Voxels`.

```python
# two images with known offsets stacked into a (layer, y, x) volume
volume = registration.stack_layers([image, drifted[2]], offsets=[(0, 0), (-2, 2)])
float(np.abs(volume[1] - volume[0])[4:-4, 4:-4].max())      # 0.0: the drift is undone
```

### bexa.analysis.projections: projections, integrated maps and RLP point clouds

Reductions of volumes in memory: projections, the intensity of a pixel ROI at every motor point,
and the reciprocal-space points of the 3-D RLP view. `project` and `integrated_map` keep the
labels of a DataArray; `bexa.acc.Projections` and `RoiIntegral` do the same while streaming.

```python
import bexa
from bexa.analysis import projections
from bexa.core.units import bragg_angle_deg

scan = bexa.demo("mosa")                     # synthetic chi x mu scan, 6 x 16 frames of 64 x 80 pixels
vol = scan.read()                            # DataArray (chi, mu, y, x)
```

#### `project`
`bexa.analysis.projections.project(volume, dims, method="sum")`

Reduces a volume over `dims` with the reducer named by `method` (`"sum"`, `"max"`, `"mean"`, ...):
dim names for a DataArray, which keeps its other coordinates, axis numbers for an array.

```python
# vol: the demo chi x mu scan in memory; projected along mu, then onto the pixels by maximum
projections.project(vol, "mu").dims                           # ('chi', 'y', 'x')
projections.project(vol.values, (0, 1), method="max").shape    # (64, 80)
```

#### `integrated_map`
`bexa.analysis.projections.integrated_map(volume, roi=None, method="sum")`

Intensity inside a pixel window at every motor point: the last two axes are cut to the pixel
ranges of `roi` and reduced. A DataArray gives a DataArray on the motor dims, like
`scan.rocking_curve(roi)` does while streaming.

```python
# vol: the demo chi x mu scan in memory; the intensity of a pixel window at every grid point
integrated = projections.integrated_map(vol, bexa.ROI(y=(24, 40), x=(30, 50)))   # (chi, mu)
projections.integrated_map(vol, method="max").dims  # ('chi', 'mu')
```

#### `rlp_points`
`bexa.analysis.projections.rlp_points(structure, intensity, geometry, crystal=None, space="angular", hkl_center=None, threshold=None, **fixed_angles_deg)`

Coordinates and intensity of every motor point for a 3-D reciprocal-space scatter, as flat numpy
arrays; `intensity` is a map on the motor grid, `structure` and `geometry` come from the scan.
`space`: `"angular"` (motor names), `"Q"` (`Qx`, `Qy`, `Qz`; needs `two_theta=` in deg, or a
`crystal` and `hkl_center`) or `"hkl"` (`h`, `k`, `l`; needs `crystal`); other keywords fix motor
angles (deg). NaN points and those below `threshold` are dropped. Uses
`bexa.geometry.transforms.build_coordinate_grids`; replaces v9's `plot_3d_rlp`. Plot with
`bexa.viz.volume.rlp_scatter`.

```python
# scan: the demo chi x mu scan; integrated: its intensity per grid point; the points in Q
two_theta = 2 * bragg_angle_deg(2.338, scan.energy_keV)      # Al 111 at 17 keV, deg
points = projections.rlp_points(scan.structure, integrated, scan.geometry, space="Q",
                                two_theta=two_theta, threshold=1e5)
sorted(points), points["intensity"].size                      # (['Qx', 'Qy', 'Qz', 'intensity'], 16)
```

### bexa.analysis.segmentation: regions from maps

k-means and DBSCAN clusters of COM maps or of whole rocking curves, statistics per cluster, and a
smoothness mask: ports of v9's `plot_kmeans_clustering`, `plot_dbscan_clustering` and
`plot_smoothness_analysis` (plots: `bexa.viz.maps.cluster_map`). The clustering functions need
scikit-learn, `smoothness_mask` scipy. All run on the host and return numpy.

```python
import numpy as np
import bexa
from bexa.analysis import segmentation
from bexa.testing.synthetic import dfxm_sample

scan = bexa.demo("mosa", **dfxm_sample((64, 80)).scan_kwargs)   # chi x mu scan of a synthetic polycrystal
res = scan.com(axes=("chi", "mu"), sigma=1.0)       # com_chi, com_mu, width_chi, width_mu, total
bright = res["total"] > 2000                         # grains in the scan range; the others stay dark
com_chi = res["com_chi"].where(bright).values        # deg, NaN outside those grains
com_mu = res["com_mu"].where(bright).values     # the COM in mu, NaN outside the bright pixels
```

#### `kmeans_com`
`bexa.analysis.segmentation.kmeans_com(com_a, com_b, n_clusters=5, features="angle_magnitude", scale=True, random_state=42)`

k-means of the pixels by their two COM values: the angle `atan2(b, a)` and magnitude
`hypot(a, b)` (`features="angle_magnitude"`, taken about zero, so centre the maps with
`fields.vector_field` when they sit far from 0) or the pair itself (`"raw"`), standardised when
`scale`. Returns a float label map, NaN where either COM is NaN.

```python
# com_mu, com_chi: the COM maps of the demo scan; pixels clustered by tilt (needs scikit-learn)
labels = segmentation.kmeans_com(com_mu, com_chi, n_clusters=3)   # grain B and the two domains of grain A
segmentation.kmeans_com(com_mu, com_chi, n_clusters=2).shape  # (64, 80)
```

#### `dbscan_map`
`bexa.analysis.segmentation.dbscan_map(com_a, com_b, eps=2.0, min_samples=5)`

DBSCAN on the standardised COM pair, for any number of orientation clusters: `eps` is the
neighbourhood radius in standard deviations, `min_samples` the neighbours of a core point. Returns
a label map, `-1` for noise and NaN where a COM is NaN.

```python
# com_mu, com_chi: the COM maps of the demo scan; clusters by density, noise as NaN
db = segmentation.dbscan_map(com_mu, com_chi, eps=0.1, min_samples=10)
np.unique(db[np.isfinite(db)])                                   # [-1, 0, 1, 2]
```

#### `kmeans_profiles`
`bexa.analysis.segmentation.kmeans_profiles(volume, n_clusters=5, scale=False, random_state=42)`

k-means on whole curves: every pixel of a `(frames, y, x)` volume is a sample with one feature per
frame (`scale` standardises them), which groups pixels by the shape and position of their rocking
curves. Returns a `(y, x)` label map, NaN where a curve holds a NaN.

```python
# the rocking curves of every pixel of the demo scan; pixels clustered by curve shape
profiles = segmentation.kmeans_profiles(scan.read().sum("chi").values, n_clusters=4)   # mu curves
np.unique(profiles).size                        # 4
```

#### `cluster_stats`
`bexa.analysis.segmentation.cluster_stats(labels, maps)`

`{label: {map name: {"mean", "std", "median", "min", "max", "n"}}}` for every map inside every
cluster, NaN pixels left out.

```python
# labels: the k-means clusters; the mean and spread of every map inside each cluster
stats = segmentation.cluster_stats(labels, {"com_mu": com_mu, "com_chi": com_chi})
{k: round(v["com_mu"]["mean"], 2) for k, v in stats.items()}   # {0: -0.12, 1: 0.36, 2: 0.08} deg
```

#### `smoothness_mask`
`bexa.analysis.segmentation.smoothness_mask(image, window=6, threshold=None)`

Pixels whose standard deviation over a `(2h + 1)` square, `h = window // 2`, is below `threshold`
(default: the mean of the non-zero metric); returns `(mask, metric, threshold)`, the metric 0 on
the border where the window does not fit. NaN spreads over its window. Same numbers as v9's
sliding-window loop, from two uniform filters.

```python
# res: the COM maps of the demo scan; where the COM varies smoothly (inside a grain)
smooth, metric, threshold = segmentation.smoothness_mask(res["com_mu"].values, window=6)
round(float(smooth.mean()), 2)                                   # 0.74 of the pixels are smooth
```

### bexa.analysis.fields: vector fields, boundaries and strain integration

Two COM maps (mu and chi, or mu and energy) as the components of a vector field: angle and
magnitude, boundaries where a map changes sharply, and the compatibility and integration steps of
the `Yue idea` notebook. Everything returns numpy; `edge_mask` needs scipy.

```python
import numpy as np
import bexa
from bexa.analysis import fields
from bexa.testing.synthetic import dfxm_sample

scan = bexa.demo("mosa", **dfxm_sample((64, 80)).scan_kwargs)   # chi x mu scan of a synthetic polycrystal
res = scan.com(axes=("chi", "mu"), sigma=1.0)   # the COM maps of the demo scan
com_chi, com_mu = res["com_chi"].values, res["com_mu"].values   # deg, (64, 80)
```

#### `vector_field`
`bexa.analysis.fields.vector_field(com_a, com_b, center="median")`

The field `(u, v)`: the two maps minus their centre, `"median"` or `"mean"` (NaN ignored), `"none"`
or a given `(a0, b0)`. Plot with `bexa.viz.maps.vector_field`; v9's `plot_vector_field` step.

```python
# com_mu, com_chi: the COM maps of the demo scan; the tilt of every pixel as a vector
u, v = fields.vector_field(com_mu, com_chi)            # u along mu, v along chi, deg
fields.vector_field(com_mu, com_chi, center="mean")[0].shape  # (64, 80)
```

#### `angle_magnitude`
`bexa.analysis.fields.angle_magnitude(u, v)`

`(atan2(v, u), hypot(u, v))`: the direction in radians (-pi to pi) and the length of the field.

```python
# u, v: the tilt vectors; their direction and length
angle, magnitude = fields.angle_magnitude(u, v)
angle.shape, bool(np.nanmax(magnitude) > 0)     # ((64, 80), True)
```

#### `edge_mask`
`bexa.analysis.fields.edge_mask(field, method="sobel", sigma=0.9, threshold=1.0)`

Boundaries where a map changes sharply, `(mask, strength)`: `"sobel"` gradient magnitude (a step of
height `h` gives `4 h`) or `"laplace"`, the absolute Laplacian of Gaussian of `sigma` pixels; `mask`
is `strength > threshold`. NaN pixels are filled with the mean for filtering and never count.

```python
# com_mu: the COM map in mu; where the tilt jumps (grain boundaries)
boundaries, strength = fields.edge_mask(com_mu, threshold=0.5)   # grain boundaries, domain wall
int(boundaries.sum())                                            # 330
```

#### `gradients`
`bexa.analysis.fields.gradients(field)`

`(d/dy, d/dx)` of a map with `numpy.gradient`, per pixel.

```python
# com_mu: the COM map in mu; its gradients along y and x
dmu_dy, dmu_dx = fields.gradients(com_mu)               # deg per pixel
```

#### `compatibility`
`bexa.analysis.fields.compatibility(eyz, ezx)`

Derivatives of the two shear-strain maps and their compatibility term, as in `Yue idea`:
`eyz_py`, `eyz_px`, `ezx_py`, `ezx_px` (`numpy.gradient` along y and x) and
`dev2nd = (ezx_py - eyz_px) / 2`. `examples/06` passes the chi and mu COM maps as `eyz` and `ezx`.

```python
# com_chi, com_mu: the COM maps as strain components; the compatibility terms
parts = fields.compatibility(eyz=com_chi, ezx=com_mu)   # dict of five maps
```

#### `integrate_displacement`
`bexa.analysis.fields.integrate_displacement(dev2nd, eyz)`

The notebook's "first integration": `df/dy`, the cumulative sum of `dev2nd` along y, minus, row by
row, its last-column value and `eyz` there.

```python
# parts: the compatibility terms; the displacement integrated from them
dfdy = fields.integrate_displacement(parts["dev2nd"], com_chi)
dfdy.shape                                      # (64, 80)
```

#### `integrate_second`
`bexa.analysis.fields.integrate_second(ezx, eyz)`

The notebook's "second integration", a dict: `f1` (cumulative sum of `ezx` along y), `f2` (of
`eyz` along x) and `f = f1 - f1[:, 0] + f2[:, 0]` row by row.

```python
# com_mu, com_chi: the COM maps; the field whose second derivatives they are
f = fields.integrate_second(com_mu, com_chi)["f"]
f.shape                                         # (64, 80)
```

### bexa.analysis.pump_probe: laser on/off analysis

Splitting shots by the laser flag, differential signals, binning shots and points on the scan
axis, combining runs and a linearity check: the scalar logic of the PAL-XFEL scripts
(`Scan_Combiner.py`, the cube builder) and the LCLS `kieran_analysis` binning. `differential` and
`bin_shots` keep cupy frames on the GPU; `bin_by_axis` and `linearity_check` accept numpy only.

```python
import tempfile
import bexa
from bexa.analysis import pump_probe
from bexa.pipelines import xfel_cube
from bexa.testing.synthetic import make_lcls_cube, make_pal_run

work = tempfile.mkdtemp()                       # a scratch folder for the files the examples write
run = make_pal_run(f"{work}/raw", run=1, n_points=6, shots_per_point=12)   # synthetic PAL-XFEL run
scan = bexa.open(run.measurement_dir, cache=False)  # the PAL-XFEL run as a scan
shots = scan.source.shot_table()             # pandas, one row per shot
roi = bexa.ROI(y=(2, 22), x=(4, 28))            # a pixel window
cube = xfel_cube.build_cube(scan.source, roi=roi, workers=1)   # frames(laser, delay, y, x), signal(laser, delay)
lcls = bexa.load(make_lcls_cube(f"{work}/lcls").root, squeeze_single=False)   # frames(laser, shot, y, x)
flag, i0 = "event_info.THIRTY_HERTZ", "qbpm:eh1:qbpm1:sum"   # columns of the shot table
roi_sum = "detector:eh1:jungfrau2:ROI1_stat.sum"  # the per-shot ROI sum column
```

#### `split_on_off`
`bexa.analysis.pump_probe.split_on_off(table, flag_key, dropna=True)`

Splits a per-shot pandas table into the rows with the laser on (flag 1) and off (flag 0), `(on,
off)`; NaN flags (dropped laser shots) belong to neither. `dropna` also drops rows with a NaN in
any column, as the legacy cube builder did.

```python
# shots, flag: the shots of a PAL-XFEL point and their laser flag; split in two
on, off = pump_probe.split_on_off(shots, flag)
len(on), len(off), len(shots)                           # (32, 34, 72)
```

#### `differential`
`bexa.analysis.pump_probe.differential(on, off, mode="diff")`

`on - off` (`mode="diff"`), `on / off` (`"ratio"`) or `100 (on - off) / off` (`"percent"`), for
numbers, numpy or cupy arrays and DataArrays (coordinates kept). ValueError for another mode.

```python
# cube: a laser on/off cube; the change of the signal with the laser, in percent
signal = cube["signal"]                                  # ROI signal per point (laser, delay)
change = pump_probe.differential(signal.sel(laser="on"), signal.sel(laser="off"), mode="percent")
change.values.round(1)                                   # [-29.3, -0.4, 29.2, 30.1, 29.9, 30.3]
```

#### `bin_by_axis`
`bexa.analysis.pump_probe.bin_by_axis(axis, values, decimals=2)`

Averages arrays over the points whose axis values agree after rounding to `decimals`, as
`Scan_Combiner.py`: every array of the list `values` (one row per axis point) is averaged per bin
(NaN propagates), and each bin coordinate is the mean of its members. Returns
`(centres, [binned, ...])`.

```python
# on: the laser-on shots; their normalised signal binned by delay
ratio = (on[roi_sum] / on[i0]).values                   # I0-normalised ROI sum per laser-on shot
delays, (per_delay,) = pump_probe.bin_by_axis(on["delay_input"].values, [ratio])
delays.round(2)                                          # [-2, 0, 2, 4, 6, 8]
```

#### `combine_runs`
`bexa.analysis.pump_probe.combine_runs(datasets, axis=None, decimals=2)`

Merges the cubes of several runs on one scan axis (by default `scan_axis_of` the first; `"delays"`
means `"delay"`), averaging points whose values agree after rounding to `decimals`. Keeps the
variables every run has; the frame shapes must match (same ROI). Records the sources,
`attrs["scan_axis"]` and `attrs["combined_runs"]`. Replaces `Scan_Combiner.py`; `bexa cube combine`.

```python
# cube, cube2: the cubes of two runs on the same axis; combined into one
run2 = make_pal_run(f"{work}/raw", run=2, n_points=6, shots_per_point=12, seed=1)
cube2 = xfel_cube.build_cube(bexa.open(run2.measurement_dir, cache=False).source, roi=roi, workers=1)
combined = pump_probe.combine_runs([cube, cube2], axis="delay")
dict(combined.sizes)                                     # {'laser': 2, 'delay': 6, 'y': 20, 'x': 24}
```

#### `scan_axis_of`
`bexa.analysis.pump_probe.scan_axis_of(ds)`

The scan dimension of a cube: `ds.attrs["scan_axis"]`, else the first dim of `ds["frames"]` other
than `laser`, `y` and `x` (ValueError when there is none).

```python
# cube, lcls: a PAL-XFEL cube and an LCLS cube; the axis each was scanned along
pump_probe.scan_axis_of(cube), pump_probe.scan_axis_of(lcls)     # ('delay', 'scanvar')
```

#### `bin_lcls`
`bexa.analysis.pump_probe.bin_lcls(ds, axis="scanvar")`

Bins an LCLS per-shot dataset (`bexa.load` of a `Run*_onStk.npy` folder,
`bexa.io.cube.load_lcls_cube`) on the scan variable: `frames(laser, scanvar, y, x)` sums the
I0-normalised shots of every value (`bin_shots`, I0 = 0 skipped), `i0` sums I0 and `n_shots`
counts the shots, to divide by for means. Keeps the attrs. Replaces the `kieran_analysis` binning.

```python
# lcls: an LCLS cube of shots; the shots binned by delay into laser on and off
binned = pump_probe.bin_lcls(lcls)
binned["frames"].dims, binned["n_shots"].values        # (('laser', 'scanvar', 'y', 'x'), [3, 4, 4, 4, 4, 4])
```

#### `bin_shots`
`bexa.analysis.pump_probe.bin_shots(on, off, scanvar, i0_on, i0_off)`

The array form of `bin_lcls`: sums `on[k] / i0_on[k]` and `off[k] / i0_off[k]` over the shots of
every unique `scanvar`, skipping shots where either I0 is zero. Returns
`(bins, sum_on, sum_off, counts)`, the sums `(bins, y, x)` of the kind of `on`.

```python
# frames, shot_i0: the shots and monitor of the LCLS cube; binned by delay, on and off apart
frames, shot_i0 = lcls["frames"], lcls["i0"]
bins, sum_on, sum_off, counts = pump_probe.bin_shots(
    frames.sel(laser="on").values, frames.sel(laser="off").values, lcls["scanvar"].values,
    shot_i0.sel(laser="on").values, shot_i0.sel(laser="off").values)
len(bins), sum_on.shape                         # (6, (6, 24, 24))
```

#### `linearity_check`
`bexa.analysis.pump_probe.linearity_check(det, i0, laser=None)`

Straight-line fit of a detector signal against I0 to spot saturation and bad shots: `{"all": ...}`,
or `{"on": ..., "off": ...}` with a laser flag per shot, each with `slope`, `intercept`, `r2` and
`n`; NaN shots are skipped and fewer than 2 points give NaN.

```python
# shots: a PAL-XFEL point; whether the signal follows the monitor, laser on and off
check = pump_probe.linearity_check(shots[roi_sum], shots[i0], shots[flag])
round(check["off"]["r2"], 3), round(check["on"]["r2"], 3)      # (0.998, 0.239): on varies with delay
```

Constants:

- `LASER_COORD`: the `laser` coordinate of every cube, `np.array(["off", "on"])`.

### bexa.viz.style: plot conventions, themes and colour limits

`bexa.viz` draws with matplotlib (plotly for 3-D renderings, napari for viewers), and every plot
function follows one rule. A single-panel function takes `ax=None` and draws into that axes when
one is given, otherwise into a new figure; the multi-panel ones (`tiles`, `projections_panel`,
`slices_grid`, `on_off_diff`, `moments_vs_axis`, `auto.grid`) make their own grid of axes, the
browsers and `animate` their own figure. Each returns its figure and the artists it made (listed
on every entry) and never calls `plt.show()`: a notebook shows the figure by itself, a script
calls `plt.show()` or `fig.savefig(...)`. Images get percentile colour limits by default,
`clim=("p", 1, 99)`: the 1st and 99th percentiles of the finite values set the ends of the colour
scale, so a few hot or dead pixels do not flatten the image. Every plot function takes the same
keywords for its looks (`clim`, `cmap`, `colorbar`, `scalebar`, `figsize`) and leaves them at
`style.DEFAULT`, which means "take it from the style defaults": `style.set(...)` changes those
once for a whole notebook, `style.context(...)` for a few cells, and `style.apply(fig, ...)`
restyles a figure already drawn. `bexa.viz.style` holds those defaults, the colour-limit
grammar, the two themes and the helpers the plotting modules share.

```python
import bexa
from bexa.viz import style

scan = bexa.demo("mosa")                         # synthetic chi x mu scan, 6 x 16 frames of 64 x 80 pixels
res = scan.com(axes=("chi", "mu"), sigma=1.0)    # xarray.Dataset: com_chi, width_chi, com_mu, width_mu, total
prev = scan.preview(downsample=(1, 1, 4, 4))     # DataArray (chi, mu, y, x) = (6, 16, 16, 20)
```

#### Customising plots

What to change, and the three places to change it: the keyword of one call, `style.set` for
every figure from then on (`style.context` for a `with` block), or `style.apply` on a figure
already drawn. `bexa.plot`, the `.bexa` accessors, the panels of a reduce result
(`res.bexa.plot()`), `tiles`, the browsers, `render` and the DFXM report all read the same
defaults, so one settings cell at the top of a notebook sets every figure.

| what | one call | every figure | afterwards |
|---|---|---|---|
| colour limits of an image | `clim=("p", 2, 98)`, `"p2-98"`, `(vmin, vmax)`, `None` | `style.set(clim="p2-98")` | `style.apply(fig, clim=(0, 500))` |
| a stack (tiles, browsers, `animate`) | `clim="shared"`, `("shared", 2, 98)`, `"each"`, `(vmin, vmax)` | the same `clim` | `browser.set_clim("each")` |
| colormap | `cmap="magma"` | `style.set(cmap="magma")` (COM maps: `diverging_cmap`) | `style.apply(fig, cmap="gray")` |
| colourbars | `colorbar=False` | `style.set(colorbar=False)` | `style.apply(fig, colorbar_label="counts")` |
| scalebar | `scalebar=235` (nm per pixel), `False` | `style.set(scalebar=False)` | |
| figure size | `figsize=(5, 4)` | `style.set(figsize=(5, 4))` (single panels), `panel_size=(3, 3)` (each pane of a grid) | `style.apply(fig, figsize=(8, 8))` |
| the span of a COM map | `span=0.1`, `span_fraction=0.5` | `style.set(com={"span": 0.1})` | |
| one plot kind only | | `style.set(tiles={"per_row": 4, "title_size": 7})` | |
| fonts, dpi, ticks | | `style.use("paper")`, `plt.rcParams` | |

```python
style.set(cmap="magma", clim="p2-98", colorbar=False)   # every image from here on
fig, ax, im = bexa.plot(res["total"])            # magma, 2nd to 98th percentile, no colourbar
with style.context(com={"span": 0.1}):           # a few cells with other defaults
    fig, ax, im = bexa.plot(res["com_mu"])       # +/- 0.05 deg around the mean
style.apply(fig, clim=(-0.2, 0.2), cmap="RdBu")  # change a figure already drawn
style.reset()                                    # the shipped defaults again
```

#### `use`
`bexa.viz.style.use(theme="screen")`

Apply a theme to matplotlib's global settings (`plt.rcParams`) for the rest of the session:
`"screen"` (100 dpi on screen, 150 dpi saved, viridis, constrained layout) or `"paper"` (the
settings of the lab's XFEL scripts: ticks pointing in on all four sides, minor ticks, 150 dpi on
screen, 300 dpi saved). bexa never applies a theme by itself. `use("screen")` after `"paper"`
keeps the paper tick settings (`matplotlib.rcdefaults()` resets all); an unknown name raises
`KeyError`.

```python
# the paper theme (fonts, ticks, dpi) for the figures from here on
style.use("paper")                               # the figures from here on
style.use("screen")
```

#### `set`
`bexa.viz.style.set(**options)`

Change the style defaults for every figure from here on. A global option (`clim`, `render_clim`,
`cmap`, `diverging_cmap`, `span_fraction`, `figsize`, `panel_size`, `colorbar`,
`colorbar_fraction`, `colorbar_pad`, `scalebar`, `title_size`: the fields of `PlotDefaults`)
applies to every plot kind; a plot kind (the keys of `KINDS`: `image`, `tiles`, `com`, `width`,
`projections`, `browse`, `volume`, `render`, `grid`, ...) takes a dict of keywords of its
function, or of global options for that kind only, which win over the global values. An unknown
option, kind or keyword raises `ValueError`.

```python
# other defaults for every figure, then for COM maps and tile grids only
style.set(cmap="magma", colorbar=False)          # every image
style.set(com={"span": 0.1}, tiles={"per_row": 4, "title_size": 7})   # COM maps and tile grids only
style.defaults.cmap, style.defaults.kinds        # ('magma', {'com': {'span': 0.1}, 'tiles': {'per_row': 4, 'title_size': 7}})
style.reset()
```

#### `reset`
`bexa.viz.style.reset()`

Back to the shipped defaults (`PlotDefaults()`); `rcParams` are not touched
(`matplotlib.rcdefaults()` does that).

```python
# a default changed, then the shipped defaults again
style.set(cmap="gray")
style.reset()
style.defaults.cmap                              # None
```

#### `context`
`bexa.viz.style.context(**options)`

A `with` block with other defaults, the same `options` as `set`; the previous defaults come
back when the block ends, whatever happens inside.

```python
# other defaults for one figure, restored after the block
with style.context(clim="p2-98", colorbar=False):
    fig, ax, im = bexa.plot(res["total"])        # 2nd to 98th percentile, no colourbar
im.get_clim()                                    # (10781.7, 27625.8)
```

#### `option`
`bexa.viz.style.option(kind, name, value=DEFAULT, *, global_name=None, fallback=None)`

Resolve one keyword the way every plot function does: the argument when it was given (anything
but `DEFAULT` or `None`), else the kind's override from `style.set`, else the global default of
that name (`global_name` when the global is called differently: a COM map's `cmap` is
`diverging_cmap`), else `fallback` (the function's own literal). For plotting helpers of your
own.

```python
# how a plot function resolves one keyword: given, the kind's override, the global, a fallback
style.option("image", "cmap", "gray")            # 'gray': given
style.option("image", "cmap")                    # None: the global cmap is None (rcParams decide)
with style.context(cmap="magma", tiles={"cmap": "gray"}):
    style.option("image", "cmap"), style.option("tiles", "cmap")   # ('magma', 'gray')
```

#### `local_option`
`bexa.viz.style.local_option(kind, name, value=DEFAULT, fallback=None)`

Like `option` without the global: the argument, the kind's override or `fallback`. For keywords
whose global value would be wrong for a kind: `tiles` and `VolumeBrowser` draw no colourbar
unless asked, whatever `colorbar` says, and a COM map's `clim` overrides centre and span only
when given.

```python
# the same without the global: tile grids draw no colourbar unless asked
style.local_option("tiles", "colorbar", fallback=False)   # False, although style.defaults.colorbar is True
style.local_option("tiles", "colorbar", True)   # True
```

#### `limits`
`bexa.viz.style.limits(data, clim=DEFAULT, kind="image")`

The `(vmin, vmax)` of one image: `clim` when given (a `resolve_clim` spec), else the kind's or
the global `clim` default (`render_clim` for the `render` kind); `clim=None` gives
`(None, None)` (matplotlib scales by itself).

```python
# res: the COM maps of the demo scan; the colour limits of a map under the defaults
style.limits(res["total"])                       # (10171.6, 27822.9): the default percentiles
style.limits(res["total"], "p5-95")              # (12044.2, 27016.0)
style.limits(res["total"], None)                 # (None, None)
```

#### `stack_mode`
`bexa.viz.style.stack_mode(clim=DEFAULT, kind="image")`

How a stack of frames (`tiles`, the browsers, `animate`) is scaled, from its `clim`:
`("shared", (lo, hi))` for one scale over every frame at those percentiles (`"shared"`,
`("shared", lo, hi)`, `("p", lo, hi)`, `"p2-98"`), `("each", (lo, hi))` for every frame on its
own (`"each"`, `("each", lo, hi)`, `None`) and `("fixed", (vmin, vmax))` for a pair of numbers.
Left at `DEFAULT`, the kind's or the global `clim` decides, its percentiles for the bare words.

```python
# how a stack of frames is scaled, from a clim spec
style.stack_mode("shared")                       # ('shared', (1.0, 99.0))
style.stack_mode(("each", 2, 98))                # ('each', (2.0, 98.0))
style.stack_mode((0, 500))                       # ('fixed', (0.0, 500.0))
```

#### `percentiles_of`
`bexa.viz.style.percentiles_of(spec)`

The `(lo, hi)` percentiles named by a spec (`("p", 2, 98)`, `("shared", 2, 98)`, `"p2-98"`);
`(1, 99)` for a `(vmin, vmax)` pair; an unknown word raises `ValueError`.

```python
# the percentiles a clim spec names
style.percentiles_of("p5-95")                    # (5.0, 95.0)
style.percentiles_of((0, 500))                   # (1.0, 99.0)
```

#### `colormap`
`bexa.viz.style.colormap(kind, cmap=DEFAULT)`

The colormap of a plot kind: the argument, else the kind's override, else the global `cmap`
(`diverging_cmap`, RdBu, for the `com` kind). `None` leaves `rcParams["image.cmap"]` in charge.

```python
# the colormap of an image, and of a COM map (the diverging one)
style.colormap("image")                          # None
style.colormap("com")                            # 'RdBu'
```

#### `colorbar`
`bexa.viz.style.colorbar(fig, im, ax, kind="image", show=DEFAULT, label=None)`

Draw the colourbar of `im` next to `ax` (`fig.colorbar` with the style's `colorbar_fraction`
and `colorbar_pad`) when `show`, the kind's override or the global `colorbar` says so; returns
the `Colorbar`, or `None` when none is drawn.

```python
# fig, im, ax: a figure drawn without a colourbar; one added afterwards
fig, ax, im = bexa.plot(res["total"], colorbar=False)
bar = style.colorbar(fig, im, ax, label="counts")   # add it afterwards
```

#### `figure_size`
`bexa.viz.style.figure_size(kind, figsize=None, rows=1, cols=1, panel=None)`

The figure size a plot function uses: `figsize` when given, else the kind's `figsize` override,
else, for a single-panel figure (`panel=None`), the global `figsize` or `None` (the function's
own size or `rcParams`); for a panel figure, `panel` inches per pane (the kind's or the global
`panel_size` instead, when set) times `cols` and `rows`.

```python
# the size of a single-panel figure and of a 2 x 3 grid of tiles
style.figure_size("image")                       # None
style.figure_size("tiles", rows=2, cols=3, panel=(3.2, 3.0))   # (9.6, 6.0)
with style.context(panel_size=(2, 2)):
    style.figure_size("tiles", rows=2, cols=3, panel=(3.2, 3.0))   # (6.0, 4.0)
```

#### `title_font`
`bexa.viz.style.title_font(kind)`

`{"fontsize": size}` for the pane titles of a kind (`title_size`, 9 by default), or `{}` when
`title_size` is `None` (matplotlib's own size): `ax.set_title(name, **style.title_font("tiles"))`.

```python
# the font of the pane titles of a tile grid
style.title_font("tiles")                        # {'fontsize': 9}
```

#### `apply`
`bexa.viz.style.apply(fig, clim=None, cmap=None, figsize=None, colorbar_label=None)`

Restyle a figure already drawn: every image of every axes gets `clim` (a `resolve_clim` spec,
applied to that image's own data in the units shown) and `cmap`, every colourbar the
`colorbar_label`, the figure its `figsize` in inches. Arguments left `None` change nothing.
Returns the figure.

```python
# fig: the panel of every map of the demo scan; its limits, colormap and size changed
fig, axes = bexa.plot(res)                       # the panel of every map
style.apply(fig, clim="p5-95", cmap="magma", figsize=(10, 7))  # Figure
```

#### `resolve_clim`
`bexa.viz.style.resolve_clim(data, clim=("p", 1, 99))`

Turn a colour-limit spec into `(vmin, vmax)` for `data`: `("p", lo, hi)` gives the `lo` and
`hi` percentiles of the finite values, `"p2-98"` is the short form, `(vmin, vmax)` comes back as
floats, `None` gives `(None, None)` (matplotlib then scales by itself) and `DEFAULT` the global
`clim`. The stack modes `"shared"` and `"each"` raise `ValueError` here: they belong to
`stack_mode`. The `clim` of every image goes through it (`limits` adds the defaults).

```python
# res: the COM maps of the demo scan; percentile and fixed limits for its total map
style.resolve_clim(res["total"], ("p", 5, 95))   # (12044.2, 27016.0)
style.resolve_clim(res["total"], "p2-98")        # (10781.7, 27625.8)
style.resolve_clim(res["total"], (0, 1000))      # (0.0, 1000.0)
```

#### `percentile_limits`
`bexa.viz.style.percentile_limits(data, low=1.0, high=99.0)`

The `low` and `high` percentiles of the finite values of `data` as two floats, ignoring NaN and
infinities. Gives `(0.0, 1.0)` when nothing is finite and moves `vmax` just above `vmin` for a
constant image, so the range is always usable.

```python
# res: the COM maps of the demo scan; the 1st and 99th percentiles of a COM map
lo, hi = style.percentile_limits(res["com_mu"], 1, 99)   # (-0.533, 0.534) deg
style.percentile_limits(res["com_mu"], 5, 95)   # (-0.50147, 0.50277)
```

#### `as_array`
`bexa.viz.style.as_array(data)`

A plain numpy array from a DataArray (its values), a cupy array (copied from the GPU) or
anything `np.asarray` accepts. Every plot function calls it, so all of them take DataArrays,
numpy and cupy arrays alike.

```python
# res: the COM maps of the demo scan; one map as a plain numpy array
arr = style.as_array(res["com_mu"])              # numpy array (64, 80)
```

#### `new_axes`
`bexa.viz.style.new_axes(ax=None, figsize=None, **kwargs)`

`(fig, ax)`: the given axes and its figure, or a new figure from
`plt.subplots(figsize=figsize, **kwargs)`. The single-panel plot functions start with it; your
own plotting helpers can too.

```python
# a new figure of a given size, then the same axes passed back
fig, ax = style.new_axes(figsize=(5, 4))
fig, ax = style.new_axes(ax)                     # the same axes and figure back
```

#### `dim_label`
`bexa.viz.style.dim_label(da, dim)`

Axis label for the dim `dim` of `da`: `"mu (deg)"` when its coordinate has a `units`
attribute, the bare name otherwise (also for plain arrays).

```python
# prev: a preview (chi, mu, y, x); the axis label of its mu dim, with and without units
style.dim_label(prev, "mu")                      # 'mu'
prev.coords["mu"].attrs["units"] = "deg"
style.dim_label(prev, "mu")                      # 'mu (deg)'
```

#### `pixel_size_nm`
`bexa.viz.style.pixel_size_nm(da)`

The effective (sample-plane) pixel size in nm from `da.attrs["effective_pixel_nm"]`, or `None`.
The `scalebar=True` default of `images.show` and of the `maps` functions reads it. Results of
`scan.reduce` do not carry this attribute: pass the size (`scalebar=235`,
`scalebar=scan.geometry.effective_pixel_nm`) or set the attribute.

```python
# res: the COM maps of the demo scan; the pixel size in nm, once set in the attrs
style.pixel_size_nm(res["com_mu"])               # None
res["com_mu"].attrs["effective_pixel_nm"] = 235.0
style.pixel_size_nm(res["com_mu"])               # 235.0
```

Constants:

- `THEMES`: dict of theme name to settings, `{"paper": PAPER, "screen": SCREEN}`.
- `PAPER`, `SCREEN`: the two rcParams dicts; update a copy and pass it to `plt.rcParams.update`
  for a variant.
- `DEFAULT`: the marker every plot keyword starts at ("take it from the style defaults"); its
  `repr` is `DEFAULT` and it cannot be used as a boolean.
- `defaults`: the `PlotDefaults` instance that `set`, `context` and `reset` change; read it for
  the current values (`style.defaults.clim`).
- `PlotDefaults`: the dataclass of the defaults: `clim=("p", 1, 99)`, `render_clim=("p", 5, 99.5)`,
  `cmap=None`, `diverging_cmap="RdBu"`, `span_fraction=0.2`, `figsize=None`, `panel_size=None`,
  `colorbar=True`, `colorbar_fraction=0.046`, `colorbar_pad=0.04`, `scalebar=True`,
  `title_size=9`, `kinds={}` (the per-kind overrides).
- `GLOBAL_OPTIONS`: the names of those fields but `kinds`.
- `KINDS`: dict of plot kind to `(module, function)`, what `set` validates a kind's keywords
  against.

### bexa.viz.auto: one call picks the figure

`plot(obj)` looks at an object's type, dims, name and attrs and calls the matching function of
the other `bexa.viz` modules. `bexa.plot` is the same function, and the `.bexa` accessor calls
it: `da.bexa.plot()` for a DataArray, `res.bexa.plot()` for a Dataset.

```python
import bexa
from bexa.viz import auto

scan = bexa.demo("mosa")                         # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
res = scan.reduce([bexa.acc.Sum(), bexa.acc.MotorCOM(axes=("chi", "mu"), sigma=1.0)])  # sum, com_chi, com_mu, width_chi, width_mu, total
prev = scan.preview(downsample=(1, 1, 4, 4))     # DataArray (chi, mu, y, x) = (6, 16, 16, 20)
```

#### `plot`
`bexa.viz.auto.plot(obj, kind=None, **kwargs)`

Plot `obj` with the function that fits it and return what that function returns. `kind=None`
takes `guess_kind(obj)`; a name from `KINDS` forces the choice. Other keywords go to the chosen
function. A numpy array with three or more dims gets the dims `axis0, ..., y, x` for `browse`,
`projections`, `volume` and `render`; an unknown kind raises `ValueError`.

| `kind` | calls | returns |
|---|---|---|
| `"image"` | `bexa.viz.images.show` | `(fig, ax, im)` |
| `"com"` | `bexa.viz.maps.com_map` | `(fig, ax, im)` |
| `"width"` | `maps.com_map` with `center="median"`, the `width` kind's overrides and `cmap` from the style (viridis) | `(fig, ax, im)` |
| `"curve"` | `bexa.viz.curves.curve` | `(fig, ax)` |
| `"rocking"` | `curves.rocking_curves` | `(fig, ax)` |
| `"tiles"` | `images.tiles` | `(fig, axes)` |
| `"browse"` | `bexa.viz.interactive.browse` | `Browser` |
| `"projections"` | `bexa.viz.volume.projections_panel` | `(fig, axes)` |
| `"volume"` | `interactive.browse_volume` | `VolumeBrowser` |
| `"on_off"` | `curves.on_off_diff` | `(fig, axes)` |
| `"grid"` | `bexa.viz.auto.grid` | `(fig, axes)` |
| `"render"` | `volume.render` | plotly `Figure` |

```python
# res: the COM maps of the demo scan; the figure that fits each object
fig, ax, im = auto.plot(res["com_mu"])                          # com_map: RdBu around the mean
fig, ax, im = bexa.plot(res["sum"], log=True, title="summed")   # images.show
fig, axes = res.bexa.plot()                                     # grid: one panel per 2-D map
fig, axes = prev.bexa.plot(kind="tiles", axis="mu", n=8, per_row=4)
```

#### `guess_kind`
`bexa.viz.auto.guess_kind(obj)`

The kind `plot` would choose for `obj`, as a string:

- an `xarray.Dataset`: `"on_off"` when it has a `frames` variable and a `laser` dim (a
  pump-probe cube), otherwise `"grid"`;
- 0 or 1 dims: `"curve"`;
- 2 dims: `"width"` when the name starts with `width_`, `skew_` or `kurtosis_`; `"com"` when it
  starts with `com_` or `argmax_` or the attrs have a `motor` key; otherwise `"image"`;
- 3 dims: `"tiles"` along the first dim; 4 or more: `"projections"`.

`"rocking"`, `"browse"`, `"volume"` and `"render"` are never guessed; ask for them with `kind=`.

```python
# res, prev: the COM maps and a preview of the demo scan; the kind plot would choose
auto.guess_kind(res["width_mu"])                 # 'width'
auto.guess_kind(prev)                            # 'projections'
auto.guess_kind(res)                             # 'grid'
```

#### `grid`
`bexa.viz.auto.grid(ds, per_row=3, max_panels=9, figsize=None, names=None, panes=None, **options)`

One panel per 2-D variable of a reduce result, each drawn by `plot` with the variable name as
title, so COM and width maps get their colours and the rest are images. Takes the first
`max_panels` 2-D variables or those in `names`, `per_row` to a row; spare axes are switched off.
`options` (`cmap`, `clim`, `colorbar`, `scalebar`, ...) go to every pane and
`panes={"com_mu": {"span": 0.1}}` to one variable; `figsize` defaults to 4.4 x 3.8 inches per
pane (`style.set(panel_size=...)`). Returns `(fig, axes)`, `axes` a 2-D numpy array;
`ValueError` when there is no 2-D variable. `plot(res)` and `res.bexa.plot()` call it (public,
although not in the module's `__all__`).

```python
# res: the COM maps of the demo scan; one panel per map
fig, axes = auto.grid(res, names=["sum", "com_chi", "com_mu", "width_mu"], per_row=2)
fig, axes = res.bexa.plot(cmap="magma", colorbar=False, panes={"com_mu": {"span": 0.1}})
axes.shape                                      # (2, 3)
```

Constants:

- `KINDS`: tuple of the kind names `plot` accepts, in the order of the table above.

### bexa.viz.images: single images, tiles, ROI boxes and scalebars

```python
import bexa
from bexa.viz import images

scan = bexa.demo("mosa")                         # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
image = scan.sum()                               # summed image, DataArray (y, x) = (64, 80)
prev = scan.preview(downsample=(1, 1, 4, 4))     # DataArray (chi, mu, y, x) = (6, 16, 16, 20)
peak = bexa.ROI(y=(16, 48), x=(20, 60))          # pixel window in full-resolution pixels
```

#### `show`
`bexa.viz.images.show(image, ax=None, clim=DEFAULT, cmap=DEFAULT, log=False, roi=None, scalebar=DEFAULT, title=None, colorbar=DEFAULT, figsize=None, **imshow_kwargs)`

Display a 2-D image (DataArray, numpy or cupy array) with percentile colour limits and a
colourbar; returns `(fig, ax, im)`, `im` the matplotlib `AxesImage`. A DataArray is drawn in the
coordinates of its last two dims, the pixels whatever their names (`y`/`x`, or `yb`/`xb` for a
preview binned on its own), so a preview's axes read in full-resolution pixels and ROI boxes
line up; the axes are labelled after those dims and the title defaults to the array's name.

- `clim`, `cmap`, `scalebar`, `colorbar` and `figsize` left at `DEFAULT` come from the style
  defaults (`bexa.viz.style.set`); `clim` is a `bexa.viz.style.resolve_clim` spec, `None` lets
  matplotlib scale; `log=True` shows `log10(1 + image)`, negative values clipped to 0; `roi`:
  an `ROI` or `{name: ROI}`, drawn by `roi_overlay`.
- `scalebar`: `True` draws one when `attrs["effective_pixel_nm"]` is set, a number gives the nm
  per pixel (per unit of the `y`/`x` coordinates), `False` none; needs matplotlib-scalebar.
- Other keywords go to `ax.imshow`; an input that is not 2-D raises `ValueError`.

```python
# image: the summed image of the demo scan; peak: a pixel window; the image with the window drawn
fig, ax, im = images.show(image, log=True, roi={"peak": peak}, scalebar=235)
fig, ax, im = images.show(prev.isel(chi=3, mu=8), clim=None, cmap="gray", interpolation="nearest")
fig, ax, im = image.bexa.plot(log=True)          # the same function through the accessor
```

#### `tiles`
`bexa.viz.images.tiles(volume, axis=0, n=9, per_row=3, indices=None, clim=DEFAULT, cmap=DEFAULT, log=False, titles=True, figsize=None, colorbar=DEFAULT)`

A grid of images along one axis of a stack. For a DataArray, `axis` is a dim name or index, all
other dims except the last two (the pixels, whatever their names) are summed away and each tile
is titled from the coordinate
(`mu = 0.3333`); for a numpy array the axis is moved to the front and tiles are titled
`index = 5`. Shows `n` evenly spaced frames (or exactly `indices`), `per_row` to a row.
`clim` follows the stack grammar of `bexa.viz.style.stack_mode`: `"shared"` (the default: one
scale for every tile, the style's percentiles of the frames shown), `("shared", 2, 98)`,
`"each"` (every tile on its own), `("each", lo, hi)`, `(vmin, vmax)`; `log=True` shows
`log10(1 + I)`. `colorbar=True` adds one colourbar for a shared scale, one per tile otherwise
(none unless asked, or `style.set(tiles={"colorbar": True})`); `figsize` defaults to 3.2 x 3.0
inches per tile. Returns `(fig, axes)`, `axes` a 2-D numpy array (ticks removed, spares off).

```python
# prev: a preview (chi, mu, y, x); eight frames along mu side by side
fig, axes = images.tiles(prev, axis="mu", n=8, per_row=4)                  # chi summed away
fig, axes = images.tiles(prev.values[3], indices=[0, 5, 10, 15], clim="each", log=True)
```

#### `roi_overlay`
`bexa.viz.images.roi_overlay(ax, rois, color="r", lw=1.0)`

Draw a rectangle on `ax` for the `y`/`x` pixel ranges of each ROI, with its name at the top-left
corner. `rois` is one `ROI` (named `roi`), a dict `{name: ROI}` or a list (`roi0`, `roi1`, ...);
an ROI without both bounds in `y` and in `x` is skipped. The boxes use the data coordinates of
`ax`, so draw the image in full-resolution pixels (as `show` does for bexa's DataArrays).
Returns the list of matplotlib `Rectangle` patches.

```python
# image: the summed image of the demo scan, shown; two windows drawn on it
fig, ax, im = images.show(image)
boxes = images.roi_overlay(ax, {"peak": peak, "background": bexa.ROI(y=(0, 10), x=(0, 10))}, color="w")
len(boxes)                                      # 2
```

#### `add_scalebar`
`bexa.viz.images.add_scalebar(ax, pixel_size_nm, location="lower right", **kwargs)`

Add a matplotlib-scalebar `ScaleBar` to `ax` for `pixel_size_nm` nanometres per data unit; other
keywords go to `ScaleBar` (`color`, `box_alpha`, `length_fraction`, ...). Returns the scalebar,
or `None` when matplotlib-scalebar is not installed.

```python
# image: the summed image shown without a scalebar; one added by hand
fig, ax, im = images.show(image, scalebar=False)
bar = images.add_scalebar(ax, 235, location="lower left", color="white", box_alpha=0)
type(bar).__name__                              # 'ScaleBar'
```

#### `scalebar_nm`
`bexa.viz.images.scalebar_nm(scalebar, source)`

Nanometres per pixel for a `scalebar` argument: a number as given, `True` from
`source.attrs["effective_pixel_nm"]` (`bexa.viz.style.pixel_size_nm`), `None` for `False` or
when the attribute is missing. `show` and the `maps` functions resolve their `scalebar`
through it after the style defaults.

```python
# image: the summed image with its pixel size in the attrs; the nm per pixel of a scalebar argument
images.scalebar_nm(235, image), images.scalebar_nm(True, image), images.scalebar_nm(False, image)   # (235.0, 235.0, None)
```

### bexa.viz.curves: rocking curves, pump-probe panels and moments

1-D figures: intensity against a motor, laser on/off comparisons along the scan axis of a
pump-probe cube, and per-frame statistics.

```python
import bexa
from bexa.testing import make_legacy_cube
from bexa.viz import curves

scan = bexa.demo("rocking")                      # mu scan, 25 frames of 64 x 80 pixels
peak = bexa.ROI(y=(16, 48), x=(20, 60))         # a pixel window
curve = scan.rocking_curve(peak)                 # DataArray (mu,): intensity inside the window
stats = scan.stats()                             # frame_sum, frame_mean, frame_max, frame_com_y, frame_com_x
cube = bexa.load(make_legacy_cube("run42.h5"))   # pump-probe cube: frames (laser, delay, y, x), signal
```

#### `rocking_curves`
`bexa.viz.curves.rocking_curves(curves, fits=None, ax=None, normalize=False, xlabel=None, figsize=None)`

Rocking curves as dots against the motor, one colour per curve. `curves` is a 1-D DataArray or
`{label: DataArray}`. `fits` maps the same labels to `(x, y)` arrays, drawn as a line in the
curve's colour (lmfit `ModelResult`s fitted with an `x=` keyword also work); for a `FitResult`
of `bexa.analysis.fitting` pass `(x, fit.best_fit)`, as the object itself raises
`AttributeError`. `normalize=True` scales every curve and fit to 0..1 (minimum to maximum). The
x label comes from the coordinate unless `xlabel` is given. Returns `(fig, ax)`.

```python
# curve: the rocking curve of the demo mu scan; shown with its Gaussian fit
from bexa.analysis.fitting import fit_multi_gaussian
x = curve.coords["mu"].values
fit = fit_multi_gaussian(x, curve.values, n=1, background="constant")    # needs lmfit
fig, ax = curves.rocking_curves({"peak": curve}, fits={"peak": (x, fit.best_fit)})
fig, ax = curves.rocking_curves({"peak": curve, "detector": stats["frame_sum"]}, normalize=True)
```

#### `curve`
`bexa.viz.curves.curve(da, ax=None, label=None, figsize=None, **kwargs)`

Line plot of a 1-D DataArray against its coordinate (the index when the dim has none). The
legend label defaults to the array's name, the x label carries the coordinate's units, the y
label is the name; other keywords go to `ax.plot`. Returns `(fig, ax)`; pass `ax` to add lines.

```python
# stats: the per-frame statistics of the demo scan; the centre of the spot against mu
fig, ax = curves.curve(stats["frame_com_x"])
fig, ax = curves.curve(stats["frame_com_y"], ax=ax, ls="--")
len(ax.lines)                                   # 2
```

#### `on_off_diff`
`bexa.viz.curves.on_off_diff(ds, roi=None, axis=None, mode="diff", axes=None, figsize=None, title=None)`

The pump-probe figure of a cube with `frames(laser, <axis>, y, x)` (from `bexa.load` or
`bexa.analysis.pump_probe`), in three panels: the image summed over laser and scan axis with the
ROI box; the ROI intensity with laser on (red) and off (black) against the scan axis; and their
difference, `mode="diff"` (on - off), `"ratio"` (on / off) or `"percent"` (100 (on - off) / off).
The scan axis is `axis`, else `ds.attrs["scan_axis"]`, else the first dim of `frames` other than
`laser`, `y`, `x`. Without `roi` the whole frame counts; `axes` takes three existing axes.
Returns `(fig, axes)`. `cube.bexa.plot()` draws the same figure.

```python
# cube: a laser on/off cube; the summed image, the on and off curves and their difference
fig, axes = curves.on_off_diff(cube, roi=bexa.ROI(y=(5, 15), x=(6, 18)), mode="percent", title="run 42")
fig, axes = cube.bexa.plot(roi=bexa.ROI(y=(5, 15), x=(6, 18)))
axes.shape                                      # (3,)
```

#### `moments_vs_axis`
`bexa.viz.curves.moments_vs_axis(stats, keys=None, axes=None, difference=True, figsize=None)`

Per-frame statistics (sum, centre of mass, variance, skew, ...) against the scan axis, one panel
per quantity, up to three to a row. `stats` maps names to 1-D DataArrays (a Dataset such as
`scan.stats()` works); `keys` picks and orders them. Names ending in `_on` and `_off` share a
panel (on red, off black) with, when `difference`, a dashed green on - off line. `axes` takes
existing axes. Returns `(fig, axes)`.

```python
# stats: the per-frame statistics of the demo scan; three of them against mu
fig, axes = curves.moments_vs_axis(stats, keys=["frame_com_y", "frame_com_x", "frame_max"])
on = cube["frames"].sel(laser="on").sum(("y", "x"))
off = cube["frames"].sel(laser="off").sum(("y", "x"))
fig, axes = curves.moments_vs_axis({"sum_on": on, "sum_off": off})      # one panel: on, off, on - off
```

#### `linearity`
`bexa.viz.curves.linearity(detector, i0, laser=None, ax=None, figsize=None)`

Detector-linearity check: one point per shot or frame, detector signal on the x axis and the
incident monitor I0 on the y axis. With `laser` flags (1 or `True` for on), laser-on points are
red dots and laser-off points black crosses. Returns `(fig, ax)`.

```python
# cube: a laser on/off cube; the detector signal against the monitor, laser on and off
det = cube["frames"].sum(("y", "x"))                                  # (laser, delay)
flag = (det["laser"] == "on").broadcast_like(det)
fig, ax = curves.linearity(det.values.ravel(), cube["signal"].values.ravel(), laser=flag.values.ravel())
```

### bexa.viz.maps: centre-of-mass, bivariate, HSV and vector-field maps

The per-pixel maps of the lab's notebooks: the RdBu centre-of-mass map, the bivariate blend with
its key, the darfix-style HSV image, quiver plots and cluster labels.

```python
import bexa
from bexa.viz import maps

scan = bexa.demo("mosa")                         # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
res = scan.com(axes=("chi", "mu"), sigma=1.0)    # com_chi, width_chi, com_mu, width_mu, total (y, x), deg
```

#### `com_map`
`bexa.viz.maps.com_map(com, ax=None, center="mean", span=None, span_fraction=DEFAULT, cmap=DEFAULT, scalebar=DEFAULT, title=None, colorbar=DEFAULT, figsize=None, clim=None)`

A centre-of-mass map with colour limits `center +/- span / 2`, so tilts around a reference read
as red and blue. `center` is `"mean"`, `"median"` or a value in the map's units; `span`
defaults to `span_fraction` (a fifth) of the map's range (`style.set(com={"span": 0.1})` sets it
for every COM map); `clim` (a `resolve_clim` spec) replaces centre and span. `cmap` defaults
to the style's `diverging_cmap` (RdBu). The colourbar is labelled with the
name and `attrs["units"]` (`com_mu (deg)`), the title defaults to the name, ticks are removed and
`scalebar` works as in `images.show`. For width maps use `center="median", cmap="viridis"`, as
`bexa.plot` does for `width_*`. Returns `(fig, ax, im)`.

```python
# res: the COM maps of the demo scan; the COM in mu, then a width map
fig, ax, im = maps.com_map(res["com_mu"])                               # RdBu around the mean
fig, ax, im = maps.com_map(res["width_mu"], center="median", cmap="viridis")
fig, ax, im = maps.com_map(res["com_chi"], center=0.0, span=0.1, scalebar=235, title="COM chi")
im.get_clim()                                                           # (-0.05, 0.05)
```

#### `bivariate`
`bexa.viz.maps.bivariate(map_a, map_b, ax=None, cmaps=(BLUE_RED, GREEN_PURPLE), labels=None, limits=(None, None), key=True, scalebar=DEFAULT, title=None, figsize=None)`

Two maps in one RGB image, for example the centre of mass in chi and in mu. Each map is
normalised to 0..1 over its minimum and maximum (or the `(low, high)` pair in `limits`; NaN goes
to the low end), sent through its colormap (blue-white-red and green-white-purple by default),
and the two RGB images are averaged. `key=True` adds an inset (top right) with the colour of
every pair of values, its axes in the maps' units and labelled with `labels` (default: the
DataArray names). `scalebar` as in `images.show`. Returns `(fig, ax)`.

```python
# res: the COM maps of the demo scan; the tilts in chi and mu in one colour image
fig, ax = maps.bivariate(res["com_chi"], res["com_mu"], labels=("chi (deg)", "mu (deg)"))
fig, ax = maps.bivariate(res["com_chi"], res["com_mu"], limits=((-0.2, 0.2), None), key=False)
len(ax.images), len(fig.axes)                   # (1, 1)
```

#### `hsv_mosaicity`
`bexa.viz.maps.hsv_mosaicity(map_a, map_b, ax=None, labels=None, key=True, scalebar=DEFAULT, title=None, figsize=None)`

darfix-style mosaicity image: hue from `map_a` and saturation from `map_b`, each normalised over
its minimum and maximum, at full brightness, with an inset key; `labels`, `scalebar` and
`title` as in `bivariate`. Returns `(fig, ax)`.

```python
# res: the COM maps of the demo scan; chi as hue, mu as saturation
fig, ax = maps.hsv_mosaicity(res["com_chi"], res["com_mu"], labels=("chi", "mu"))
fig, ax = maps.hsv_mosaicity(res["com_chi"], res["com_mu"], key=False, scalebar=False)
len(ax.images)                                  # 1
```

#### `vector_field`
`bexa.viz.maps.vector_field(u, v, ax=None, step=8, background=None, cmap="twilight", wheel=True, scale=None, title=None, figsize=None)`

Quiver plot of the field `(u, v)`, one arrow every `step` pixels, coloured by direction
`atan2(v, u)` with a cyclic colormap. `background` is drawn in grey underneath, `wheel` adds a
`color_wheel` key (lower left), `scale` is matplotlib's quiver scale; the y axis points down as
in images and NaN arrows are left out. A new figure is 8 x 6 inches. Returns `(fig, ax)`.

```python
# res: the COM maps of the demo scan; the tilt of every pixel as an arrow
from bexa.analysis import fields
u, v = fields.vector_field(res["com_mu"].values, res["com_chi"].values, center="median")
fig, ax = maps.vector_field(u, v, step=4, background=res["total"], title="tilt direction")
len(ax.collections)                             # 1
```

#### `color_wheel`
`bexa.viz.maps.color_wheel(ax, cmap="twilight", n=100)`

Draw a ring of the colormap against angle (-pi to pi in `n` steps) on a polar axes: the key of
`vector_field`. Returns `None`.

```python
# the hue wheel that keys the arrows of a vector field
import matplotlib.pyplot as plt
maps.color_wheel(plt.figure(figsize=(2, 2)).add_subplot(projection="polar"))   # the hue wheel on a polar axes
```

#### `contour_quiver_overlay`
`bexa.viz.maps.contour_quiver_overlay(background, u, v, ax=None, levels=8, step=8, cmap="viridis", figsize=None)`

The image `background` (colormap `cmap`) with `levels` white contour lines and the field `(u, v)`
as black arrows every `step` pixels. Returns `(fig, ax)`.

```python
# res, u, v: the total map and the tilt vectors of the demo scan; contours with arrows on top
fig, ax = maps.contour_quiver_overlay(res["total"], u, v, levels=6, step=4)
fig, ax = maps.contour_quiver_overlay(res["total"], u, v, cmap="gray", step=8)   # grey background, an arrow every 8 pixels
```

#### `cluster_map`
`bexa.viz.maps.cluster_map(labels, ax=None, cmap="tab10", title=None, figsize=None, colorbar=DEFAULT)`

Integer cluster labels (from `bexa.analysis.segmentation`, for example) with a discrete
colormap, no interpolation and a colourbar labelled `cluster`. Returns `(fig, ax, im)`.

```python
# u, v: the tilt vectors of the demo scan; pixels clustered by tilt, shown as labels
from bexa.analysis import segmentation
labels = segmentation.kmeans_com(u, v, n_clusters=3)            # by tilt direction and size; needs scikit-learn
fig, ax, im = maps.cluster_map(labels, title="k-means on the COM maps")
```

Constants:

- `BLUE_RED`, `GREEN_PURPLE`: matplotlib colormaps blue-white-red and green-white-purple, the
  default `cmaps` of `bivariate`.

### bexa.viz.volume: projections, slices, isosurfaces and 3-D renderings

Views of volumes `(motors..., y, x)` and z-stacks `(z, y, x)`; the last two dims are the pixels
whatever their names (`yb`, `xb` for a preview binned on its own). `projections_panel`,
`isosurface` and `rlp_scatter` draw interactive plotly figures with `backend="plotly"`; `render`
is plotly only. Show plotly figures in a notebook with `show_plotly`.

```python
import bexa
from bexa.viz import volume

scan = bexa.demo("mosa")                         # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
prev = scan.preview(downsample=(1, 1, 4, 4))     # DataArray (chi, mu, y, x) = (6, 16, 16, 20)
ds = bexa.demo("zstack")                         # a mosaicity scan and an energy series at 5 heights
stack = bexa.stack(ds.select(type="fscan2d"), [bexa.acc.Sum()], dim="samz")   # sum (samz, y, x) = (5, 40, 48)
```

#### `projections_panel`
`bexa.viz.volume.projections_panel(data, method="sum", keys=None, log=False, cmap=DEFAULT, figsize=None, backend="mpl", clim=DEFAULT, colorbar=DEFAULT)`

Every projection of a volume in one figure: the detector image `xy`, each motor against each
pixel axis (`chi_y`, `chi_x`, `mu_y`, `mu_x`) and each pair of motors (`chi_mu`). `data` is a
DataArray `(motors..., y, x)` (the last two dims are the pixels under any names; the panels
keep the `_y`, `_x` names) reduced with `method` (`"sum"`, `"max"`, `"mean"` or any other
DataArray reduction), or a mapping of names to 2-D DataArrays such as
`scan.reduce([bexa.acc.Projections()])` (every 2-D entry except `grid` becomes a panel). `keys`
picks and orders panels; `log=True` shows `log10(1 + I)`. Each panel has its own limits from
`clim` (one spec, the style's percentiles by default), coordinate axes and, with `colorbar`
(the style's default), a colourbar, three to a row of 4.2 x 3.6 inches each (`figsize`, or the
style's `panel_size`). Returns `(fig, axes)`, or with `backend="plotly"` a plotly figure of
heatmaps with the same limits, colormap and size (90 pixels per inch).

```python
# prev: a preview (chi, mu, y, x); every projection of it in one figure
fig, axes = volume.projections_panel(prev, log=True)             # xy, chi_y, chi_x, mu_y, mu_x, chi_mu
fig, axes = volume.projections_panel(prev, method="max", keys=["xy", "chi_mu"])
fig, axes = volume.projections_panel(scan.reduce([bexa.acc.Projections()]))   # from the full frames
fig = volume.projections_panel(prev, log=True, backend="plotly")
```

#### `slices_grid`
`bexa.viz.volume.slices_grid(volume, axis=0, n=9, per_row=3, **kwargs)`

`n` evenly spaced slices along one motor dim, titled with the motor value: `images.tiles` under
a volume name (`indices`, `clim`, `cmap`, `log`, `titles`, `figsize` go to it). Returns
`(fig, axes)`.

```python
# prev: a preview (chi, mu, y, x); six slices along chi
fig, axes = volume.slices_grid(prev, axis="chi", n=6, per_row=3, log=True)
fig, axes = volume.slices_grid(prev, axis="mu", indices=[0, 5, 10, 15], clim="each")   # four chosen slices, each on its own scale
```

#### `isosurface`
`bexa.viz.volume.isosurface(volume, threshold=None, ax=None, downsample=1, color="cyan", alpha=0.6, figsize=None, backend="mpl")`

A 3-D surface of a volume at `threshold` (default: 90th percentile of the finite values). More
than three dims are flattened to `(frames, y, x)`; `downsample` keeps every n-th voxel along
each axis. matplotlib: marching cubes (needs scikit-image) on a 3-D axes (`ax` must be one),
returns `(fig, ax)`; the mesh is drawn with the first array axis along the plot's x axis while
limits and labels assume `(z, y, x)`, so prefer plotly for volumes that are not cubes.
`backend="plotly"`: a plotly `Isosurface` figure, two surfaces from the threshold to the maximum.

```python
# stack: the summed images of a z-stack (samz, y, x); the surface at a threshold
fig, ax = volume.isosurface(stack["sum"], threshold=float(stack["sum"].quantile(0.95)))
fig = volume.isosurface(stack["sum"], backend="plotly")  # a plotly figure of the same surface
```

#### `rlp_scatter`
`bexa.viz.volume.rlp_scatter(coords, intensity, ax=None, threshold=None, cmap="viridis", point_size=20.0, alpha=0.8, log=False, backend="mpl", figsize=None)`

Integrated intensity at every motor point as a 3-D scatter in reciprocal-space (or angular)
coordinates. `coords` maps exactly three names to arrays shaped like `intensity`, for example
the Q grids of `bexa.geometry.transforms.build_coordinate_grids` or
`Roi3dRlpSession.coordinates` (other counts raise `ValueError`). Non-finite points are dropped,
`threshold` keeps intensities at or above it, `log=True` colours by `log10(1 + I)`. Returns
`(fig, ax)` (3-D axes with colourbar; `point_size` is for this view), or a plotly `Scatter3d`
figure with `backend="plotly"`.

```python
# scan: the demo chi x mu scan; its intensity as points in reciprocal space
from bexa.geometry.transforms import build_coordinate_grids
grids = build_coordinate_grids(scan.structure, scan.geometry, space="Q", two_theta=17.9)   # Qx, Qy, Qz
intensity = prev.sum(("y", "x"))                                                   # (chi, mu), like the grids
fig, ax = volume.rlp_scatter(grids, intensity, log=True)
fig = volume.rlp_scatter(grids, intensity, threshold=float(intensity.median()), backend="plotly")
```

#### `render`
`bexa.viz.volume.render(volume, mode="translucent", log=False, clim=DEFAULT, cmap=DEFAULT, scale=(1.0, 1.0, 1.0), spacing=None, units="um", downsample=None, max_voxels=300000, surface_count=None, opacity=None, title=None)`

Volume rendering of a `(z, y, x)` stack with plotly (`go.Volume`): colour and opacity follow the
intensity, and the figure rotates and zooms in the browser. Returns a plotly `Figure`; show it
with `show_plotly` or save it with `fig.write_html(path)`. `da.bexa.render(...)` calls it.

- `mode`: `"translucent"` fades dim voxels out and lets bright ones glow (as napari's
  translucent rendering and the Slices-to-Voxels app), `"mip"` keeps only the brightest voxels,
  like a maximum-intensity projection, `"iso"` draws one surface at the upper colour limit;
  `surface_count` and `opacity` override the mode's settings. `log=True` renders
  `log10(1 + I)`; `clim` is a `resolve_clim` spec, by default the style's `render_clim` (the 5th
  and 99.5th percentiles), and `cmap` the style's `cmap` (viridis).
- Axes (named after the last three dims) in pixels and layers, stretched by `scale` like napari's;
  with `spacing=(dz, dy, dx)`, the voxel size in `units`, they read in those units.
- More than `max_voxels` voxels are block-averaged in `y` and `x` first (factor automatic, or
  `downsample`); the axes keep the full-resolution scale.

```python
# stack: the summed images of a z-stack; a plotly rendering in micrometres
fig = volume.render(stack["sum"], log=True, spacing=(1.0, 0.235, 0.235))   # 1 um layers, 235 nm pixels
fig = stack["sum"].bexa.render(mode="mip", scale=(4, 1, 1), title="z-stack, summed")
fig.write_html("zstack_sum.html")                                          # opens in any browser
```

#### `show_plotly`
`bexa.viz.volume.show_plotly(fig, how="auto", height=600, include_plotlyjs="cdn")`

Show a plotly figure in a notebook without the plotly JupyterLab extension. `how="iframe"` shows
the figure as a self-contained page in an inline frame `height` pixels tall, which works in any
JupyterLab and in VS Code, and returns the IPython `HTML` object; `"native"` calls plotly's own
`fig.show()` (needs the extension) and `"browser"` opens a browser tab, both returning `None`;
`"html"` returns the page as a string and shows nothing. `"auto"` is `"iframe"` under IPython,
`"html"` when the environment variable `BEXA_HEADLESS` is set and `"browser"` otherwise; other
values raise `ValueError`. `include_plotlyjs="cdn"` loads plotly.js from the internet when the
page opens, `True` embeds it (about 5 MB per figure) to work offline.
`bexa.notebook.show_plotly` is the same function.

```python
# fig: a plotly figure; shown in a notebook without the plotly extension, or as HTML
volume.show_plotly(fig)                                         # in Jupyter: an inline frame to rotate
page = volume.show_plotly(fig, how="html", include_plotlyjs=True)   # self-contained page as text
```

### bexa.viz.interactive: sliders and ROI pickers

Browsers step through the motor dims of an array (previews and maps, not whole scans), with one
slider per extra dim: the last two dims are the image and the last three the volume, whatever
their names (`y`, `x`, or `yb`, `xb` for a preview binned on its own). They read one block per
move through `isel`, so a volume stacked on disk
(`bexa.stack(..., store=True)`, `bexa.io.cube.open_lazy`) browses without being loaded, and
`clim="shared"` takes its limits from the stack's per-frame statistics (`shared_limits`)
instead of a pass over everything. `brightest`, `shared_limits` and `total_volume` are
`bexa.core.stacks` functions re-exported here: `brightest` gives the slider positions of the
condition that lights the volume up, for `update(**peak)`, and `total_volume` the intensity
summed over the scan motors (a `(z, y, x)` volume to render or browse). `widgets=None` (default) gives ipywidgets sliders in
a Jupyter kernel (JupyterLab, VS Code interactive window) and matplotlib `Slider`s under the
figure elsewhere (scripts, terminal IPython, or no ipywidgets); `widgets=True` or `False`
forces one. With ipywidgets and a live backend (`%matplotlib widget`, ipympl) the image follows
the knob; with a static backend (`inline`, Agg) figures are pictures, so the sliders act on
release and the figure is redrawn into an output area under them. matplotlib sliders need a
window (`plt.show()` in a script) and stop responding when the browser object is
garbage-collected: keep it in a variable.

```python
import tempfile
import bexa
from bexa.viz import interactive

scan = bexa.demo("mosa")                         # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
prev = scan.preview(downsample=(1, 1, 4, 4))     # DataArray (chi, mu, y, x) = (6, 16, 16, 20)
ds = bexa.open_dataset(bexa.demo("zstack").path, cache=tempfile.mkdtemp())   # a z-stack dataset with a disk cache
vol = bexa.stack(ds.select(type="fscan2d"), [bexa.acc.Preview()], dim="auto", downsample=2, store=True)["preview"]  # the previews stacked on disk
vol = vol.transpose(..., "samz", "y", "x")       # (chi, mu, samz, y, x) = (6, 15, 5, 20, 24), read from the cache file block by block
```

#### `browse`
`bexa.viz.interactive.browse(volume, **kwargs)`

Open a `Browser` on `volume` (keywords as for `Browser`) and return it. `da.bexa.browse(...)`
does the same.

```python
# prev: a preview (chi, mu, y, x); sliders over chi and mu, one frame at a time
b = interactive.browse(prev)                     # sliders chi and mu
logged = prev.bexa.browse(log=True, clim="each")
```

#### `Browser`
`bexa.viz.interactive.Browser(volume, cmap=DEFAULT, clim=DEFAULT, log=False, figsize=None, widgets=None, colorbar=DEFAULT)`

Step through the motor dims of a DataArray `(motors..., y, x)` one image at a time. Every dim
but the last two (the pixels, whatever their names) gets a slider and the title shows the
current values
(`chi = -0.1, mu = 0.06667`); a leading dim may also index quantities (an `xr.concat` of maps
with a `quantity` coordinate), whose names then show in the title. `clim` follows the stack
grammar of `bexa.viz.style.stack_mode`: `"shared"` (the default: one scale for the whole array
at the style's percentiles, from `shared_limits`: the stack's statistics or a sample of frames,
never a full pass over a large or lazy volume), `("shared", 2, 98)`, `"each"` (every frame
rescaled), `("each", lo, hi)`, `(vmin, vmax)`; `cmap`, `figsize` (6 x 5 inches) and `colorbar`
follow the style defaults; `log=True` shows `log10(1 + I)`. A move reads one frame through
`isel` and replaces the image data, so stepping is fast and a lazy volume is never loaded whole.
Members:

- `update(**index)`: set the view from code, see `Browser.update`.
- `set_clim(clim)`: other colour limits on show, see `Browser.set_clim`.
- `redraw()`: redraw the image for the current `index` (knobs and title unchanged).
- `data`: property, the frame on show as a numpy array (after `log`), read once per move.
- `fig`, `ax`, `im` (the `AxesImage`), `title` (its `Text`), `index` (dict of dim to position),
  `motor_dims` (the slider dims), `limits` (`(vmin, vmax)`, `None` for `"each"`), `percentiles`
  (those of the scale), `volume`, `log`.

```python
# prev: a preview (chi, mu, y, x); the browser with matplotlib sliders, every frame rescaled
b = interactive.Browser(prev, clim="each", widgets=False)   # matplotlib sliders under the image
b.motor_dims, b.index                                        # (['chi', 'mu'], {'chi': 0, 'mu': 0})
b.data.shape                                                 # (16, 20): the frame on show
```

#### `Browser.update`
`bexa.viz.interactive.Browser.update(**index)`

Set the view from code: move to the given positions along slider dims (`update(mu=12)`; dims
left out stay), put the knobs there without drawing twice, redraw image and title, and refresh
the figure (in the output area with a static backend). The sliders call it, so a script can step
through conditions and save each view. Positions are integer indices, not motor values.
`VolumeBrowser` has the same method; `RenderBrowser` has its own.

```python
# b: a browser; moved from code and read back
b.update(chi=2, mu=8)
b.title.get_text()                               # 'chi = -0.1, mu = 0.06667'
b.update(mu=int(abs(prev["mu"] - 0.5).argmin()))   # the index of the mu value closest to 0.5
b.fig.savefig("mu_0.5.png")
```

#### `Browser.set_clim`
`bexa.viz.interactive.Browser.set_clim(clim)`

Change the colour limits of a browser on show, any `stack_mode` spec (`"each"`,
`("shared", 2, 98)`, `(vmin, vmax)`), and redraw. `VolumeBrowser` has the same method;
`RenderBrowser.set_clim` renders the block again.

```python
# b: a browser on show; its colour limits changed
b.set_clim("each")                               # every frame on its own 1st to 99th percentile
b.limits                                         # None
b.set_clim((0, 500))
b.limits                                         # (0.0, 500.0)
```

#### `compare`
`bexa.viz.interactive.compare(a, b, **kwargs)`

Two browsers with the same keywords on two arrays with the same dims, the second following the
first: moving a slider of the first (or calling its `update`) moves the second to the same
positions. The link is one way; the second browser's sliders move only the second. Returns
`(first, second)`.

```python
# prev: a preview; two browsers moving together, raw and background-subtracted
raw, clean = interactive.compare(prev, prev - prev.min("mu"))   # raw and background-subtracted frames
raw.update(mu=10)
clean.index                                                      # {'chi': 0, 'mu': 10}
```

#### `browse_volume`
`bexa.viz.interactive.browse_volume(volume, **kwargs)`

Open a `VolumeBrowser` on `volume` (keywords as for `VolumeBrowser`) and return it.
`da.bexa.browse_volume(...)` does the same.

```python
# vol: the previews of a z-stack (chi, mu, samz, y, x); the volume with sliders over chi and mu
view = interactive.browse_volume(vol, log=True)          # sliders chi and mu over the (samz, y, x) volume
by_height = vol.isel(chi=3).sum("mu").bexa.browse_volume(method="sum")   # (samz, y, x) alone: no slider
peak = interactive.brightest(vol, over=vol.dims[:-3])   # {'chi': 2, 'mu': 6}: the condition that lights up the whole stack
view.update(**peak)
```

#### `VolumeBrowser`
`bexa.viz.interactive.VolumeBrowser(volume, view="projections", method="max", log=False, clim=DEFAULT, cmap=DEFAULT, threshold=None, figsize=None, widgets=None, colorbar=DEFAULT)`

Show a volume and step through the conditions in front of it: the last three dims are the volume
(`samz, y, x` for a z-stack) and every leading dim (`chi`, `mu`, `energy`, ...) gets a slider.
`view="projections"` shows the current block projected over `z`, `y` and `x`, by maximum
(`method="max"`) or sum (`"sum"`), updated in place; `view="isosurface"` redraws a marching-cubes
surface at `threshold` on a 3-D axes (needs scikit-image; see the caveat of
`bexa.viz.volume.isosurface`), by default the 90th percentile of the first block, kept fixed so
conditions compare. `clim`, `cmap` and `log` as in `Browser`; `colorbar=True` draws one per
projection (none unless asked: the sliders sit under the panels); one `(z, y, x)` block is read
per move, so
a stack on disk browses without being loaded. Fewer than three dims, an unknown `view` or
`method` raise `ValueError`. Members:

- `current()`: the `(z, y, x)` numpy block at the current positions (after `log`).
- `data`: property, the same block.
- `update(**index)`: set the view from code, as `Browser.update`.
- `set_clim(clim)`: other colour limits on show, as `Browser.set_clim`.
- `redraw()`: redraw projections or surface for the current `index`.
- `fig`, `axes` (projections) or `ax` (3-D axes, isosurface), `images` (the three
  `AxesImage`s), `title`, `index`, `motor_dims`, `volume_dims`, `threshold`, `view`, `method`,
  `limits`, `log`, `volume`.

```python
# view: a volume browser; moved from code, the block on show, the isosurface view
view.update(chi=3, mu=7)
view.current().shape                             # (5, 20, 24)
view.title.get_text()                            # 'chi = 0.1, mu = 0'
surface = interactive.VolumeBrowser(vol, view="isosurface", log=True)   # needs scikit-image
```

#### `browse_render`
`bexa.viz.interactive.browse_render(volume, **kwargs)`

Open a `RenderBrowser` on `volume` (keywords as for `RenderBrowser`) and return it.
`da.bexa.browse_render(...)` does the same.

```python
# vol: the previews of a z-stack; a plotly rendering per condition under ipywidgets sliders
rb = interactive.browse_render(vol, mode="mip", log=True, spacing=(1.0, 0.47, 0.47))   # um: layers, pixels
```

#### `RenderBrowser`
`bexa.viz.interactive.RenderBrowser(volume, height=600, title=None, **render_kwargs)`

ipywidgets sliders for the leading dims of `(conditions..., z, y, x)`; the `(z, y, x)` block at
the current position is drawn by `bexa.viz.volume.render` into an output area under them,
`height` pixels tall, starting at once with the first condition. Each move costs one rendering,
so the sliders act on release. `title` is a prefix of the title (the slider positions follow
it). `render_kwargs` go to `render` (`mode`, `log`, `cmap`, `spacing`, ...); `clim` takes the
stack grammar of `bexa.viz.style.stack_mode`: by default (`"shared"`) every condition is drawn
on one colour scale, `shared_limits(volume, 5, 99.5)` (the style's `render_clim` percentiles,
of `log10(1 + I)` with `log=True`; from the stack's statistics or a sample of frames),
`("shared", lo, hi)` other percentiles, `"each"` or `("each", lo, hi)` every block on its own,
`(vmin, vmax)` a fixed scale. Needs ipywidgets (`ImportError` otherwise; then
`render(volume.isel(...))` per condition). Members:

- `update(**index)`: move to new positions (`update(chi=2, mu=5)`), put the knobs there, render
  once and show it.
- `restyle(**render_kwargs)`: render with other settings, see `RenderBrowser.restyle`;
  `set_clim(clim)` does it for the colour scale.
- `current()`: the `(z, y, x)` block at the current positions, as a DataArray.
- `show(fig)`: draw any plotly figure into the output area.
- `figure` (the last rendering), `sliders` (dict of dim to `IntSlider`), `output`, `index`,
  `motor_dims`, `height`, `render_kwargs`, `volume`. The inherited `redraw()` is not
  implemented; use `update`.

```python
# rb: a render browser; moved from code, the block on show, the shared colour scale
rb.update(chi=3, mu=7)
rb.current().dims                                # ('samz', 'y', 'x')
rb.render_kwargs["clim"]                         # (1.0414, 2.6687): one scale for every condition, log10(1 + I)
rb.figure.write_html("chi3_mu7.html")
```

#### `RenderBrowser.restyle`
`bexa.viz.interactive.RenderBrowser.restyle(**render_kwargs)`

Change how the blocks are rendered (`mode`, `cmap`, `opacity`, `spacing`, `log`, `clim`...)
and render the current block again; the new settings stay for the next moves. A new `clim`
(any `stack_mode` spec) or a change of `log` settles the colour scale again; `set_clim(clim)`
is the short form.

```python
# rb: a render browser; rendered again with other settings
rb.restyle(mode="mip", cmap="magma")
rb.set_clim(("shared", 10, 99))                  # one scale at other percentiles
rb.render_kwargs["clim"]                         # (1.0414, 2.3198)
```

#### `pick_roi`
`bexa.viz.interactive.pick_roi(image, block=False, **kwargs)`

Show `image` and let the user drag a rectangle on it with the left mouse button; returns the
`RoiPicker`, whose `roi` holds the selection once drawn. `block=True` calls
`plt.show(block=True)` and returns when the window closes (scripts); in a notebook use the
widget backend and read `picker.roi` after drawing. Other keywords (`ax`, `cmap`, `clim`) go to
`RoiPicker`.

```python
# prev: a preview; a rectangle dragged on its summed image gives an ROI
picker = interactive.pick_roi(prev.sum(("chi", "mu")))   # drag a rectangle on the summed preview
picker.roi                                               # None until a rectangle is drawn
peak = interactive.pick_roi(prev.sum(("chi", "mu")), block=True).roi   # script: waits for the window
```

#### `RoiPicker`
`bexa.viz.interactive.RoiPicker(image, ax=None, cmap=DEFAULT, clim=DEFAULT)`

A matplotlib `RectangleSelector` on an image (colour limits and colormap from `clim`, `cmap` or
the style defaults; on `ax` or a new figure). Each selection sets `roi` to `ROI(y=(y0, y1), x=(x0, x1))` with exclusive ends, in
full-resolution pixels when the image has `y`/`x` coordinates (a preview's pixels are mapped
back), else in the array's pixels, and writes the range in the title; the rectangle stays
adjustable. Attributes: `roi` (`ROI` or `None`), `selector`, `fig`, `ax`, `image`.

```python
# prev: a preview; the picker on one frame, with its own colour limits
picker = interactive.RoiPicker(prev.isel(chi=3, mu=8), cmap="gray", clim=("p", 5, 99.5))
picker.selector.extents                          # (xmin, xmax, ymin, ymax) of the rectangle, array pixels
```

### bexa.viz.sliders: what the slider browsers share

`Browser`, `VolumeBrowser` and `RenderBrowser` build on `SliderPanel`: one index per leading
dim, an ipywidgets slider per dim in a notebook (matplotlib's own sliders in a script), one
block read per move and colour limits that are shared, per frame or fixed. Nothing here is
called directly in a notebook; subclass `SliderPanel` for a browser of your own.

```python
import bexa
from bexa.viz import sliders

scan = bexa.demo("mosa")                         # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
prev = scan.preview(downsample=(1, 1, 4, 4))     # DataArray (chi, mu, y, x) = (6, 16, 16, 20)
```

#### `SliderPanel`
`bexa.viz.sliders.SliderPanel()`

The base of the browsers. A subclass sets `volume`, `motor_dims`, `index`, `fig` and `title`,
calls `_init_sliders(widgets)` and implements `redraw()` and `_apply_limits()`; it inherits
`update(**index)` (move, put the knobs there, redraw, refresh the figure), `set_clim(clim)`
(any `bexa.viz.style.stack_mode` spec, applied on show), the block on show through `_block()`
(one `isel` per move, `log10(1 + I)` when `log`) and the members `limits` (`(vmin, vmax)` for
a shared or fixed scale, `None` per frame), `percentiles`, `kind` (the style kind its defaults
come from), `_sliders` and `_output` (the output area of a static backend).

```python
# what the browsers inherit from
from bexa.viz.interactive import Browser
issubclass(Browser, sliders.SliderPanel)         # True
```

#### `log_block`
`bexa.viz.sliders.log_block(block)`

`log10(1 + block)` with negative values clipped to 0: how the browsers show intensities with
`log=True`.

```python
# prev: a preview (chi, mu, y, x); one frame as the browsers show it with log=True
sliders.log_block(prev.values[0, 0]).max()       # 1.824
```

#### `in_notebook`
`bexa.viz.sliders.in_notebook()`

`True` inside a Jupyter kernel (JupyterLab, VS Code interactive window), where ipywidgets can
show; the browsers' `widgets=None` asks it.

```python
# whether ipywidgets can show here
sliders.in_notebook()                            # False in a script
```

#### `static_backend`
`bexa.viz.sliders.static_backend()`

`True` when matplotlib's backend makes pictures (`inline`, Agg, PDF, SVG): the browsers then
redraw into an output area and the sliders act on release.

```python
# whether figures are pictures here (inline, Agg)
sliders.static_backend()                         # True under the inline backend or Agg
```

#### `plain_layout`
`bexa.viz.sliders.plain_layout(fig)`

Keep the hand-placed axes of a figure (the sliders under a browser, its colourbar) where they
are, for good: a placeholder layout engine is set on `fig`, so matplotlib never installs the
layout `rcParams` asks for (constrained layout under the screen theme) after a `savefig`, which
cannot lay slider axes out. The browsers call it on their figures; call it on a figure of your own
that places axes by hand. Returns the figure.

```python
# b: a browser whose figure holds slider axes placed by hand
b = Browser(prev, widgets=False)
type(b.fig.get_layout_engine()).__name__                     # 'PlaceHolderLayoutEngine': set by the browser
sliders.plain_layout(b.fig) is b.fig                         # True
```

#### `format_value`
`bexa.viz.sliders.format_value(value)`

A slider value for a title: numbers with four significant digits (`0.06667`), anything else as
text (a quantity name).

```python
# a slider value for a title: a number and a name
sliders.format_value(0.0666667), sliders.format_value("com")   # ('0.06667', 'com')
```

### bexa.viz.animation: GIF and MP4 of a stack

```python
import bexa
from bexa.viz import animation

scan = bexa.demo("rocking")                      # mu scan, 25 frames of 64 x 80 pixels
vol = scan.read()                                # DataArray (mu, y, x) = (25, 64, 80) in memory
```

#### `animate`
`bexa.viz.animation.animate(volume, axis=0, clim=DEFAULT, cmap=DEFAULT, interval=200, title_fmt="{dim} = {value:.3g}", normalize=None, figsize=None, colorbar=DEFAULT)`

A matplotlib `FuncAnimation` stepping through `axis` of a 3-D stack (DataArray or numpy; reduce
larger arrays to three dims first), a frame every `interval` ms when shown, with a colourbar and
the title `title_fmt` filled with `dim` and the coordinate `value` (`index` and the position for
numpy). `clim` follows the stack grammar of `bexa.viz.style.stack_mode` (`"shared"`, the
default: the style's percentiles of the whole stack; `("shared", lo, hi)`; `"each"`;
`("each", lo, hi)`; `(vmin, vmax)`); `cmap`, `figsize` (6 x 5 inches) and `colorbar` follow
the style defaults. `normalize` is applied to every frame first (for example
a division by a background box, as the legacy scripts did). Keep the returned object; save it
with `save_gif` or `save_mp4`, or show it with `IPython.display.HTML(anim.to_jshtml())`.

```python
# vol: the demo mu scan in memory; a frame per mu, every frame rescaled
anim = animation.animate(vol, axis="mu", clim="each", title_fmt="mu = {value:.3f} deg")
flat = animation.animate(vol, normalize=lambda frame: frame / frame[:10, :10].mean())   # background box
```

#### `save_gif`
`bexa.viz.animation.save_gif(anim, path, fps=5)`

Write an animation to a GIF (matplotlib's Pillow writer) at `fps` frames per second, looping,
creating the parent folders; logs and returns the path as a `Path`.

```python
# anim: the animation of the mu scan; written as a GIF
path = animation.save_gif(anim, "figures/rocking.gif", fps=8)
animation.save_gif(anim, "figures/rocking_slow.gif", fps=2).name  # 'rocking_slow.gif'
```

#### `save_mp4`
`bexa.viz.animation.save_mp4(anim, path, fps=10, dpi=150)`

Write an animation to MP4 with ffmpeg at `fps` frames per second and `dpi`, creating the parent
folders. Needs `ffmpeg` on the PATH (`RuntimeError` without it). Returns the path.

```python
# anim: the animation of the mu scan; written as an MP4 (needs ffmpeg)
path = animation.save_mp4(anim, "figures/rocking.mp4", fps=10)   # needs ffmpeg
```

#### `frames_to_gif`
`bexa.viz.animation.frames_to_gif(frames, path, fps=5)`

Write frames that are already images (uint8 arrays, RGB `(y, x, 3)` or grey `(y, x)`) to a
looping GIF with imageio, `1000 / fps` ms per frame, creating the parent folders; returns the
path.

```python
# RGB frames drawn by hand; written as a GIF
import numpy as np
import matplotlib.pyplot as plt
rgb = [(plt.cm.magma(f / f.max())[..., :3] * 255).astype(np.uint8) for f in vol.values]
path = animation.frames_to_gif(rgb, "figures/rocking_magma.gif", fps=10)  # the GIF written from the frames
```

### bexa.viz.napari_app: napari viewers and the apps behind them

napari viewers for a machine with a display. Each app keeps its state in a plain object
(`Roi3dRlpSession`, `SlicesToVoxels`) that works without napari; the `*_app` functions wrap it in
a window with magicgui controls. Only the launchers import napari and magicgui (`ImportError`
with an install hint otherwise); `import bexa.viz` does not load this module. A launcher opens
its window at once (`show=False` builds it hidden): in a script call `napari.run()` afterwards,
in Jupyter run `%gui qt` first. Volumes and layers may name their pixel dims `y`, `x` or, for a
preview binned on its own, `yb`, `xb`: the last two dims are the pixels.

```python
import bexa
from bexa.viz import napari_app

scan = bexa.demo("mosa")                         # chi x mu scan, 6 x 16 frames of 64 x 80 pixels
ds = bexa.demo("zstack")                         # a mosaicity scan and an energy series at 5 heights
mosa = ds.select(type="fscan2d")                 # the chi x mu scan of every height
layers = [s.preview(downsample=(1, 1, 2, 2)) for s in mosa]         # 5 x (chi, mu, 20, 24)
stack = bexa.stack(mosa, [bexa.acc.Sum()], dim="samz")["sum"]       # (samz, y, x) = (5, 40, 48)
```

#### `view_volume`
`bexa.viz.napari_app.view_volume(volume, log=False, show=True, **kwargs)`

Open a napari viewer in 3-D on a volume `(z, y, x)` or any stack of images, as the layer
`volume_layer(volume, log=log, **kwargs)` (keywords `name`, `scale`, `rendering`, `colormap`).
Returns the `napari.Viewer`.

```python
# stack: the summed images of a z-stack; a napari viewer of it (opens a window: shown, not run)
viewer = napari_app.view_volume(stack, log=True, scale=(4, 1, 1))   # z drawn 4 pixels per layer
```

#### `volume_layer`
`bexa.viz.napari_app.volume_layer(volume, name="volume", log=False, scale=None, rendering="mip", colormap="inferno")`

The `LayerSpec` of a volume: float32 data (`log10(1 + I)` with `log`), contrast limits at the
1st and 99th percentiles, `rendering` (`"mip"`, `"translucent"`, `"iso"`, ...), `colormap`,
additive blending and, when given, the voxel `scale` `(z, y, x)`.

```python
# stack: the summed images of a z-stack; the layer napari would show
spec = napari_app.volume_layer(stack, log=True, scale=(4, 1, 1))
spec.name, spec.data.shape                       # ('volume', (5, 40, 48))
```

#### `projection_layers`
`bexa.viz.napari_app.projection_layers(volume, log=True)`

The sums of a `(z, y, x)` volume over `z`, `y` and `x` as `{"xy", "xz", "yz"}` layer specs
named `proj_xy`, `proj_xz`, `proj_yz` (magma, percentile contrast, `log10(1 + I)` with `log`).
Arrays with more dims are flattened to `(frames, y, x)` first.

```python
# stack: the summed images of a z-stack; its three projections as layers
proj = napari_app.projection_layers(stack)
{key: spec.data.shape for key, spec in proj.items()}   # {'xy': (40, 48), 'xz': (5, 48), 'yz': (5, 40)}
```

#### `LayerSpec`
`bexa.viz.napari_app.LayerSpec(data, name, kwargs={})`

The arguments of one `viewer.add_image` call, as a dataclass: `data`, `name` and the other
keywords in `kwargs` (a new empty dict by default).

- `add_to(viewer)`: remove the layer called `name` from `viewer` if there is one, then add this
  image; returns the new napari layer. Any napari viewer model works, including
  `napari.components.ViewerModel()`, which has no window.

```python
# spec: a layer spec; added to a napari viewer model (no window)
from napari.components import ViewerModel
model = ViewerModel()                            # a napari viewer without a window
layer = spec.add_to(model)
[l.name for l in model.layers]                   # ['volume']
```

#### `roi_3drlp_app`
`bexa.viz.napari_app.roi_3drlp_app(scan, downsample=4, show=True)`

The ROI-to-3D-RLP viewer: the `xy`, `xz` and `yz` sum projections of the scan's preview (frames
stacked as z, pixels downsampled by `downsample`: 1, 2, 4, 8 or 16 in the "Projections" panel),
each with a rectangle layer (`roi_xy`, ...) to draw on. "Apply ROI and build volume" cuts the
stack to the rectangles and adds a `volume_<space>` layer whose metadata holds the ROI and the
motor-point coordinates in `angular`, `Q` or `hkl` space (or the reason they are missing). The
logic is `Roi3dRlpSession`. Returns the viewer.

```python
# scan: the demo chi x mu scan; the ROI-to-reciprocal-space app (opens a window: shown, not run)
viewer = napari_app.roi_3drlp_app(scan, downsample=4)   # draw rectangles, then "Apply ROI and build volume"
```

#### `Roi3dRlpSession`
`bexa.viz.napari_app.Roi3dRlpSession(scan, downsample=4, log=True)`

The state of the ROI-to-3D-RLP app without napari. Attributes `scan`, `downsample`, `log`,
`volume` (the preview, `None` until loaded), `roi` (empty at first), `space`. Members:

- `motor_dims`: property, the motor dims of the scan.
- `load(downsample=None)`: build and keep the preview `(motors..., y, x)`, pixels block-averaged
  by `downsample` (a new factor replaces the old), `log10` when `log`; made by `scan.preview`,
  so cached on disk when the scan has a cache. Returns the DataArray.
- `stack()`: the preview as a numpy stack `(frames, y, x)`, loaded when needed.
- `set_roi_from_shapes(shapes)`: set `roi` from rectangles on the projections (see
  `roi_from_shapes`; frame index and preview pixels) and return it.
- `roi_volume()`: the stack cut to the ROI.
- `coordinates(space="angular", crystal=None, **fixed_angles_deg)`: the coordinates of every
  motor point (`bexa.geometry.transforms.build_coordinate_grids`); `"Q"` needs `two_theta=` or a
  crystal and `hkl_center`, `"hkl"` a crystal.
- `com_maps(sigma=3.0)`: `{motor: map}` of the centre of mass along each motor inside the ROI's
  pixels, every frame smoothed first by a Gaussian of `sigma` pixels.

```python
# scan: the demo chi x mu scan; the app's state without a window: load, ROI, points
session = napari_app.Roi3dRlpSession(scan, downsample=2, log=False)
prev = session.load()                                             # DataArray (chi, mu, 32, 40)
roi = session.set_roi_from_shapes({"xy": [[8, 10], [8, 30], [24, 30], [24, 10]]})
session.roi_volume().shape                                        # (96, 16, 20)
grids = session.coordinates("Q", two_theta=17.9)                  # Qx, Qy, Qz on the (chi, mu) grid
com = session.com_maps(sigma=1.0)                                 # {'chi': (16, 20) map, 'mu': (16, 20) map}
```

#### `roi_from_shapes`
`bexa.viz.napari_app.roi_from_shapes(shapes, volume_shape)`

The ROI drawn as rectangles on the projections of a `(z, y, x)` volume of shape `volume_shape`.
`shapes` maps `"xy"`, `"xz"` or `"yz"` to the `(n, 2)` vertices of a rectangle in
`(row, column)` order, as napari shapes layers store them: `xy` gives the `y` and `x` ranges,
`xz` the `z` and `x` ranges, `yz` the `z` and `y` ranges (a later one overrides). Bounds are the
integer parts of the extreme vertices, clipped to the volume; a missing range is the whole axis.
Returns `ROI(z=..., y=..., x=...)` with `z` as an index dim.

```python
# shapes: rectangles drawn on the xy and yz views; the ROI they define
shapes = {"xy": [[2, 3], [2, 15], [12, 15], [12, 3]], "yz": [[1, 2], [4, 2], [4, 12], [1, 12]]}
roi = napari_app.roi_from_shapes(shapes, (5, 40, 48))
roi.get("z"), roi.get("y"), roi.get("x")         # ((1, 4), (2, 12), (3, 15))
```

#### `slices_to_voxels_app`
`bexa.viz.napari_app.slices_to_voxels_app(layers, downsample=4, show=True, scale=(5.0, 0.15, 0.15), dims=None)`

The Slices-to-Voxels app: one frame per z layer, chosen by the two leading dims of the layers
(`chi` and `mu` for a mosaicity scan per layer, `energy` and `mu` for an energy series) and
stacked into a `voxels` layer drawn with the voxel size `scale` `(z, y, x)`. A "GLOBAL" panel
holds two sliders and averaging radii for all layers, a "LOCAL" panel the offset and visibility
of one layer. `layers`, `dims` and the logic as in `SlicesToVoxels`. Returns the viewer.

```python
# layers: the previews of a z-stack; the slices-to-voxels app (opens a window: shown, not run)
viewer = napari_app.slices_to_voxels_app(layers, scale=(2.1, 1.0, 1.0))   # 1 um layers / 470 nm pixels
```

#### `SlicesToVoxels`
`bexa.viz.napari_app.SlicesToVoxels(layers, downsample=4, dims=None)`

The state of the Slices-to-Voxels app without napari. `layers` holds one 4-D array or DataArray
`(dim0, dim1, y, x)` per z layer (3-D layers get a second dim of length 1), or one `Scan` per
layer, previewed with pixels block-averaged by `downsample`. `dims` names the two leading dims
(default: from the first layer; `("phi", "mu")` for plain arrays). The attributes keep the legacy
names whatever the dims: `phi_idx`, `mu_idx` (global positions), `phi_rad`, `mu_rad` (averaging
radii), `offsets` (`(n_layers, 2)` integer shifts), `visible` (a flag per layer), `phi_count`,
`mu_count`, `layers` (float32 arrays), `dims`. Members:

- `mean_frame(layer)`: the mean frame `(y, x)` of layer `layer` within the radii around the
  global position plus the layer's offset (clipped to the grid).
- `build_volume(log=True)`: the `(z, y, x)` float32 volume of the visible layers' mean frames
  (hidden layers are zeros); `log10` of the values clipped at 1e-6 when `log`.

```python
# layers: the previews of a z-stack; the app's state without a window
model = napari_app.SlicesToVoxels(layers)                # dims from the previews: ('chi', 'mu')
model.phi_idx, model.mu_idx, model.mu_rad = 3, 7, 1       # global (chi, mu) position, mu averaged +/- 1
model.offsets[2] = (0, 1)                                 # layer 2 takes the frame one mu step later
model.visible[4] = False
model.build_volume(log=True).shape                        # (5, 20, 24)
```

### bexa.notebook: Jupyter and VS Code helpers

#### `setup`
`bexa.notebook.setup(backend="widget", wide=True, autoreload=False, log_level="INFO", theme=None)`

Configure a Jupyter kernel for bexa; call it once at the top of a notebook. It sends bexa's log
messages to the cell output at `log_level` (outside IPython it does only this and says so), runs
`%matplotlib <backend>` with `"widget"` (ipympl, interactive figures; `"inline"` with a warning
when ipympl is missing), `"inline"` or `"qt"`, loads `%autoreload 2` when `autoreload=True`
(edited bexa modules reload by themselves), and with `wide=True` widens the output cells to the
window. `theme="screen"` or `"paper"` applies that `bexa.viz.style` theme (fonts, dpi, ticks)
to matplotlib, in or outside IPython. Changes the IPython session's state; returns `None`.

```python
# the first cell of a notebook: backend, wide cells, logging
import bexa
bexa.notebook.setup()                                    # widget backend, wide cells, INFO logs in the cell
bexa.notebook.setup(backend="inline", autoreload=True, log_level="DEBUG")
bexa.notebook.setup(theme="paper")                       # the paper theme for every figure
```

#### `show_plotly`
`bexa.notebook.show_plotly(fig, how="auto", height=600, include_plotlyjs="cdn")`

Show a plotly figure in the notebook; the same function as `bexa.viz.volume.show_plotly`. The
default is an inline frame, which works in any JupyterLab without the plotly extension;
`how="native"` uses plotly's own display where the extension is installed.

```python
# fig: a plotly rendering of the demo mu scan; shown inline
fig = bexa.viz.volume.render(bexa.demo("rocking").read(), log=True)
bexa.notebook.show_plotly(fig)                           # drag to rotate, scroll to zoom
bexa.notebook.show_plotly(fig, how="native")             # with the plotly JupyterLab extension
```

#### `in_ipython`
`bexa.notebook.in_ipython()`

`True` inside IPython, a Jupyter kernel or a VS Code interactive window, `False` in a plain
Python script.

```python
# whether this runs under IPython
bexa.notebook.in_ipython()                       # False in a script
```

#### `in_kernel`
`bexa.notebook.in_kernel()`

`True` inside a Jupyter kernel (JupyterLab, VS Code interactive window), where ipywidgets can
show; `False` in terminal IPython and in scripts. `ScanPicker(interactive=None)` and the
browsers' `widgets=None` ask it.

```python
# whether this runs in a Jupyter kernel, where widgets show
bexa.notebook.in_kernel()                        # False in a script
```

### bexa.notebook.picker: choosing scans in a notebook

What the data cell of a notebook used to spell out by hand (the folders, `SCANS`, `FILTERS`, the
stack axes) in one object: the datasets under a `RAW_DATA` root, the scans listed and selected,
the motors to extract from them and which motor the stack is built on. The selection may span
dataset folders: listing another folder keeps the scans selected so far, and scans chosen there
join them (`bexa.notebook.listing`, `bexa.notebook.selection`). In a Jupyter kernel with
ipywidgets a `ScanPicker` shows as a widget (`bexa.notebook.picker_widgets`): a list of the
scans (shift and ctrl for ranges; `view="table"` for a checkbox per row with sortable columns,
`"gallery"` for thumbnails), the selection panel with the selected scans of every folder
(`bexa.notebook.selection_panel`), one toggle button per motor that moves within a scan or
between the selected scans (the defaults pressed: the scanned motors, and what
`Dataset.varying` finds between scans; `ccmth` next to `energy`, `samz` next to `uz`
released), and a dropdown for the stack axis. Anywhere else it shows as a table, and the same
methods drive it from code. The notebook `examples/13_esrf_dfxm_general.ipynb` uses it in
section 3.

```python
import tempfile
import bexa
from bexa.notebook.picker import ScanPicker, list_datasets

from bexa.testing import make_esrf_zstack

folder = bexa.demo("zstack").path                # a z-stack dataset: a chi x mu scan and an energy series at 5 heights
annealed = make_esrf_zstack(folder.parent, dataset="demo_annealed", tilt_offset={"mu": 0.05}).dataset_dir   # the same z-stack again, after annealing
picker = ScanPicker(root=folder.parent, folders=[folder], filters={"type": "fscan2d"},  # the chi x mu scans, no widgets
                    interactive=False, cache=tempfile.mkdtemp())   # the chi x mu scans, no widgets
```

#### `list_datasets`
`bexa.notebook.picker.list_datasets(root, depth=2, format=None)`

The dataset folders of numbered scans under `root`, `depth` levels down (`RAW_DATA`, sample,
dataset at the ESRF), of any layout whose folders hold numbered scans: an ESRF BLISS dataset (a
master `.h5` file beside `scanNNNN` folders) or an APS 7-ID SPEC session. It is a quick
`bexa.io.survey.survey`: no HDF5 file is opened, the file-system part of the fingerprints decides.
`format="auto"` sniffs every folder fully, a spec name keeps the folders that spec recognises.
Hidden folders, `bexa_cache` and `__pycache__` are skipped, a recognised folder is not entered;
a dataset folder given as `root` comes back alone; a missing root raises `FileNotFoundError`.

```python
# folder: the demo z-stack dataset; the dataset folders under it and under its parent
list_datasets(folder)                            # [the dataset folder itself]
[p.name for p in list_datasets(folder.parent)]   # ['demo_annealed', 'demo_zstack']
list_datasets(folder.parent, format="auto")      # the same, each checked against the format specs
```

#### `ScanPicker`
`bexa.notebook.picker.ScanPicker(root=None, folders=None, scans=None, filters=None, *, view="list", cache=True, cache_reductions=None, interactive=None, state_file=None, status="finished", page_size=50, depth=2, motors=None, stack_axis=None, **open_kwargs)`

The scans of a beamtime to analyse: a `bexa.notebook.listing.ScanListing` (the folders listed,
the filters, the selection across folders) with the motors, the roles, the widgets and the
state file. `root` is a `RAW_DATA` folder whose datasets are listed (`None` lists the given
`folders` only); `folders` the dataset folder(s) listed at first as one
`bexa.core.dataset.Dataset` with running ids (`None`: the first dataset under `root`); `scans`
and `filters` the code-level starting point, as `Dataset.select_numbers` takes them: `filters`
(`type`, `motors`, `where`, positioner ranges) decide what is listed at all, `scans` what is
selected at first (`None`: every listed scan that can be selected). `status` says which scans
may be selected: `"finished"` (complete and aborted, the default), `"complete"` or `"any"`;
scans still running, empty or unreadable are listed greyed and never selected. `view` is
`"list"`, `"table"` or `"gallery"`; `interactive=None` builds the widgets inside a Jupyter
kernel with ipywidgets, `False` never, `True` without a kernel (tests); `state_file` keeps the
selection, the pressed motors, the stack axis and the filters across kernel restarts as long
as the code defaults are unchanged (edited code wins); `motors` and `stack_axis` are the
code-level override of the buttons; `cache`, `cache_reductions` and `open_kwargs` go to
`bexa.open_dataset`. `depth` is how many folder levels below `root` the datasets are looked
for (`list_datasets`; 2 is `RAW_DATA`, sample, dataset). Members:

- `dataset`, `folders` (the dataset and folders listed), `root`, `datasets` (the folders
  under `root`), `rows` (the full table of the listing), `candidates` (the ids listed),
  `selectable` (those that may be selected), `ids` (the selected scans among those listed),
  `selection` (every selected scan, of every folder, as `(folder, scan number)`),
  `selected_in(folder)` (the scan numbers selected in one folder), `scans` (the selected
  `Scan`s of every folder, read with the pressed motors as the dims of their grids), `table`
  (one row per selected scan in the order of `scans`, its folder in `dataset`, the positioners
  that differ), `rows_of(ids)`.
- `motor_candidates()`, `motors` (the pressed motors), `dims` (the pressed motors that move
  within a scan, in loop order), `stack_axis`, `series_axes` (the pressed motors that differ
  between the scans and are not the stack axis), `scan_axes` (`dims + series_axes`), `axes`
  (`[stack_axis, *series_axes]`, what `bexa.stack` is given as `dim`), `varying()`,
  `roles_text()`, `status_text()`, `selection_text()`.
- `select(ids)`, `select_all()`, `remove(scans)`, `clear()`, `set_filters(...)`,
  `set_folders(folders)`, `refresh()`, `reset()`, `set_motors(names)`, `toggle_motor(name)`,
  `reset_motors()`, `set_stack_axis(name)`, `observe(callback)`, `thumbnails(size)`,
  `summary_line(row)`, `selection_lines()`, `selection_view()`, `save_state()`, `show()`,
  `close()`; `widgets` (the `PickerWidgets`, or `None`), `ui` (the bars' filters), `view`,
  `page_size`, `state_file`.

```python
# picker: the chi x mu scans of the demo z-stack, no widgets; what it selected and extracts
picker.ids                                       # [1, 7, 13, 19, 25]: the chi x mu scan at every height
picker.candidates == picker.selectable           # True: every listed scan is finished
picker.motors, picker.dims                       # (['chi', 'mu', 'samz'], ['chi', 'mu'])
picker.stack_axis, picker.axes                   # ('samz', ['samz'])
picker.roles_text()                              # 'scan axes: chi (6) x mu (15); stack axis: samz (5)'
[s.source.scan for s in picker.scans]            # [1, 7, 13, 19, 25]
picker.table.set_index("scan")["samz"].to_dict()   # {1: -0.002, 7: -0.001, 13: 0.0, 19: 0.001, 25: 0.002}
picker                                           # the table with a 'selected' column (the widgets in a kernel)
```

#### `ScanPicker.select`
`bexa.notebook.picker.ScanPicker.select(ids)`

Select these scans among those listed, and none of the other listed ones; the scans selected in
other folders, or hidden by a filter, stay selected. Ids that are not listed or not selectable
(a scan still running) are reported in a warning and left out. `select_all()` takes every
listed scan that can be selected, `select([])` unselects the listed ones, `remove(scans)`
unselects ids of the listed scans or `(folder, scan number)` pairs from `selection`, and
`clear()` empties the whole selection, every folder. Every change calls the `observe`
callbacks and writes the state file.

```python
# picker: the picker of the demo z-stack; two scans chosen, then all
picker.select([1, 7])
picker.status_text()                             # '2 of 5 scans selected'
picker.varying()                                 # {'samz': array([-0.002, -0.001])}
picker.select_all()
len(picker.ids)                                  # 5
```

#### `ScanPicker.set_filters`
`bexa.notebook.picker.ScanPicker.set_filters(**ui)`

Narrow the listing the way the bars do: `type="fscan3d"`, `motors="chi x mu"` (the exact
text of the motors column), `status="finished"` (`"any"`, the default listing, `"finished"`
or `"complete"`). Filters change what is listed, never the selection: a selected scan a filter
hides stays selected (the selection panel shows it). Unknown names raise `ValueError`. The
code-level `filters` of the constructor stay underneath; in another folder whose table lacks a
column they name, every scan is listed, with a warning.

```python
# picker: the picker of the demo z-stack; the listing narrowed by status
picker.set_filters(status="complete")            # the aborted scans leave the list
picker.candidates                                # [1, 7, 13, 19, 25]
picker.set_filters(status="any")
picker.ui                                        # {'type': 'any', 'motors': '', 'status': 'any'}
```

#### `ScanPicker.set_motors`
`bexa.notebook.picker.ScanPicker.set_motors(names, notify=True)`

Press exactly these motors (`toggle_motor(name)` flips one, `reset_motors()` goes back to the
defaults and the default stack axis). The candidates are those of `motor_candidates()`: the
scanned motors of the first selected scan and every other recorded motor that moves within it
(an energy mosa lists `energy`, `chi`, `mu` and `ccmth`), then every positioner that differs
between the selected scans (a z-stack at several energies lists `uz`, `energy`, `ccmth` and
the stage readback `samz`). A name that is not a candidate raises `ValueError`. The pressed
motors that move within a scan become the dims of every selected scan (`Scan.with_dims` when
they differ from bexa's choice: `ccmth` instead of `energy` reads the mono angle in degrees).

```python
# series: a picker on the energy series of the demo z-stack; the mono angle pressed instead of the energy
series = ScanPicker(folders=[folder], filters={"type": "fscan1d"}, interactive=False, cache=False)   # the energy series
{n: (c.n_values, c.where, c.default) for n, c in series.motor_candidates().items()}  # {'mu': (8, 'scan', True), 'samz': (5, 'stack', True), 'energy': (5, 'stack', True), 'cc...
# {'mu': (8, 'scan', True), 'samz': (5, 'stack', True), 'energy': (5, 'stack', True), 'ccmth': (5, 'stack', False)}
series.axes                                      # ['samz', 'energy']
series.toggle_motor("ccmth")
series.toggle_motor("energy")                    # the mono angle instead of the energy
series.axes, series.varying()["ccmth"].round(4)  # (['samz', 'ccmth'], array([6.6705, 6.6744, 6.6784, 6.6823, 6.6863]))
series.reset_motors()
series.motors                                    # ['mu', 'samz', 'energy']
```

#### `ScanPicker.set_stack_axis`
`bexa.notebook.picker.ScanPicker.set_stack_axis(name, notify=True)`

Make `name` the stack axis: one of the pressed motors that differ between the scans (the
default is the leading one in `Dataset.varying`'s order: `uz` for a z-stack, `nanodac1_temp`
for a temperature series); `None` or `"none"` for no stack axis (every pressed motor is then a
scan axis), `"default"` for bexa's choice. Another name raises `ValueError`.

```python
# series: the picker of the energy series; the energy as the stack axis, then none, then the default
series.set_stack_axis("energy")
series.axes, series.series_axes                  # (['energy', 'samz'], ['samz'])
series.set_stack_axis(None)
series.stack_axis, series.scan_axes              # (None, ['mu', 'samz', 'energy'])
series.set_stack_axis("default")
series.stack_axis                                # 'samz'
```

#### `ScanPicker.varying`
`bexa.notebook.picker.ScanPicker.varying()`

The values of every axis of `axes` over the selected scans, `{dim: sorted values}`, as
`bexa.core.dataset.varying(scans, axes=picker.axes)` gives them; `{}` when nothing is stacked.
The notebook's `position_of` and the stacks read it.

```python
# picker: the picker of the demo z-stack; the heights its selection is stacked on
picker.varying()                                 # {'samz': array([-0.002, -0.001, 0., 0.001, 0.002])}
```

#### `ScanPicker.set_folders`
`bexa.notebook.picker.ScanPicker.set_folders(folders)`

List other dataset folder(s) (several give one list with running ids); the scans selected so
far stay selected, so scans chosen in the new folders join them and `scans` reads them all.
Each set of folders is opened once and kept: listing it again is instant and shows its
selection as it was. A folder that is not a dataset raises `ValueError`, naming the dataset
folders under it when there are some (a sample folder given instead of a dataset folder).
`refresh()` looks at every folder again (scans that finished or started since) and keeps the
selection; `reset()` goes back to the code defaults: the folders, the selection (the scans of
other folders are dropped), the motors and the filters.

```python
# picker: the picker of the demo z-stack; scans of a second dataset folder join its selection
picker.set_folders(annealed)                     # the other dataset listed: the selection stays
picker.ids, len(picker.selection)                # ([], 5): nothing chosen here yet, five scans in the first folder
picker.select([1, 7])                            # two scans of this folder join them
picker.status_text()                             # '2 of 5 scans selected in demo_annealed; 7 in all, from 2 folders'
picker.table[["dataset", "scan"]].values.tolist()[-2:]   # [['demo_annealed', 1], ['demo_annealed', 7]]
picker.set_folders(folder)                       # back, at once: its five scans still selected
picker.remove([(annealed, 1), (annealed, 7)])    # the scans of the other folder leave
picker.refresh()
picker.reset()
picker.ids                                       # [1, 7, 13, 19, 25]
```

#### `ScanPicker.selection_view`
`bexa.notebook.picker.ScanPicker.selection_view()`

Another live view of the selection panel, to display in a cell of its own: in JupyterLab,
right-click that cell's output and choose "Create New View for Cell Output" to get it in a
window beside the notebook. It follows every change, and its Remove, Clear all and List folder buttons
act on the picker. Without widgets it returns `table`.

```python
# picker: the picker of the demo z-stack, no widgets: the table of the selection instead
picker.selection_view()["scan"].tolist()         # [1, 7, 13, 19, 25]
```

#### `ScanPicker.observe`
`bexa.notebook.picker.ScanPicker.observe(callback)`

Call `callback(picker)` after every change of the selection, the motors, the stack axis, the
filters or the folders; the widgets register themselves this way.

```python
# picker: the picker of the demo z-stack; a callback after every change
seen = []
picker.observe(lambda p: seen.append(p.status_text()))
picker.select([13])
seen                                             # ['1 of 5 scans selected']
```

#### `ScanPicker.summary_line`
`bexa.notebook.picker.ScanPicker.summary_line(row)`

One line of the list view for a row of `rows`: the id, the dataset (several folders), the
status, type, motors and shape, the positioners that vary over the listing (the energy as
`17 keV`) and the frames.

```python
# picker: the picker of the demo z-stack; the list line of its first scan
picker.summary_line(picker.rows_of([1]).iloc[0])   # '  1  complete   fscan2d   chi x mu   6 x 15   17 keV  ccmth 6.678  samz -0.002  90 frames'
```

#### `ScanPicker.thumbnails`
`bexa.notebook.picker.ScanPicker.thumbnails(size=128, images=None)`

PNG thumbnails (`{id: bytes}`) of the summed image of every listed scan, for the gallery view:
one pass per scan with a motor stride of 4 and the pixels binned to about `size` across, kept
once computed (and cached by the dataset's cache); `images` maps ids to summed images already
at hand.

```python
# picker: the picker of the demo z-stack; a thumbnail per listed scan
pngs = picker.thumbnails(size=48)
sorted(pngs), pngs[1][:4]                        # ([1, 7, 13, 19, 25], b'\x89PNG')
```

#### `ScanPicker.save_state`
`bexa.notebook.picker.ScanPicker.save_state()`

Write the state file (when the picker has one) and return its path: the hash of the code
defaults, the folders listed, every selected scan as a `[dataset folder, scan number]` pair
(of any folder; ids change when folders are added), the bars' filters, the pressed motors and
the stack axis (`bexa.notebook.picker_state.PickerState`). Called after every change. The file
is read back by the next picker built with the same code defaults: changed defaults are
reported and the file ignored, listed folders that are gone give way to the code folders, and
selected scans whose folder is gone are left out. A file written before the selection could
span folders (folder names instead of paths) is read too.

```python
# a picker with a state file; its selection survives a kernel restart
remembered = ScanPicker(folders=[folder], filters={"type": "fscan2d"}, interactive=False,
                        cache=False, state_file=tempfile.mkdtemp() + "/picker.json")
remembered.select([7, 13])
remembered.save_state().name                     # 'picker.json'
ScanPicker(folders=[folder], filters={"type": "fscan2d"}, interactive=False, cache=False,
           state_file=remembered.state_file).ids   # [7, 13]: the selection survives a restart
```

#### `pick_scans`
`bexa.notebook.picker.pick_scans(root=None, folders=None, **kwargs)`

Build a `ScanPicker` (keywords as above), show it (`show()`: the widgets in a kernel, the table
elsewhere) and return it.

```python
# the demo z-stack; a picker built and shown in one call
from bexa.notebook.picker import pick_scans
shown = pick_scans(folders=[folder], filters={"type": "fscan2d"}, interactive=False, cache=False)
shown.ids                                        # [1, 7, 13, 19, 25]
```

Constants:

- `VIEWS`: `("list", "table", "gallery")`.
- `STATUS_SHOWN`: what the bars' `status` filter lists: `{"any": None, "finished": ("complete",
  "aborted"), "complete": ("complete",)}` (defined in `bexa.notebook.listing`).

### bexa.notebook.listing: the scans listed, and a selection across folders

The part of the scan picker without motors, widgets or state file: the dataset folders under a
root, the scans of the folders listed, the code-level filters and those of the bars, and the
selection, which keeps the scans chosen in any folder. `list_datasets` and `STATUS_SHOWN` are
defined here and imported by `bexa.notebook.picker`.

```python
import bexa
from bexa.notebook.listing import ScanListing

folder = bexa.demo("zstack").path                # a z-stack dataset: a chi x mu scan and an energy series at 5 heights
```

#### `ScanListing`
`bexa.notebook.listing.ScanListing(root=None, folders=None, scans=None, filters=None, *, status="finished", depth=2, **open_options)`

The scans of the dataset folders listed and a selection across folders; `ScanPicker` is a
`ScanListing` with motors, roles, widgets and a state file, and documents the members they
share (`dataset`, `folders`, `rows`, `candidates`, `selectable`, `ids`, `selection`,
`selected_in`, `table`, `select`, `select_all`, `remove`, `clear`, `set_filters`,
`set_folders`, `refresh`, `reset`, `observe`, `status_text`, `selection_text`,
`summary_line`, `selection_lines`, `rows_of`, `close`). The parameters are those of
`ScanPicker` that decide what is listed; `open_options` go to `bexa.core.dataset.Dataset`.

```python
# folder: the demo z-stack; its chi x mu scans listed and two of them selected, no motors, no widgets
listing = ScanListing(folders=[folder], filters={"type": "fscan2d"}, cache=False)
listing.select([1, 7])
listing.status_text(), [number for _, number in listing.selection]   # ('2 of 5 scans selected', [1, 7])
```

### bexa.notebook.selection: scans chosen in several dataset folders

The selection of the scan picker. A scan is named by its dataset folder and number, so the
selection survives listing other folders; each set of folders listed is opened once as a
`Dataset` and kept, and a selected scan is opened through whichever of these datasets holds it.
Used through `ScanPicker`; the class is public for scripts that keep a selection of their own.

```python
import bexa
from bexa.notebook.selection import Selection, folder_path

folder = bexa.demo("zstack").path                # a z-stack dataset: a chi x mu scan and an energy series at 5 heights
```

#### `Selection`
`bexa.notebook.selection.Selection(open_options, statuses)`

The scans chosen in any number of dataset folders, `keys` a set of `(folder, scan number)`
pairs. `open_options` are the keywords of `Dataset` for every folder opened, `statuses` the
statuses a scan must have to be selected (`bexa.core.status.status_filter` gives them; `None`
allows any). Members: `open(folders)` (the `OpenedDataset` of these folders, opened once and
kept in `opened`), `ordered()` (the keys folder by folder, in the order the folders were
opened: `order`), `locate(key)` (the dataset holding a scan and its id there; a folder not
opened yet is opened alone; `KeyError` when no dataset holds it), `scan(key)`, `allowed(key)`
(the scan is on disk, opens and has an allowed status), `row(key)` (its table row),
`label(folder, root=None)` (its name, or its path under `root` when two folders share a name),
`table(root=None)` (one row per selected scan, the positioners that differ), `refresh()` (every
opened folder again; scans gone or no longer allowed leave) and `close()`.

```python
# folder: the demo z-stack; two of its scans chosen by (folder, number), then read
chosen = Selection({"cache": False}, statuses=None)
opened = chosen.open([folder])
chosen.keys = {(folder_path(folder), 13), (folder_path(folder), 1)}
[number for _, number in chosen.ordered()], chosen.allowed((folder_path(folder), 99))   # ([1, 13], False)
chosen.table()[["dataset", "scan", "samz"]].values.tolist()   # [['demo_zstack', 1, -0.002], ['demo_zstack', 13, 0.0]]
```

#### `OpenedDataset`
`bexa.notebook.selection.OpenedDataset(dataset, ids=<factory>)`

A dataset the selection opened, and `ids`, the id of each of its scans by `(folder, number)`;
`index()` maps them again (after a refresh new scans move the running ids of several folders).

```python
# opened: the demo z-stack as the selection opened it; the id of scan 13
opened.ids[(folder_path(folder), 13)]            # 13: one folder, the id is the scan number
```

#### `folder_path`
`bexa.notebook.selection.folder_path(path)`

A folder as the selection names it: absolute, links not followed, so a mapped drive stays a
drive letter.

```python
# a relative folder name made absolute against the working directory
folder_path("RAW_DATA").is_absolute()            # True
```

### bexa.notebook.picker_state: what a picker remembers

The state file of the scan picker: the folders listed, every selected scan by folder and
number, the bars' filters, the pressed motors, the stack axis and the view, with a hash of the
code defaults it was saved under. A state saved under other `FOLDERS`, `SCANS` or `FILTERS` is
ignored: edited code wins.

```python
import tempfile
from bexa.notebook.picker_state import PickerState, defaults_key, stable

folder = tempfile.mkdtemp()                      # any folder for the state file
```

#### `PickerState`
`bexa.notebook.picker_state.PickerState(defaults, root, folders, selected, ui=<factory>, motors=None, stack_axis=None, view="list")`

A picker's choices as its state file holds them: `selected` has one `[folder, scan number]` per
selected scan. `write(path)` writes it as JSON and returns the path; `PickerState.read(path,
defaults)` returns it, or `None` when the file is missing, unreadable or saved under other
code defaults.

```python
# a state written and read back under the same code defaults, then under others
key = defaults_key("RAW_DATA", ["RAW_DATA/sample/dataset"], None, {"type": "fscan2d"}, "finished")
path = PickerState(key, "RAW_DATA", ["RAW_DATA/sample/dataset"], [["RAW_DATA/sample/dataset", 7]]).write(folder + "/picker.json")
PickerState.read(path, key).selected             # [['RAW_DATA/sample/dataset', 7]]
PickerState.read(path, "another key") is None    # True: saved under other code defaults
```

#### `defaults_key`
`bexa.notebook.picker_state.defaults_key(root, folders, scans, filters, status)`

A short hash of a picker's code defaults: the root, the folders, the scans, the filters and the
status. A lambda in the filters counts by its code, so re-running the cell keeps the state.

```python
# the same filters written twice, as a re-run cell would: the same key
same = defaults_key("RAW_DATA", ["a"], None, {"where": lambda row: " x " in row["motors"]}, "finished")
same == defaults_key("RAW_DATA", ["a"], None, {"where": lambda row: " x " in row["motors"]}, "finished")   # True
```

#### `stable`
`bexa.notebook.picker_state.stable(value)`

A JSON-able stand-in for a code default: mappings and sequences element by element, a callable
by its code (not its address), anything else by its `repr`.

```python
# a filter mapping with a lambda: its code stands for it
stable({"type": "fscan2d", "scans": (3, 9)})     # {'type': "'fscan2d'", 'scans': ['3', '9']}
```

### bexa.notebook.picker_widgets: the widget views of the picker

The ipywidgets side of `ScanPicker`: the bars (the root and the datasets under it, each with
the number of scans selected in it, the type, motors and status filters, the view, the
selection buttons and the status line), the scan list in one of three views, the selection
panel (`bexa.notebook.selection_panel`), the motor buttons and the roles row. Built by
`ScanPicker` when it is interactive; every click goes through the picker's methods and the
picker calls `refresh()` back. A root or folder that cannot be opened is reported in red in the
status line. Needs ipywidgets.

```python
import tempfile
import bexa
from bexa.notebook.picker import ScanPicker
from bexa.notebook.picker_widgets import PickerWidgets

folder = bexa.demo("zstack").path                # a z-stack dataset: a chi x mu scan and an energy series at 5 heights
live = ScanPicker(folders=[folder], filters={"type": "fscan2d"}, interactive=True, cache=False)   # the widgets without a kernel
```

#### `PickerWidgets`
`bexa.notebook.picker_widgets.PickerWidgets(picker)`

The widget box of a picker: `box` is what a cell displays (`picker.show()` does), `refresh()`
rebuilds what the picker's state decides (the bars, the status line, the scan list, the
selection panels, the motor buttons and the roles row), `selection_view()` makes another live
selection panel and returns its box (`ScanPicker.selection_view` calls it). The parts are
attributes for scripts and tests: `root_text`, `folder_select`, `open_button`, `type_dd`,
`motors_text`, `status_dd`, `view_dd`, `all_button`, `none_button` (unselects the scans
listed), `invert_button`, `reset_button`, `refresh_button`, `status`, `body` (the view: a
`SelectMultiple` of summary lines, or the table rows with `checkboxes` per id, or the gallery
tiles), `panels` (the `SelectionPanel` under the list, then the extra views), `motor_row`
(one `ToggleButton` per motor), `roles_row` (the stack-axis dropdown, the "Default motors"
button, `roles_text`).

```python
# live: the picker of the demo z-stack with its widgets built; the list and a button driven from code
widgets = live.widgets
type(widgets).__name__, len(widgets.box.children)   # ('PickerWidgets', 8)
widgets.body.children[0].value = (1, 7)          # as a click in the list would
live.ids                                         # [1, 7]
[b.description for b in widgets.motor_row.children]   # ['chi (6)', 'mu (15)', 'samz (2)']
widgets.none_button.click()
live.ids, widgets.status.value                   # ([], '<b>&nbsp;&nbsp;0 of 5 scans selected</b>')
```

### bexa.notebook.selection_panel: the selected scans of every folder

The panel under the scan list of the picker's widgets: one line per selected scan of every
folder (the folder, the scan number, then as in the list), with Remove (the highlighted scans),
Clear all (every folder) and List folder (the folders of the highlighted scans, in the list
above). `ScanPicker.selection_view()` shows another live copy in a cell of its own, which
JupyterLab's "Create New View for Cell Output" turns into a window beside the notebook. Needs
ipywidgets.

```python
import bexa
from bexa.notebook.picker import ScanPicker

folder = bexa.demo("zstack").path                # a z-stack dataset: a chi x mu scan and an energy series at 5 heights
live = ScanPicker(folders=[folder], filters={"type": "fscan2d"}, scans=[1, 7], interactive=True, cache=False)   # the widgets without a kernel
```

#### `SelectionPanel`
`bexa.notebook.selection_panel.SelectionPanel(picker)`

The selected scans of a picker in a `SelectMultiple` (`scans`, whose values are the
`(folder, scan number)` pairs), its `title` (how many scans in how many folders), and
`remove_button`, `clear_button`, `list_button`; `box` is what a cell displays, `refresh()`
shows the picker's selection again (the picker's widgets call it after every change). The
constants `CSS`, `EMPTY` and `HELP` are its style and texts.

```python
# live: the picker of the demo z-stack with two scans selected; its panel driven from code
panel = live.widgets.panels[0]
[label.split()[:2] for label, _ in panel.scans.options]   # [['demo_zstack', '1'], ['demo_zstack', '7']]
panel.scans.value = (panel.scans.options[0][1],)  # highlight the first, as a click would
panel.remove_button.click()
[number for _, number in live.selection]         # [7]
```

### bexa.pipelines.dfxm_report: the standard DFXM figure set

One call turns a dark-field microscopy scan into the figures looked at after every scan, plus an
HDF5 file of the arrays behind them. On the command line: `bexa scan report`.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.pipelines.dfxm_report import report

scan = bexa.demo("mosa")                 # synthetic chi x mu scan, 6 x 16 frames of 64 x 80 pixels
work = Path(tempfile.mkdtemp())          # any folder for the outputs
```

#### `report`
`bexa.pipelines.dfxm_report.report(scan, out_dir, roi=None, downsample=4, sigma=3.0, device="auto", apply_log=False, dpi=150, prefix=None, show_progress=False, plot_options=None)`

One streaming pass with the `Sum`, `Max`, `Projections`, `Preview` and (for a scan with motors)
`MotorCOM(axes=<every motor>, sigma=sigma, moments=2)` accumulators, then the figures.
`downsample` applies to the whole pass, so every map comes out binned: an int or a `(y, x)` pair
bins the pixels (default 4), one factor per dim bins the motors too, `None` keeps full
resolution. `roi` limits the pass to a pixel window (and motor ranges), `apply_log` stores
`log10(1 + I)` in the preview, `device` is `"auto"`, `"cpu"` or `"cuda"`.

Writes into `out_dir` (created if missing) `<prefix>_sum.png`, `_max.png`, `_projections.png`,
`_tiles.png` (up to 9 preview frames along the first motor), `_com_<motor>.png` and
`_width_<motor>.png` per motor, `_bivariate.png` (the centres of mass of the first two motors in
one colour image) and `_results.h5` (every array, with provenance, for `bexa.load`). `prefix`
defaults to the scan name; maps get a scalebar when the effective pixel size is known (needs
matplotlib-scalebar). `plot_options` are style defaults for these figures only, as
`bexa.viz.style.set` takes them (`{"cmap": "magma", "colorbar": False, "com": {"span": 0.1}}`);
the session's defaults are untouched. Returns `{name: Path}`. The command is `bexa scan report PATH -o figures/
[--roi y=700:1500,x=800:1800] [--downsample 4] [--sigma 3] [--apply-log]`.

```python
# scan: the demo chi x mu scan; work: an output folder; the standard figures and arrays written
written = report(scan, work / "report", downsample=(1, 1, 2, 2), device="cpu")
sorted(written)          # ['bivariate', 'com_chi', 'com_mu', 'max', 'projections', 'results', 'sum', 'tiles', 'width_chi', 'width_mu']
written["com_mu"].name   # 'demo_mosa_com_mu.png'
res = bexa.load(written["results"])        # xarray.Dataset: sum, max, xy, chi_mu, preview, com_mu, ...
res["com_mu"].shape                        # (32, 40): binned 2 x 2 like the pass
report(scan, work / "grain", roi=bexa.ROI(y=(8, 56), x=(10, 70)), downsample=None, prefix="grain")  # {'sum': Path('.../grain_sum.png'), 'max': Path('.../grain_max.png'), 'projections': Pat...
report(scan, work / "magma", device="cpu", plot_options={"cmap": "magma", "com": {"span": 0.1}})  # {'sum': Path('.../demo_mosa_sum.png'), 'max': Path('.../demo_mosa_max.png'), 'projectio...
```

### bexa.pipelines.xfel_cube: PAL-XFEL point files to a laser on/off cube

Port of `Aaron_allscan_cube_parallel.py`. A PAL-XFEL run of the September 2025 layout has, per
motor point, a shot table (`type=measurement/run=NNN/scan=NNN/pNNNN.h5`) and a frame file
(`type=raw/...`). Every point is reduced to a mean frame per laser state, the median of each
motor readback and the mean ROI/I0 signal, and the points are stacked into a cube, an
`xarray.Dataset` with `frames(laser, <axis>, y, x)`, `signal(laser, <axis>)` (mean of
`roi_stat / i0`), `n_shots(laser, <axis>)`, coordinates `laser` (`'off'`, `'on'`), `point`,
`index` and the readbacks (`delay`, `phi`, `chi`, `th`, `tth`, `laser_v`, `laser_h`), and attrs
`scan_axis`, `run`, `scan`, `format`, `energy_keV` (when known) and the provenance. The axis is
the first readback of the spec's `axis_priority` that is finite at every point and varies, else
`index`. A `profile` argument is a profile name, a YAML path or a `BeamtimeProfile`.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.config import BeamtimeProfile
from bexa.pipelines import xfel_cube
from bexa.testing import make_pal_run

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
run1 = make_pal_run(work / "raw", run=1, n_points=6)    # 6 delay points of 16 shots, 24 x 32 pixels
profile = BeamtimeProfile(name="demo_pal", format="pal_xfel_points_2025_09",   # or a profile name
                          raw_root=work / "raw", processed_root=work / "reduced")
```

#### `reduce_run`
`bexa.pipelines.xfel_cube.reduce_run(profile, run, scan=1, roi=None, workers=None, out=None, legacy=False, show_progress=False, **kwargs)`

Builds the cube of run `run` (`open_run`, then `build_cube`), saves it to `out` or to
`cube_path(profile, run)`, `<processed_root>/data/run<N>.h5`, and returns it. The file is in the
bexa format (`bexa.load` reads it), or with `legacy=True` in the old `runN.h5` layout
(`save_legacy`). `roi` cuts the frames before averaging; `workers` and `**kwargs` (`clip`,
`axis`) go to `build_cube`. Replaces `python Aaron_allscan_cube_parallel.py 42`; the command is
`bexa cube build -p PROFILE --run 42 [--workers 16] [--roi 400:550,200:375] [--legacy]`.

```python
# profile: a PAL-XFEL beamtime profile; run 1 reduced to a laser on/off cube
cube = xfel_cube.reduce_run(profile, 1, workers=1)      # writes <processed_root>/data/run1.h5
cube["frames"].dims                                      # ('laser', 'delay', 'y', 'x')
cube.attrs["scan_axis"], cube.sizes["delay"]            # ('delay', 6)
cube["signal"].sel(laser="on").values                    # mean ROI / I0 per delay, laser on
bexa.load(work / "reduced" / "data" / "run1.h5")        # the same cube from disk
xfel_cube.reduce_run(profile, 1, workers=1, out=work / "run1_old.h5", legacy=True)  # Dataset: frames, signal, n_shots
```

#### `build_cube`
`bexa.pipelines.xfel_cube.build_cube(source, roi=None, workers=None, clip=None, axis=None, show_progress=False)`

Reduces every point of an open point-file source (an `Hdf5PointsSource` from `open_run` or
`bexa.open(folder).source`) with `reduce_point` and returns the cube of `assemble`; writes
nothing. `workers=None` uses a spawn-based process pool with all usable cores but one (in a
script, guard the calling code with `if __name__ == "__main__":`); `workers=1` stays in the
calling process. `clip` replaces non-positive pixels (default the spec's `clip_nonpositive`,
1e-4); `axis` forces the scan dim.

```python
# scan: run 1 opened as a scan; the cube built from its source inside a pixel window
scan = bexa.open(run1.measurement_dir)          # the run as a stack of shots
cube = xfel_cube.build_cube(scan.source, roi=bexa.ROI(y=(4, 20), x=(8, 24)), workers=1)
cube["frames"].shape                             # (2, 6, 16, 16)
bexa.save(cube, work / "run1_roi.h5")           # Path('.../run1_roi.h5')
```

#### `open_run`
`bexa.pipelines.xfel_cube.open_run(profile, run, scan=1, **kwargs)`

The `Hdf5PointsSource` of one run and scan of a profile: the spec named by `format` (with
`format_overrides`), the data root (`root`, else `raw_root`), the profile's `detector` and
`geometry.energy_keV`; `**kwargs` (`detector`, `energy_keV`) override them. `ValueError` when the
spec does not use the `hdf5_points` engine. Close the source when done.

```python
# profile: a PAL-XFEL beamtime profile; the source of run 1 (points, shots, frames)
source = xfel_cube.open_run(profile, 1)
source.n_points, source.n_frames, source.frame_shape     # (6, 96, (24, 32))
```

#### `cube_path`
`bexa.pipelines.xfel_cube.cube_path(profile, run)`

Where the cube of a run lives: `<processed_root>/data/run<N>.h5` for a number,
`<processed_root>/data/<name>.h5` for a name such as `"run79-88Combined"` (the output of
`bexa cube combine --runs 79-88`, which replaces `Scan_Combiner.py`). Takes a `BeamtimeProfile`
object; `ValueError` without `processed_root`. `bexa cube plot` and the watcher look there.

```python
# profile: a PAL-XFEL beamtime profile; where the cube of a run is written
xfel_cube.cube_path(profile, 1).name                       # 'run1.h5'
xfel_cube.cube_path(profile, "run79-88Combined").name      # 'run79-88Combined.h5'
```

#### `save_legacy`
`bexa.pipelines.xfel_cube.save_legacy(ds, path)`

Writes a cube in the `runN.h5` layout of `Aaron_allscan_cube_parallel.py` (the arrays of
`to_legacy_arrays`), for scripts that still read it; creates the folder and returns the path.
`bexa.load` reads both layouts into the same dataset.

```python
# cube: the cube of run 1; written in the layout of the old XFEL scripts and read back
path = xfel_cube.save_legacy(cube, work / "legacy" / "run1.h5")
bexa.load(path)["frames"].dims                   # ('laser', 'delay', 'y', 'x')
```

#### `to_legacy_arrays`
`bexa.pipelines.xfel_cube.to_legacy_arrays(ds)`

The datasets of a legacy `runN.h5` as a dict of numpy arrays: `delays`, `phi`, `chi`, `th`,
`tth`, `laser_v`, `laser_h` (NaN values dropped, as the old builder did), `index`, `signals_on`,
`signals_off`, `jungfrau_on` and `jungfrau_off` (`(points, y, x)`).

```python
# cube: the cube of run 1; the arrays the old scripts named
arrays = xfel_cube.to_legacy_arrays(cube)
arrays["jungfrau_on"].shape, arrays["delays"].shape      # ((6, 16, 16), (6,))
```

#### `reduce_point`
`bexa.pipelines.xfel_cube.reduce_point(table, frames, keys, roi=None, clip=0.0001, number=0)`

One point, exactly as the legacy `_process_one`: `table` is its shot table (pandas), `frames` its
`(shots, y, x)` array, `keys` a `PointKeys`. Scalars are column medians; the signal is the mean
of `roi_stat / i0` per laser state after dropping rows with a NaN in a signal column; frames are
cut to `roi`, non-positive pixels set to `clip`, shots with a NaN laser flag dropped and the rest
averaged per state (NaN when a state has no shot). Returns a `PointResult`.

```python
# source, keys: the source of run 1 and its column keys; one point reduced to on and off
source = xfel_cube.open_run(profile, 1)
keys = xfel_cube.PointKeys.from_spec(source.spec, source.table(0).columns)
res = xfel_cube.reduce_point(source.table(0), source.read_point(0), keys, number=1)
res.n_on, res.n_off, res.n_shots, res.frame_on.shape    # (8, 7, 16, (24, 32))
```

#### `PointKeys`
`bexa.pipelines.xfel_cube.PointKeys(laser, i0, roi_stat, signal_columns, scalars)`

Dataclass of the column names of one beamtime: laser flag, I0 and ROI sum (`None` when absent),
`signal_columns` (must be finite for a shot to count in the signal: laser flag, both I0s, ROI
sum, beam position) and `scalars` (each of `SCALAR_KEYS` to its column).

- `from_spec(spec, columns)`: classmethod; each logical key becomes the first of the format
  spec's aliases present in `columns`.

```python
# keys: the column names of run 1: the laser flag, the monitor, the scalars
keys.laser, keys.i0                  # ('event_info.THIRTY_HERTZ', 'qbpm:eh1:qbpm1:sum')
keys.scalars["delay"]                # 'delay_input'
```

#### `PointResult`
`bexa.pipelines.xfel_cube.PointResult(number, scalars, signal_on, signal_off, frame_on, frame_off, n_on, n_off, n_shots, extra={})`

Dataclass of one reduced point: `number`, readback medians `scalars`, the mean signal and mean
`(y, x)` frame per laser state, shots used per state (`n_on`, `n_off`) and in all (`n_shots`),
and a free `extra` dict that `assemble` ignores.

```python
# res: one reduced point; its scalars and the on and off signals
res.scalars["th"], res.signal_on < res.signal_off        # (12.5, True): the pump lowers the peak at delay -2
```

#### `assemble`
`bexa.pipelines.xfel_cube.assemble(results, axis=None, priority=None, attrs=None)`

Stacks `PointResult`s (sorted by number) into the cube described above, on `axis` when given (a
readback name or `"index"`), else on `detect_axis(results, priority or [*SCALAR_KEYS, "index"])`;
`attrs` are copied and `scan_axis` is set. The last step of `build_cube`, for points reduced by
hand.

```python
# results: every point of run 1 reduced; assembled into one cube
results = [xfel_cube.reduce_point(source.table(i), source.read_point(i), keys, number=i + 1)
           for i in range(source.n_points)]
xfel_cube.assemble(results)["frames"].dims                  # ('laser', 'delay', 'y', 'x')
xfel_cube.assemble(results, axis="index")["frames"].dims    # ('laser', 'index', 'y', 'x')
```

#### `detect_axis`
`bexa.pipelines.xfel_cube.detect_axis(results, priority)`

The first name of `priority` whose values are finite at every point and span more than
`AXIS_TOLERANCE`, else `"index"`; names after `"index"` in `priority` are not tried.

```python
# results: the reduced points; which scalar the run was scanned along
xfel_cube.detect_axis(results, ["th", "delay", "index"])     # 'delay': th is constant
xfel_cube.detect_axis(results, ["th", "index"])              # 'index'
```

Constants:

- `SCALAR_KEYS`: the readbacks kept per point, `('delay', 'phi', 'chi', 'th', 'tth', 'laser_v', 'laser_h')`.
- `SIGNAL_KEYS`: the logical columns a shot needs finite to count in the signal.
- `LEGACY_NAMES`: `{'delay': 'delays'}`, names that differ in the legacy layout.
- `LASER_COORD`: `array(['off', 'on'])`, the `laser` coordinate (index 0 is laser off).
- `AXIS_TOLERANCE`: `1e-9`, the smallest span that counts as a varying axis.

### bexa.pipelines.cube_report: figures and GIFs of one cube

Panel-for-panel port of `1d_plot_allscan.py` and `animation_allscan.py` for any laser on/off cube
with `frames(laser, <axis>, y, x)`: from `xfel_cube`, or `bexa.load` of a legacy or combined
`runN.h5`. File names follow the legacy pattern `<label>_ROI_r0_r1_c0_c1_<kind>.png`, by which the
watcher recognises processed runs. Figures are saved and closed, never shown.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.pipelines import cube_report
from bexa.testing import make_legacy_cube

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
cube = bexa.load(make_legacy_cube(work / "run42.h5"))     # 6 delays, frames of 20 x 24 pixels
peak = bexa.ROI(y=(5, 15), x=(4, 20))                      # rows 5 to 15, columns 4 to 20
```

#### `figure_set`
`bexa.pipelines.cube_report.figure_set(ds, out_dir, label, roi=None, axis=None, background_box=None, dpi=100, kinds=("image", "diffplot", "COMplot", "FWHMplot", "skewplot"))`

Writes one PNG per entry of `kinds` into `out_dir` and returns `{kind: Path}`. `image`: the
laser-on ROI summed over the axis. `diffplot`: the summed laser-on frame with the ROI and the
background box drawn in red, the ROI intensity on and off against the axis, and on minus off.
`COMplot`, `FWHMplot`, `skewplot`: the per-frame centre of mass (pixels from the ROI corner),
second and third central moments of the ROI in x and y, on and off (and on minus off).

`roi` is a `bexa.ROI` of pixel ranges (default the full frame); `axis` a coordinate along the scan
dim (default the cube's `scan_axis`; `"delays"` means `delay`); `background_box` is only drawn,
given as `{"y": [y0, y1], "x": [x0, x1]}`, a `ROI` or `((y0, y1), (x0, x1))` (default
`DEFAULT_BACKGROUND_BOX`). Replaces `1d_plot_allscan.py 42 delays --ROI 400 550 200 375`; the
command is `bexa cube plot run42.h5 --roi 400:550,200:375 -o outputs/`.

```python
# cube: a laser on/off cube; work: an output folder; the figure set of one cube written
written = cube_report.figure_set(cube, work / "outputs", label="run42")
written["diffplot"].name          # 'run42_ROI_0_20_0_24_diffplot.png'
cube_report.figure_set(cube, work / "outputs", label="run42", roi=peak, kinds=("diffplot", "COMplot"))  # {'diffplot': Path('.../run42_ROI_5_15_4_20_diffplot.png'), 'COMplot': Path('.../run42_R...
```

#### `animations`
`bexa.pipelines.cube_report.animations(ds, out_dir, label, roi=None, axis=None, background_box=None, fps=5)`

Writes `<label>_animation_on.gif` and `<label>_animation_off.gif`, which step through the scan
axis, and returns `{"animation_on": Path, "animation_off": Path}`. Every frame is cut to `roi`
and divided by the mean of its background box (laser off: the legacy robust mean, values above
the 90th and below the 10th percentile replaced by the median), with its own colour limits. The
box defaults to `{"y": [0, 20], "x": [0, 20]}` here. The ROI is not in the file names, so a call
with another ROI overwrites the GIFs. Replaces `animation_allscan.py`; the command is
`bexa cube animate CUBE -o outputs/`.

```python
# cube, peak: a cube and a pixel window; the GIFs of the on and off frames
gifs = cube_report.animations(cube, work / "outputs", label="run42", roi=peak, fps=8)
gifs["animation_off"].name        # 'run42_animation_off.gif'
```

#### `roi_tag`
`bexa.pipelines.cube_report.roi_tag(roi, frame_shape)`

The ROI part of the legacy file names, `ROI_r0_r1_c0_c1` (rows, then columns), for the pixel
window of `roi` in a frame of `frame_shape`.

```python
# peak: a pixel window; the tag the file names carry for it
cube_report.roi_tag(peak, (20, 24))                    # 'ROI_5_15_4_20'
```

#### `full_roi`
`bexa.pipelines.cube_report.full_roi(frame_shape)`

`ROI(y=(0, H), x=(0, W))`, the whole frame: the default ROI of `figure_set` and `animations`.

```python
# the ROI of the whole frame, for files of the full detector
cube_report.full_roi((20, 24))                         # ROI(y=(0, 20), x=(0, 24))
```

Constants:

- `FIGURE_KINDS`: the kinds of `figure_set`, `('image', 'diffplot', 'COMplot', 'FWHMplot', 'skewplot')`.
- `DEFAULT_BACKGROUND_BOX`: `{'y': [10, 30], 'x': [10, 30]}`, the box `figure_set` draws when none is given.

### bexa.pipelines.auto_process: the beamtime watcher

Port of `scan_auto_process.py` without `os.system`. For every raw run without outputs it builds
the cube (`xfel_cube.reduce_run`, unless `<cube_dir>/<run>.h5` exists), writes the
`cube_report.figure_set` of every ROI of the run and the `cube_report.animations`, and logs the
run; a failed run is tried once more, then put on the skip list. `watch` repeats this at a fixed
interval. Default places: raw runs in `<raw_root>/type=raw`, cubes in `<processed_root>/data`,
figures, GIFs and `logs/` in `<processed_root>/outputs`, and `ROI_dict.csv`, `Skip_dict.csv` in
`<processed_root>`. The command is `bexa auto -p PROFILE [--watch --interval 60] [--dry-run]
[--runs 10-20] [--legacy] [--no-animations]`.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.config import BeamtimeProfile
from bexa.pipelines import auto_process as ap
from bexa.testing import make_pal_run

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
make_pal_run(work / "raw", run=1, n_points=4, shots_per_point=8)  # SyntheticPalRun
make_pal_run(work / "raw", run=2, n_points=3, shots_per_point=8, seed=1)  # SyntheticPalRun
profile = BeamtimeProfile(name="demo_pal", format="pal_xfel_points_2025_09", raw_root=work / "raw",  # a profile for the synthetic PAL-XFEL run
                          processed_root=work / "reduced", background_box={"y": [0, 8], "x": [0, 8]})
```

#### `config_from_profile`
`bexa.pipelines.auto_process.config_from_profile(profile, raw_dir=None, cube_dir=None, outputs_dir=None, roi_table=None, skip_list=None, **kwargs)`

The `AutoConfig` of a session from a profile (name, YAML path or `BeamtimeProfile`); explicit
paths win. Defaults: `<raw_root>/type=raw`, `<processed_root>/data`, the profile's
`auto_process.outputs` (else `outputs`), `auto_process.roi_table` (else `ROI_dict.csv` if it
exists) and `auto_process.skip_list` (else `Skip_dict.csv`), the last three relative to
`processed_root` (`<raw_root>/../reduced` when unset). The profile's `background_box` and `rois`
(for runs missing from the ROI table) are copied; `**kwargs` set `workers`, `scan`,
`legacy_cubes`, `animations` or `fps`.

```python
# profile: a PAL-XFEL beamtime profile; the watcher's settings derived from it
cfg = ap.config_from_profile(profile, workers=1)
cfg.raw_dir.name, cfg.cube_dir.name, cfg.outputs_dir.name   # ('type=raw', 'data', 'outputs')
cfg.skip_list.name, cfg.log_dir.name                        # ('Skip_dict.csv', 'logs')
```

#### `AutoConfig`
`bexa.pipelines.auto_process.AutoConfig(raw_dir, cube_dir, outputs_dir, profile=None, roi_table=RoiTable(), skip_list=None, scan=1, workers=None, legacy_cubes=False, background_box=None, default_rois=[], animations=True, fps=5)`

Dataclass of the folders and settings of one session. Cubes are built only with a `profile`
(`scan`, `workers`, `legacy_cubes` go to `xfel_cube.reduce_run`); `background_box` is drawn by
`figure_set` (the GIFs keep their own default box); `default_rois` lists `(name, ROI)` for runs
the `roi_table` lacks (else the full frame, named `full`); `animations` switches the GIFs, `fps`
sets their speed; `skip_list` is the skip-list file.

- `log_dir`: property, `<outputs_dir>/logs`.

```python
# cfg: the watcher's settings; a copy for a quick look with its own output folder
quick = ap.AutoConfig(raw_dir=cfg.raw_dir, cube_dir=cfg.cube_dir, outputs_dir=work / "quicklook",
                      profile=profile, workers=1, animations=False)
quick.log_dir.name, quick.default_rois                      # ('logs', [])
```

#### `run_once`
`bexa.pipelines.auto_process.run_once(cfg, only=None, dry_run=False)`

Processes every pending run (`pending_runs`) once with `process_run` and returns the list of
`RunRecord`; `only` keeps some run names (`["run12", "run15"]`), `dry_run=True` only lists them
(status `"pending"`). Failed runs are then added to the skip list, and a ROI table loaded from a
file is saved as YAML next to it with the axis used for each run. This is `bexa auto -p PROFILE`.

```python
# cfg: the watcher's settings; the runs waiting, then every one of them processed
[r.run for r in ap.run_once(cfg, dry_run=True)]         # ['run1', 'run2']
records = ap.run_once(cfg)
[(r.run, r.status) for r in records]                    # [('run1', 'ok'), ('run2', 'ok')]
sorted(records[0].outputs)                              # ['full:COMplot', 'full:FWHMplot', 'full:animation_off', 'full:animation_on', 'full:diffplot', 'full:image', 'full:skewplot']
```

#### `watch`
`bexa.pipelines.auto_process.watch(cfg, interval_s=60.0, only=None, dry_run=False, max_cycles=None)`

`run_once` in a loop, `interval_s` seconds apart, so runs are processed as they appear during the
beamtime. Without `max_cycles` it never returns (stop it with Ctrl+C); with it, it returns the
records of all cycles after that many, without a final sleep. Replaces `scan_auto_process.py`;
the command is `bexa auto -p PROFILE --watch --interval 60`.

```python
# cfg: the watcher's settings; one cycle of the watcher (a new run appeared)
make_pal_run(work / "raw", run=3, n_points=3, shots_per_point=8)          # a new run appears
[(r.run, r.status) for r in ap.watch(cfg, interval_s=60, max_cycles=1)]   # [('run3', 'ok')]
# ap.watch(cfg, interval_s=60)            during the beamtime: polls until interrupted
```

#### `process_run`
`bexa.pipelines.auto_process.process_run(run, cfg, retries=1)`

Everything for one run name (`"run12"`, or a combined cube's stem): the cube (built unless it
exists; combined cubes never are), the axis (`cube_axis`), the `figure_set` of every ROI of the
run (ROI table, else `cfg.default_rois`, else the full frame) and the GIFs. An exception is
caught and the run tried again, `retries` times at most. Appends to `logs/bexa_auto.csv`, writes
`logs/<run>.json` and returns a `RunRecord`; the skip list is left to `run_once`.

```python
# cfg: the watcher's settings; run 1 reduced and its figures written, with a record
rec = ap.process_run("run1", cfg)        # processes again even though outputs exist
rec.status, rec.attempts                 # ('ok', 1)
```

#### `RunRecord`
`bexa.pipelines.auto_process.RunRecord(run, status, seconds, outputs={}, error="", attempts=1)`

Dataclass of one run's outcome: `status` (`"ok"`, `"failed"`, `"pending"` in a dry run), wall
time in `seconds`, `outputs` as `{"<roi name>:<kind>": path}`, the last `error` as
`"<exception type>: <message>"` and the number of `attempts`. Saved as `logs/<run>.json`.

```python
# rec: the record of run 1; what it wrote, and its JSON on disk
Path(rec.outputs["full:diffplot"]).name      # 'run1_ROI_0_24_0_32_diffplot.png'
(cfg.log_dir / "run1.json").exists()         # True
```

#### `pending_runs`
`bexa.pipelines.auto_process.pending_runs(cfg, only=None)`

The runs of `discover_runs` that have no output file yet (`processed_runs`) and are not on the
skip list, restricted to `only` when given.

```python
# cfg: the watcher's settings; the runs on disk that have no cube yet
ap.pending_runs(cfg)                     # []: every run has its figures
ap.pending_runs(cfg, only=["run1"])             # []
```

#### `discover_runs`
`bexa.pipelines.auto_process.discover_runs(raw_dir, cube_dir=None)`

`run<N>` for every folder of `raw_dir` with digits in its name (`run=012` gives `run12`), then the
stem of every `*Combined*.h5` in `cube_dir` (the cubes of `bexa cube combine`, formerly
`Scan_Combiner.py`), without duplicates.

```python
# cfg: the watcher's settings; the raw runs and the cubes found
ap.discover_runs(cfg.raw_dir, cfg.cube_dir)      # ['run1', 'run2', 'run3']
ap.discover_runs(cfg.raw_dir)                   # ['run1', 'run2', 'run3']
```

#### `processed_runs`
`bexa.pipelines.auto_process.processed_runs(outputs_dir)`

The set of run names with an output: the part before the first `_` of every file name in
`outputs_dir` (empty when the folder does not exist).

```python
# cfg: the watcher's settings; the runs whose figures exist
sorted(ap.processed_runs(cfg.outputs_dir))       # ['run1', 'run2', 'run3']
```

#### `read_skip_list`
`bexa.pipelines.auto_process.read_skip_list(path)`

The run names of a skip-list file (the legacy `Skip_dict.csv`: a `0` line, then one name per
line; numbers and empty lines are ignored), without duplicates; `[]` for `None` or a missing file.

```python
# cfg: the watcher's settings; the runs listed to be left alone
ap.read_skip_list(cfg.skip_list)                 # []: nothing failed
```

#### `write_skip_list`
`bexa.pipelines.auto_process.write_skip_list(path, runs)`

Writes run names in the same format, replacing the file (the folder is created); returns the
path. Listed runs are never processed: delete a line to retry a run.

```python
# cfg: the watcher's settings; two runs added to the skip list
ap.write_skip_list(cfg.skip_list, ["run7", "run8"])  # Path('.../Skip_dict.csv')
ap.read_skip_list(cfg.skip_list)                 # ['run7', 'run8']
```

#### `cube_axis`
`bexa.pipelines.auto_process.cube_axis(ds, requested)`

The axis to plot a cube against: `requested` (the ROI table's; `"delays"` means `delay`) if the
cube has it as a dim or coordinate, else the cube's `scan_axis` (with a warning if one was asked).

```python
# cube: the cube of run 1; the values of its scan axis, by name or by default
cube = bexa.load(cfg.cube_dir / "run1.h5")
ap.cube_axis(cube, "delays"), ap.cube_axis(cube, "th"), ap.cube_axis(cube, None)   # ('delay', 'th', 'delay')
```

#### `RoiTable`
`bexa.pipelines.auto_process.RoiTable(entries={}, path=None)`

The per-run scan axis and named ROIs of the watcher, `entries` =
`{run: {"axis": name, "rois": {name: [r0, r1, c0, c1]}}}` (rows, then columns), read from the
legacy `ROI_dict.csv` (one column per run: the axis, then `r0 r1 c0 c1` quadruples, named `roi1`,
`roi2`, ...) or from YAML (`run42: {axis: delays, rois: {peak: [400, 550, 200, 375]}}`).

- `load(path)`: classmethod; reads a CSV or YAML file; `None` gives an empty table, a missing file
  an empty table that keeps the path.
- `axis(run)`: the axis of `run`; `None` when missing, empty or `index`.
- `rois(run)`: list of `(name, ROI)`, `[]` when the run has none.
- `set_axis(run, axis)`: records the axis used for `run` (the watcher does this for every run).
- `save(path=None)`: writes YAML to `path` or to the loaded path with the suffix changed to
  `.yaml` (the CSV is never overwritten); returns the path, or `None` without one.

```python
# a ROI_dict.csv of the old scripts; the pixel window of every run read from it
import pandas as pd
pd.DataFrame({"run1": ["delays", 4, 20, 8, 24]}).to_csv(work / "ROI_dict.csv", index=False)
table = ap.RoiTable.load(work / "ROI_dict.csv")
table.axis("run1"), table.rois("run1")     # ('delays', [('roi1', ROI(y=(4, 20), x=(8, 24)))])
table.set_axis("run2", "th")
table.save().name                          # 'ROI_dict.yaml'
```

Constants:

- `LEGACY_MOTORS`: the axis names of a legacy `runN.h5`, `('delays', 'phi', 'chi', 'th', 'tth', 'laser_v', 'laser_h', 'index')` (for reference; the watcher does not use it).

### bexa.pipelines.benchmark: timings and baselines

Times the steps of a typical session (metadata, full read, preview, one reduction, the DFXM
report) on a synthetic ESRF scan, and optionally on a real frame stack, and compares them with a
per-machine baseline. Results are plain dicts that save as JSON. The wrappers are
`bexa bench [--full] [--real stack.npy] [--save]` (quick unless `--full`) and
`python benchmarks/run.py [--quick]` (full size unless `--quick`), both with the baselines in
`benchmarks/baselines/<machine>.json`.

```python
import tempfile
from pathlib import Path
import bexa
from bexa.pipelines import benchmark

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
```

#### `run_suite`
`bexa.pipelines.benchmark.run_suite(quick=True, device="auto", real_path=None, work_dir=None, plots=True)`

Writes a synthetic ESRF chi x mu scan, 4 x 12 frames of 48 x 64 pixels when `quick` (a few
seconds) or 2 x 16 frames of 2160 x 2560 pixels otherwise, into a temporary folder deleted
afterwards (or `work_dir`, kept), and times it with `bench_scan` (and `bench_plots` when
`plots`). `real_path`, any stack the generic reader opens (`.npy`, tiff, `file.h5::/path`), is
timed as well. Returns `{"machine", "quick", "device", "cupy", "cpu_count", "timestamp",
"cases"}`, with `cases` holding `"esrf_synthetic"` (and `"real_stack"`).

```python
# the quick benchmark suite on synthetic data; the timings of one case
results = benchmark.run_suite(quick=True)
case = results["cases"]["esrf_synthetic"]
case["frames"], case["frame_shape"]              # (48, [48, 64])
case["reduce_s"], case["report_s"]               # seconds
```

#### `format_results`
`bexa.pipelines.benchmark.format_results(results)`

The text `bexa bench` prints: per case the frames and frame shape, the timings in seconds
(`info_s`, `read_s`, `preview_s`, `reduce_s`, `report_s`), `read_mb_s`, `reduce_fps`,
`compute_to_read`, `peak_rss_mb` and `gpu_used_mb`.

```python
# results: the suite's timings; as a text table
print(benchmark.format_results(results))        # one line per case, the timings in seconds
```

#### `save_baseline`
`bexa.pipelines.benchmark.save_baseline(results, folder)`

Writes `results` as JSON to `<folder>/<results["machine"]>.json`, creating the folder, and
returns the path. `bexa bench --save` writes to `benchmarks/baselines/`.

```python
# results: the suite's timings; written as this machine's baseline
path = benchmark.save_baseline(results, work / "baselines")
path.name == benchmark.machine_id() + ".json"    # True
```

#### `load_baseline`
`bexa.pipelines.benchmark.load_baseline(folder, machine=None)`

The results stored for `machine` (default this one, `machine_id()`) in
`<folder>/<machine>.json`, or `None` when there are none.

```python
# work: a folder with a baseline; read back, and a folder without one
baseline = benchmark.load_baseline(work / "baselines")
benchmark.load_baseline(work / "nowhere")        # None
```

#### `compare`
`bexa.pipelines.benchmark.compare(results, baseline, tolerance=0.2)`

The regressions: one line per case and timing of `TIMED_KEYS` slower than the baseline by more
than `tolerance` (0.2 is 20 percent), such as
`'esrf_synthetic.reduce_s: 1.279 s vs baseline 0.139 s'`; `[]` when nothing is slower.
`bexa bench` then exits with code 1. Timings of a few milliseconds are noisy.

```python
# results, baseline: the timings and the baseline; a slower run flagged
benchmark.compare(results, baseline)             # []
slower = {"cases": {"esrf_synthetic": {**case, "reduce_s": 2 * case["reduce_s"] + 1}}}
benchmark.compare(slower, baseline)              # ['esrf_synthetic.reduce_s: <now> s vs baseline <before> s']
```

#### `machine_id`
`bexa.pipelines.benchmark.machine_id()`

`"<hostname>-<device>"`, the device being what `"auto"` resolves to (`cuda` with a usable GPU,
else `cpu`), spaces replaced by `_`: the name of the baseline file.

```python
# the name the baseline file carries for this machine
benchmark.machine_id()                           # '<hostname>-cuda' or '<hostname>-cpu'
```

#### `bench_scan`
`bexa.pipelines.benchmark.bench_scan(scan, device="auto", batch_frames=None)`

Times an open scan: `scan.info()` (`info_s`), a read of every frame into memory (`read_s`,
`read_fps`, `read_mb_s`), a preview with downsample 4 and no cache (`preview_s`) and one pass of
`Sum`, `Max` and `MotorCOM` over all motors (`reduce_s`, `reduce_fps`, and `compute_to_read`,
the reduce over the read time). Also returns `frames`, `frame_shape`, `stored_gb`, the `device`
used, `peak_rss_mb` and `gpu_used_mb` (held by the cupy pool at the end; `None` without cupy).
`batch_frames` sets the batch size of the pass.

```python
# the demo mu scan; the read and reduce timings of one scan
timings = benchmark.bench_scan(bexa.demo("rocking"), device="cpu")
timings["frames"], timings["device"]             # (25, 'cpu')
```

#### `bench_plots`
`bexa.pipelines.benchmark.bench_plots(scan)`

Times `dfxm_report.report(scan, <temporary folder>, downsample=4, device="cpu")`; returns
`{"report_s": seconds}`. Switches matplotlib to the Agg backend (no windows) for the session.

```python
# the demo mu scan; the timings of the standard figures
benchmark.bench_plots(bexa.demo("rocking"))      # {'report_s': seconds}
```

Constants:

- `TIMED_KEYS`: the timings `compare` checks, `('info_s', 'read_s', 'preview_s', 'reduce_s', 'report_s')`.

### bexa.testing.demo: `bexa.demo`, synthetic data in one call

A small synthetic ESRF ID03 dataset written and opened in one call, to try the API without
beamtime data. `bexa.demo` is this module's `demo`.

#### `demo`
`bexa.testing.demo.demo(kind="mosa", root=None, **kwargs)`

Writes a dataset with the generators of `bexa.testing.synthetic` (2026 BLISS layout, read by the
`esrf_id03_bliss_2026` spec) into `root`, or a new temporary folder `bexa_demo_*` that is kept,
and opens it without cache. `**kwargs` go to the generator (`make_esrf_scan`,
`make_esrf_energy_series` or `make_esrf_zstack`), for example `frame_shape` or `noise`. An
unknown `kind` raises `ValueError`. The kinds:

- `"mosa"`: a `Scan`, fscan2d, dims `(chi, mu, y, x)`, shape `(6, 16, 64, 80)`, chi -0.5 to 0.5
  deg, mu -1 to 1 deg, 17 keV;
- `"rocking"`: a `Scan`, fscan1d, dims `(mu, y, x)`, shape `(25, 64, 80)`;
- `"energy"`: a `Scan`, three chi x mu scans at 17.00, 17.05 and 17.10 keV opened as one series,
  dims `(energy, chi, mu, y, x)`, shape `(3, 3, 10, 48, 64)`;
- `"zstack"`: a `Dataset` of 30 scans: at five heights (`samz` -0.002 to 0.002 mm) one chi x mu
  scan of 6 x 15 frames and five mu scans of 8 frames at 16.98 to 17.02 keV; 40 x 48 pixels;
- `"alignment"`: a `Dataset` of four 1-D scans of 48 x 64 pixels: mu (25 points), chi (21), samz
  (16, an edge) and ccmth (15).

```python
# a synthetic mu scan with smaller frames and more noise than the default
import bexa
scan = bexa.demo("rocking", frame_shape=(32, 40), noise=5.0)     # kwargs go to make_esrf_scan
scan.dims, scan.shape                            # (('mu', 'y', 'x'), (25, 32, 40))
ds = bexa.demo("zstack")
len(ds), ds.groups("samz")[0.0]                  # (30, [13, 14, 15, 16, 17, 18])
```

Constants:

- `KINDS`: the kinds `demo` accepts, `('mosa', 'rocking', 'energy', 'zstack', 'alignment')`; import it with `from bexa.testing.demo import KINDS`.

### bexa.testing.synthetic: small valid files for every layout, with the planted truth

Generators that write tiny files with the paths, dtypes, chunking and metadata of each layout
bexa reads, plant a signal whose answer is known, and return a record of what they wrote with
that answer in `truth`: the tests, the examples and `bexa.demo` run on them, and so can anyone
without beamtime data. Everything is also importable from `bexa.testing`. `polycrystal`,
`dfxm_sample`, `Region` and `SAMPLE_REGIONS` model the sample the ESRF generators image: a 3-D
polycrystal whose sections show a grain split into two domains, its neighbour and dark grains.

```python
import tempfile
from pathlib import Path
import numpy as np
import bexa
from bexa.testing import (dfxm_sample, make_esrf_beamtime, make_esrf_energy_series, make_esrf_scan,
                          make_esrf_zstack, make_lcls_cube, make_legacy_cube, make_pal_run, make_aps_session,
                          make_pal_spec_scan, polycrystal)

work = Path(tempfile.mkdtemp())                 # a scratch folder for the files the examples write
```

#### `make_esrf_scan`
`bexa.testing.synthetic.make_esrf_scan(root, dataset="synth_dfxm", scan=1, detector="pco_ff", motors=(("chi", 4), ("mu", 12)), ranges=None, frame_shape=(48, 64), n_files=2, energy_keV=17.0, layout="2026", order="slow_major", partial=0, amplitude=4000.0, background=10.0, noise=0.0, seed=0, dtype=np.uint16, positioners=None, curve="gaussian", centers=None, widths=None, monitors=None, state="finished", motor_lag=0, empty_last_file=False, command=None)`

Writes one BLISS scan: the master `<root>/<dataset>/<dataset>.h5` (entry `<scan>.1` with
`fscan_parameters`, per-frame motor values, positioners with `ccmth` for `energy_keV` on Si 111,
LIMA's `acq_nb_frames` and `saving_frame_per_file`, and the start and end fields BLISS writes;
later scans of the dataset are added to it) and up to `n_files` detector files
`<dataset>/scan<NNNN>/<detector>_<i>.h5` (frames at `ESRF_FRAMES_PATH`, one per chunk,
bitshuffle-lz4 when hdf5plugin is installed; every file but the last holds the same number of
frames, as LIMA writes them, and a partial scan fills fewer files). Read by
`esrf_id03_bliss_2026` (`esrf_id03_bliss_2024` for `layout="2024"`); returns a
`SyntheticEsrfScan`.

- Signal: along each motor every pixel has a Gaussian (with `curve="edge"` an error-function step,
  as in a knife-edge scan) of height `amplitude` (a number or an `(H, W)` map) on `background`
  counts, plus Gaussian `noise`, rounded to `dtype` (an integer type). The first motor's centre
  moves along x and the second's along y over the inner 60 % of the range, sigma 12 % of the
  range, unless `centers`, `widths` (a value or `(H, W)` map per motor, as `dfxm_sample` gives)
  set them; a curve narrower than half a step is widened.
- `motors`: `(name, points)`, slow first (one motor gives an fscan1d, two an fscan2d, three an
  fscan3d written with BLISS's `slow1_motor`/`slow2_motor`/`fast_motor`; `ccmth` among them makes
  an energy mosa that `bexa.open` reads with an `energy` dim in keV); `ranges`:
  `{motor: (start, stop)}`, default mu -1 to 1, chi -0.5 to 0.5, phi -0.3 to 0.3 deg, ccmth
  0.005 deg either side of the angle of `energy_keV`, else 0 to 1; `monitors`: slow channels
  sampled during the scan, `{"nanodac1_temp": 50.0}`, written to the monitoring sub-scan
  `<scan>.2` as BLISS does for a furnace (200 samples around the value), which the autumn 2026
  spec reads as positioners;
  `positioners`: fixed motor values such as `{"samz": 0.5}`; `layout`: `"2026"` or `"2025"` (with
  `fscan_parameters`, both read as 2026), `"2024"` (without, plus `obpitch`) or `"F2026"` (the
  2026 layout plus the frames linked into the master as a virtual dataset
  `N.1/instrument/<detector>/image`, read by `esrf_id03_bliss_F2026`); `order="snake"` reverses
  the fast motor on every other slow step; `partial` frames are missing at the end (every frame
  may be: a scan that has just started, whose only detector file is empty).
- The state of the scan, as a beamtime leaves it: `state="finished"` (the default) writes
  `start_time`, `end_time`, `end_reason` (`SUCCESS`, or `USER_ABORT` when frames are missing)
  and the writer status `SUCCEEDED`; `"running"` writes `start_time` and the status `RUNNING`
  only, as for a scan still going; `"legacy"` writes none of them (older files). `motor_lag`
  frames at the end have no readbacks in the master yet; `empty_last_file` adds a detector
  file of zero bytes, as the one the detector has opened but not flushed. The time stamps are
  `SCAN_STARTED` and `SCAN_ENDED`.
- `command`: the scan command to write as the title instead of the fscan family: `"dmesh"` or
  `"amesh"` (a step mesh, whose title counts intervals and whose master has no
  `fscan_parameters`), `"ascan"`, a lockstep command of `LOCKSTEP_COMMANDS` (`"a2scan"`: every
  motor steps at every frame, so all motors need the same number of points), `"loopscan"`, or
  any other word (a beamline macro, read from its readbacks).

```python
# work: a scratch folder; a chi x mu scan written as BLISS would, then opened
made = make_esrf_scan(work, dataset="grain", motors=(("chi", 5), ("mu", 20)), noise=3.0)
scan = bexa.open(made.dataset_dir)               # esrf_id03_bliss_2026, dims (chi, mu, y, x)
res = scan.com(axes=("chi", "mu"), sigma=0)
float(np.median(np.abs(res["com_mu"].values - made.truth["mu_center"])))   # 0.0158...: the planted centres come back
autumn = make_esrf_scan(work, dataset="autumn", layout="F2026")
bexa.open(autumn.dataset_dir).spec.name          # 'esrf_id03_bliss_F2026'
going = make_esrf_scan(work, dataset="going", motors=(("chi", 3), ("mu", 8)), n_files=1, partial=9, state="running")   # 15 of 24 frames so far
bexa.open(going.dataset_dir).status              # <ScanStatus.RUNNING: 'running'>
mesh = make_esrf_scan(work, dataset="mesh", motors=(("samy", 3), ("samz", 5)), n_files=1, command="dmesh")   # a step mesh of the stage
bexa.open(mesh.dataset_dir).dims, bexa.open(mesh.dataset_dir).structure.scan_type   # (('samy', 'samz', 'y', 'x'), 'dmesh')
```

#### `make_esrf_beamtime`
`bexa.testing.synthetic.make_esrf_beamtime(root, dataset="beamtime", frame_shape=(24, 32), **kwargs)`

A dataset as a beamtime leaves it, for reading unfinished scans: scans 1 to 3 are complete
chi x mu scans (3 x 8 frames), scan 4 was aborted (its last five frames are missing), scan 5 is
running with 15 frames taken, scan 6 has just started and holds no frame, and a folder named
`scan0004_old` that is not a scan sits next to them. `kwargs` go to every `make_esrf_scan` call
(`layout`, `n_files`...). Returns a `SyntheticBeamtime`.

```python
# work: a scratch folder; a beamtime with complete, aborted, running and empty scans
live = make_esrf_beamtime(work)
ds = bexa.open_dataset(live.dataset_dir, cache=False)
ds.table()["status"].tolist()                    # ['complete', 'complete', 'complete', 'aborted', 'running', 'running']
ds.select_numbers(), ds.problems                 # ([1, 2, 3, 4], {6: 'scan 6 of beamtime is running: no frame to read yet'})
```

#### `SyntheticBeamtime`
`bexa.testing.synthetic.SyntheticBeamtime(root, dataset, master, scans, complete, aborted, running, empty, stray)`

Dataclass of what `make_esrf_beamtime` wrote: the `master`, `scans` (`{number:
SyntheticEsrfScan}`), the ids of the `complete` scans, the `aborted`, `running` and `empty`
(just started) ones, and the `stray` folder that is not a scan; `dataset_dir` is the folder
`bexa.open_dataset` takes.

```python
# live: the synthetic beamtime; which scan is which, and the frames of the running one
live.complete, live.aborted, live.running, live.empty, live.stray.name   # ([1, 2, 3], 4, 5, 6, 'scan0004_old')
live.scans[5].n_frames                           # 15
```

#### `SyntheticEsrfScan`
`bexa.testing.synthetic.SyntheticEsrfScan(root, dataset, scan, detector, master, detector_files, motor_dims, motor_shape, coords, per_frame, frame_shape, n_frames, energy_keV, frames, truth={})`

Dataclass of what `make_esrf_scan` wrote: the paths (`master`, `detector_files`), the grid
(`motor_dims`, `motor_shape`, `coords` as motor to grid values, `per_frame` as motor to the value
of every frame in acquisition order), `frame_shape`, `n_frames` (after `partial`), `energy_keV`,
`frames` (the float32 frames before rounding, `(n_frames, H, W)`) and `truth`
(`"<motor>_center"`: the `(H, W)` map of planted curve centres).

- `dataset_dir`: property, `<root>/<dataset>`, the folder `bexa.open` takes.
- `scan_folder`: property, `<root>/<dataset>/scan<NNNN>`.

```python
# made: the synthetic chi x mu scan; its grid, frames and files
made.motor_shape, made.n_frames, made.frames.shape      # ((5, 20), 100, (100, 48, 64))
made.scan_folder.name, made.master.name                 # ('scan0001', 'grain.h5')
sorted(made.truth)                                      # ['chi_center', 'mu_center']
```

#### `make_esrf_energy_series`
`bexa.testing.synthetic.make_esrf_energy_series(root, dataset="synth_energy", scans=(1, 2, 3), energies=(17.0, 17.05, 17.1), **kwargs)`

Several scans of one dataset, one per energy (`ccmth` set from it; seeds 0, 1, 2, ...), as an
energy series is recorded; `**kwargs` go to `make_esrf_scan`, whose layout and spec apply. Returns
a list of `SyntheticEsrfScan`; `bexa.open(dataset_dir, scan=(1, 3))` stacks them on `energy`.

```python
# work: a scratch folder; one mu scan per energy, opened as a series
series = make_esrf_energy_series(work, motors=(("mu", 8),), n_files=1)
bexa.open(series[0].dataset_dir, scan=(1, 3)).dims     # ('energy', 'mu', 'y', 'x')
```

#### `make_esrf_zstack`
`bexa.testing.synthetic.make_esrf_zstack(root, dataset="synth_zstack", z_values=(-0.002, -0.001, 0.0, 0.001, 0.002), mosa=(("chi", 6), ("mu", 15)), energies=(16.98, 16.99, 17.0, 17.01, 17.02), energy_motor=("mu", 8), frame_shape=(40, 48), first_scan=1, n_files=1, seed=0, energy_scan="rocking", layer_px=4.0, z_motor="samz", tilt_offset=None, **kwargs)`

A z-stack in one dataset: at every height of `z_values` (the positioner `z_motor`, `samz` by
default or `uz` as at ma7352, in mm) a mosaicity scan and an energy series, numbered from
`first_scan`. With `energy_scan="rocking"` a `mosa` scan at 17 keV comes first, then one
`energy_motor` scan (a mu rocking curve) per energy; with `"mosa"` the `mosa` scan is repeated
at every energy (dims `(energy, chi, mu, y, x)`; the one nearest the mean energy counts as the
height's mosaicity scan). One energy (`energies=(17.0,)`) gives heights only, one height
(`z_values=(0.0,)`) an energy series only. The sample is `dfxm_sample` cut `layer_px` pixels
deeper per layer step, so grain sections change with the height; the energy centre adds each
region's strain to a gradient along x and with the height. `tilt_offset` (`{"mu": 0.05}`,
degrees) tilts the whole sample, the same grain measured again after annealing, say, for
difference maps between two datasets (the planted `chi_center`/`mu_center` move with it). With
`z_motor="uz"` the positioner `samz` follows at `uz - 0.03`, as the ID03 stage does. Files as in
`make_esrf_scan` (spec `esrf_id03_bliss_2026`); returns a `SyntheticZStack`.

```python
# work: a scratch folder; a mosaicity scan and an energy series at every height
zs = make_esrf_zstack(work)                     # 5 heights x (90 + 5 x 8 frames), 40 x 48 pixels
zs.mosa_scans, zs.energy_scans[0]               # ([1, 7, 13, 19, 25], (2, 6))
bexa.open(zs.dataset_dir, scan=zs.energy_scans[0]).dims     # ('energy', 'mu', 'y', 'x')
zs.truth["energy_center"].shape                 # (5, 40, 48)
uz = make_esrf_zstack(work / "uz", z_values=(0.0, 0.5), energies=(17.0,), z_motor="uz")   # two heights on uz, no energy series
uz.mosa_scans, bexa.open_dataset(uz.dataset_dir).varying(type="fscan2d")   # ([1, 3], {'uz': array([0. , 0.5])})
after = make_esrf_zstack(work / "after", z_values=(0.0, 0.5), energies=(17.0,), z_motor="uz", tilt_offset={"mu": 0.1})
float((after.truth["mu_center"] - uz.truth["mu_center"]).mean())   # 0.1: the same grain, tilted
```

#### `SyntheticZStack`
`bexa.testing.synthetic.SyntheticZStack(root, dataset, z_values, mosa_scans, energy_scans, energies, layers, truth={})`

Dataclass: `z_values`, `energies`, `mosa_scans` (the mosaicity scan of every height),
`energy_scans` (first and last scan of every height's energy series), `layers` (the
`SyntheticEsrfScan` of every mosaicity scan) and `truth`: `grain_center` (`(heights, 2)`, row and
column of grain A) and the `(heights, H, W)` maps `energy_center` (keV), `chi_center`,
`mu_center` (deg), `labels` (region labels) and `grains` (grain indices).

- `dataset_dir`: property, `<root>/<dataset>`, for `bexa.open_dataset`.

```python
# zs: the synthetic z-stack; its folder opened as a dataset, the scans per height
ds = bexa.open_dataset(zs.dataset_dir)
len(ds), ds.groups("samz")[0.0]                 # (30, [13, 14, 15, 16, 17, 18])
```

#### `dfxm_sample`
`bexa.testing.synthetic.dfxm_sample(frame_shape, z=None, *, microstructure=None, seed=0, peak=4000.0, regions=SAMPLE_REGIONS)`

The polycrystal (`polycrystal(frame_shape, seed=seed)`, or `microstructure`, whose `(H, W)` must
be `frame_shape`) cut at height `z` (pixels from the top of the box; default the middle, where
grains A and B are largest), as per-pixel maps: grain A split by its wall into `domain_a1` and
`domain_a2`, grain B at a clearly different orientation, and grains that stay dark in chi +-0.5
and mu +-1 deg scans. Inside a grain the orientation varies by hundredths of a degree and the
intensity by about 10 %; a wider curve is lower, keeping the area. The brightest domain peaks
near `peak` counts. Returns a `SampleModel`; `**sample.scan_kwargs` passes it to `make_esrf_scan`.

```python
# a sample model (grains, boundaries) for a scan of 48 x 56 pixels
sample = dfxm_sample((48, 56))
np.unique(sample.labels)                        # array([0, 1, 2, 3])
made = make_esrf_scan(work, dataset="grains", motors=(("chi", 9), ("mu", 21)), frame_shape=(48, 56),
                      n_files=1, background=0.0, **sample.scan_kwargs)
```

#### `SampleModel`
`bexa.testing.synthetic.SampleModel(amplitude, centers, widths, energy_offset, labels, grain_center, grains, microstructure)`

Dataclass of `(H, W)` maps: `amplitude` (peak counts), `centers` and `widths` (`{"chi": map,
"mu": map}`, deg), `energy_offset` (keV from the nominal energy: the strain of each region),
`labels` (the `Region` label, 0 to 3) and `grains` (grain index); plus `grain_center` (row,
column of grain A; NaN when absent) and the `microstructure` it was cut from.

- `scan_kwargs`: property, `{"amplitude": ..., "centers": ..., "widths": ...}` for `make_esrf_scan`.

```python
# sample: the sample model; what make_esrf_scan takes from it, the grain centre
sorted(sample.scan_kwargs)                      # ['amplitude', 'centers', 'widths']
sample.grain_center                             # (24.66..., 19.91...): row, column of grain A
```

#### `polycrystal`
`bexa.testing.synthetic.polycrystal(frame_shape, depth=None, seed=0, regions=SAMPLE_REGIONS)`

A 3-D polycrystal of Voronoi grains in a box `depth` pixels deep (default 1.2 times the smaller
side) behind an `(H, W)` field of view: grain A in the middle, grain B beside it and a little
higher, and a jittered grid of grains oriented outside the scans (|chi| above 1 deg or |mu| above
1.6 deg). The same `seed` gives the same polycrystal. Returns a `Microstructure`;
`dfxm_sample(..., microstructure=micro)` cuts it at any height.

```python
# a 3-D polycrystal of 48 x 56 pixels with seeded grains
micro = polycrystal((48, 56), seed=0)
micro.shape, len(micro.seeds)                   # ((58, 48, 56), 14)
upper = dfxm_sample((48, 56), z=10, microstructure=micro)    # another section of the same grains
```

#### `Microstructure`
`bexa.testing.synthetic.Microstructure(shape, seeds, tilts, gradients, wall_normal, texture, grain_a=0, grain_b=1)`

Dataclass of a polycrystal in a `(depth, H, W)` box: `seeds` (`(n, 3)`, z y x in pixels),
`tilts` (`(n, 2)`, chi and mu per grain, deg), `gradients` (`(n, 2, 2)`, d(chi, mu)/d(y, x) in deg
per pixel), `wall_normal` (unit vector of the domain wall through grain A's seed), `texture` (a
smooth random field at a quarter of the resolution) and the indices of grains A and B.

- `slice(z)`: at height `z` (pixels), three `(H, W)` arrays: the grain of every pixel, its side
  of the domain wall (bool) and the texture.

```python
# micro: the polycrystal; one slice through it at a height
grains, side, texture = micro.slice(29.0)
grains.shape, side.dtype                        # ((48, 56), dtype('bool'))
```

#### `Region`
`bexa.testing.synthetic.Region(label, tilt, width, brightness, energy_offset)`

Frozen dataclass of one kind of region of the synthetic sample: `label` (its value in the label
map), `tilt` (chi and mu centre of the rocking curve, deg; `None`: each grain its own), `width`
(chi and mu sigma, deg), `brightness` (integrated intensity relative to the brightest) and
`energy_offset` (energy centre relative to the nominal energy, keV). `SAMPLE_REGIONS` holds the
four defaults; pass a changed copy as `regions`.

```python
# the regions of the standard sample plus one of your own
from bexa.testing import SAMPLE_REGIONS, Region
regions = {**SAMPLE_REGIONS, "grain_b": Region(3, (0.3, 0.6), (0.08, 0.12), 0.75, 0.01)}
sample_b = dfxm_sample((48, 56), regions=regions)       # grain B further off and more strained
```

#### `make_pal_run`
`bexa.testing.synthetic.make_pal_run(root, run=1, scan=1, n_points=5, shots_per_point=16, frame_shape=(24, 32), laser_key="event_info.THIRTY_HERTZ", delays=None, missing_shots=1, seed=0)`

A PAL-XFEL run in the September 2025 layout, two files `pNNNN.h5` per point: a pandas table (key
`measurements`, one row per shot: `delay_input`, the laser flag `laser_key`, qbpm I0s, the
Jungfrau ROI sum, beam position, `th`, `phi`, `chi`, `tth`, `laser_v`, `laser_h`) in
`<root>/type=measurement/run=RRR/scan=SSS/`, and float32 frames at `PAL_FRAMES_PATH` with a
`metadata` table in `<root>/type=raw/run=RRR/scan=SSS/`. The laser alternates shot by shot,
`missing_shots` flags per point are NaN, pixel (0, 0) is negative (a dead pixel) and the pumped
peak is scaled by 1 + 0.3 tanh(delay); `delays` default to `linspace(-2, 8, n_points)`. Read by
`pal_xfel_points_2025_09` (`bexa.open`, `xfel_cube`); returns a `SyntheticPalRun`.

```python
# work: a scratch folder; a PAL-XFEL run of four points, opened as a scan
pal = make_pal_run(work / "pal", run=7, n_points=4)
bexa.open(pal.measurement_dir).dims             # ('frame', 'y', 'x'): every shot is a frame
pal.truth["frames_on"].shape, pal.truth["signals_off"].shape     # ((4, 24, 32), (4,))
```

#### `SyntheticPalRun`
`bexa.testing.synthetic.SyntheticPalRun(root, run, scan, point_files, raw_files, laser_key, frame_shape, delays, tables, frames, truth={})`

Dataclass: `point_files` and `raw_files`, the planted `delays`, `tables` (pandas) and `frames`
(per point, `(shots, H, W)` float32), and `truth`, what the legacy cube builder computes:
`delays` (median readback per point), `frames_on`, `frames_off` (`(points, H, W)`), `signals_on`
and `signals_off` (`(points,)`).

- `measurement_dir`: property, `<root>/type=measurement/run=RRR/scan=SSS`.
- `raw_dir`: property, `<root>/type=raw/run=RRR/scan=SSS`.

```python
# pal: the synthetic PAL-XFEL run; its folders and the columns of its first point
pal.measurement_dir.parent.name, pal.raw_dir.name       # ('run=007', 'scan=001')
list(pal.tables[0].columns)[:3]         # ['timestamp', 'delay_input', 'event_info.THIRTY_HERTZ']
```

#### `make_pal_spec_scan`
`bexa.testing.synthetic.make_pal_spec_scan(root, run=1, scan=1, shot=1, motors=(("delay", 8, -1.0, 6.0),), frame_shape=(20, 24), partial=0, detector="jungfrau", seed=0)`

A PAL-XFEL scan file of the April 2025 layout, `<root>/type=measurement/run=RRR/scan=SSS/pNNNN.h5`
(`NNNN` is `shot`): group `run/scan<NNNN>` with the attributes `scanMode` and `scanHistory` (the
command, `a1scan` for one motor, `a2scan` for two, first outer), motor arrays `motor/m1`, `m2`,
I0 in `det/ohqbpm2_totalsum` and images in `det/<detector>/data`. `motors` holds
`(name, points, start, stop)`; `partial` points go unrecorded (whole rows for two motors). Read
by `pal_xfel_spec_2025_04`; returns a `SyntheticSpecScan`.

```python
# work: a scratch folder; a SPEC-style PAL-XFEL scan file, opened
spec = make_pal_spec_scan(work / "pal_spec")
bexa.open(spec.path).dims                       # ('delay', 'y', 'x')
```

#### `SyntheticSpecScan`
`bexa.testing.synthetic.SyntheticSpecScan(root, run, scan, shot, path, motors, n_recorded, frame_shape, images, i0)`

Dataclass: the file `path`, `motors` (name to planned positions), `n_recorded` points, `images`
(`(n_recorded, H, W)` float32) and `i0` (`(n_recorded,)`).

```python
# spec: the synthetic SPEC-style scan; its file, the points recorded, its images
spec.path.name, spec.n_recorded, spec.images.shape      # ('p0001.h5', 8, (8, 20, 24))
```

#### `make_aps_session`
`bexa.testing.synthetic.make_aps_session(root, name="wse2_1_1", scans=DEFAULT_APS_SCANS, frame_shape=(24, 32), energy_keV=11.5, seed=0)`

Write an APS 7-ID session: `<root>/<name>.spec` with the 7-ID-C motor and counter lists, and
one areaDetector TIFF per scan point, `<name>_S<scan:03d>_<point:05d>.tif`. `scans` holds
`(number, command, options)`: SPEC step scans (`ascan`, `a2scan`) and `loopscan`; the options are
`partial` (points not taken, with SPEC's "aborted" comment), `frames=False` (rows without frames,
as a session copied without its TIFFs) and `missing` (point numbers whose frame is left out).
Every frame holds a Gaussian spot whose centre moves with the first motor and whose intensity
peaks mid-scan; the `r1` counter column is the frame's total. The default session has a
loopscan, two ascans (one aborted), an a2scan and a scan without frames.

```python
# work: a scratch folder; the default session, then one of your own with a missing frame
session = make_aps_session(work / "aps")
session.path.name, sorted(session.scans), session.rows   # ('wse2_1_1.spec', [1, 2, 3, 4, 5], {1: 5, 2: 9, 3: 3, 4: 6, 5: 5})
mine = make_aps_session(work / "aps2", scans=[(7, "ascan phi 0 1 4 1", {"missing": [2]})])
len(mine.frames[7]), bexa.open(mine.path, cache=False).structure.n_missing   # (4, 1)
```

#### `SyntheticApsSession`
`bexa.testing.synthetic.SyntheticApsSession(root, name, path, frame_shape, energy_keV, scans=<factory>, frames=<factory>, coords=<factory>, rows=<factory>, truth=<factory>, totals=<factory>)`

What `make_aps_session` wrote: `path` (the SPEC file), per scan number the `scans` command,
the `frames` written, the `coords` (motor values per point recorded), the `rows` recorded,
the `truth` (`(n, 2)` spot centre `(y, x)` per point) and the `totals` (the `r1` counter).

```python
# session: the default synthetic session; the planted truth of scan 2
session.coords[2]["phi"][:3], session.truth[2][4]   # (array([8.8, 8.925, 9.05]), array([12., 16.]))
```

#### `make_lcls_cube`
`bexa.testing.synthetic.make_lcls_cube(root, run=296, n_steps=6, repeats=4, frame_shape=(24, 24), seed=0)`

An LCLS XCS cube triple in `root`: `Run<NNNN>_onStk.npy`, `Run<NNNN>_offStk.npy` (float32 stacks,
one frame per shot, `n_steps` scan values times `repeats` shots) and `Run<NNNN>_stats.csv`
(columns `shot`, `scanvar`, `OnI0`, `OffI0`; the first shot has `OnI0` 0, a dropped shot).
Read by `lcls_xcs_cube` (`bexa.load` or `bexa.open` of the folder); returns a `SyntheticLclsCube`.

```python
# work: a scratch folder; an LCLS cube of shots, loaded back
lcls = make_lcls_cube(work / "lcls")
bexa.load(lcls.root)["frames"].shape            # (2, 24, 24, 24): laser, shot, y, x
```

#### `SyntheticLclsCube`
`bexa.testing.synthetic.SyntheticLclsCube(root, run, on_path, off_path, stats_path, scanvar, i0_on, i0_off, truth)`

Dataclass: the three paths, `scanvar`, `i0_on` and `i0_off` per shot, and `truth`, the legacy
binning: `bins` (the unique scan values), `avg_on` and `avg_off` (`(bins, H, W)` sums of the
I0-normalised images per bin, shots with an I0 of 0 skipped; sums despite the name).

```python
# lcls: the synthetic LCLS cube; the planted truth it was made from
lcls.truth["bins"], lcls.truth["avg_on"].shape   # (array([-1.,  0.,  1.,  2.,  3.,  4.]), (6, 24, 24))
```

#### `make_legacy_cube`
`bexa.testing.synthetic.make_legacy_cube(path, n_points=6, frame_shape=(20, 24), axis="delays", seed=0)`

A reduced cube in the `runN.h5` layout of `Aaron_allscan_cube_parallel.py` (and `Scan_Combiner.py`,
`xfel_cube.save_legacy`): `delays`, `phi`, `chi`, `th`, `tth`, `laser_v`, `laser_h` (`axis` runs
from -1 to 4, the others are 1.5), `index`, `signals_on`, `signals_off` and float64 frames
`jungfrau_on`, `jungfrau_off` (`(n_points, H, W)`, the pumped peak scaled by 1 + 0.2 tanh of the
axis). Read by `bexa_cube_legacy` (`bexa.load`, `bexa.open`); returns the path.

```python
# work: a scratch folder; a cube in the layout of the old XFEL scripts, loaded back
path = make_legacy_cube(work / "reduced" / "run42.h5", axis="th")
bexa.load(path)["frames"].dims                  # ('laser', 'th', 'y', 'x')
```

Constants:

- `SAMPLE_REGIONS`: the four `Region`s of `dfxm_sample`: `others` (label 0), `domain_a1` (1), `domain_a2` (2), `grain_b` (3).
- `ESRF_FRAMES_PATH`: `'entry_0000/ESRF-ID03/{detector}/data'`, the frames in an ESRF detector file.
- `PAL_FRAMES_PATH`: `'/detector/eh1/jungfrau2/image/block0_values'`, the frames in a PAL-XFEL raw file.

### bexa.geometry.conventions: named lab frames and motor stacks

Every transform and solver takes a named convention instead of hard-coding the beam axis, the
sample-motor axes and the zero of the detector azimuth eta; `bexa.geometry` re-exports
`CONVENTIONS`, `Convention` and the `rotation_*` helpers. Angles are in degrees. The registered
conventions, all with `eta_zero_offset = 0` (the solvers take the offset as an argument):

| Name | Beam | Horizontal, vertical | Sample stack, bottom to top (axis) | eta | Used for |
|---|---|---|---|---|---|
| `id03` | +z | +x, +y | mu (y), chi (z), phi (y), about fixed lab axes | `vertical` mode | ESRF ID03 DFXM (v9's transform), the `esrf_*` formats; the default when a format names none |
| `pal_fourc` | +x | +y, +z | mu (y), chi (x), phi (y), omega (z), each carried by the motors below | `vertical`: 0 for vertical scattering, 90 towards +y | PAL-XFEL (`pal_xfel_*`, `bexa_cube_legacy`), `four_circle_diffractometer.py`, `Motor_Solver_v1.py` |
| `sacla` | +x | +y, +z | as `pal_fourc` | `sacla`: from -y, anticlockwise looking downstream, 0 to 360 | SACLA, `Motor_Solver_SACLA.py` (with `eta_zero_offset=-180`) |
| `lcls_xcs` | alias of `pal_fourc` | | | | LCLS XCS (`lcls_smalldata`, `lcls_xcs_cube`) |
| `euxfel_mid` | alias of `id03` | | | | EuXFEL MID (`euxfel_extra_data`) |

```python
import numpy as np
from bexa.core.units import wavevector
from bexa.geometry.conventions import (CONVENTIONS, Convention, eta_of_kf, get_convention,
                                       rotation_axis, rotation_x, rotation_y, rotation_z)
k = wavevector(11.1)                  # 5.6252 1/Angstrom at 11.1 keV
```

#### `Convention`
`bexa.geometry.conventions.Convention(name, beam, horizontal, vertical, motor_axes={}, motor_order=(), coupled=True, eta_mode="vertical", eta_zero_offset=0.0, description="")`

Frozen dataclass for one instrument. `beam`, `horizontal` and `vertical` are lab unit vectors
(2theta turns `k_f` from `beam` towards `horizontal`, delta towards `vertical`); `motor_axes` maps
each sample motor to its axis at zero and `motor_order` lists the stack from the bottom. With
`coupled=True` each axis is carried by the motors below it (a goniometer, `R = R_1 R_2 ... R_n`
from the bottom up); `False` rotates about fixed lab axes, `R = R_n ... R_1` (`id03`:
`R_phi R_chi R_mu`). Rotations are right-handed. `eta_mode` is `"vertical"` or `"sacla"`.

- `sample_rotation(**angles_deg)`: 3 x 3 rotation of the stack for angles in degrees given by
  motor name; motors left out stay at 0, names outside `motor_order` are ignored.
- `kf_direction(two_theta_deg, delta_deg=0.0)`: unit `k_f`, shape `(..., 3)`; arrays broadcast.
- `detector_angles(Q_lab, k)`: `(two_theta, eta)` in degrees for a lab Q and wavevector `k`
  (1/Angstrom); a Q off the Ewald sphere is projected onto it.
- `with_eta_offset(eta_zero_offset)`: a copy with another eta zero (degrees).

```python
# pal: the PAL-XFEL four-circle convention; the sample rotation for two angles
pal = CONVENTIONS["pal_fourc"]
R = pal.sample_rotation(mu=10.0, omega=30.0)               # 3 x 3; chi and phi stay at 0
Q = k * (pal.kf_direction(40.0) - np.array(pal.beam))      # lab Q for 2theta = 40 deg
pal.detector_angles(Q, k)                                  # (40.0, 90.0): scattering towards +y
CONVENTIONS["sacla"].with_eta_offset(-180).detector_angles(Q, k)   # (40.0, 0.0)
mine = Convention("my_stack", (1, 0, 0), (0, 1, 0), (0, 0, 1), {"th": (0, 0, 1)}, ("th",))
```

#### `get_convention`
`bexa.geometry.conventions.get_convention(name)`

The registered `Convention` for a name; a `Convention` passes through and `None` gives `id03`.
Unknown names raise `KeyError` listing the known ones. Every `convention=` argument goes through it.

```python
# a convention by name, and the default one
get_convention("lcls_xcs").name, get_convention(None).name     # ('pal_fourc', 'id03')
```

#### `eta_of_kf`
`bexa.geometry.conventions.eta_of_kf(kf_h, kf_v, convention)`

Azimuth eta in degrees from the horizontal and vertical components of `k_f` (numbers or arrays).
`"vertical"` mode: `atan2(kf_h, kf_v) - eta_zero_offset`, 0 straight up and 90 towards
+horizontal; `"sacla"` mode: `atan2(kf_v, -kf_h) - eta_zero_offset` wrapped to [0, 360), 0
towards -horizontal and 90 straight up.

```python
# the azimuth of a scattered beam in two conventions
eta_of_kf([0.0, 0.5], [1.0, 0.5], CONVENTIONS["pal_fourc"])    # [0.0, 45.0]
eta_of_kf(0.0, 1.0, CONVENTIONS["sacla"])                      # 90.0
```

#### `rotation_axis`
`bexa.geometry.conventions.rotation_axis(axis, angle_deg)`

3 x 3 right-handed rotation about `axis` (normalised inside) by `angle_deg` degrees (Rodrigues).

```python
# a 120 degree rotation about the body diagonal applied to x
rotation_axis((1, 1, 1), 120.0) @ [1, 0, 0]      # [0, 1, 0]
```

#### `rotation_x`
`bexa.geometry.conventions.rotation_x(angle_deg)`

Rotation about the lab x axis by `angle_deg` degrees.

```python
# a 90 degree rotation about x applied to y
rotation_x(90.0) @ [0, 1, 0]                     # [0, 0, 1]
```

#### `rotation_y`
`bexa.geometry.conventions.rotation_y(angle_deg)`

Rotation about the lab y axis by `angle_deg` degrees.

```python
# a 90 degree rotation about y applied to z
rotation_y(90.0) @ [0, 0, 1]                     # [1, 0, 0]
```

#### `rotation_z`
`bexa.geometry.conventions.rotation_z(angle_deg)`

Rotation about the lab z axis by `angle_deg` degrees.

```python
# a 90 degree rotation about z applied to x
rotation_z(90.0) @ [1, 0, 0]                     # [0, 1, 0]
```

Constants:

- `CONVENTIONS`: dict of name to `Convention`, as in the table above (`lcls_xcs` and `euxfel_mid`
  are the same objects as `pal_fourc` and `id03`).

### bexa.geometry.detector: detector and beam geometry

`DetectorGeometry` holds the energy and where the detector is. Every `Scan` carries one as
`scan.geometry`, built from the format's detector table and the profile's `geometry:` section.

```python
import bexa
from bexa.geometry import DetectorGeometry

geom = DetectorGeometry(energy_keV=17.0, pixel_size_um=6.5, distance_mm=850.0,  # the far-field camera at 17 keV: pixel size, distance, beam centre
                        beam_center=(1080, 1280), detector_shape=(2160, 2560))
```

#### `DetectorGeometry`
`bexa.geometry.detector.DetectorGeometry(energy_keV=None, pixel_size_um=None, distance_mm=None, beam_center=None, two_theta_offset_deg=0.0, delta_offset_deg=0.0, effective_pixel_nm=None, detector_shape=None, convention="id03", detector=None, extra={})`

Dataclass: energy in keV, pixel pitch in um, sample-detector distance in mm, `beam_center` as
the `(y, x)` pixel of the direct beam, detector arm angles in degrees (added to the pixel
angles), `effective_pixel_nm` (pixel size on the sample, for scalebars), `detector_shape`
`(height, width)`, the convention name, the detector name and a free `extra` dict.

- `wavelength_A`, `k0`, `pixel_size_mm`: properties, the wavelength in Angstrom, `2 pi / lambda`
  in 1/Angstrom and the pixel in mm; `ValueError` when the input is unset.
- `center()`: `beam_center`, or the middle of `detector_shape`; `ValueError` if neither is set.
- `with_energy(energy_keV)`: a copy at another energy.

```python
# geom: the detector geometry of the demo scan; its wavelength, wavevector and pixel size
print(geom)                                          # 17.0000 keV, 6.5 um pixels, 850 mm, convention id03
geom.wavelength_A, geom.k0, geom.pixel_size_mm       # (0.7293, 8.6151, 0.0065)
DetectorGeometry(detector_shape=(2160, 2560)).center()     # (1080.0, 1280.0)
geom.with_energy(19.0).energy_keV                    # 19.0; geom keeps 17.0
```

#### `DetectorGeometry.from_spec`
`bexa.geometry.detector.DetectorGeometry.from_spec(spec, detector=None, energy_keV=None, profile=None)`

Classmethod behind `bexa.open`: pixel size, effective pixel and shape from the format spec's
`detectors` entry for `detector`, the spec's convention (`id03` if none), then every field set in
the profile's `geometry` section (a pydantic model or a dict) on top.

```python
# the geometry of the ESRF far-field camera, from the format spec at 17 keV
from bexa.io.formats import load_spec
ff = DetectorGeometry.from_spec(load_spec("esrf_id03_bliss_2026"), "pco_ff", energy_keV=17.0,
                                profile={"distance_mm": 850.0})
print(ff)                   # 17.0000 keV, 6.5 um pixels, 850 mm, 235 nm effective pixel, convention id03
print(bexa.demo("mosa").geometry)    # what bexa.open attached (no distance)
```

### bexa.geometry.transforms: pixels to angles to Q to hkl

Ported from v9's `pixels_to_angles`, `angles_to_Q` and `motor_angles_to_Q`, with the convention
explicit (`convention=None` means the geometry's). Units: `pixels_to_angles` returns radians
and `angles_to_Q` takes radians; the other functions take degrees. Q is `k_f - k_i` in
1/Angstrom, 2 pi included. With `id03`, Q agrees with v9's only while mu and phi are 0: v9
composed `R_mu R_chi R_phi` and turned mu and phi the other way (see `Convention`).

```python
import numpy as np
import bexa
from bexa.core.units import bragg_angle_deg
from bexa.crystal import Crystal, LatticeParameters
from bexa.geometry import DetectorGeometry
from bexa.geometry.transforms import (angles_to_Q, build_coordinate_grids, motor_angles_to_Q,
                                      pixels_to_angles, q_magnitude, scattering_vector)

geom = DetectorGeometry(energy_keV=17.0, pixel_size_um=6.5, distance_mm=850.0,  # the far-field camera at 17 keV: pixel size, distance, beam centre
                        beam_center=(1080, 1280), convention="id03")
scan = bexa.demo("mosa")                 # synthetic chi x mu scan (6 x 16 points) at 17 keV
```

#### `pixels_to_angles`
`bexa.geometry.transforms.pixels_to_angles(pixel_x, pixel_y, geometry)`

Scattering angles `(two_theta, delta)` in radians of pixels (column `x`, row `y`; numbers or
arrays): `two_theta = atan(dx / L)`, `delta = atan(dy / sqrt(dx^2 + L^2))` from the beam
centre, plus the arm offsets. Needs `distance_mm`, `pixel_size_um` and a beam centre or shape.

```python
# geom: the detector geometry; two pixels as scattering and azimuth angles
tt, delta = pixels_to_angles([1280, 1780], [1080, 1080], geom)
np.degrees(tt)                                  # [0.0, 0.2191]: 500 pixels = 3.25 mm at 850 mm
yy, xx = np.mgrid[0:2160:8, 0:2560:8]
tt, delta = pixels_to_angles(xx, yy, geom)      # radians, (270, 320) each
```

#### `angles_to_Q`
`bexa.geometry.transforms.angles_to_Q(two_theta, delta, geometry, convention=None, **motor_angles_deg)`

Sample-frame `(Qx, Qy, Qz)` in 1/Angstrom for scattering angles in radians and sample motor
angles in degrees (`mu=`, `chi=`, `phi=`): `Q_sample = R^T Q_lab`, `R` the convention's sample
rotation, `k` from `geometry.energy_keV`. Components have the shape of the angle arrays.

```python
# tt, delta: the angles of two pixels; their Q at a sample tilt, and the straight-through case
Qx, Qy, Qz = angles_to_Q(tt, delta, geom, mu=0.1, chi=-0.2)   # (270, 320) each
angles_to_Q(np.radians(20.0), 0.0, geom)                      # (2.9466, 0.0, -0.5196)
```

#### `scattering_vector`
`bexa.geometry.transforms.scattering_vector(two_theta_deg, delta_deg, k0, convention="id03")`

Lab-frame `Q = k_f - k_i`, shape `(..., 3)`, in the units of `k0`, for angles in degrees.

```python
# the scattering vector of a 20 degree beam in the default and the PAL conventions
scattering_vector(20.0, 0.0, geom.k0)                  # [2.9466, 0.0, -0.5196]: id03, beam along z
scattering_vector(20.0, 0.0, geom.k0, "pal_fourc")     # [-0.5196, 2.9466, 0.0]: beam along x
```

#### `q_magnitude`
`bexa.geometry.transforms.q_magnitude(two_theta_deg, k0)`

`|Q| = 2 k0 sin(two_theta / 2)` for 2theta in degrees; numbers or arrays.

```python
# the length of Q for two scattering angles
q_magnitude([10.0, 20.0], geom.k0)       # [1.5017, 2.992] 1/Angstrom
```

#### `motor_angles_to_Q`
`bexa.geometry.transforms.motor_angles_to_Q(two_theta_deg, geometry, convention=None, **motor_angles_deg)`

Sample-frame Q of one Bragg condition (2theta in degrees, delta 0) at many sample orientations,
the 3-D reciprocal-lattice-point view of a mosaicity scan: each motor keyword is an array of
degrees, the arrays broadcast, and `(Qx, Qy, Qz)` take their shape. One rotation per point.

```python
# geom: the detector geometry; Q over a chi x mu grid at a fixed scattering angle
chi, mu = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-0.5, 0.5, 5), indexing="ij")
Qx, Qy, Qz = motor_angles_to_Q(20.0, geom, chi=chi, mu=mu)     # each (3, 5)
np.sqrt(Qx**2 + Qy**2 + Qz**2).max()                           # 2.992: |Q| is set by 2theta
```

#### `build_coordinate_grids`
`bexa.geometry.transforms.build_coordinate_grids(structure, geometry, crystal=None, space="angular", hkl_center=None, **fixed_angles_deg)`

Coordinates of every motor point of a scan (`scan.structure`) as a dict of arrays shaped like the
motor grid: the coordinate modes of v9's `plot_3d_rlp`, used by
`bexa.analysis.projections.rlp_points` and the ROI-3DRLP napari app.

- `"angular"`: the motor values.
- `"Q"`: `Qx`, `Qy`, `Qz` from `motor_angles_to_Q`. The Bragg angle is `two_theta=` (degrees) or
  comes from `crystal.d_spacing(hkl_center)` and the energy (`ValueError` without either). Dims
  that are not sample motors of the convention are ignored; keywords such as `phi=0.5` fix others.
- `"hkl"`: `h`, `k`, `l` = `B^-1 Q` with the crystal axes along the sample axes (no orientation
  matrix); needs `crystal`. `hkl_center` only sets the Bragg angle, the indices are not shifted.

```python
# scan: the demo chi x mu scan; the Q coordinates of every grid point
two_theta = 2 * bragg_angle_deg(2.338, scan.energy_keV)          # Al 111 at 17 keV: 17.95 deg
grids = build_coordinate_grids(scan.structure, scan.geometry, space="Q",
                               two_theta=two_theta)             # Qx, Qy, Qz, each (6, 16)
al = Crystal(LatticeParameters.cubic(4.0495), centering="F")
hkl = build_coordinate_grids(scan.structure, scan.geometry, crystal=al, space="hkl",
                             hkl_center=(1, 1, 1))               # h, k, l, each (6, 16)
```

### bexa.crystal.lattice: cells, d-spacings and reflections

Cell parameters, the B matrix, centering rules and equivalent reflections for all the solvers;
`bexa.crystal` re-exports every name, so `from bexa.crystal import Crystal` works. Lengths in
Angstrom, angles in degrees.

```python
import numpy as np
from bexa.crystal import (BravaisLattice, Crystal, LatticeParameters, centering_allowed,
                          d_spacing, detect_bravais)

si = LatticeParameters.silicon()                 # cubic, a = 5.43102 Angstrom
crystal = Crystal(si, centering="F", name="Si")  # silicon as a Crystal
```

#### `LatticeParameters`
`bexa.crystal.lattice.LatticeParameters(a, b, c, alpha=90.0, beta=90.0, gamma=90.0)`

Frozen dataclass of the cell: lengths in Angstrom, angles in degrees.

- `cubic(a)`, `tetragonal(a, c)`, `hexagonal(a, c)` (gamma 120), `orthorhombic(a, b, c)`,
  `monoclinic(a, b, c, beta)`, `diamond()` (a = 3.5668), `silicon()` (a = 5.43102): classmethods.
- `volume`, `angles_rad`: properties, cubic Angstrom and `(alpha, beta, gamma)` in radians.
- `metric_tensor()`: the 3 x 3 direct metric tensor in Angstrom^2.
- `d_spacing(hkl)`: d in Angstrom from the reciprocal metric tensor (the module's `d_spacing`).
- `as_dict()`: the six values by name, a form `create_diffractometer` also accepts.

```python
# si: the silicon cell; a spacing and the volume; an orthorhombic cell's spacing
si.d_spacing((1, 1, 1)), si.volume                              # (3.1356, 160.19)
LatticeParameters.orthorhombic(3.4979, 6.3383, 15.4319).d_spacing((0, 4, 0))   # Td-WTe2: 1.5846
ws2 = LatticeParameters.hexagonal(3.1532, 12.323)
ws2.d_spacing((1, 1, 0)), ws2.as_dict()["gamma"]                # (1.5766, 120.0)
```

#### `Crystal`
`bexa.crystal.lattice.Crystal(lattice, bravais=None, centering="P", name="")`

A lattice with its B matrix: `Q = crystal.B @ (h, k, l)` in 1/Angstrom (2 pi included), a* along
x and b* in the xy plane. `bravais` (a `BravaisLattice` or its value) sets the symmetry of
`equivalent_reflections` and is guessed with `detect_bravais` when omitted; `centering` is used
by `is_allowed` and `reflections`. Attributes: `lattice`, `bravais`, `centering`, `name`, `B`,
`a_star`, `b_star`, `c_star`.

Caveat: `B` is built as in the legacy motor solvers, with direct-cell angles where the reciprocal
ones belong. It is exact when all cell angles are 90 deg; for hexagonal, trigonal, monoclinic and
triclinic cells `Crystal.d_spacing` differs from `LatticeParameters.d_spacing` for mixed indices
(WS2 (110): 2.7308 against 1.5766 Angstrom), and so do the solvers built on `Crystal`.

- `Q_vector(hkl)`: Q of a reflection in the crystal frame, 1/Angstrom.
- `hkl_from_Q(Q)`: continuous `(h, k, l)` arrays for Q vectors of shape `(..., 3)`.
- `is_allowed(hkl)`: `centering_allowed(hkl, crystal.centering)`.

```python
# crystal: silicon as a Crystal; a reciprocal lattice vector
crystal                                   # Crystal(Si: a=5.4310, b=5.4310, c=5.4310, alpha=90, beta=90, gamma=90, F)
crystal.Q_vector((1, 1, 1))               # [1.1569, 1.1569, 1.1569]
h, k, l = crystal.hkl_from_Q(crystal.Q_vector((2, -1, 3)))     # 2.0, -1.0, 3.0
crystal.is_allowed((2, 0, 0)), crystal.is_allowed((1, 0, 0))    # (True, False)
```

#### `Crystal.d_spacing`
`bexa.crystal.lattice.Crystal.d_spacing(hkl)`

d-spacing in Angstrom, `2 pi / |B hkl|` (`inf` for 000). With `Crystal.bragg_angle_deg` it
replaces the Bragg's-law cells of `Usefulthings`; in a terminal
`bexa crystal dspacing --lattice silicon --hkl 1 1 1 --energy 17` prints d, |Q|, the Bragg angle
and the number of equivalents.

```python
# crystal: silicon; the spacing of the 111 planes
crystal.d_spacing((1, 1, 1))               # 3.1356 Angstrom
```

#### `Crystal.bragg_angle_deg`
`bexa.crystal.lattice.Crystal.bragg_angle_deg(hkl, energy_keV)`

Bragg angle theta (not 2theta) in degrees at an energy in keV, from `lambda = 2 d sin(theta)`;
`ValueError` when the reflection is out of reach.

```python
# crystal: silicon; the Bragg angle of 111 at 17 keV
crystal.bragg_angle_deg((1, 1, 1), 17.0)         # 6.678 deg, so 2theta = 13.357 deg
```

#### `Crystal.equivalent_reflections`
`bexa.crystal.lattice.Crystal.equivalent_reflections(hkl)`

Sorted `(h, k, l)` tuples related to `hkl` by the Laue symmetry of `crystal.bravais`: permutations
and signs for cubic, `(h, k, l)` and `(k, h, l)` with signs for tetragonal, signs for
orthorhombic, the Friedel pair for triclinic. The hexagonal, trigonal and monoclinic rules also
return reflections with another d-spacing (hexagonal (1, 0, 2): 16 entries instead of 12), so
filter those with `LatticeParameters.d_spacing`. The solvers use this list for a family.

```python
# crystal: silicon; the reflections equivalent to 111 by symmetry, and to 311
crystal.equivalent_reflections((1, 1, 1))       # [(-1, -1, -1), (-1, -1, 1), ..., (1, 1, 1)]: 8
len(crystal.equivalent_reflections((3, 1, 1)))   # 24
```

#### `Crystal.reflections`
`bexa.crystal.lattice.Crystal.reflections(max_index=5, allowed_only=True)`

Every `(h, k, l)` with indices from `-max_index` to `max_index` except 000, only those the
centering allows unless `allowed_only=False`. In a terminal,
`bexa crystal reflections --lattice diamond --energy 11.1 --max-hkl 3 --centering F` lists the
reachable ones with d and 2theta.

```python
# crystal: silicon; every reflection up to index 2, allowed ones and all
len(crystal.reflections(2)), len(crystal.reflections(2, allowed_only=False))   # (34, 124)
```

#### `d_spacing`
`bexa.crystal.lattice.d_spacing(lattice, hkl)`

d-spacing in Angstrom of a `LatticeParameters`, `1/d^2 = h G* h` (any cell; `inf` for 000).

```python
# si: the silicon cell; the spacing of 220
d_spacing(si, (2, 2, 0))                         # 1.9202
```

#### `detect_bravais`
`bexa.crystal.lattice.detect_bravais(lattice, tol=0.001)`

Lattice system guessed from the cell (relative tolerance `tol`), as a `BravaisLattice`: with
right angles cubic, tetragonal (a = b; the unique axis must be c) or orthorhombic; then hexagonal
(a = b, gamma = 120), trigonal (a = b = c, alpha = beta = gamma), monoclinic (alpha = gamma =
90), triclinic. Pass `bravais=` to `Crystal` when the crystal is less symmetric than its cell.

```python
# the Bravais lattice of a hexagonal cell and of a rhombohedral one
detect_bravais(LatticeParameters.hexagonal(3.1532, 12.323))     # BravaisLattice.HEXAGONAL
detect_bravais(LatticeParameters(5, 5, 5, 80, 80, 80)).value    # 'trigonal'
```

#### `centering_allowed`
`bexa.crystal.lattice.centering_allowed(hkl, centering="P")`

True when `hkl` meets the lattice-centering condition: P all, I h+k+l even, F all even or all
odd, A k+l even, B h+l even, C h+k even, R -h+k+l divisible by 3 (hexagonal axes). Glide and
screw extinctions are not applied (silicon's 200 passes); other letters raise `ValueError`.

```python
# whether 111 and 100 are allowed in a face-centred lattice
centering_allowed((1, 1, 1), "F"), centering_allowed((1, 0, 0), "F")    # (True, False)
centering_allowed((1, 1, 1))                    # True
```

#### `BravaisLattice`
`bexa.crystal.lattice.BravaisLattice(value)`

String enum of the lattice systems (`CUBIC`, `TETRAGONAL`, `ORTHORHOMBIC`, `HEXAGONAL`,
`TRIGONAL`, `MONOCLINIC`, `TRICLINIC`) with lower-case values; build one as
`BravaisLattice("cubic")`. Members equal their strings, so `bravais="cubic"` works everywhere.

```python
# the lattice enum from a name, compared with a string
BravaisLattice("hexagonal"), BravaisLattice.CUBIC == "cubic"    # (BravaisLattice.HEXAGONAL, True)
```

### bexa.crystal.cif: cell and space group from CIF files

`crystal_from_cif` reads the structure with pymatgen when it is installed; otherwise, or when
pymatgen fails on the file, a built-in parser takes the cell and the space-group symbol, which is
all the hkl conversions and centering rules need. A profile sample's `cif:` becomes
`scan.crystal` this way.

```python
from pathlib import Path
from bexa.crystal import Crystal
from bexa.crystal.cif import CifInfo, crystal_from_cif, read_cif

path = Path("Si.cif")                           # a CIF written by the setup
path.write_text("data_Si\n_cell_length_a 5.43102(8)\n_cell_length_b 5.43102(8)\n"
                "_cell_length_c 5.43102(8)\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
                "_cell_angle_gamma 90\n_symmetry_space_group_name_H-M 'F d -3 m'\n"
                "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n"
                "_atom_site_fract_y\n_atom_site_fract_z\nSi1 Si 0 0 0\n")
```

#### `crystal_from_cif`
`bexa.crystal.cif.crystal_from_cif(path, use_pymatgen=True)`

A `Crystal` named after the file stem. Through pymatgen: the cell of the parsed structure,
`bravais` from its crystal system and `centering` from the space-group symbol (pymatgen may warn
about missing symmetry operations). With `use_pymatgen=False`, without pymatgen or when it fails
(a warning is logged), `read_cif` supplies the cell and `bravais` is guessed from it.

```python
# path: a CIF of silicon; the Crystal it describes
si = crystal_from_cif(path)                          # Crystal(Si: a=5.4310, ..., F) via pymatgen
si.bravais, si.centering                             # (BravaisLattice.CUBIC, 'F')
plain = crystal_from_cif(path, use_pymatgen=False)   # built-in parser: same cell and centering
```

#### `read_cif`
`bexa.crystal.cif.read_cif(path)`

The built-in parser: `_cell_length_*` and `_cell_angle_*` (uncertainties dropped, missing angles
90) and the symbol of `_symmetry_space_group_name_H-M` or `_space_group_name_H-M_alt`, whose first
letter is the centering (P if none). Returns a `CifInfo`; `ValueError` without the three lengths.
It reads the whole file, so with several data blocks the last cell wins but the first symbol is
kept: split such files first.

```python
# path: a CIF of silicon; its lattice, space group and centering
info = read_cif(path)
info.lattice.a, info.space_group, info.centering    # (5.43102, 'F d -3 m', 'F')
```

#### `CifInfo`
`bexa.crystal.cif.CifInfo(path, lattice, space_group=None, centering="P", formula=None)`

Dataclass returned by `read_cif`: the `path`, the `LatticeParameters`, the symbol as written, the
centering letter and `formula`, which the parser leaves at `None`.

```python
# info: what read_cif found; a Crystal built from it by hand
Crystal(info.lattice, centering=info.centering, name=info.path.stem)   # what use_pymatgen=False builds
```

### bexa.crystal.diffractometer: four-circle motor solutions

Motor angles that bring a reflection onto the detector of a four-circle diffractometer (sample
stack mu, chi, phi, omega; detector two_theta and eta). Port of `four_circle_diffractometer.py`,
`Motor_Solver_v1.py` and `Motor_Solver_SACLA.py`, which differ only in the convention. Angles in
degrees. The solver runs many local optimisations, a second or two per reflection on the lab
workstation: fix the motors you know.

```python
import numpy as np
import pandas as pd
from bexa.crystal import Crystal, LatticeParameters
from bexa.crystal.diffractometer import (FourCircleDiffractometer, MotorConfig, Solution,
                                         create_diffractometer, format_solutions, orientation_matrix)

# Motor_Solver_SACLA.py's case: diamond at 11.1 keV, (100) up, (011) horizontal
diff = create_diffractometer(LatticeParameters.diamond(), 11.1, out_of_plane=(1, 0, 0),  # a four-circle diffractometer for diamond at 11.1 keV
                             zone_axis=(0, 1, 1), bravais="cubic", convention="sacla",
                             eta_zero_offset=-180)
for name, (low, high) in {"mu": (-45, -10), "two_theta": (20, 70), "eta": (20, 120)}.items():
    diff.set_motor_range(name, low, high)
for name, value in {"chi": 0.0, "phi": 0.0, "omega": 35.5}.items():
    diff.set_motor_fixed(name, value)
```

#### `create_diffractometer`
`bexa.crystal.diffractometer.create_diffractometer(lattice, energy_keV, out_of_plane, zone_axis, bravais=None, convention="pal_fourc", motor_order=None, eta_zero_offset=0.0, centering="P")`

A `FourCircleDiffractometer` from cell parameters (`LatticeParameters` or a dict) in one call:
`out_of_plane` holds the Miller indices along the lab vertical and `zone_axis` those along the
horizontal perpendicular to the beam at zero angles; `eta_zero_offset` (degrees) is subtracted
from every eta. Replaces `Motor_Solver_SACLA.py` (`convention="sacla", eta_zero_offset=-180`, then
`solve_for_peak_family`); `bexa crystal solve` is the terminal version.

```python
# a four-circle diffractometer for a cubic crystal at 11 keV with a known orientation
pal = create_diffractometer({"a": 3.56, "b": 3.56, "c": 3.56}, 11.0, (1, 1, 0), (1, -1, 0),
                            eta_zero_offset=-90)                 # pal_fourc
pal.crystal.bravais, pal.motors["eta"].bounds()                  # (BravaisLattice.CUBIC, (-20.0, 20.0))
```

#### `FourCircleDiffractometer`
`bexa.crystal.diffractometer.FourCircleDiffractometer(crystal, energy_keV, out_of_plane, zone_axis, convention="pal_fourc", motor_order=None, eta_zero_offset=0.0)`

The solver for a `Crystal`. Attributes: `crystal`, `energy_keV`, `wavelength` (Angstrom), `k`
(1/Angstrom), `convention` (its eta offset replaced by `eta_zero_offset`), `U`, `UB` and `motors`,
a `MotorConfig` per motor. Default limits (deg): mu -10 to 90, chi -5 to 95, phi and omega
-180 to 180, two_theta 0 to 170, eta -20 to 20 (`pal_fourc`) or 0 to 360 (`sacla`).
`motor_order` reorders the stack (bottom first; exactly mu, chi, phi, omega).

- `set_motor_fixed(name, value)`, `set_motor_range(name, range_min, range_max)`,
  `set_motor_free(name, range_min=-180, range_max=180)`, `set_motor(name, config)`: constrain a
  motor; unknown names raise `KeyError`. A fixed motor may move by up to 0.001 deg in a solution.
- `default_positions()`: start angles of the sample motors (fixed value, middle of a range, or 0).
- `describe_motors()`: the constraints as text, one line per motor.
- `motor_order`, `eta_zero_offset`: properties.
- `sample_rotation(mu=0.0, chi=0.0, phi=0.0, omega=0.0)`: 3 x 3 rotation of the stack.
- `q_lab(hkl, mu, chi, phi, omega)`: lab-frame Q (1/Angstrom) of a reflection at those angles.
- `detector_angles(Q_lab)`: `(two_theta, eta)`, projected onto the Ewald sphere.
- `bragg_two_theta(hkl)`: 2theta from Bragg's law, `None` when out of reach.
- `residual(angles, hkl, include_detector_penalty=True)`: `(|k_f| - k)^2` for the sample angles in
  `motor_order`, plus 0.01 times the squared excursion of eta and two_theta outside their limits.
- `accessible_reflections(max_hkl=5)`: reflections Bragg's law allows (centering not applied).

```python
# diff: the diffractometer; its motors and their rest positions
print(diff.describe_motors())         # mu: -45 to -10 deg (range) ... eta: 20 to 120 deg (range)
diff.default_positions()              # {'mu': -27.5, 'chi': 0.0, 'phi': 0.0, 'omega': 35.5}
diff.bragg_two_theta((2, 2, 0)), diff.bragg_two_theta((8, 8, 8))     # (52.575, None)
Q = diff.q_lab((0, 2, 2), mu=-40.297, chi=0.0, phi=0.0, omega=35.5)
diff.detector_angles(Q)               # (52.575, 24.765)
len(diff.accessible_reflections(max_hkl=2))    # 124
```

#### `FourCircleDiffractometer.solve_for_peak_family`
`bexa.crystal.diffractometer.FourCircleDiffractometer.solve_for_peak_family(hkl, use_equivalents=True, current_positions=None, movement_threshold=0.1, max_solutions=50)`

Solves every equivalent of `hkl` (only `hkl` with `use_equivalents=False`) and returns the
`Solution`s sorted by the number of motors moved more than `movement_threshold` degrees from
`current_positions` (default `default_positions()`), then by total travel; at most
`max_solutions`, which does not shorten the search. The main loop of `Motor_Solver_SACLA.py`.

```python
# diff: the diffractometer; every way to reach a 220 reflection
solutions = diff.solve_for_peak_family((2, 2, 0), max_solutions=4)    # 12 equivalents: about 15 s
solutions[0]      # Solution(hkl=(-2, 0, 2), mu=-18.123, chi=0.001, ..., eta=82.521, d=1.2611 A, ...)
```

#### `FourCircleDiffractometer.solve_for_reflection`
`bexa.crystal.diffractometer.FourCircleDiffractometer.solve_for_reflection(hkl, current_positions=None, movement_threshold=0.1)`

Motor angles for one reflection, moving as few motors as possible: bounded local optimisations
from the current positions, single-motor and pair sweeps, a coarse grid and 20 seeded random
points, then a differential-evolution search only if none succeeds. Returns a `Solution` within
the eta and two_theta limits, or `None` (with a `UserWarning` if Bragg's law cannot be met).

```python
# diff: the diffractometer; the motor angles of one reflection
sol = diff.solve_for_reflection((0, 2, 2))         # about a second
sol.mu, sol.eta                                    # (-40.297, 24.765)
diff.solve_for_reflection((2, 2, 0)) is None       # True: out of reach within these limits
```

#### `format_solutions`
`bexa.crystal.diffractometer.format_solutions(solutions, diffractometer=None, movement_threshold=0.1, max_display=10)`

Text table of solutions (hkl, sample motors, 2theta, eta, d in Angstrom), at most `max_display`
rows. Given the diffractometer, motors moved more than `movement_threshold` degrees from its
default positions are marked `*`. An empty list gives "No solutions found.".

```python
# solutions: the 220 solutions; as a text table
print(format_solutions(solutions, diff))
# Found 2 solution(s):
#          hkl        mu       chi       phi     omega    2theta       eta      d(A)
# ----------------------------------------------------------------------------------
#     (-2,0,2)   -18.12*     0.00      0.00     35.50      52.57     82.52    1.2611
#      (0,2,2)   -40.30*    -0.00     -0.00     35.50      52.57     24.77    1.2611
format_solutions(solutions[:1], diff).splitlines()[0]  # 'Found 1 solution(s):'
```

#### `Solution`
`bexa.crystal.diffractometer.Solution(hkl, mu, chi, phi, omega, two_theta, eta, q_magnitude, d_spacing, energy_keV, residual, motor_order=("mu", "chi", "phi", "omega"))`

Dataclass: `hkl`, the six angles in degrees, `q_magnitude` (1/Angstrom), `d_spacing`
(Angstrom), `energy_keV`, `residual` (the solver's objective: elastic mismatch plus 1e-4 per
moved motor and 1e-6 per degree of travel) and the stack order for display.

- `angles()`: dict of the sample angles, `two_theta` and `eta`.
- `to_dict()`: `hkl`, the angles, `q_magnitude`, `d_spacing`, `energy_keV`, `residual`.

```python
# sol: one solution; its angles, and a table of several
sol.angles()                                        # {'mu': -40.297, 'chi': ..., 'eta': 24.765}
table = pd.DataFrame([s.to_dict() for s in solutions])     # one row per solution
```

#### `MotorConfig`
`bexa.crystal.diffractometer.MotorConfig(name, constraint="free", value=None, range_min=-180.0, range_max=180.0)`

Constraint of one motor: `constraint` is `"fixed"` (at `value`), `"range"` or `"free"` (both
between `range_min` and `range_max` degrees; they differ only in the default position). The
`set_motor_*` methods of `FourCircleDiffractometer` build these.

- `fixed(name, value)`, `ranged(name, range_min, range_max)`,
  `free(name, range_min=-180.0, range_max=180.0)`: classmethod constructors.
- `is_fixed`: property.
- `bounds()`: `(value, value)` when fixed, otherwise `(range_min, range_max)`.
- `default_position()`: the value, the middle of a range, or 0 when free.
- `penalty(angle)`: squared distance from the allowed values, 0 inside.
- `describe()`: one line of text.

```python
# a motor limited to a range; its bounds, rest position and the penalty outside
mu = MotorConfig.ranged("mu", -45, -10)
mu.bounds(), mu.default_position(), mu.penalty(-50.0)     # ((-45.0, -10.0), -27.5, 25.0)
MotorConfig.fixed("omega", 35.5).describe()               # 'omega: fixed at 35.5 deg'
pal.set_motor("chi", MotorConfig.free("chi", -5, 5))      # as pal.set_motor_free("chi", -5, 5)
```

#### `orientation_matrix`
`bexa.crystal.diffractometer.orientation_matrix(crystal, out_of_plane, zone_axis)`

U matrix with `out_of_plane` along the lab vertical and the perpendicular part of `zone_axis`
along the horizontal at zero angles (both as reciprocal vectors): `U @ crystal.Q_vector(hkl)` is
the lab Q (beam x, horizontal y, vertical z). `ValueError` when the two are parallel.

```python
# cubic: a cubic crystal; the matrix that puts two reflections along the beamline axes
cubic = Crystal(LatticeParameters.cubic(3.56))
U = orientation_matrix(cubic, (1, 1, 0), (1, -1, 0))
U @ cubic.Q_vector((1, 1, 0)) / np.linalg.norm(cubic.Q_vector((1, 1, 0)))    # [0, 0, 1]: (110) up
```

Constants:

- `SAMPLE_MOTORS`: `("mu", "chi", "phi", "omega")`; `DETECTOR_MOTORS`: `("two_theta", "eta")`.

### bexa.crystal.zone_axis: the zone axis from one reflection

Which crystal direction lies along the lab horizontal (the zone axis), from one indexed
reflection and the angles at which it was found. Port of `ZoneAxis_Solver_v1.py`, vectorised over
the candidate directions; same geometry as the diffractometer. Angles in degrees, residuals in
1/Angstrom.

```python
from bexa.crystal import Crystal, LatticeParameters
from bexa.crystal.zone_axis import (AngleFit, ZoneAxisCandidate, ZoneAxisSolver,
                                    normalize_zone_axis, reduce_indices, solve_zone_axis)

observed = {"mu": -38.0, "chi": 0.0, "phi": 0.0, "omega": 0.0, "two_theta": 53.2, "eta": -67.5}  # where a peak was seen: the motor angles
solver = ZoneAxisSolver(Crystal(LatticeParameters.cubic(3.56), bravais="cubic"), 11.0,  # the solver for a cubic crystal
                        out_of_plane=(1, 1, 0), eta_zero_offset=-90)
```

#### `solve_zone_axis`
`bexa.crystal.zone_axis.solve_zone_axis(lattice, energy_keV, out_of_plane, hkl_observed, angles, bravais=None, convention="pal_fourc", eta_zero_offset=0.0, search_range=10, num_results=1, angle_tolerances=None)`

One call: a `ZoneAxisSolver` for the cell (`LatticeParameters` or dict) and
`ZoneAxisSolver.find_zone_axis`. `angles` must hold `two_theta` and `eta`; mu, chi, phi, omega
default to 0. Returns up to `num_results` `ZoneAxisCandidate`s. Replaces `ZoneAxis_Solver_v1.py`;
the terminal version of the example below is
`bexa crystal zone-axis --lattice cubic:3.56 --energy 11 --hkl -2 0 -2 --oop 1 1 0 --angles mu=-38,chi=0,phi=0,omega=0,two_theta=53.2,eta=-67.5 --eta-offset -90`.

```python
# observed: the angles where a peak was seen; the zone axis that explains them
results = solve_zone_axis(LatticeParameters.cubic(3.56), 11.0, (1, 1, 0), (-2, 0, -2), observed,
                          bravais="cubic", eta_zero_offset=-90, search_range=3, num_results=3)
[(r.zone_axis, round(r.residual, 3)) for r in results]  # [(tuple of 3, 1.266), (tuple of 3, 1.266), (tuple of 3, 2.628)]
# [((-1, 0, 0), 1.266), ((0, 1, 0), 1.266), ((0, 0, -1), 2.628)]
```

#### `ZoneAxisSolver`
`bexa.crystal.zone_axis.ZoneAxisSolver(crystal, energy_keV, out_of_plane, convention="pal_fourc", motor_order=None, eta_zero_offset=0.0)`

The solver behind `solve_zone_axis`, with the geometry arguments of `FourCircleDiffractometer`.
Attributes: `crystal`, `energy_keV`, `wavelength`, `k`, `out_of_plane`, `convention`.

- `kf_direction(two_theta_deg, eta_deg)`: unit `k_f` for detector angles in degrees.
- `detector_to_q_lab(two_theta_deg, eta_deg)`: lab Q (1/Angstrom) of a peak seen there.
- `sample_rotation(mu, chi, phi, omega)`: 3 x 3 rotation of the stack.
- `candidate_axes(search_range)`: `(N, 3)` integer directions up to `search_range`, minus those
  within about 8 deg of the surface normal.
- `orientation_matrices(axes)`: `(N, 3, 3)` U matrices, as `orientation_matrix`, all at once.

```python
# solver: the solver; a detector position as Q, and the candidate axes
solver.detector_to_q_lab(53.2, -67.5)             # [-2.2352, -1.7082, -4.1239]
axes = solver.candidate_axes(3)                   # (336, 3)
solver.orientation_matrices(axes).shape           # (336, 3, 3)
```

#### `ZoneAxisSolver.find_zone_axis`
`bexa.crystal.zone_axis.ZoneAxisSolver.find_zone_axis(hkl_observed, mu, chi, phi, omega, two_theta, eta, search_range=10, num_results=1, angle_tolerances=None)`

Tries every direction up to `search_range` as the zone axis with every equivalent of
`hkl_observed`, scoring the smallest `|Q_predicted - Q_observed|`; a direction and its negative
count once. Candidates within ten times the best residual (at least 1/Angstrom) are sorted by
the axis's d-spacing, largest first, then by residual, so read `residual`, not only the rank.
With `angle_tolerances` (degrees per angle; 180 if missing, any two_theta), only axes that
`refine_angles` can fit survive, with the fit attached (seconds instead of milliseconds). The
fit clips angles to -180..180, so a `sacla` eta over 180 plus its tolerance raises `ValueError`.

```python
# solver: the solver; tol: the tolerances; the axes that fit the observed angles
tol = {"mu": 10.0, "chi": 10.0, "phi": 0.0, "omega": 10.0, "two_theta": 10.0, "eta": 10.0}
found = solver.find_zone_axis((-2, 0, -2), -38.0, 0.0, 0.0, 0.0, 53.2, -67.5,
                              search_range=3, num_results=3, angle_tolerances=tol)
[(c.zone_axis, c.matched_hkl) for c in found]   # [(tuple of 3, tuple of 3), (tuple of 3, tuple of 3), (tuple of 3, tuple of 3)]
# [((-1, 0, 0), (0, -2, 2)), ((0, 1, 0), (0, -2, 2)), ((-1, 0, -1), (-2, 0, 2))]
```

#### `ZoneAxisSolver.refine_angles`
`bexa.crystal.zone_axis.ZoneAxisSolver.refine_angles(zone_axis, hkl, observed, tolerances)`

Angles within `tolerances` (degrees) of the `observed` dict that put `hkl` exactly on the
detector for this zone axis: two_theta comes from Bragg's law and is only checked, the other five
move in a bounded local optimisation. Returns an `AngleFit`, or `None` if there is none.

```python
# solver: the solver; the angles refined for one axis against the observation
fit = solver.refine_angles((-1, 0, -1), (-2, 0, 2), observed, tol)
fit.angles["mu"], fit.differences["mu"]            # (-31.035, 6.965)
```

#### `ZoneAxisSolver.validate`
`bexa.crystal.zone_axis.ZoneAxisSolver.validate(zone_axis, hkl, mu, chi, phi, omega, two_theta, eta, tolerance=0.5)`

Forward check: computes the detector angles of `hkl` at the given motor angles and compares them
with `two_theta` and `eta`. Returns a dict: `valid` (both within `tolerance` degrees), the
calculated, observed and difference values of each, `Q_lab` and `k_f` (or `valid` False and an
`error` when the axis is parallel to the surface normal). The eta difference is not wrapped at 360.

```python
# solver: the solver; fit: the refined angles; checked against the reflection
check = solver.validate((-1, 0, -1), (-2, 0, 2), **fit.angles)
check["valid"], check["two_theta_diff"] < 1e-6      # (True, True)
```

#### `ZoneAxisCandidate`
`bexa.crystal.zone_axis.ZoneAxisCandidate(zone_axis, residual, d_spacing, matched_hkl, fit=None, details={})`

One result: `zone_axis` (reduced, sign kept), `residual` (1/Angstrom), `d_spacing` of the axis,
`matched_hkl` (the equivalent that fits), `fit` (an `AngleFit` with tolerances) and `details`
(observed Q, directions tested, search range, family size). Property `within_tolerance`: `fit`
is set.

```python
# best: the first axis found; its indices, whether it fits, what was tested
best = found[0]
best.zone_axis, best.within_tolerance, best.details["candidates_tested"]    # ((-1, 0, 0), True, 336)
```

#### `AngleFit`
`bexa.crystal.zone_axis.AngleFit(angles, residual, differences)`

Result of `ZoneAxisSolver.refine_angles`: `angles` (degrees, two_theta from Bragg's law),
`residual` (1/Angstrom) and `differences` from the observed angles (degrees; phi, omega and eta
wrapped at 360).

```python
# best.fit: the refined angles of the best axis; how far each is from the observation
{name: round(value, 3) for name, value in best.fit.differences.items()}  # {'mu': 9.998, 'chi': 0.809, 'phi': 0.0, 'omega': 8.027, 'two_theta': 0.001, 'eta': 4.317}
# {'mu': 9.998, 'chi': 0.809, 'phi': 0.0, 'omega': 8.027, 'two_theta': 0.001, 'eta': 4.317}
```

#### `normalize_zone_axis`
`bexa.crystal.zone_axis.normalize_zone_axis(hkl)`

A direction regardless of sign: reduced indices with the first non-zero one positive.

```python
# zone-axis indices reduced and with their sign fixed
normalize_zone_axis((-3, 3, 0)), normalize_zone_axis((0, -2, 4))    # ((1, -1, 0), (0, 1, -2))
```

#### `reduce_indices`
`bexa.crystal.zone_axis.reduce_indices(hkl)`

Indices divided by their greatest common divisor, sign kept.

```python
# indices divided by their common factor
reduce_indices((-3, 3, 0)), reduce_indices((2, 4, 6))       # ((-1, 1, 0), (1, 2, 3))
```

Constants:

- `ANGLE_NAMES`: `("mu", "chi", "phi", "omega", "two_theta", "eta")`, keys of observations,
  tolerances and fits; `WRAPPING_ANGLES`: `("phi", "omega", "eta")`, wrapped at 360 deg.

### bexa.crystal.omega_search: omega where two peaks share a mu window

Omega settings at which two reflections of one family are close in mu, so that one mu scan
catches both. Port of `omega_finder.py`: with chi = phi = 0 the elastic condition gives mu
analytically, every omega of a grid is evaluated at once and the minima of the closest-pair gap
are refined. Needs mu about the horizontal and omega about the vertical (`pal_fourc`, `sacla`).

```python
from bexa.crystal import LatticeParameters
from bexa.crystal.diffractometer import create_diffractometer
from bexa.crystal.omega_search import (DEFAULT_RANGES, OmegaCandidate, PeakHit,
                                       find_omega_offsets, format_candidates, peaks_at_omega)

diff = create_diffractometer(LatticeParameters.diamond(), 11.1, (1, 1, 0), (0, 0, 1),  # a four-circle diffractometer for diamond at 11.1 keV
                             bravais="cubic", convention="sacla", eta_zero_offset=-180)
ranges = {"mu": (-45.0, -10.0), "eta": (35.0, 100.0), "two_theta": (20.0, 70.0)}  # the motor limits
```

#### `find_omega_offsets`
`bexa.crystal.omega_search.find_omega_offsets(diffractometer, family, ranges=None, omega_scan=(-180.0, 180.0), step=0.01, top_n=5, omega_base=0.0, verify=False)`

The `top_n` best omegas (degrees), smallest mu gap first, as `OmegaCandidate`s. `ranges` gives
the mu, eta and two_theta windows (missing ones from `DEFAULT_RANGES`); the diffractometer gives
UB, energy and eta convention, not its motor limits. omega runs over `omega_scan` in `step`
degrees, minima are refined 1000 times finer, candidates closer than 0.3 deg merged.
`extra["omega_offset"]` is `omega - omega_base`. `verify=True` re-solves each candidate with
`FourCircleDiffractometer.solve_for_peak_family` (slow; uses the motor limits) into `solver_gap`
and `solver_solutions`. `[]` when the family's 2theta is outside the two_theta window. Replaces
`omega_finder.py`; the terminal version is `bexa crystal omega-search`.

```python
# diff, ranges: the diffractometer and the motor limits; the omegas where two 311 peaks share a mu window
candidates = find_omega_offsets(diff, (3, 1, 1), ranges, step=0.05, top_n=3)   # under 1 s
best = candidates[0]
best.omega, best.gap, best.first.hkl, best.second.hkl    # (-10.029, 6.674, (1, -3, -1), (1, -3, 1))
```

#### `format_candidates`
`bexa.crystal.omega_search.format_candidates(candidates, omega_base=0.0)`

The table `omega_finder.py` printed (offset from `omega_base`, omega, mu gap, both reflections
with mu and eta, eta difference, solver check) and a "Recommended:" line; "No candidate omega
found." for an empty list.

```python
# candidates: the omegas found; as a text table
print(format_candidates(candidates))
#  #   offset    omega      dmu       reflA      muA    etaA       reflB      muB    etaB  d_eta     solver
#  1  -10.029  -10.029   6.6745 (1, -3, -1)  -32.781  100.00  (1, -3, 1)  -26.106   58.59   41.4          -
#  ... two more rows
# Recommended: omega_offset = -10.029 (omega = -10.029), |delta mu| = 6.6745 deg, 2theta = 62.57
format_candidates(candidates, omega_base=10.0).splitlines()[0]  # ' #   offset    omega      dmu       reflA      muA    eta...'
```

#### `peaks_at_omega`
`bexa.crystal.omega_search.peaks_at_omega(diffractometer, family, omega, ranges=None)`

The equivalents of `family` on the Ewald sphere at one omega (chi = phi = 0) with mu and eta in
`ranges`, as `PeakHit`s; `[]` when the family's 2theta is outside the window. Handy to pick a
target for `FourCircleDiffractometer.solve_for_reflection`.

```python
# diff: the diffractometer; the 311 peaks reachable at the best omega
peaks_at_omega(diff, (3, 1, 1), best.omega, ranges)  # [PeakHit(hkl=(1, -3, -1), mu=-32.780872497209, eta=99.9999565356555), PeakHit(hkl=(1, -...
# [PeakHit(hkl=(1, -3, -1), mu=-32.78..., eta=99.99...), PeakHit(hkl=(1, -3, 1), mu=-26.10..., eta=58.58...)]
len(peaks_at_omega(diff, (3, 1, 1), best.omega))  # 4
```

#### `OmegaCandidate`
`bexa.crystal.omega_search.OmegaCandidate(omega, gap, first, second, two_theta, solver_gap=float("nan"), solver_solutions=-1, extra={})`

Dataclass: `omega` and mu `gap` in degrees, the two `PeakHit`s, the family's `two_theta`, the
`verify` results (NaN and -1 when not verified) and `extra`. Property `eta_difference`: the angle
between the two peaks' eta, 0 to 180 deg.

```python
# best: the best omega; its scattering angle, eta difference and the other peaks
best.two_theta, best.eta_difference, best.extra     # (62.573, 41.413, {'omega_offset': -10.029})
```

#### `PeakHit`
`bexa.crystal.omega_search.PeakHit(hkl, mu, eta)`

One equivalent on the Ewald sphere: `hkl` and the `mu` and `eta` (degrees) at which it is seen.

```python
# best.first: the first of the two peaks at the best omega
best.first                  # PeakHit(hkl=(1, -3, -1), mu=-32.78..., eta=99.99...)
```

Constants:

- `DEFAULT_RANGES`: `{"mu": (-45.0, -10.0), "eta": (0.0, 360.0), "two_theta": (0.0, 170.0)}`.

### bexa.crystal.sample_frame: sample-to-grain matrix

Port of `Sample2Grain.m`: the matrix between the sample (lab) frame and a grain's Cartesian frame.

```python
from bexa.crystal import LatticeParameters
from bexa.crystal.sample_frame import LEVI_CIVITA_SIGN, lattice_vectors, sample_to_grain
```

#### `sample_to_grain`
`bexa.crystal.sample_frame.sample_to_grain(lattice, surface_vectors)`

Matrix converting sample-frame vectors into the grain's Cartesian frame (that of
`lattice_vectors`). The columns of the 3 x 3 `surface_vectors` are the lab x, y and z as
direct-lattice indices `[u, v, w]`; one may be all zeros and is completed as the right-handed
cross product (fewer than two raise `ValueError`). As in the MATLAB, a cubic cell gives `a` times
a rotation; for non-orthogonal cells the columns do not follow the given directions.

```python
# surface: the sample surface and in-plane directions; the matrix from sample to grain frame
surface = [[0, 1, 1], [0, -1, 1], [0, 1, 0]]   # columns x, y, z: (110) surface, y along [1 -1 1]
M = sample_to_grain(LatticeParameters.silicon(), surface)
M / 5.43102      # columns x = [-1 1 2]/sqrt(6), y = [1 -1 1]/sqrt(3), z = [1 1 0]/sqrt(2)
```

#### `lattice_vectors`
`bexa.crystal.sample_frame.lattice_vectors(lattice, decimals=12)`

Direct-lattice vectors a, b, c as columns (Angstrom), a along x and b in the xy plane, rounded to
`decimals` places.

```python
# the real-space lattice vectors of a hexagonal cell
lattice_vectors(LatticeParameters.hexagonal(3.1532, 12.323))  # array (3, 3) float64, -1.577..12.32
# [[3.1532, -1.5766, 0.0], [0.0, 2.7308, 0.0], [0.0, 0.0, 12.323]]
lattice_vectors(LatticeParameters.cubic(3.56), decimals=3)  # array (3, 3) float64, 0..3.56
```

Constants:

- `LEVI_CIVITA_SIGN`: the sign table `[[0, 1, -1], [-1, 0, 1], [1, -1, 0]]` that orients the
  completed axis.

### bexa.core.units: physical constants and unit conversions

The conversions the rest of bexa uses: photon energy and wavelength, Bragg and monochromator
angles, photons per pulse, Gaussian widths. Energies are in keV, wavelengths and d-spacings in
Angstrom, angles in degrees, unless a parameter name says otherwise.

```python
import numpy as np
from bexa.core import units
```

#### `energy_to_wavelength`
`bexa.core.units.energy_to_wavelength(energy_keV)`

Wavelength in Angstrom of a photon of `energy_keV` keV, `lambda = hc / E`; a number or an
array in, a numpy float or array out. Also importable as
`bexa.optics.conversions.energy_to_wavelength`.

```python
# the wavelength of a 17 keV beam, and of three energies at once
units.energy_to_wavelength(17.0)                          # 0.72932
units.energy_to_wavelength(np.array([8.0, 17.0, 30.0]))   # array([1.5498, 0.72932, 0.41328])
```

#### `wavelength_to_energy`
`bexa.core.units.wavelength_to_energy(wavelength_A)`

Photon energy in keV for a wavelength in Angstrom, a number or an array. Also importable as
`bexa.optics.conversions.wavelength_to_energy`.

```python
# the energy of the copper K-alpha line
units.wavelength_to_energy(1.5406)       # 8.0478 keV (Cu K-alpha1)
```

#### `wavevector`
`bexa.core.units.wavevector(energy_keV)`

Wavevector magnitude `k = 2 pi / lambda` in 1/Angstrom for one energy in keV, as a float (an
array raises TypeError). Also importable as `bexa.optics.conversions.wavevector`.

```python
# the wavevector of a 17 keV beam
units.wavevector(17.0)                   # 8.6151 1/Angstrom
```

#### `bragg_angle_deg`
`bexa.core.units.bragg_angle_deg(d_spacing_A, energy_keV)`

Bragg angle theta in degrees of a spacing `d_spacing_A` (Angstrom) at `energy_keV`; scalars
only, ValueError if `lambda > 2d`. Replaces the Bragg's-law cell of `Usefulthings`; the
spacing of a reflection hkl comes from `bexa.crystal.Crystal.d_spacing` or
`bexa crystal dspacing`. Also importable as `bexa.optics.conversions.bragg_angle_deg`.

```python
# the Bragg angle of silicon 111 at 17 keV
units.bragg_angle_deg(3.1356, 17.0)      # 6.6784 deg: Si(111) at 17 keV
```

#### `d_spacing_from_angle`
`bexa.core.units.d_spacing_from_angle(theta_deg, energy_keV)`

Lattice spacing in Angstrom from a Bragg angle theta in degrees (half of 2theta) and an energy
in keV. For a measured 2theta use `bexa.optics.conversions.d_spacing_from_two_theta`.

```python
# the spacing behind a Bragg angle of 6.6784 degrees at 17 keV
units.d_spacing_from_angle(6.6784, 17.0)     # 3.1356 Angstrom
```

#### `mono_angle_to_energy`
`bexa.core.units.mono_angle_to_energy(theta_deg, crystal="Si111")`

Energy in keV passed by a monochromator crystal at Bragg angle `theta_deg` (degrees, a number
or an array). `crystal` is a name from `MONO_D_SPACINGS_ANGSTROM` or a d-spacing in Angstrom.
The ID03 reader converts the channel-cut angle `ccmth` with it. Also importable as
`bexa.optics.conversions.mono_angle_to_energy`.

```python
# the energy an Si111 and an Si311 monochromator pass at 6.7 degrees
units.mono_angle_to_energy(6.7), units.mono_angle_to_energy(6.7, "Si311")   # (16.945, 32.448) keV
units.mono_angle_to_energy(np.array([6.0, 6.7, 7.5]))  # array([18.914, 16.945, 15.147])
```

#### `energy_to_mono_angle`
`bexa.core.units.energy_to_mono_angle(energy_keV, crystal="Si111")`

Monochromator Bragg angle in degrees that selects `energy_keV` (`crystal` as above); scalars
only. ValueError if unreachable, KeyError for a name missing from `MONO_D_SPACINGS_ANGSTROM`.
Also importable as `bexa.optics.conversions.energy_to_mono_angle`.

```python
# the monochromator angle for 17 keV, Si111 and Si311
units.energy_to_mono_angle(17.0), units.energy_to_mono_angle(17.0, "Si311")   # (6.6784, 12.867) deg
```

#### `photons_from_pulse_energy`
`bexa.core.units.photons_from_pulse_energy(pulse_energy_J, photon_energy_eV)`

Number of photons in a pulse of `pulse_energy_J` joules at `photon_energy_eV` (eV, not keV).
Replaces the pulse-energy-to-photons cell of `Usefulthings`;
`bexa optics photons --pulse-energy 5e-6 --energy 16` does the same with keV. Also importable
as `bexa.optics.conversions.photons_from_pulse_energy`.

```python
# the photons in a 5 uJ pulse at 16 keV
units.photons_from_pulse_energy(5e-6, 16000)     # 1.95e9 photons: 5 uJ at 16 keV
```

#### `sigma_to_fwhm`
`bexa.core.units.sigma_to_fwhm(sigma)`

Full width at half maximum of a Gaussian of standard deviation `sigma` (2.3548 sigma, same
unit); a number or an array.

```python
# the FWHM of a Gaussian of sigma 0.01
units.sigma_to_fwhm(0.01)                # 0.023548
```

#### `fwhm_to_sigma`
`bexa.core.units.fwhm_to_sigma(fwhm)`

Standard deviation of a Gaussian with full width at half maximum `fwhm`, same unit; a number
or an array.

```python
# the sigma of a Gaussian of FWHM 0.05
units.fwhm_to_sigma(0.05)                # 0.021233
```

Constants:

- `HC_KEV_ANGSTROM`: 12.398419843, h c in keV Angstrom (also
  `bexa.optics.conversions.HC_KEV_ANGSTROM`).
- `D_SI111_ANGSTROM`: 3.1356, d-spacing of Si(111) in Angstrom (channel-cut monochromators).
- `EV_PER_JOULE`: 6.241509074e18 eV per joule.
- `FWHM_PER_SIGMA`: 2.35482 (`2 sqrt(2 ln 2)`), used for the widths in `bexa.analysis.rocking`
  and `bexa.analysis.fitting`.
- `MONO_D_SPACINGS_ANGSTROM`: dict of monochromator reflection to d-spacing in Angstrom: `Si111`
  3.1356, `Si220` 1.9201, `Si311` 1.6375, `Si333` 1.0452, `Ge111` 3.2664, `C111` 2.0593 (diamond).

### bexa.optics.conversions: energy, wavelength and monochromator angles

A one-call summary of a beam energy and the 2theta to d-spacing conversions of the
`Usefulthings` notebook. It also re-exports, unchanged, `HC_KEV_ANGSTROM`,
`energy_to_wavelength`, `wavelength_to_energy`, `wavevector`, `bragg_angle_deg`,
`mono_angle_to_energy`, `energy_to_mono_angle` and `photons_from_pulse_energy` from
`bexa.core.units` (entries above). The `bexa.optics` package defines nothing itself, and
`import bexa` does not load its modules: import them by name.

```python
import numpy as np
from bexa.optics import conversions
```

#### `summary`
`bexa.optics.conversions.summary(energy_keV=None, wavelength_A=None)`

A dict of floats for one beam energy: `energy_keV`, `wavelength_A`, `wavevector_1_A`
(1/Angstrom) and `theta_<crystal>_deg`, the Bragg angle of every `MONO_D_SPACINGS` reflection
that can reach it. Give the energy or the wavelength (the energy wins if both are given,
neither raises ValueError). `bexa optics convert --energy 17` (or `--wavelength 0.73`) prints it.

```python
# everything about a 17 keV beam: wavelength, monochromator angles
s = conversions.summary(energy_keV=17.0)
s["wavelength_A"], s["theta_Si111_deg"], s["theta_Si311_deg"]   # (0.72932, 6.6784, 12.867)
conversions.summary(wavelength_A=1.0)["energy_keV"]             # 12.398
```

#### `d_spacing_from_two_theta`
`bexa.optics.conversions.d_spacing_from_two_theta(two_theta_deg, energy_keV)`

Lattice spacing in Angstrom of a reflection seen at the scattering angle `two_theta_deg`
(degrees) at `energy_keV`; numbers or arrays. Replaces the d-spacing from two theta cell of
`Usefulthings`.

```python
# the spacing behind a 2theta of 26.03 degrees at 19.2 keV, and of two angles
conversions.d_spacing_from_two_theta(26.03, 19.2)                   # 1.4337 Angstrom
conversions.d_spacing_from_two_theta(np.array([20.0, 30.0]), 19.2)  # array([1.8594, 1.2475])
```

#### `two_theta_from_d_spacing`
`bexa.optics.conversions.two_theta_from_d_spacing(d_spacing_A, energy_keV)`

Scattering angle 2theta in degrees of a spacing in Angstrom at `energy_keV`, twice
`bragg_angle_deg`. Scalars only; raises ValueError if `lambda > 2d`.

```python
# the 2theta of silicon 111 at 17 keV
conversions.two_theta_from_d_spacing(3.1356, 17.0)     # 13.357 deg
```

Constants:

- `MONO_D_SPACINGS`: dict of reflection to d-spacing in Angstrom used by `summary`:
  `bexa.core.units.MONO_D_SPACINGS_ANGSTROM` plus `Ge220` 2.0002, which `energy_to_mono_angle`
  does not know.

### bexa.optics.attenuation: transmission and attenuation lengths

Transmission of filters, windows, scintillators and samples from the xraydb tables (Elam total
attenuation: photoabsorption plus coherent and incoherent scattering); needs xraydb. A material
is an element symbol (`"Si"`), a `KNOWN_MATERIALS` name, a name from xraydb's list (`"silicon"`,
`"beryllium"`, `"polyimide"`) or any formula given with `density=` (g/cm3). Energies are in
keV, numbers or arrays; thicknesses are in um or strings with a unit (`"500um"`, `"0.3mm"`).

```python
import numpy as np
from bexa.optics import attenuation
```

#### `transmission`
`bexa.optics.attenuation.transmission(material, thickness, energy_keV, density=None)`

Fraction transmitted through `thickness` of `material`, `exp(-mu t)`: a numpy float, or an
array for an array of energies; `density` (g/cm3) replaces the tabulated value. Replaces the
attenuation cell of `Usefulthings`. The command
`bexa optics transmission --material Si --thickness 500um --energy 10:30:5` prints it with the
1/e length for a range of energies.

```python
# the transmission of 500 um of silicon at 17 keV, and of a kapton window at two energies
attenuation.transmission("Si", "500um", 17.0)                    # 0.434
attenuation.transmission("kapton", 25, np.array([8.0, 17.0]))    # array([0.978, 0.997])
attenuation.transmission("Fe2O3", "10um", 10.0, density=5.24)    # 0.530
```

#### `attenuation_length_um`
`bexa.optics.attenuation.attenuation_length_um(material, energy_keV, density=None)`

Attenuation length in um, the thickness that transmits 1/e (`1 / mu`).

```python
# the attenuation length of WTe2 at two energies
attenuation.attenuation_length_um("WTe2", [8.0, 17.0])   # array([4.642, 16.82]) um
attenuation.attenuation_length_um("Fe2O3", 17.0, density=5.24)  # 66.591
```

#### `linear_attenuation_1_cm`
`bexa.optics.attenuation.linear_attenuation_1_cm(material, energy_keV, density=None)`

Linear attenuation coefficient mu in 1/cm, the mass attenuation coefficient times the density.

```python
# the linear attenuation of silicon at 17 keV
attenuation.linear_attenuation_1_cm("Si", 17.0)          # 16.68 1/cm
attenuation.linear_attenuation_1_cm("Si", np.array([8.0, 17.0]))  # array([150.66, 16.682])
```

#### `mass_attenuation_cm2_g`
`bexa.optics.attenuation.mass_attenuation_cm2_g(material, energy_keV, density=None)`

Mass attenuation coefficient mu/rho in cm2/g. It does not depend on the density, but a formula
without a tabulated density still needs `density=` to be accepted.

```python
# the mass attenuation of silicon at three energies
attenuation.mass_attenuation_cm2_g("Si", [8.0, 17.0, 30.0])   # array([64.69, 7.1625, 1.4365])
```

#### `resolve_material`
`bexa.optics.attenuation.resolve_material(material, density=None)`

`(formula, density_g_cm3)` for a material string, looked up in `KNOWN_MATERIALS`
(case-insensitive; `density` replaces the table value), then taken as a formula if `density` is
given, then in xraydb's materials and element densities. Anything else raises ValueError.

```python
# a scintillator by name, and a compound with its density given
attenuation.resolve_material("LuAG")                  # ('Lu3Al5O12', 6.73)
attenuation.resolve_material("Fe2O3", density=5.24)   # ('Fe2O3', 5.24)
```

#### `material_density`
`bexa.optics.attenuation.material_density(material)`

Tabulated density in g/cm3, the second item of `resolve_material(material)`.

```python
# the density of silicon by name
attenuation.material_density("silicon")      # 2.329
```

#### `parse_length_um`
`bexa.optics.attenuation.parse_length_um(value)`

A thickness in um from a number (taken as um) or a string with an optional unit: `nm`, `um`
(or the micro sign), `mm`, `cm`, `m`; spaces are allowed. Raises ValueError otherwise.

```python
# thicknesses written with units, in micrometres
attenuation.parse_length_um("0.3mm"), attenuation.parse_length_um("25 um")   # (300.0, 25.0)
```

Constants:

- `KNOWN_MATERIALS`: dict of lower-case name to (formula, density g/cm3), looked up before
  xraydb: `diamond` and `c-diamond` (C, 3.515), `glassy carbon` and `c-glassy` (C, 1.5), `luag`
  (Lu3Al5O12, 6.73), `yag` (Y3Al5O12, 4.56), `gagg` (Gd3Al2Ga3O12, 6.63), `lyso`
  (Lu1.8Y0.2SiO5, 7.1), `kapton` (C22H10N2O5, 1.42), `mylar` (C10H8O4, 1.4), `water` (H2O, 1.0),
  `air` (N1.562O0.42C0.0003Ar0.0094, 0.0012), `sapphire` (Al2O3, 3.98), `quartz` (SiO2, 2.65),
  `wte2` (WTe2, 9.43), `ws2` (WS2, 7.5), `mos2` (MoS2, 5.06).

### bexa.optics.nist: NIST mass attenuation tables

An alternative to xraydb: the NIST X-ray mass attenuation tables of the elements H to U,
downloaded once, then read from two text files (port of `Database_Extraction.py`).
`bexa.optics.attenuation` does not use them; combine `nist_mass_attenuation` with a density
yourself. The files, which can also be written by hand:

- `nist_xcom.dat`: per element a line `#S  Z 14  Si`, then lines `energy_eV  mu_rho_cm2_per_g`;
  other lines starting with `#` are comments.
- `element_densities.dat`: lines `symbol  density_g_per_cm3`; `#` starts a comment.

```python
import numpy as np
from bexa.optics import nist
```

#### `fetch_nist_tables`
`bexa.optics.nist.fetch_nist_tables(out_dir, delay_s=0.2)`

Downloads the element densities and the tables of the 92 elements from physics.nist.gov (one
request per element, `delay_s` seconds apart) and writes the two files into `out_dir` (created
if needed, files overwritten); returns `(xcom_path, density_path)`. Energies are converted from
MeV to eV; elements that fail are logged and skipped. Needs the network, `requests` and
`beautifulsoup4`. Replaces `Database_Extraction.py`.

```python
# the NIST XCOM tables downloaded once (uses the network: shown, not run)
xcom_path, density_path = nist.fetch_nist_tables("nist_tables")    # once, downloads
```

#### `load_nist_tables`
`bexa.optics.nist.load_nist_tables(folder)`

Reads the two files in `folder` into `(tables, densities)`: `tables[symbol]` is an `(n, 2)`
array of energy (eV) and mu/rho (cm2/g), `densities[symbol]` a density in g/cm3.

```python
# nist_tables: the folder of the tables; the table of silicon and its density
tables, densities = nist.load_nist_tables("nist_tables")
tables["Si"].shape, densities["Si"]          # ((n, 2), 2.33)
```

#### `nist_mass_attenuation`
`bexa.optics.nist.nist_mass_attenuation(table, energy_keV)`

Mass attenuation coefficient in cm2/g at `energy_keV` (a number or an array), interpolated
linearly in log(energy) and log(mu/rho) from one element's table; outside the table the end
values are returned.

```python
# tables: the NIST tables; the mass attenuation of silicon at 17 keV, and its transmission
mu_rho = nist.nist_mass_attenuation(tables["Si"], 17.0)
mu_rho, np.exp(-mu_rho * densities["Si"] * 500e-4)       # (7.17 cm2/g, 0.434 through 500 um of Si)
```

Constants:

- `XCOM_FILE` (`"nist_xcom.dat"`) and `DENSITY_FILE` (`"element_densities.dat"`): the file names.
- `URL_ELEMENT` (`"https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z{Z:02d}.html"`)
  and `URL_DENSITIES` (`"https://physics.nist.gov/PhysRefData/XrayMassCoef/tab1.html"`).
- `SYMBOLS`: the 92 element symbols H to U; `SYMBOLS[Z - 1]` is element Z.

### bexa.optics.bragg_magnifier: asymmetric Bragg magnifier

Port of `Bragg_Mag_Efficiency_Calcs.py`: dynamical-theory (Bragg case) rocking curves of
silicon, the magnification of a crystal cut at a miscut angle to its planes, and the throughput
of a Bragg magnifier, a symmetric S0 monochromator followed by two orthogonal pairs of
asymmetric crystals S1 and S2. Each pair magnifies one direction by `total_mag = M1 * M2`,
shared by the split ratio `s` as `M1 = total_mag**s`. Units differ from the rest of bexa:
energies in eV, rocking-angle offsets in arcsec, divergences in urad, miscuts in degrees except
in `mag_from_miscut`, `miscut_from_mag` and `reflectivity` (radians). Needs xraydb.

```python
import numpy as np
from bexa.optics import bragg_magnifier as bm

dth = np.linspace(-60, 60, 1201)     # offsets in arcsec, coarser than the default grid
```

#### `efficiency`
`bexa.optics.bragg_magnifier.efficiency(total_mags, splits, E0=9251.0, dE=0.5, n_E=21, spectrum="gaussian", sigma_theta_urad=1.5, hkl=(2, 2, 0), pol="sigma", dth=None, mono=True, align=True)`

Throughput in percent for every total magnification (of one pair) and split ratio: returns
`(grid, info)`, `grid` of shape `(len(total_mags), len(splits))`. At each energy of
`energy_grid(E0, dE, n_E, spectrum)` (`dE` a FWHM in eV) and each offset of `dth` (default
`default_angle_grid()`), the reflectivities of S0, S1 (at incidence and exit) and S2 are
multiplied, squared for the two pairs and weighted by the Gaussian divergence
`sigma_theta_urad`, then divided by what S0 alone passes. Port of `compute_efficiency_v2` in
`Bragg_Mag_Efficiency_Calcs.py`.

- `align=True` sets S0 at its refraction-shifted peak and each S1/S2 pair at its best overlap;
  with `align=False` the crystals stay at their nominal angles and the throughput is near zero.
- `mono=False` leaves S0 out and normalises by the divergence profile alone.
- `info`: `E_arr`, `E_weights`, `theta_B_arr`, `S0_R_exit_2d`, `input_area_arr`, `G_div`,
  `sigma_arcsec`, `dphi_mono` (arcsec), `mono`, `align`.
- Cost: about 20 ms per (magnification, split) point with the default 8001 offsets and 21
  energies, so a 20 x 50 map takes about 20 s; the example below takes well under a second.

```python
# dth: the angle grid; the efficiency of a magnifier over magnifications and bandwidths
grid, info = bm.efficiency([5.0, 20.0], [0.2, 0.5, 0.8], n_E=5, dth=dth)
grid.round(2)            # [[55.15, 74.51, 3.79], [0.15, 0.05, 0.08]] percent
info["dphi_mono"]        # 1.5 arcsec: the refraction shift of the S0 peak
```

#### `optimal_split`
`bexa.optics.bragg_magnifier.optimal_split(total_mag, n_splits=50, **kwargs)`

Best split ratio for one total magnification: runs `efficiency` for `n_splits` ratios from 0 to
1 (keywords go to `efficiency`) and returns `(split, efficiency_percent, dphi_arcsec)`, `dphi`
being the offset that aligns the pair at `E0` (0 with `align=False`). About 1 s with the
defaults. `bexa optics magnifier --mag 20` runs it with 25 splits, 11 energies and 2001 offsets
and also prints the two magnifications and miscuts.

```python
# the best split of a magnification of 5 over several crystals
bm.optimal_split(5.0, n_splits=5, n_E=3, dth=dth)     # (0.5, 74.51, 2.93)
```

#### `rocking_curve`
`bexa.optics.bragg_magnifier.rocking_curve(alpha_deg, dtheta_arcsec, E_eV=9251.0, hkl=(2, 2, 0), pol="sigma")`

Rocking curve of a Si `hkl` crystal whose surface is cut at `alpha_deg` to the planes (0 is
symmetric; a positive miscut gives grazing incidence and magnifies by `1 / |b|`), against
offsets `dtheta_arcsec` (arcsec) from the nominal glancing angle `theta_B - alpha`. `pol` is
`"sigma"` or `"pi"`; `hkl` must be a tuple. Returns a dict: `R_inc` and `R_exit` (the curve
against incidence and exit offsets), `theta_inc_deg`, `theta_exit_deg` (glancing angles to the
surface), `dtheta`, `dtheta_exit`, `b` (asymmetry factor), `FWHM_inc`, `FWHM_exit` (arcsec),
`R_peak`.

```python
# dth: the angle grid; the rocking curve of a symmetric crystal
curve = bm.rocking_curve(0.0, dth)                   # symmetric Si 220 at 9251 eV
curve["R_peak"], curve["FWHM_inc"]                   # (0.992, 4.6)
curve = bm.rocking_curve(18.0, dth)
curve["FWHM_inc"], curve["FWHM_exit"], 1 / abs(curve["b"])   # (17.8, 1.22, 14.69)
```

#### `rocking_curve_absolute`
`bexa.optics.bragg_magnifier.rocking_curve_absolute(alpha_deg, angles_deg, E_eV=9251.0, hkl=(2, 2, 0), pol="sigma")`

`rocking_curve` against absolute glancing incidence angles `angles_deg` (degrees, to the
surface) instead of offsets; the same dict.

```python
# angles: absolute Bragg angles; the rocking curve on them
angles = np.linspace(20.40, 20.46, 601)
curve = bm.rocking_curve_absolute(0.0, angles)
angles[np.argmax(curve["R_inc"])]          # 20.4259 deg: theta_B 20.4255 plus refraction
```

#### `mag_from_miscut`
`bexa.optics.bragg_magnifier.mag_from_miscut(miscut_rad, E_eV=9251.0, hkl=(2, 2, 0))`

Magnification `sin(theta_B + alpha) / sin(theta_B - alpha)` for a miscut in radians; numbers or
arrays.

```python
# the magnification of an 18 degree miscut
bm.mag_from_miscut(np.radians(18.0))       # 14.69
bm.mag_from_miscut(np.radians(18.0), E_eV=8000.0)  # 6.5946
```

#### `miscut_from_mag`
`bexa.optics.bragg_magnifier.miscut_from_mag(mag, E_eV=9251.0, hkl=(2, 2, 0))`

Miscut in radians that gives the magnification `mag`, the inverse of `mag_from_miscut`; numbers
or arrays.

```python
# the miscuts for three magnifications
np.degrees(bm.miscut_from_mag([5, 10, 20]))     # array([13.94, 16.95, 18.62]) deg
np.degrees(bm.miscut_from_mag(10, hkl=(1, 1, 1)))  # 10.148
```

#### `mag_split`
`bexa.optics.bragg_magnifier.mag_split(total_mag, split_ratio)`

Magnifications `(M1, M2) = (total_mag**s, total_mag**(1 - s))` of S1 and S2 for a split ratio
`s` from 0 to 1.

```python
# a magnification of 20 split over two crystals of bandwidth 0.3
bm.mag_split(20.0, 0.3)                    # (2.456, 8.142)
```

#### `miscuts_from_split`
`bexa.optics.bragg_magnifier.miscuts_from_split(total_mag, split_ratio, E_eV=9251.0, hkl=(2, 2, 0))`

Miscuts in degrees of S1 and S2 for a total magnification and split ratio.

```python
# the miscuts of that split
bm.miscuts_from_split(20.0, 0.3)           # (8.918, 16.221) deg
bm.miscuts_from_split(20.0, 0.5)                # (13.295, 13.295)
```

#### `track_phase_space`
`bexa.optics.bragg_magnifier.track_phase_space(total_mag, split_ratio, sigma_x_um=35.0, sigma_theta_urad=1.5, E_eV=9251.0, hkl=(2, 2, 0))`

Beam size and divergence through the magnifier: a list of
`(stage, sigma_x_um, sigma_theta_urad, product)` for the source and after S0, S1 and S2, then
for the second pair (the other direction, from the source again). Each crystal multiplies the
size by its magnification and divides the divergence by it, so the product stays constant.

```python
# the beam size and divergence through a magnifier of 20 at 0.5 bandwidth
for stage, size_um, div_urad, product in bm.track_phase_space(20.0, 0.5):
    print(f"{stage:16s} {size_um:7.1f} um {div_urad:6.3f} urad")   # after S2: 700.0 um, 0.075 urad
```

#### `crystal_params`
`bexa.optics.bragg_magnifier.crystal_params(E_eV, hkl=(2, 2, 0))`

The Si `hkl` reflection at `E_eV`: a dict of `lam` and `d` (Angstrom), `theta_B` (radians), the
complex susceptibilities `chi_0` and `chi_h`, and `Gamma`, from xraydb's f0 and Chantler f',
f'' with a Debye-Waller factor. Cached (64 entries), so `hkl` must be a tuple.

```python
# the silicon 220 parameters at 9251 eV
p = bm.crystal_params(9251.0, (2, 2, 0))
np.degrees(p["theta_B"]), p["d"]           # (20.4255, 1.9202)
```

#### `reflectivity`
`bexa.optics.bragg_magnifier.reflectivity(alpha, dth, theta_B, chi_0, chi_h, lam, d, C)`

The kernel behind `rocking_curve` (legacy `calc_R_inc_exit`): reflectivity of a crystal with
miscut `alpha` (radians) at offsets `dth` (arcsec), from the values of `crystal_params` and the
polarisation factor `C` (1 for sigma, `cos(2 theta_B)` for pi). Returns `(R, R_exit)`, the
curve against incidence offsets and resampled onto exit offsets.

```python
# p: the silicon parameters; the reflectivity of an 18 degree miscut over dth
p = bm.crystal_params(9251.0)
R, R_exit = bm.reflectivity(np.radians(18.0), dth, p["theta_B"], p["chi_0"], p["chi_h"],
                            p["lam"], p["d"], 1.0)
R.max()                                    # 0.9016, as rocking_curve(18.0, dth)["R_peak"]
```

#### `si_structure_factor`
`bexa.optics.bragg_magnifier.si_structure_factor(h, k, l)`

Complex geometric structure factor of the diamond lattice (fcc times the two-atom basis): 8 for
220, `4 - 4j` for 111, zero for forbidden reflections such as 200 and 222.

```python
# the structure factors of silicon 220 and 111
abs(bm.si_structure_factor(2, 2, 0)), abs(bm.si_structure_factor(1, 1, 1))   # (8.0, 5.657)
```

#### `energy_grid`
`bexa.optics.bragg_magnifier.energy_grid(E0, dE_fwhm, n_E=21, spectrum="gaussian")`

Energies (eV) and normalised weights of the spectrum (legacy `build_energy_grid`): `"gaussian"`
spans 3 sigma either side of `E0` for a FWHM `dE_fwhm`, `"flat"` the band `E0 +- dE_fwhm / 2`,
and a callable `w(E)` is sampled on the 3-sigma grid. The weights include the trapezoid steps and
sum to 1. Use an odd `n_E` of at least 3 so that the middle point, where `efficiency` cuts and
aligns the crystals, is `E0`.

```python
# an energy grid of five points around 9251 eV with flat weights
E, w = bm.energy_grid(9251.0, 0.5, n_E=5, spectrum="flat")
E, w      # [9250.75, 9250.875, 9251.0, 9251.125, 9251.25], [0.125, 0.25, 0.25, 0.25, 0.125]
```

#### `gaussian_profile`
`bexa.optics.bragg_magnifier.gaussian_profile(dth, sigma_arcsec, center=0.0)`

Gaussian of standard deviation `sigma_arcsec` centred on `center` (arcsec), with unit area on
the grid `dth`: the divergence weight in `efficiency`.

```python
# dth: the angle grid; a Gaussian of 1.5 urad on it, normalised
G = bm.gaussian_profile(dth, 1.5 * bm.URAD_TO_ARCSEC)     # 1.5 urad = 0.309 arcsec
G.sum() * (dth[1] - dth[0])                               # 1.0
```

#### `default_angle_grid`
`bexa.optics.bragg_magnifier.default_angle_grid()`

The offsets used when `dth=None`: -100 to 100 arcsec in 8001 steps of 0.025 arcsec.

```python
# the angle grid the magnifier functions use by default
bm.default_angle_grid().shape              # (8001,)
```

Constants:

- `DEFAULT_E0` 9251.0 eV, `DEFAULT_DE` 0.5 eV FWHM, `DEFAULT_SIGMA_THETA_URAD` 1.5 urad,
  `DEFAULT_SIGMA_X_UM` 35.0 um, `DEFAULT_HKL` (2, 2, 0): the defaults above.
- `URAD_TO_ARCSEC` 0.206265 and `ARCSEC_TO_RAD` 4.84814e-6.
- `R_E` 2.8179403e-5 Angstrom (classical electron radius), `A_SI` 5.43102 Angstrom (Si lattice
  constant), `V_SI` 160.193 Angstrom^3 (`A_SI**3`), `B_DW` 0.4632 Angstrom^2 (Debye-Waller B of
  Si).

### bexa.core.structure: dims, coordinates and frame-to-grid mapping of a scan

`Structure` describes the grid of a scan: the motor dims (slow first), their coordinates, and
which detector frame sits at each grid point. Every `Scan` carries one as `scan.structure`, built
by the engine from the motor readbacks; it matters when you write your own reduction over
`scan.batches()` or need the frames of a grid region.

```python
import numpy as np
import bexa
from bexa.core.structure import Structure

scan = bexa.demo("mosa")                # synthetic chi x mu scan, 6 x 16 frames of 64 x 80 pixels
st = scan.structure                             # the structure of the demo chi x mu scan
```

#### `Structure`
`bexa.core.structure.Structure(motor_dims, motor_shape, coords, frame_index, frame_shape, n_frames, per_frame={}, scalars={}, scan_type="grid", units={}, energy_keV=None, title="", planned_shape=None, planned_frames=None, motor_names=frozenset(), _grid_of_frame=None)`

A dataclass. `motor_dims` names the scanned dims, slow first (`("chi", "mu")`, or `("frame",)`
for a plain sequence); `motor_shape` gives their lengths and `coords` their 1-D coordinate
arrays. `frame_index` is an int64 array of shape `motor_shape` with the flat frame id at each
grid point, `MISSING` (-1) where a partial scan has no frame yet. `frame_shape` is
`(height, width)`, `n_frames` the frames the source can deliver, `per_frame` the per-frame
channels (motor readbacks, `elapsed_time`), `scalars` the fixed positioners (`samz`, `ccmth`),
`units` the unit per motor (`"deg"`, `"mm"`), `energy_keV` the photon energy in keV when known,
`title` the recorded scan command, `planned_shape` and `planned_frames` what the scan was
meant to be (points per declared motor, frames in all) when the files say, `motor_names` every
name the file lists as a motor (the per-frame channels among them are the dims a reader may
choose, `Scan.with_dims`), and `scan_type` the declared kind (`"fscan1d"`, `"fscan2d"`,
`"fscan3d"`), else the command word (`"amesh"`, `"dmesh"`, `"ascan"`, ...), or `"list"`,
`"points"`, `"cube"`, `"multi"` (stacked scans) or `"grid"` (default). `_grid_of_frame` is an
internal cache. Members:

- `dims`, `shape`, `ndim`: properties, `motor_dims + ("y", "x")`, `motor_shape + frame_shape`
  and `len(dims)`.
- `n_grid`, `n_missing`, `is_complete`: properties, the number of grid points, how many have no
  frame or, when the plan is known, how many planned frames were not taken (whichever is more:
  a scan still being written, or stopped early), and whether that is zero.
- `frame_ids(region=None)`: sorted frame ids inside a grid region, one slice or index array per
  motor dim in grid indices (not motor values); missing points are skipped, `None` gives all.
- `used_frame_ids()`: `frame_ids()` of the whole grid.
- `grid_positions(frame_ids)`: tuple of index arrays, one per motor dim (-1 for frames off the
  grid); `grid_of_frame()` is the `(n_frames, len(motor_dims))` table behind it.
- `motor_values(name, frame_ids)`: position of motor `name` per frame: the per-frame readback
  when there is one, else the grid coordinate.
- `sub(region)`: a copy restricted to a grid region, keeping the frame ids of the full scan (how
  a motor ROI becomes a smaller grid; a part of the grid is not the plan any more, so
  `planned_shape` and `planned_frames` are dropped). `with_frame_shape(frame_shape)`: a copy
  with another frame shape.
- `describe()`: the summary printed by `bexa info` and `scan.info()` (type, dims, shape, frames
  of the planned ones, missing points, title, motor ranges with units, energy, fixed
  positioners).
- `frames_only(n_frames, frame_shape, **kwargs)`: class method, a plain sequence with a `frame`
  dim and `scan_type="list"` (`.npy` and tiff stacks).
- `stack(parts, dim, coords)`: class method, structures with identical grids and frame shapes
  stacked on a new leading dim (energy series); frame ids are offset, per-frame channels
  concatenated, `scan_type` is `"multi"`, the planned frames add up and the motor names are
  pooled. Other grids raise `ValueError`.

```python
# st: the structure of the demo chi x mu scan; its grid, a frame's place, the frames missing
st.dims, st.shape                               # ('chi', 'mu', 'y', 'x'), (6, 16, 64, 80)
st.frame_index[2, 5], st.n_missing              # 37 (chi index 2, mu index 5), 0
st.planned_frames, sorted(st.motor_names)[:4]   # 96, ['ccmth', 'chi', 'ffx', 'ffz']
ids = st.frame_ids((slice(2, 3), slice(None)))  # chi index 2, every mu: frames 32 to 47
st.grid_positions(np.array([17])), st.motor_values("mu", ids[:2])   # ([1], [1]), [-1.0, -0.867] deg
part = st.sub((slice(0, 2), slice(4, 12)))      # chi indices 0-1, mu indices 4-11: motor_shape (2, 8)
Structure.stack([st, st], "energy", [17.00, 17.02]).dims   # ('energy', 'chi', 'mu', 'y', 'x')
```

#### `Structure.from_grid`
`bexa.core.structure.Structure.from_grid(motor_dims, coords, frame_shape, n_frames=None, **kwargs)`

Class method: a regular grid acquired slow-major (frames numbered row by row). With `n_frames`
smaller than the grid, the last points are `MISSING`, as for an interrupted or running scan.
`kwargs` set the other fields (`scan_type`, `units`, `energy_keV`, `title`).

```python
# a structure from a regular grid of two motors and a frame count
s = Structure.from_grid(("chi", "mu"), {"chi": np.arange(3.0), "mu": np.linspace(-1, 1, 5)},
                        frame_shape=(2160, 2560), n_frames=12)
s.frame_index.tolist()                  # [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, -1, -1, -1]]
```

#### `Structure.from_per_frame`
`bexa.core.structure.Structure.from_per_frame(motors, frame_shape, tolerance=None, order=None, **kwargs)`

Class method: the grid detected from per-frame motor readbacks (a dict of equal-length arrays).
Frames are placed by value, not by order, so slow-major, snake and cyclic acquisitions all work.
Readbacks closer than `tolerance` (a number or a dict per motor; default 1e-3 of the motor's
range) form one grid point with their mean as coordinate. The slow dim is the motor that changes
least often between frames unless `order` lists the dims; a position recorded twice keeps its
first frame. The readbacks go to `per_frame`, `scan_type` to `fscan<N>d`.

```python
# a structure found from per-frame readbacks of two motors (a snake scan)
chi = np.repeat([0.0, 0.1], 4)                       # slow motor
mu = np.r_[np.arange(4.0), np.arange(4.0)[::-1]]     # fast motor in snake order
s = Structure.from_per_frame({"mu": mu, "chi": chi}, frame_shape=(64, 80))
s.motor_dims, s.frame_index.tolist()                 # ('chi', 'mu'), [[0, 1, 2, 3], [7, 6, 5, 4]]
```

#### `same_partition`
`bexa.core.structure.same_partition(a, b)`

True when two label arrays split the items into the same groups: two motors whose grid labels
(from `_group_values`) coincide step in lockstep (a motor and its encoder, the two motors of an
`a2scan`, two positioners that move together between scans), so one of them is enough as a
dim. `varying` and the BLISS engine use it.

```python
# whether two readbacks step together (one dim) or not
from bexa.core.structure import _group_values, same_partition
same_partition(_group_values(mu, None)[1], _group_values(mu * 2 + 1, None)[1])   # True: the same steps
same_partition(_group_values(mu, None)[1], _group_values(chi, None)[1])          # False
```

Constants:

- `PIXEL_DIMS`: `("y", "x")`, the two detector dims. `BINNED_PIXEL_DIMS`: `("yb", "xb")`, the
  pixel dims of a `bexa.acc.Preview(downsample=...)` binned on its own, so that one result can
  hold images of two sizes. `MISSING`: `-1`, the id of an absent frame.

### bexa.core.backend: numpy or cupy, devices, memory budget, batch sizes

Accumulators and `bexa.analysis` functions run on the array module of their input. This module
picks numpy or cupy, moves arrays between host and GPU, and sizes batches from the memory budget.
cupy is optional; the GPU examples need it and a CUDA device.

```python
import numpy as np
import bexa
from bexa.core import backend

scan = bexa.demo("mosa")                # synthetic chi x mu scan
```

#### `to_host`
`bexa.core.backend.to_host(obj)`

Also `bexa.to_host`. Copies cupy arrays back to numpy, recursing through dicts, lists, tuples and
xarray objects (attrs kept); anything else comes back unchanged. Results of `scan.reduce` and the
scan verbs are already on the host: use it on what `bexa.analysis` returns for GPU input.

```python
# vol: the demo mu scan on the GPU (if any); smoothed there and brought back
from bexa.analysis import rocking
vol = backend.to_device(scan.read().values, "cuda")        # (chi, mu, y, x) on the GPU
com = rocking.motor_com(vol, scan.coords["mu"], axis=1)    # cupy array (chi, y, x)
com = bexa.to_host(com)                                    # numpy array
```

#### `resolve_device`
`bexa.core.backend.resolve_device(device="auto")`

`"cpu"` or `"cuda"` for a request: `"auto"` gives `"cuda"` when `cupy_available()`; `"cuda"` or
`"gpu"` raise `RuntimeError` without cupy or a device; other names raise `ValueError`. The
`BEXA_DEVICE` variable is read only for `device=None` (or empty): an explicit `"auto"`, the
default of `scan.reduce` and the scan verbs, ignores it.

```python
# the device auto picks here, and the default
backend.resolve_device("auto")          # 'cuda' where cupy finds a GPU, else 'cpu'
backend.resolve_device(None)            # what BEXA_DEVICE says, 'auto' when it is not set
```

#### `array_module`
`bexa.core.backend.array_module(target="cpu")`

numpy or cupy for a device name (resolved by `resolve_device`) or for an existing array; the
usual start of device-agnostic code is `xp = array_module(frames)`.

```python
# numpy or cupy, for the device auto picks
xp = backend.array_module("auto")       # cupy where a GPU answers, else numpy
backend.array_module("cpu").__name__            # 'numpy'
```

#### `to_device`
`bexa.core.backend.to_device(a, device="auto")`

The array on `device`: cupy for `"cuda"` (and for `"auto"` on a GPU machine), numpy for
`"cpu"`; no copy when it is already there.

```python
# an array moved to the GPU (a copy stays numpy without cupy) and back
gpu = backend.to_device(np.ones((10, 64, 80), np.float32), "cuda")   # cupy array
host = backend.to_device(gpu, "cpu")                                  # numpy array
```

#### `memory_budget`
`bexa.core.backend.memory_budget(device="cpu", fraction=None)`

Bytes one operation may use: `fraction` (default `BEXA_MEMORY_FRACTION`, else 0.8) of the memory
this process can still use (`bexa.core.resources.memory_info`: the SLURM job's limit on a
cluster, the free memory elsewhere), and on CUDA at most `fraction` of the free GPU memory; the
rest is headroom for the pipeline's temporaries, the session's other variables and the system.
`BEXA_MEMORY_BYTES`, when set, is the budget itself: `bexa.stack(workers=)` hands every worker
its share of the parent's budget that way, so a worker does not size itself from a job whose
accounting the parent and the other workers fill. `scan.read`, previews, `bexa.stack` and
`bexa.load` refuse anything larger (`check_fits`); batch sizes, the prefetch depth and the
in-memory cache level follow from it.

```python
# the memory one step may use on the CPU, and a fraction of the GPU
backend.memory_budget("cpu") / 1e9              # GB: 0.8 of what this process can still use
backend.memory_budget("cuda", fraction=0.2)     # also at most 0.2 of the free GPU memory
```

#### `check_fits`
`bexa.core.backend.check_fits(nbytes, what, device="cpu", hint="")`

Raise `BudgetError` (a `MemoryError`) when `nbytes` is more than `memory_budget(device)`, else
return the budget. The message names `what`, the size and the budget in GB, the `hint` (the way
out: an ROI, a larger downsample, `store=True`) and that `BEXA_MEMORY_FRACTION` raises the
budget, so a result that cannot fit stops the call instead of the kernel. `scan.read`,
`Preview`, the in-memory `bexa.stack` and `bexa.load` call it before allocating.

```python
# a small array passes; a petabyte raises BudgetError
backend.check_fits(10**6, "a small array")                       # the budget in bytes: it fits
try:
    backend.check_fits(10**15, "reading 10 TB", hint="add an ROI")
except MemoryError as exc:
    print(exc)      # reading 10 TB needs 1000000.00 GB but the memory budget is 8.88 GB; add an ROI (BEXA_MEMORY_FRACTION raises the budget)
```

#### `choose_batch_frames`
`bexa.core.backend.choose_batch_frames(frame_bytes, budget=None, live_copies=4, min_frames=1, max_frames=None, device="cpu", max_bytes=None)`

Frames per batch so that `live_copies` batch-sized arrays (raw batch, float copy, filtered copy,
work space) fit in `budget` bytes (default `memory_budget(device)`), the batch itself capped at
`max_bytes` (default `MAX_BATCH_BYTES`, 256 MiB), then clipped to `min_frames` and `max_frames`.
`frame_bytes` is one frame after ROI, downsampling and dtype conversion. The cap matters on
large nodes: with 400 GB the budget alone made batches of whole scans, which are no faster (the
frames stream through once either way) and multiply every batch-sized temporary of the
pipeline. Measured on the lab workstation (160 frames of 1024 x 1024, Sum + MotorCOM(sigma=3)
on the CPU), batches of 16, 47 and 160 frames run at the same speed, and peak memory is about
4.5 batches plus the results.

```python
# how many 2160 x 2560 frames fit a batch under a budget of 8 GiB
backend.choose_batch_frames(frame_bytes=2160 * 2560 * 4, budget=8 * 1024**3)   # 12 float32 PCO frames: 256 MiB
backend.choose_batch_frames(frame_bytes=2160 * 2560 * 4, budget=8 * 1024**3, max_bytes=2 * 1024**3)   # 97: the budget alone
```

#### `device_info`
`bexa.core.backend.device_info()`

A dict of what this process may use: `cpu_count`, `cpu_source`, `cpu_count_machine`,
`ram_total_gb`, `ram_available_gb`, `memory_source`, `cuda`, and with a GPU `gpu_name`,
`gpu_total_gb`, `gpu_free_gb`. `bexa version` prints it.

```python
# what devices this machine has
backend.device_info()   # {'cpu_count': 16, 'cpu_source': 'machine', ..., 'cuda': True, 'gpu_name': 'NVIDIA GeForce RTX 4090', ...}
```

#### `cupy_available`
`bexa.core.backend.cupy_available()`

True when cupy imports and a CUDA device answers; checked once per process.

```python
# whether cupy is installed and a GPU answers
backend.cupy_available()                # True on a GPU machine: device="auto" then means the GPU
```

#### `device_of`
`bexa.core.backend.device_of(a)`

`"cuda"` for a cupy array, `"cpu"` for anything else.

```python
# gpu, host: the same array on the GPU and on the CPU; where each lives
backend.device_of(gpu), backend.device_of(host)          # ('cuda', 'cpu')
```

#### `is_cupy_array`
`bexa.core.backend.is_cupy_array(a)`

True for a cupy ndarray; it checks the type's module, so it never imports cupy.

```python
# gpu, host: the same array on the GPU and on the CPU
backend.is_cupy_array(gpu), backend.is_cupy_array(host)  # (True, False)
```

#### `ndimage_module`
`bexa.core.backend.ndimage_module(xp)`

`scipy.ndimage` for numpy, `cupyx.scipy.ndimage` for cupy; `xp` is the module or an array.

```python
# gpu: an array on the GPU; the ndimage of its device, used to smooth it
smooth = backend.ndimage_module(gpu).gaussian_filter(gpu, sigma=(0, 2, 2))   # on the GPU
```

#### `cast_frames`
`bexa.core.backend.cast_frames(frames, dtype)`

`frames` as `dtype` (numpy), the conversion of a large array (from `CAST_IN_THREADS_FROM`,
8 MiB) split over `bexa.core.parallel.compute_threads` threads; numpy's casting releases the
GIL. The streaming engine reads frames as stored and converts them here (or on the GPU), so the
reading thread only reads. An array that already has the dtype is returned as it is.

```python
# raw: uint16 frames; cast to float32 in threads
raw = np.random.default_rng(0).integers(0, 60000, (24, 512, 512), dtype=np.uint16)
backend.cast_frames(raw, np.float32).dtype          # float32, equal to raw.astype(np.float32)
backend.cast_frames(raw, np.uint16) is raw          # True
```

#### `gaussian_frames`
`bexa.core.backend.gaussian_frames(frames, sigma)`

Every frame of an `(n, y, x)` array smoothed by a Gaussian of `sigma` pixels along the last two
axes, numpy or cupy, the same kind back; `sigma <= 0` returns the input itself. numpy float
frames are split over `bexa.core.parallel.compute_threads` threads of the shared
`bexa.core.parallel.compute_pool`: scipy's filters release the GIL, and eight threads smooth
about five times faster than one. A single image, an integer array or a cupy array goes through
`gaussian_filter` directly. The weights of `MotorCOM` and the `sigma` of
`bexa.analysis.rocking.motor_com` go through it.

```python
# frames: eight frames; each smoothed with a Gaussian of 3 pixels, in threads
frames = np.random.default_rng(0).random((8, 64, 80), dtype=np.float32)
smooth = backend.gaussian_frames(frames, 3.0)                 # (8, 64, 80), smoothed in threads
backend.gaussian_frames(frames, 0) is frames                  # True
```

#### `BudgetError`
`bexa.core.backend.BudgetError`, a subclass of `MemoryError`

The `MemoryError` subclass `check_fits` raises: a request larger than the budget, refused before
anything was allocated. A `MemoryError` for callers (`except MemoryError` catches it), but not
an out-of-memory condition, so `reduce` does not retry with smaller batches on it.

```python
# the error check_fits raises for a petabyte volume
try:
    backend.check_fits(10**15, "a petabyte volume")
except backend.BudgetError as exc:
    str(exc)[:38]                                # 'a petabyte volume needs 1000000.00 GB'
```

#### `is_oom_error`
`bexa.core.backend.is_oom_error(exc)`

True for `MemoryError` (also raised by `scan.read` above the budget) and cupy out-of-memory
errors. `reduce` uses it to halve the batch and retry, then to fall back to the CPU, except for
a `BudgetError`, whose refusal would not change with smaller batches.

```python
# whether an exception is the GPU running out of memory
try:
    vol = scan.read(device="cuda")
except Exception as exc:
    if not backend.is_oom_error(exc):
        raise
    vol = scan.read(downsample=(1, 1, 2, 2))         # retry smaller
```

#### `free_device_memory`
`bexa.core.backend.free_device_memory()`

Returns the blocks cached by cupy's memory pools (device and pinned host) to the driver; does
nothing without cupy or a GPU.

```python
# gpu, smooth: arrays on the GPU; dropped, then the pool freed
del gpu, smooth
backend.free_device_memory()                     # the cupy pool gives its blocks back; nothing without cupy
```

Constants:

- `DEFAULT_MEMORY_FRACTION`: `0.8`, the budget fraction without `fraction=` or
  `BEXA_MEMORY_FRACTION`. `MAX_BATCH_BYTES`: `256 * 2**20`, the cap on one batch of frames
  whatever the budget.

### bexa.core.resources: memory and CPUs of the process, SLURM cgroups

psutil reports the whole node, but a SLURM job (a Jupyter-SLURM session too) owns part of it
and is killed when it exceeds its memory allocation. These functions read the job's limit from
its cgroup (v2, or v1), fall back to the SLURM variables, then to the machine. `memory_budget`,
`default_workers` and `bexa settings` rely on them.

```python
from bexa.core import resources
```

#### `memory_info`
`bexa.core.resources.memory_info(root=Path("/sys/fs/cgroup"), proc_cgroup=Path("/proc/self/cgroup"))`

The memory this process can use, as a `MemoryInfo` in bytes: the machine's available and total
memory (source `"machine"`), replaced by a cgroup limit from `cgroup_memory` when its headroom
(limit minus use) is smaller (source `"cgroup v2"` or `"cgroup v1"`, `total` the limit). With no
readable cgroup inside a SLURM job (`SLURM_JOB_ID` set), the allocation from
`SLURM_MEM_PER_NODE`, or `SLURM_MEM_PER_CPU` times `SLURM_CPUS_ON_NODE` (or
`SLURM_CPUS_PER_TASK`), in MB, minus this process's resident memory (source
`"SLURM allocation"`). `root` and `proc_cgroup` point at a fake cgroup tree in tests.

```python
# the memory of this job: available and total, and where the numbers come from
mem = resources.memory_info()
print(f"{mem.available / 1e9:.1f} of {mem.total / 1e9:.1f} GB from the {mem.source}")
# 26.0 of 68.6 GB from the machine            (in a SLURM job: ... from the cgroup v2)
```

#### `usable_cpus`
`bexa.core.resources.usable_cpus()`

`(cores, source)`: the cores this process may run on. The CPU affinity on Linux
(`"CPU affinity"`), `os.cpu_count()` elsewhere (`"machine"`), lowered to `SLURM_CPUS_PER_TASK`
when that is smaller.

```python
# the CPUs this job may use (its cgroup or affinity)
resources.usable_cpus()                 # (16, 'machine') on the lab workstation
```

#### `cgroup_memory`
`bexa.core.resources.cgroup_memory(root=Path("/sys/fs/cgroup"), proc_cgroup=Path("/proc/self/cgroup"))`

`(limit, used, version)` in bytes of the tightest memory limit on this process, or `None` outside
Linux or without a limit. It walks from the process's cgroup (read from `/proc/self/cgroup`) up to
the root, since SLURM sets the limit on the job and runs the process in a task below it: v2
`memory.max` and `memory.current`, else v1 `memory.limit_in_bytes` and `memory.usage_in_bytes`.
`"max"` and values of `UNLIMITED` or more mean no limit. `used` leaves out the file cache,
active and inactive (`active_file` and `inactive_file` of `memory.stat`, `total_...` in v1):
clean file data the kernel reclaims before killing anything, which a pass over a few hundred GB
of frames fills the job's accounting with.

```python
# the memory limit of the cgroup this job runs in, or None
resources.cgroup_memory()               # None on Windows and macOS, or on Linux without a limit
# in a SLURM job with --mem=64G using 5 GB: (68719476736, 5368709120, 'cgroup v2')
```

#### `MemoryInfo`
`bexa.core.resources.MemoryInfo(available, total, source)`

Frozen dataclass returned by `memory_info`: `available` bytes this process can still use, `total`
bytes of its limit, and the `source` of that limit (`"machine"`, `"cgroup v2"`, `"cgroup v1"`,
`"SLURM allocation"`).

```python
# mem: the memory info; its fields
mem.available, mem.total, mem.source            # (28645289984, 68572536832, 'machine')
```

Constants:

- `CGROUP_ROOT`: `Path("/sys/fs/cgroup")`. `PROC_CGROUP`: `Path("/proc/self/cgroup")`.
- `UNLIMITED`: `2**60`; cgroup limits at or above it count as no limit.

### bexa.core.settings: what bexa uses right now (`bexa.settings()`)

Device, memory budget, cache folder and profile come from defaults, the profile, environment
variables and call arguments. These functions resolve them as the library does and name the
source of each value: run them first on a new machine, in a new kernel or in a batch job.

```python
import bexa
from bexa.core.settings import describe, settings
```

#### `show`
`bexa.core.settings.show()`

This is `bexa.settings()`: prints `describe()` and returns `settings()`. The command line
equivalent is `bexa settings`.

```python
# what bexa uses right now: device, memory, cache, profile
s = bexa.settings()     # device: cuda (requested auto, from default), GPU NVIDIA GeForce RTX 4090, 16 CPU cores (from machine)
                        # memory: budget 20.7 GB = fraction 0.8 (from default) of 25.9 GB available and 24.2 GB free on the GPU
```

#### `settings`
`bexa.core.settings.settings()`

The resolved settings as nested dicts: `version`, `repository`, `log_level`; `device`
(`requested`, `resolved`, `source`, `cpu_count`, `cpu_source`, `gpu`); `memory` (`fraction` and
`source`, `available_gb`, `limit_gb`, `limit_source`, `slurm_job`, `gpu_free_gb`, `budget_gb`);
`cache` (`dir`, `source`: `BEXA_CACHE_DIR`, `processed_root of profile <name>` or
`user cache folder`); `profile` (`name` from `BEXA_PROFILE`, `source`, `error` when it does not
load, `search_path`, `profile_dir_env`, `available`). The device shown resolves `BEXA_DEVICE`,
which calls keeping `device="auto"` do not read (see `resolve_device`).

```python
# s: the settings as a dict; the memory budget, the cache folder, the profiles found
s = settings()
s["memory"]["budget_gb"], s["cache"]["dir"], s["profile"]["available"]  # (19.31, 'C:\\Users\\sport\\AppData\\Local\\Temp\\claude\\x--Dresselhaus-Lab-Code-Genera...
```

#### `describe`
`bexa.core.settings.describe()`

`settings()` as text, one line per value with its source: version and repository, device,
memory budget, memory limit, cache, profile, profile search path, profiles available, log level.

```python
# the settings as text
print(describe())                                # device, memory budget, cache folder and profile, one per line
```

### bexa.core.cache: the result cache

Previews and, when enabled, reductions are cached in two levels: an in-memory LRU and a folder of
bexa `.h5` files, keyed by a fingerprint of the source files (path, size, modification time; the
master file of a BLISS scan by its entry, see `Hdf5StackSource.cache_records`) and of the
request, so a result is reused only while both are unchanged. `scan.preview` uses the cache of
its scan, `scan.reduce` too when the scan was opened with `cache_reductions=True` or through a
profile with a `processed_root` (or with `cache=`); the scans of a `Dataset` share one cache, and
`bexa.stack(..., store=True)` keeps its files under `<root>/stacks`.

```python
import bexa
from bexa.core.cache import Cache, default_cache, default_memory_bytes

scan = bexa.demo("rocking")                  # synthetic mu scan
total = scan.sum()                           # DataArray (y, x)
```

#### `Cache`
`bexa.core.cache.Cache(root=None, memory_bytes="auto")`

Two-level cache: `root` is the folder of entries (`None`: memory only), `memory_bytes` the budget
of the LRU level: `"auto"` (`default_memory_bytes`: a quarter of the usable memory, at most 32 GiB),
a number of bytes, or 0 or `None` to turn it off; larger objects stay on disk only. Entries are
`<root>/<first 2 characters of the key>/<key>.h5`, written through a temporary file with lzf
compression and byte shuffle (`FILE_COMPRESSION`: a preview is written in a tenth of gzip's
time and read three times faster; the files are bexa's own, so portability does not matter).
Members (and attributes `root`, `memory_bytes`):

- `key(files, params)`: static method, `bexa.core.provenance.fingerprint(files, params)`, a
  40-character hex key that changes with any file's size or modification time or any parameter.
- `put(key, obj)`: store a DataArray, Dataset or dict of DataArrays in memory and, with a root,
  on disk with `bexa.save`; returns the file path or `None`.
- `get(key)`: the object or `None`. A memory hit is the stored object itself; a disk hit loads
  the file as an `xarray.Dataset` (a DataArray comes back as a one-variable Dataset) and keeps it
  in memory. An unreadable file (a job killed while writing) is deleted and counts as a miss; a
  file over the memory budget right now is left in place and counts as a miss too.
- `key in cache`: in memory or on disk. `path_for(key)`: the file of an entry, `None` without root.
- `entries()`: every file as a `CacheEntry`, sorted by path (`bexa cache ls`).
- `clear(memory=True, disk=True)`: forget the memory level and/or delete every file and empty
  sub-folder; returns the number of files deleted (`bexa cache clear`).

```python
# cache: a cache folder; the key of a result of the demo scan
cache = Cache("processed/bexa_cache")
key = Cache.key(scan.files, {"op": "sum"})
cache.put(key, total)                        # processed/bexa_cache/<2 characters>/<key>.h5
hit = cache.get(key)                         # the DataArray itself, from memory
from_disk = Cache("processed/bexa_cache", memory_bytes=0).get(key)   # Dataset with variable 'sum'
cache.clear(disk=False), cache.clear()       # (0, 1): memory forgotten, then the file deleted
```

#### `default_cache`
`bexa.core.cache.default_cache(processed_root=None)`

The process-wide `Cache` at `bexa.config.paths.cache_root(processed_root)`: `BEXA_CACHE_DIR`,
else `<processed_root>/bexa_cache` when writable, else `~/.cache/bexa`. Made on the first call; a
`processed_root` that maps to another folder replaces it. `bexa cache path|ls|clear` use it
without a profile, that is `BEXA_CACHE_DIR` or `~/.cache/bexa`.

```python
# the cache bexa opens by default; its folder
default_cache().root                         # BEXA_CACHE_DIR when set, else ~/.cache/bexa
default_cache("processed").root.name            # 'cache'
```

#### `default_memory_bytes`
`bexa.core.cache.default_memory_bytes()`

The budget of the in-memory level of a `Cache(memory_bytes="auto")`: a quarter of the memory
this process can use (`bexa.core.resources.memory_info`, `MEMORY_SHARE`), at most 32 GiB
(`DEFAULT_MEMORY_BYTES`). Results read back from this level cost nothing, so it takes what a
session can spare: 16 GiB in a 64 GB session, 32 GiB on a 400 GB node, 1 GB on a 4 GB laptop;
being an LRU it never holds more than that whatever the dataset.

```python
# how much of a cache stays in memory by default
default_memory_bytes() <= 32 * 1024**3       # True
Cache("processed/bexa_cache").memory_bytes == default_memory_bytes()   # True
```

#### `CacheEntry`
`bexa.core.cache.CacheEntry(key, path, size_bytes, created)`

Dataclass for one file of the cache: `key`, `path`, `size_bytes`, and `created`, the file's
modification time in seconds since the epoch.

```python
# cache: a cache folder with one result stored; its entries
cache.put(key, total)                           # Path('.../2b41a6715f91507432c1a4a4ecce075c44484256.h5')
for entry in cache.entries():
    print(entry.key[:8], f"{entry.size_bytes / 1e3:.0f} kB", entry.path)
```

Constants:

- `DEFAULT_MEMORY_BYTES`: `2 * 1024**3`, the most the memory level takes by default; `MEMORY_SHARE`: `0.1`, its share of the usable memory; `FILE_COMPRESSION`: `"lzf"`, the HDF5 compression of the cache files.

### bexa.core.parallel: threads, processes and prefetching

HDF5 reads and hdf5plugin decompression release the GIL, so threads overlap reading with
computation, and so do scipy's filters, so the CPU work of one process (the Gaussian smoothing
of the centre of mass) is split over `compute_threads` threads. h5py lets one thread at a time
into HDF5, though, so reading several scans at once takes processes: CPU-bound work per file
(one XFEL point file) and `bexa.stack(..., workers=)` run in spawned processes.

```python
import bexa
from bexa.core.parallel import (
    Prefetcher, compute_pool, compute_threads, default_workers, pin_blas_threads, process_pool, thread_map,
)

ds = bexa.demo("alignment")                  # a dataset of four quick scans
```

#### `thread_map`
`bexa.core.parallel.thread_map(fn, items, workers=None)`

`fn` applied to every item in a thread pool, results as a list in the order of `items`.
`workers` defaults to `default_workers("io")` and never exceeds the number of items; one worker
runs in the calling thread. Exceptions propagate.

```python
# ds: the demo z-stack dataset; every scan summed in threads
sums = thread_map(lambda n: ds.scan(n).sum(device="cpu"), ds.scans)   # one summed image per scan
len(thread_map(len, [[1], [2, 3]], workers=2))  # 2
```

#### `Prefetcher`
`bexa.core.parallel.Prefetcher(iterable, depth=2)`

Runs an iterator in a background thread with up to `depth` items waiting in a bounded queue, so
the next item is produced while the current one is used. Iterate over it once; an exception of
the producer is re-raised in the loop, and `close()` stops the producer early. The streaming
engine reads its batches through one, `bexa.core.reductions.prefetch_depth` batches deep.

```python
# ds: the demo z-stack dataset; the batches of scan 1 read ahead in a thread
total = 0.0
for batch in Prefetcher(ds.scan(1).batches(batch_frames=4)):   # the next 4 frames load meanwhile
    total += float(batch.frames.sum())
```

#### `default_workers`
`bexa.core.parallel.default_workers(kind="io")`

Worker count from `bexa.core.resources.usable_cpus` (a SLURM job's allocation, not the node):
for `kind="io"` the usable cores capped at 8, for any other kind all usable cores but one.

```python
# how many workers an I/O-bound and a CPU-bound job get
default_workers("io"), default_workers("cpu")   # (8, 15) on a 16-core workstation
```

#### `process_pool`
`bexa.core.parallel.process_pool(workers=None, env=None)`

A `ProcessPoolExecutor` with spawned workers (HDF5 and CUDA state never cross a fork), each
running `pin_blas_threads(1)` at start; `workers` defaults to `default_workers("cpu")`. `env` is
a dict of environment variables set in every worker before it imports numpy: `bexa.stack(...,
workers=)` passes a `BEXA_MEMORY_FRACTION` divided by the number of workers and a
`BEXA_THREADS` of `min(8, cores // workers)`, so the workers share the job's memory and cores.
Functions and arguments must be picklable, and a script creates the pool under
`if __name__ == "__main__":`. The pinning runs after the worker has imported the main module and
keeps variables already set, so set `OMP_NUM_THREADS=1` and the others before starting Python to
be sure.

```python
# a pool of processes (under a __main__ guard, for spawn)
import math
if __name__ == "__main__":
    with process_pool(workers=4, env={"BEXA_THREADS": "2"}) as pool:  # the workers start here, after the pins
        roots = list(pool.map(math.sqrt, range(8)))
```

#### `pin_blas_threads`
`bexa.core.parallel.pin_blas_threads(n=1)`

Sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to
`n`, only where not set already; BLAS reads them when it loads, so call it before numpy is
imported.

```python
# one BLAS thread per process, before the pool starts
pin_blas_threads(1)                          # first thing in a worker script
pin_blas_threads(2)
```

#### `compute_threads`
`bexa.core.parallel.compute_threads()`

Threads for the CPU work inside one process: `BEXA_THREADS` when set (at least 1), else the
usable cores of the job capped at 8, beyond which the gain flattens.
`bexa.core.backend.gaussian_frames` splits its frames over that many threads, and
`bexa.stack(..., workers=)` sets `BEXA_THREADS` to `min(8, cores // workers)` in every worker
so that the workers share the cores.

```python
# how many threads the accumulators use (BEXA_THREADS)
compute_threads()                            # 8 on a 16-core workstation; what BEXA_THREADS says when it is set
```

#### `compute_pool`
`bexa.core.parallel.compute_pool()`

One `ThreadPoolExecutor` of `compute_threads()` threads per process, made on the first call and
shared by every frame-parallel numpy job (`gaussian_frames`), so that the threads are not
started again for every batch.

```python
# the shared thread pool of the process; a map over it
pool = compute_pool()
list(pool.map(abs, [-1, -2, -3]))            # [1, 2, 3]
compute_pool() is pool                       # True: the same pool on every call
```

Constants:

- `BLAS_THREAD_VARIABLES`: the four variables above.

### bexa.core.provenance: provenance attrs and fingerprints

Every array from a reader, reduction or pipeline carries provenance attrs: `bexa_version`,
`git_hash`, `created`, `python`, `source_files` and `parameters` (JSON text), plus the timing of
reductions (`elapsed_s`, `frames`, `frames_per_s`, `peak_rss_mb`, `device`). `bexa.save` writes
them into the file.

```python
import numpy as np
import bexa
from bexa.core import provenance

scan = bexa.demo("rocking")                  # synthetic mu scan
res = scan.com(axes=("mu",), sigma=0.0)      # Dataset with provenance attrs
```

#### `build_attrs`
`bexa.core.provenance.build_attrs(source_files=(), parameters=None, **extra)`

The standard attrs: `bexa_version`, `git_hash` (empty outside a checkout), `created` (ISO time
with zone), `python`, `source_files` (JSON of `file_records`), `parameters` (JSON, `{}` for
`None`), and every `extra` keyword, strings, numbers and booleans as they are, others as JSON,
`None` skipped. All values are plain strings or numbers, as HDF5 attributes need.

```python
# scan: the demo chi x mu scan; the attrs a result of it carries
attrs = provenance.build_attrs(scan.files, {"sigma": 3.0, "roi": np.array([700, 1100])}, device="cpu")
attrs["parameters"]                          # '{"roi":[700,1100],"sigma":3.0}'
```

#### `parameters_of`
`bexa.core.provenance.parameters_of(attrs)`

The `parameters` of provenance attrs as a dict (they are stored as JSON text); `{}` when absent.

```python
# res: the COM maps of the demo scan; the parameters stored in their attrs
provenance.parameters_of(res.attrs)["plan"]["window"]   # [0, 64, 0, 80]: y0, y1, x0, x1 in pixels
```

#### `attach`
`bexa.core.provenance.attach(obj, **attrs)`

Adds the keywords (except `None` values) to `obj.attrs` of a DataArray or Dataset in place and
returns `obj`.

```python
# res: the COM maps of the demo scan; a note added to the attrs of one map
com = provenance.attach(res["com_mu"], operator="DL", note="first look")
com.attrs["operator"], com.attrs["note"]        # ('DL', 'first look')
```

#### `fingerprint`
`bexa.core.provenance.fingerprint(paths, params=None)`

SHA-1 hex digest of the `file_records` of `paths` and of `params` as sorted JSON: it changes when
a file is rewritten or a parameter changes. `Cache.key` uses it; the reductions pass the
`cache_records()` of the source, so a dict record (a master file keyed by its entry) is hashed
as it is.

```python
# scan: the demo chi x mu scan; the hash of its files and parameters (a cache key)
provenance.fingerprint(scan.files, {"sigma": 3.0}) == provenance.fingerprint(scan.files, {"sigma": 2.0})  # False
provenance.fingerprint(scan.source.cache_records(), {"op": "sum"})      # the key a cached sum of this scan gets
```

#### `file_records`
`bexa.core.provenance.file_records(paths)`

One dict per path with `path`, `size` (bytes) and `mtime` (seconds since the epoch); a missing
file gets size -1. A record given as a dict passes through unchanged: sources describe a file
that way when a part of it identifies the data better than its size and modification time
(`BaseSource.cache_records`).

```python
# what the attrs record about input files, missing ones included
provenance.file_records(["missing.h5", {"path": "master.h5", "entry": "7.1"}])  # [{'path': 'missing.h5', 'size': -1, 'mtime': 0.0}, {'path': 'master.h5', 'entry': '7.1'}]
# [{'path': 'missing.h5', 'size': -1, 'mtime': 0.0}, {'path': 'master.h5', 'entry': '7.1'}]
```

#### `RunStats`
`bexa.core.provenance.RunStats(device="cpu")`

Timer for one run: `start()` returns the started object; `stop(frames=None)` returns
`elapsed_s`, `peak_rss_mb` and `device`, plus `frames` and `frames_per_s` when `frames` is given.
Attributes `device`, `t0`, `elapsed_s`. `reduce` makes its timing attrs with it.

```python
# scan: the demo chi x mu scan; the time and memory of reading it
stats = provenance.RunStats(device="cpu").start()
vol = scan.read()
stats.stop(frames=vol.shape[0])              # {'elapsed_s': ..., 'peak_rss_mb': ..., 'device': 'cpu', 'frames': 25, 'frames_per_s': ...}
```

#### `peak_rss_mb`
`bexa.core.provenance.peak_rss_mb()`

Peak resident memory of this process so far, in MB (1e6 bytes).

```python
# the peak memory of this process so far
provenance.peak_rss_mb()                        # 725.98
```

#### `git_hash`
`bexa.core.provenance.git_hash(root=None)`

Short hash of the checked-out commit at `root` (default: the bexa repository), from
`git rev-parse --short HEAD`; `None` without git or a repository. Git runs once per checkout and
process and the answer is kept: every cache write and every saved file records it.

```python
# the commit of the bexa checkout, or None
provenance.git_hash()                        # a short hash such as '546f5f7'
provenance.git_hash(root="nowhere")
```

#### `to_json`
`bexa.core.provenance.to_json(obj)`

Compact JSON with sorted keys, converting numpy values, paths, sets, dates, slices and pydantic
models through `json_default`.

```python
# numpy values, arrays and slices made JSON-able
provenance.to_json({"roi": np.array([700, 1100]), "sigma": np.float32(3.0), "rows": slice(0, 10)})  # '{"roi":[700,1100],"rows":[0,10,null],"sigma":3.0}'
# '{"roi":[700,1100],"rows":[0,10,null],"sigma":3.0}'
```

#### `json_default`
`bexa.core.provenance.json_default(obj)`

The `default=` hook of `json.dumps` behind `to_json`: numpy scalar to number, array to list,
path to text, set to sorted list, date to ISO text, slice to `[start, stop, step]`, pydantic
model to `model_dump()`, anything else to `str(obj)`.

```python
# the same, as the default of json.dumps
import json
json.dumps({"n": np.int64(7)}, default=provenance.json_default)   # '{"n": 7}'
```

### bexa.core.registry: plugin registries for engines, accumulators and plots

Named collections of plugins: `engines` (the readers of file layouts, chosen by the `engine:` of
a format spec), `accumulators` (the `bexa.acc` classes by short name) and `plots`. Built-in
members register when their module is imported; lookups by name load them first.

```python
import bexa
from bexa.core.registry import (
    Registry, accumulators, engines, load_builtins, plots,
    register_accumulator, register_engine, register_plot,
)

scan = bexa.demo("rocking")                  # synthetic mu scan
```

#### `Registry`
`bexa.core.registry.Registry(kind)`

A dictionary of plugins with a decorator; `kind` (also an attribute) names its content in error
messages. Members:

- `add(name, obj, replace=False)`: register `obj` and return it; `KeyError` when `name` holds
  another object, unless `replace=True`.
- `register(name=None, replace=False)`: decorator form of `add`; without `name` the key is the
  object's `name` attribute if it has one, else its `__name__`.
- `get(name)`: the plugin, loading the built-ins first when `name` is unknown; `KeyError` listing
  the names otherwise. `names()`: sorted names, built-ins loaded.
- `name in registry`, `len(registry)`, `iter(registry)` (sorted): without loading the built-ins.

```python
# a registry of your own: detectors by name
detectors = Registry("detector")
detectors.add("pco_ff", {"shape": (2160, 2560)})  # {'shape': (2160, 2560)}
detectors.register("eiger4m")({"shape": (2162, 2068)})     # the decorator, called directly
detectors.names(), "pco_ff" in detectors, len(detectors)    # (['eiger4m', 'pco_ff'], True, 2)
detectors.get("eiger4m")                                    # {'shape': (2162, 2068)}
```

#### `register_engine`
`bexa.core.registry.register_engine(name=None, replace=False)`

Decorator: registers a `bexa.io.base.Source` implementation in `engines`. A format spec with that
name as `engine:` then opens with it (`bexa.open`, `open_source`).

```python
# an engine of your own, registered under a layout name
@register_engine("my_layout")
class MyEngine(engines.get("hdf5_stack")):   # change what differs in your layout
    pass
```

#### `register_accumulator`
`bexa.core.registry.register_accumulator(name=None, replace=False)`

Decorator: registers an `Accumulator` class in `accumulators`. The built-ins are `sum`, `mean`,
`max`, `min`, `preview`, `projections`, `roi_integral`, `motor_com`, `energy_com`,
`argmax_motor`, `frame_stats` and `histogram`.

```python
# an accumulator of your own, registered under a name
@register_accumulator("sum_squares")
class SumSquares(bexa.acc.Sum):              # its result "sum" holds the sum of squared frames
    def update(self, frames, frame_ids):
        super().update(frames * frames, frame_ids)
res = scan.reduce([accumulators.get("sum_squares")()])
```

#### `register_plot`
`bexa.core.registry.register_plot(name=None, replace=False)`

Decorator: registers a plotting function in `plots` for lookup by name. No built-in plot is
registered in this version.

```python
# a plot function of your own, registered for bexa.plot
@register_plot()
def com_with_contours(da, ax=None):
    return da.plot(ax=ax)
plots.get("com_with_contours")                  # function
```

#### `load_builtins`
`bexa.core.registry.load_builtins()`

Imports `bexa.io.engines` and `bexa.core.reductions` once, registering the built-in engines and
accumulators; an engine whose optional dependency is missing (`extra_data`) is skipped. Call it
before `len()` or `in` on a registry.

```python
# every built-in engine and accumulator registered; the engine names
load_builtins()
engines.names()   # array_cube, generic, hdf5_points, hdf5_stack, smalldata, spec_h5 (and my_layout from above)
```

Module objects:

- `engines`, `accumulators`, `plots`: the registries, `Registry("engine")`,
  `Registry("accumulator")` and `Registry("plot")` (empty until you register a plot).

### bexa.config: profiles, models and paths in one import

`bexa.config` re-exports `load_profile`, `list_profiles`, `profile_dirs`, `env_settings` and
`write_profile_skeleton` (`bexa.config.loader`); `BeamtimeProfile`, `Sample`, `Geometry`,
`Defaults`, `DarkReference` and `AutoProcessSettings` (`bexa.config.models`); and `cache_root`,
`apply_aliases`, `ensure_dir`, `repo_root` and `package_configs` (`bexa.config.paths`). Format
specs (file layouts) live in `bexa.io.formats`.

```python
from bexa.config import BeamtimeProfile, cache_root, load_profile
```

### bexa.config.loader: finding, loading and writing beamtime profiles

A beamtime profile is a YAML file, usually `configs/beamtimes/<name>.yaml`, naming the format
spec and holding what is specific to one beamtime (fields under
`bexa.config.models.BeamtimeProfile`). `bexa.open(profile=...)`, `bexa.open_dataset`,
`bexa cube build -p` and `bexa auto -p` load it with `load_profile`.

```python
import os
from bexa.config import env_settings, list_profiles, load_profile, profile_dirs, write_profile_skeleton
```

#### `load_profile`
`bexa.config.loader.load_profile(name_or_path=None, **overrides)`

A `BeamtimeProfile` read by name (searched in `profile_dirs()`), by path to a `.yaml` or `.yml`
file, or for `None` from the name in `BEXA_PROFILE` (`ValueError` when unset). `name` defaults to
the file name, `overrides` replace top-level fields, `source_path` records the file, and `root`,
`raw_root` and `processed_root` go through `apply_aliases` with the profile's `path_aliases`. An
unknown name raises `FileNotFoundError` listing the profiles and folders; bad content raises
pydantic's `ValidationError`.

```python
# the example ESRF profile shipped with bexa; its format and a sample's dataset
prof = load_profile("example_esrf_id03")                    # configs/beamtimes/example_esrf_id03.yaml
prof.format, prof.sample("WS2G_100").dataset                # ('esrf_id03_bliss_2026', 'WS2G_100/WS2G_100_DFXM')
nf = load_profile("example_esrf_id03", detector="pco_nf")   # a field replaced for this session
os.environ["BEXA_PROFILE"] = "example_esrf_id03"            # or export BEXA_PROFILE in the shell
prof = load_profile()                                       # the profile named by BEXA_PROFILE
```

#### `list_profiles`
`bexa.config.loader.list_profiles()`

`{name: path}` of the profiles in `profile_dirs()`; for a name found twice the first folder wins.
`bexa profile list` prints them.

```python
# every profile bexa can find
sorted(list_profiles())                     # ['example_esrf_id03', 'example_pal_xfel', ...]
```

#### `profile_dirs`
`bexa.config.loader.profile_dirs()`

Folders searched for profiles, in order: those of `BEXA_PROFILE_DIR` (separated by `os.pathsep`,
`;` on Windows, `:` on Linux), `./configs/beamtimes`, and the repository's `configs/beamtimes`;
existing folders only, each once.

```python
# where profiles are looked for, in order
profile_dirs()                              # [..., Path('<repository>/configs/beamtimes')]
```

#### `write_profile_skeleton`
`bexa.config.loader.write_profile_skeleton(path, name, format_name, root=None, force=False)`

Writes `profile_skeleton(...)` to `path` as YAML (folders created) and returns the path;
`FileExistsError` for an existing file unless `force=True`. Fill in the `TODO` entries.
`bexa profile new` calls it, by default for `configs/beamtimes/<name>.yaml` of the repository.

```python
# a profile for a new beamtime written as YAML (shown, not run)
write_profile_skeleton("configs/beamtimes/hc6293.yaml", "hc6293", "esrf_id03_bliss_2026",
                       root="/data/visitor/hc6293/id03/20260119/RAW_DATA")  # Path('.../hc6293.yaml')
```

#### `profile_skeleton`
`bexa.config.loader.profile_skeleton(name, format_name, root=None)`

The skeleton as a dict: `name`, `format`, `root` (or `TODO`), `processed_root: TODO`, empty
`path_aliases`, a `SAMPLE` with a `TODO` dataset, `detector`, `geometry` (`distance_mm`,
`effective_pixel_nm`), `defaults` (`float32`, downsample `[1, 1, 4, 4]`, device `auto`), and
empty `rois` and `format_overrides`.

```python
# the text of a profile skeleton for a PAL-XFEL beamtime
from bexa.config.loader import profile_skeleton
body = profile_skeleton("ue_250913_FXS", "pal_xfel_points_2025_09")
body["name"], sorted(body)[:3]                  # ('ue_250913_FXS', ['defaults', 'detector', 'format'])
```

#### `env_settings`
`bexa.config.loader.env_settings()`

The settings given by environment variables, only those set: `device` (`BEXA_DEVICE`),
`memory_fraction` (float, `BEXA_MEMORY_FRACTION`), `cache_dir` (Path, `BEXA_CACHE_DIR`),
`profile` (`BEXA_PROFILE`). For scripts; the library reads each variable where it needs it.

```python
# the BEXA_ environment variables that are set
env_settings()                              # {'memory_fraction': 0.3} with only BEXA_MEMORY_FRACTION=0.3 set
```

Constants:

- `PROFILE_ENV`: `"BEXA_PROFILE"`. `PROFILE_DIR_ENV`: `"BEXA_PROFILE_DIR"`.
  `PROFILE_SUFFIXES`: `(".yaml", ".yml")`.

### bexa.config.models: the fields of a beamtime profile

Pydantic models of a profile file. A profile names one format spec and holds data roots,
samples, geometry, darks, defaults and named ROIs, never HDF5 paths or keys (those belong to the
format spec, `bexa.io.formats`). `BeamtimeProfile`, `Sample`, `Geometry` and `Defaults` keep
unknown keys as extra attributes (`model_extra`); `DarkReference` and `AutoProcessSettings` drop
them. All have pydantic's `model_validate(dict)`, `model_dump()` and `model_copy(update=...)`.
An ESRF ID03 profile, as in `configs/beamtimes/example_esrf_id03.yaml`:

```yaml
name: hc6293                              # default: the file name
format: esrf_id03_bliss_2026              # required: a spec in configs/formats/
root: /data/visitor/hc6293/id03/20260119/RAW_DATA
processed_root: /data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa
path_aliases:                             # the same file on the lab PC
  /data/visitor/hc6293/id03/20260119: X:/Beamtimes/hc6293
samples:
  WS2G_100:
    dataset: WS2G_100/WS2G_100_DFXM       # under root
    cif: cifs/WS2.cif
    hkl_center: [0, 0, 6]
  D2: D2_dfxm/D2_dfxm_mosa_test           # a plain string is the dataset
detector: pco_ff
geometry: {effective_pixel_nm: 235}
defaults: {downsample: [1, 1, 4, 4], device: auto}
rois:
  grain: {y: [700, 1500], x: [800, 1800]}
darks:
  pco_ff: {dataset: WS2G_100/WS2G_100_darks, scan: 1}
```

A PAL-XFEL profile (`example_pal_xfel.yaml`) has `raw_root` instead of `root`, plus
`geometry: {energy_keV: 9.7}`, `background_box` and an `auto_process` section.

```python
import bexa
from bexa.config import (
    AutoProcessSettings, BeamtimeProfile, DarkReference, Defaults, Geometry, Sample, load_profile,
)

prof = load_profile("example_esrf_id03")        # the example ESRF profile shipped with bexa
```

#### `BeamtimeProfile`
`bexa.config.models.BeamtimeProfile(*, name="", format, root=None, raw_root=None, processed_root=None, path_aliases={}, samples={}, detector=None, geometry=Geometry(), darks={}, defaults=Defaults(), rois={}, background_box=None, auto_process=None, format_overrides={}, source_path=None, **extra_data)`

Everything specific to one beamtime, usually from `load_profile`:

| Field | Meaning |
|---|---|
| `name` | Profile name; the file name when absent. |
| `format` | Required: the format spec of the files (`esrf_id03_bliss_2026`, `pal_xfel_points_2025_09`). |
| `root` | Folder of the dataset folders (frame-stack beamlines such as ESRF). |
| `raw_root` | Raw-data root of point-file beamlines (PAL-XFEL: `type=measurement/`, `type=raw/`), used without `root`. |
| `processed_root` | Results: the cache is `<processed_root>/bexa_cache` and scans opened through the profile cache their reductions there; PAL cubes go to `data/runN.h5`, figures to `outputs/`. |
| `path_aliases` | Beamline prefix to local prefix for `root`, `raw_root` and `processed_root` (see `apply_aliases`). |
| `samples` | `Sample` per name; a plain string is the dataset. |
| `detector` | Default detector (`pco_ff`, `pco_nf`, `jungfrau2`); a sample's `detector` and a call's `detector=` win. |
| `geometry` | A `Geometry`, merged over the spec's detector table into `scan.geometry`. |
| `darks` | A `DarkReference` per detector. |
| `defaults` | A `Defaults`. |
| `rois` | Named pixel windows at full resolution, `[start, stop]` per dim; `bexa.ROI.from_dict(prof.rois["grain"])` makes an ROI, and `bexa auto` uses them as default ROIs. |
| `background_box` | Corner box for the per-frame background of pump-probe cubes, `{y: [0, 50], x: [0, 50]}`. |
| `auto_process` | An `AutoProcessSettings` for `bexa auto`. |
| `format_overrides` | Deep-merged over the format spec for this beamtime (a renamed HDF5 key, another detector). |
| `source_path` | Set by `load_profile` to the file read. |

Methods: `sample(name)` returns the `Sample` or raises a `KeyError` listing the samples;
`data_root()` returns `root`, else `raw_root` (`ValueError` without both), where datasets are
opened as `data_root() / sample.dataset`.

```python
# prof: the example ESRF profile; its detector, regions and a sample's reflection
prof.detector, prof.rois["grain"]            # ('pco_ff', {'y': [700, 1500], 'x': [800, 1800]})
prof.sample("WS2G_100").hkl_center           # (0, 0, 6)
prof.data_root() / prof.sample("WS2G_100").dataset  # Path('.../WS2G_100_DFXM')
grain = bexa.ROI.from_dict(prof.rois["grain"])
p = BeamtimeProfile(format="esrf_id03_bliss_2026", root="/data/visitor/hc6293/id03/20260119/RAW_DATA",
                    samples={"WS2G_100": "WS2G_100/WS2G_100_DFXM"})
```

#### `Sample`
`bexa.config.models.Sample(*, dataset, cif=None, hkl_center=None, detector=None, notes="", **extra_data)`

One sample (one dataset folder): `dataset` (required, relative to `root`); `cif` (relative to
`root` or absolute; `bexa.open` reads it into `scan.crystal`, which enables hkl coordinates, and
warns when it cannot); `hkl_center`, the reflection `[h, k, l]` the scans are centred on;
`detector` when it differs from the profile's; `notes`, free text.

```python
# one sample of a profile: its dataset, CIF and reflection
Sample(dataset="WS2G_100/WS2G_100_DFXM", cif="cifs/WS2.cif", hkl_center=(0, 0, 6))  # Sample
Sample(dataset="D2/D2_dfxm").cif
```

#### `Geometry`
`bexa.config.models.Geometry(*, energy_keV=None, pixel_size_um=None, distance_mm=None, effective_pixel_nm=None, beam_center=None, two_theta_offset_deg=0.0, delta_offset_deg=0.0, convention=None, **extra_data)`

Detector and beam geometry; the values set replace the spec's detector table in `scan.geometry`.
`energy_keV`: photon energy in keV for the geometry and for PAL-XFEL cubes (`scan.energy_keV`
keeps the files' value). `pixel_size_um`: detector pixel in um. `distance_mm`: sample to
detector in mm. `effective_pixel_nm`: sample-plane pixel in nm after the objective (DFXM
scalebars). `beam_center`: `(y, x)` pixel of the direct beam. `two_theta_offset_deg`,
`delta_offset_deg`: deg added to the 2theta and delta of each pixel. `convention`: diffractometer
convention (`id03`, `pal_fourc`, `sacla`) instead of the spec's.

```python
# the geometry block of a profile
Geometry(energy_keV=17.0, effective_pixel_nm=235, beam_center=(1080, 1280))  # Geometry
Geometry(energy_keV=17.0).effective_pixel_nm
```

#### `Defaults`
`bexa.config.models.Defaults(*, precision="float32", downsample=None, device="auto", batch_scans=4, parallel=True, memory_fraction=None, apply_log=False, **extra_data)`

The `defaults:` section: `precision` (`"float32"` or `"float64"`, others rejected), `downsample`,
`device`, `batch_scans`, `parallel`, `memory_fraction`, `apply_log`. In this version they are
validated and stored, but no bexa function applies them: pass `downsample=` and `device=` to the
calls and set `BEXA_MEMORY_FRACTION`.

```python
# prof: the example ESRF profile; its default downsample
prof.defaults.downsample                     # [1, 1, 4, 4]
```

#### `DarkReference`
`bexa.config.models.DarkReference(*, dataset, scan=1)`

Where the darks of a detector were recorded (`darks: {<detector>: ...}`): `dataset` relative to
`root` and `scan` number. No bexa function reads `darks` yet; subtract darks with
`bexa.analysis.preprocess.subtract_dark`.

```python
# the dark reference block of a profile
DarkReference(dataset="WS2G_100/WS2G_100_darks", scan=1)  # DarkReference(dataset='WS2G_100/WS2G_100_darks', scan=1)
DarkReference(dataset="WS2G_100/WS2G_100_darks").scan  # 1
```

#### `AutoProcessSettings`
`bexa.config.models.AutoProcessSettings(*, roi_table=None, skip_list=None, outputs=None, interval_s=60.0)`

Files of the beamtime watcher (`bexa auto`), relative to `processed_root`: `roi_table`, per-run
ROIs and axes (legacy `ROI_dict.csv` or YAML; default `ROI_dict.csv` if present); `skip_list`,
runs that failed twice (default `Skip_dict.csv`); `outputs`, the figure folder (default
`outputs`). `interval_s` is stored but not read; `bexa auto --interval` sets the polling interval.

```python
# the watcher block of the example PAL-XFEL profile
load_profile("example_pal_xfel").auto_process.roi_table   # 'ROI_dict.csv'
```

### bexa.config.paths: repository, cache folder and path aliases

Where things are: the checkout and its `configs/` folder, the cache folder, and the mapping of
beamline paths onto this machine.

```python
from bexa.config import apply_aliases, cache_root, ensure_dir, package_configs, repo_root
from bexa.config.paths import is_writable, user_cache_dir
```

#### `cache_root`
`bexa.config.paths.cache_root(processed_root=None)`

The cache folder: `BEXA_CACHE_DIR` when set, else `<processed_root>/bexa_cache` when that folder
(or its parent, before it exists) is writable, else `user_cache_dir()`. Nothing is created.

```python
# where the cache goes by default, and under a PROCESSED_DATA folder
cache_root()                                 # BEXA_CACHE_DIR, else ~/.cache/bexa
cache_root("/data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa")   # without BEXA_CACHE_DIR: .../bexa/bexa_cache
```

#### `apply_aliases`
`bexa.config.paths.apply_aliases(path, aliases)`

A path recorded at the beamline mapped onto this machine: `aliases` maps a prefix as written in
the profile to where the same data lives here (forward and back slashes alike). The original path
wins when it exists, then the first alias whose mapped path exists, else the original. Returns a
`Path`.

```python
# aliases: a beamline path mapped to a local mount; a path rewritten
aliases = {"/data/visitor/hc6293/id03/20260119": "X:/Beamtimes/hc6293"}
apply_aliases("/data/visitor/hc6293/id03/20260119/RAW_DATA", aliases)   # X:/Beamtimes/hc6293/RAW_DATA once copied there
```

#### `repo_root`
`bexa.config.paths.repo_root()`

The checkout folder holding `bexa/` and `configs/`, resolved (a mapped Windows drive shows as its
UNC path).

```python
# the folder of the bexa checkout
repo_root()                                     # Path('.../Experiment-Toolbox')
```

#### `package_configs`
`bexa.config.paths.package_configs()`

`repo_root() / "configs"`, with the `beamtimes/` profiles and `formats/` specs of the code.

```python
# the configs folder of the package; the example profiles in it
sorted(p.name for p in (package_configs() / "beamtimes").glob("*.yaml"))  # ['example_esrf_id03.yaml', 'example_pal_xfel.yaml']
```

#### `user_cache_dir`
`bexa.config.paths.user_cache_dir()`

`~/.cache/bexa`, the cache without `BEXA_CACHE_DIR` or a writable `processed_root`.

```python
# the per-user cache folder
user_cache_dir()                                # Path('.../bexa')
```

#### `ensure_dir`
`bexa.config.paths.ensure_dir(path)`

Creates the folder with its parents if needed and returns it as a `Path`.

```python
# a results folder created if missing
out = ensure_dir("results/scan7")
out.is_dir(), out.name                          # (True, 'scan7')
```

#### `is_writable`
`bexa.config.paths.is_writable(path)`

True when `path` (or, if it does not exist, its parent) exists and this user may write there.

```python
# out: the results folder; whether it, and a folder not yet made, can be written
is_writable(out), is_writable(out / "new_folder")   # (True, True)
```

Constants:

- `CACHE_DIR_ENV`: `"BEXA_CACHE_DIR"`. `CACHE_DIR_NAME`: `"bexa_cache"`.

### bexa._log: logging and progress bars

The library never prints: modules log to children of the `bexa` logger (`bexa.core.scan`, ...),
which writes `HH:MM:SS LEVEL name: message` to stderr at `BEXA_LOG_LEVEL`, INFO by default;
`bexa.notebook.setup()` sends it to the notebook cell.

```python
import io
import os
import bexa
from bexa._log import configure, get_logger, in_ipython, interactive_session, progress, set_level, in_kernel
```

#### `set_level`
`bexa._log.set_level(level)`

Sets the level of the `bexa` logger and its children (`"DEBUG"`, `"INFO"`, `"WARNING"`,
`"ERROR"` or a number). In this version any later `get_logger` call, such as the first import of
a bexa module, resets it to `BEXA_LOG_LEVEL` (INFO when unset), so set that variable too.

```python
# the log level from the environment, then set by hand
os.environ["BEXA_LOG_LEVEL"] = "DEBUG"       # survives modules loaded later
set_level("DEBUG")                           # takes effect now
```

#### `configure`
`bexa._log.configure(level=None, stream=None, force=False)`

One stream handler on the `bexa` logger (no propagation to the root logger), at `level` (name
or number; `None` reads `BEXA_LOG_LEVEL`, else INFO), writing to `stream` (default `sys.stderr`).
Once configured, calls only change the level unless `force=True` rebuilds the handler.

```python
# buffer: a text stream; bexa's messages sent to it
buffer = io.StringIO()
configure("INFO", stream=buffer, force=True)     # bexa messages go to the buffer
get_logger("bexa.io.formats").info("hello")      # buffer: '<time> INFO    bexa.io.formats: hello'
configure(force=True)                            # back to stderr
```

#### `get_logger`
`bexa._log.get_logger(name=None)`

The `bexa` logger (`None` or `"bexa"`) or a child; other names get the `bexa.` prefix, so your
own messages follow bexa's level and handler. It calls `configure()` (see `set_level`).

```python
# a logger of your own under bexa's root
log = get_logger("my_analysis")              # logger 'bexa.my_analysis'
log.info("reduced %d scans", 12)             # <time> INFO    bexa.my_analysis: reduced 12 scans
```

#### `progress`
`bexa._log.progress(iterable, total=None, desc="", enabled=True)`

A generator over `iterable` with a `tqdm.auto` progress bar (a widget in notebooks, text on
stderr, removed when done); plain iteration with `enabled=False` or without tqdm.

```python
# a progress bar over twelve scans (tqdm when installed)
for number in progress(range(1, 13), desc="scans"):
    pass                                     # one step per scan
```

#### `in_ipython`
`bexa._log.in_ipython()`

True inside IPython, Jupyter or a VS Code interactive window (as `bexa.notebook.in_ipython`).

```python
# whether this runs under IPython
in_ipython()                                 # False in a plain script
```

#### `in_kernel`
`bexa._log.in_kernel()`

True inside a Jupyter kernel (where ipywidgets can show), as `bexa.notebook.in_kernel`.

```python
# whether this runs in a Jupyter kernel
in_kernel()                                  # False in a plain script
```

#### `interactive_session`
`bexa._log.interactive_session()`

True in IPython or when stderr is a terminal; `reduce` then shows a progress bar for passes of
200 frames or more.

```python
# whether someone is watching (a kernel or a terminal)
interactive_session()                        # False in a batch job writing to a log file
```

Constants:

- `ROOT_NAME`: `"bexa"`, the name of the package's root logger.

### bexa._profiling: timing a block with cProfile (`bexa.profile`)

Where the time of a slow cell or script goes, function by function.

```python
import bexa

scan = bexa.demo("mosa")                     # synthetic chi x mu scan
```

#### `profile`
`bexa._profiling.profile(output=None, top=25, sort="cumulative")`

This is `bexa.profile`: a context manager that runs the block under cProfile and yields the
`cProfile.Profile`. With `output`, the raw stats go to that file (`.prof`, for `snakeviz` or
`pstats`) and a summary of the `top` functions sorted by `sort` (`"cumulative"`, `"tottime"`,
any pstats key) to the same name with `.txt`; without it the summary is logged at INFO. The
command-line flag `bexa --profile <command>` writes `bexa-<command>.prof` and `.txt` to the
current folder.

```python
# scan: the demo chi x mu scan; where the time of a COM pass goes
with bexa.profile("profiles/com_run.prof", top=10):
    scan.com(axes=("chi", "mu"), sigma=3.0)
# writes profiles/com_run.prof and the summary profiles/com_run.txt
```

### bexa._help: the API listing (`bexa.help`)

The names of the API with their one-line summaries, from inside Python.

#### `help`
`bexa._help.help(section=None, verbose=False)`

This is `bexa.help`: prints and returns a listing. Without `section`, the everyday names with
their one-line summaries; `"all"`, every public function and class of the documented modules with
signatures; other text, the sections whose title contains it, ignoring case (`"plots"`,
`"analysis"`, `"optics"`, `"crystal"`, `"pipelines"`, `"settings"`). `verbose=True` adds names
imported from elsewhere and, in modules without `__all__`, private names.

```python
# the API listing, every module or one
import bexa
bexa.help()                                  # the everyday names
bexa.help("plots")                           # Plots: auto, images, curves, maps, volumes, interactive, napari
text = bexa.help("all")                      # everything, also returned as a string
```

### Environment variables

Every `BEXA_*` variable read by the package or by `bin/esrf_setup.sh`. Set them in the shell or,
in a notebook, with `os.environ[...]` before the call that reads them.

| Variable | Sets | Default | Read by |
|---|---|---|---|
| `BEXA_DEVICE` | Device for calls with `device=None`: `auto`, `cpu`, `cuda` (or `gpu`). Set by `bexa --device`. An explicit `device="auto"`, the default of `scan.reduce` and the scan verbs, does not read it. | `auto` | `bexa.core.backend.resolve_device`; shown by `bexa.core.settings.settings`, `bexa.config.loader.env_settings` |
| `BEXA_MEMORY_FRACTION` | Fraction of the usable memory one operation may use (batch sizes, the `scan.read` limit). Set by `bexa --memory-fraction`; `~/.bexa_env` from `esrf_setup.sh` sets 0.3. | `0.8` | `bexa.core.backend.memory_budget`, `settings`, `env_settings` |
| `BEXA_MEMORY_BYTES` | The memory budget itself, in bytes, instead of a fraction of what is available: `bexa.stack(..., workers=)` sets it in every worker process to its share of the parent's budget. | not set | `bexa.core.backend.memory_budget`, `settings` |
| `BEXA_THREADS` | Threads for the CPU work of one process: the Gaussian smoothing of `MotorCOM` and `bexa.analysis.rocking` (`bexa.core.backend.gaussian_frames`). `bexa.stack(..., workers=)` sets it to `min(8, cores // workers)` in every worker process. | the usable cores, at most 8 | `bexa.core.parallel.compute_threads` |
| `BEXA_CACHE_DIR` | Cache folder of previews and cached reductions; when set, scans opened through a profile cache their reductions. | `<processed_root>/bexa_cache`, else `~/.cache/bexa` | `bexa.config.paths.cache_root`, `bexa.core.scan.open_profile`, `settings`, `env_settings` |
| `BEXA_PROFILE` | Profile used when none is given: `load_profile()`, `bexa.open(sample=...)`, `bexa.open_dataset(sample=...)`. | not set | `bexa.config.loader.load_profile`, `settings`, `env_settings` |
| `BEXA_PROFILE_DIR` | Folders searched first for profiles, separated by `os.pathsep`. | not set | `bexa.config.loader.profile_dirs` |
| `BEXA_FORMAT_DIR` | Folders searched first for format specs, separated by `os.pathsep`. | not set | `bexa.io.formats.format_dirs` |
| `BEXA_LOG_LEVEL` | Level of the `bexa` logger, applied again at every `get_logger` call. | `INFO` | `bexa._log.configure`, `settings` |
| `BEXA_HEADLESS` | Any value: `show_plotly(how="auto")` outside IPython returns HTML instead of opening a browser. `bexa --headless` does not set it (it only selects the Agg backend). | not set | `bexa.viz.volume.show_plotly` (also `bexa.notebook.show_plotly`) |
| `BEXA_ENV` | Folder of the virtualenv the setup script builds. | `~/bexa-env-<architecture>` | `bin/esrf_setup.sh` |
| `BEXA_NO_CUPY` | Any value: the setup script skips cupy on a GPU machine. | not set | `bin/esrf_setup.sh` |
| `BEXA_LINK_LABEXTENSIONS` | Any value: the setup script links the environment's JupyterLab extensions (widgets, ipympl, plotly) into the user's Jupyter folder. | not set | `bin/esrf_setup.sh` |

bexa also reads `SLURM_JOB_ID`, `SLURM_MEM_PER_NODE`, `SLURM_MEM_PER_CPU`, `SLURM_CPUS_ON_NODE`
and `SLURM_CPUS_PER_TASK` (`memory_info`, `usable_cpus`), and `pin_blas_threads` sets the BLAS
thread variables. The test suite reads `BEXA_LEGACY_SCRIPTS` (folder of the legacy scripts for
the `legacy` tests); the `bexa.cmd` shim sets `BEXA_ROOT` for itself.

```bash
export BEXA_PROFILE=hc6293
export BEXA_CACHE_DIR=/data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa_cache
BEXA_MEMORY_FRACTION=0.2 bexa settings       # memory: budget ... = fraction 0.2 (from BEXA_MEMORY_FRACTION) ...
```

## Command-line reference

Every command wraps a public Python function, named on the `Python:` line of its entry, so
whatever the terminal does a notebook can do too. Three ways to start it:

- `python -m bexa ...` from the repository folder (nothing is installed; `-m` finds the package
  in the current folder).
- `bexa.cmd ...` on Windows, `bin/bexa ...` on Linux and macOS, from any folder: the shims put
  the repository on `PYTHONPATH` and run `python -m bexa` (`bin/bexa` prefers `python3`).
- On the ESRF cluster, after `bash ~/bexa/bin/esrf_setup.sh` once: `source ~/.bexa_env`, then
  `bexa ...`. The file puts the environment of the machine's architecture and the checkout's
  `bin/` on `PATH`, the checkout on `PYTHONPATH`, and exports `BEXA_MEMORY_FRACTION=0.3`.

`bexa --help` lists the commands and the global options, `bexa scan --help` one group,
`bexa scan reduce --help` one command with its defaults; `bexa` or a group alone prints the same
help. A wrong argument exits with code 2 and a one-line message, a problem in the data with
code 1 and a traceback. The examples use the profiles of the walkthrough: `hc6293` (ESRF ID03,
sample `WS2G_100`: scan 7, and scans 20 to 40 an energy series) and
`ue_250913_FXS` (PAL-XFEL point files, runs 42 and 79 to 88). Relative output paths are
relative to the current folder.

### Global options

Python: `bexa.cli.main.main` (the callback that runs before every command)

Global options go between `bexa` and the command name: `bexa --headless scan report ...` works,
`bexa scan report --headless ...` fails with "No such option". The callback keeps `--device`,
`--memory-fraction` and `--headless` in a `CliState` on the Typer context and exports the first
two as environment variables, so the library code a command calls sees them.

- `--device TEXT` (default `auto`): `cpu`, `cuda` or `auto`. Sets `BEXA_DEVICE`, also to `auto`
  when left out, which replaces an exported `BEXA_DEVICE`. `bexa settings` and library calls
  with `device=None` follow it; in this version the scan commands do not, as they ask the
  reductions for `device="auto"`, which takes the GPU whenever cupy finds one.
- `--memory-fraction FLOAT` (default: not set, so an exported `BEXA_MEMORY_FRACTION`, else
  0.8): fraction of the free memory (the SLURM job's allocation on a cluster) that one operation
  may use; batch sizes follow from it. Sets `BEXA_MEMORY_FRACTION`.
- `--headless` (default off): switches matplotlib to the Agg backend for this process, so no
  window opens and `--plot` or `--show` show nothing. No environment variable (in Python, set
  `MPLBACKEND=Agg` before matplotlib is imported). `bexa scan report` and `bexa auto` always
  use Agg.
- `--log-level TEXT` (default `INFO`): `DEBUG`, `INFO`, `WARNING` or `ERROR` for the `bexa`
  logger on stderr. No environment variable; the Python API reads `BEXA_LOG_LEVEL`. In this
  version the level is reset from `BEXA_LOG_LEVEL` (default INFO) as soon as the command imports
  its modules, so `BEXA_LOG_LEVEL=DEBUG bexa ...` is the form that takes effect.
- `--profile` (default off): runs the command under cProfile and writes `bexa-<command>.prof`
  and `bexa-<command>.txt` (the 30 slowest calls) to the current folder; `<command>` is the
  first word (`bexa-scan.prof` for every `bexa scan ...`). No environment variable. Not the
  beamtime profile, which is `-p`/`--profile` after the command name.

```bash
bexa --device cpu --memory-fraction 0.3 settings
bexa --headless --profile scan reduce -p hc6293 -s WS2G_100 --scan 7 --acc sum,com -o scan7.h5
BEXA_LOG_LEVEL=DEBUG bexa --headless scan info -p hc6293 -s WS2G_100 --scan 7
```

### General commands

#### `bexa version`
Python: `bexa.cli.main.version`

Prints `bexa <version>` and the resources this process may use
(`bexa.core.backend.device_info`): usable CPU cores and their source (machine or SLURM job),
total and available RAM in GB, whether cupy finds a GPU and, if so, its name and its total and
free memory. No options.

```bash
bexa version
```

#### `bexa settings`
Python: `bexa.cli.main.settings`

Prints what the library would use now and where each value comes from
(`bexa.core.settings.describe`, as `bexa.settings()`): the device and CPU cores, the memory
budget and limit, the cache folder (`BEXA_CACHE_DIR`, the `processed_root` of `BEXA_PROFILE`, or
`~/.cache/bexa`), the profile of `BEXA_PROFILE` with the search path and the profiles found, and
the log level of `BEXA_LOG_LEVEL`. Under the command line the device source always reads
`BEXA_DEVICE`, since `--device` sets it. No options.

```bash
bexa settings
BEXA_PROFILE=hc6293 bexa --memory-fraction 0.3 settings
```

#### `bexa info`
Python: `bexa.cli.main.info`

Describes a path (`bexa.io.inspect.describe`): the matching format spec (or the closest ones
with their score), then the first 40 entries of a folder, the tree of an HDF5 file (`.h5`,
`.hdf5`, `.nxs`, `.nx`: shape, dtype, chunks, compression and ratio of every dataset) or the
shape of a `.npy` array. When the path opens as a scan (`bexa.open`), `scan.info()` follows (see
`bexa scan info`).

- `PATH` (argument, required, must exist): scan folder, dataset folder or file.
- `--depth INTEGER` (default 3): deepest HDF5 level printed; the top-level groups are level 0.
- `--attrs` / `--no-attrs` (default `--no-attrs`): meant to add the HDF5 attributes; in this
  version it is not passed on and changes nothing.
- `--scan INTEGER`: the scan to open in a dataset folder with several (one number).
- `--detector TEXT`: detector to open when the layout has several (`pco_ff`, `pco_nf`).

```bash
bexa info /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM --scan 7
bexa info /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM/scan0007/pco_ff_0000.h5 --depth 4
```

### bexa profile: beamtime profiles and format specs

A beamtime profile is a YAML file with the format spec, data roots, samples, detector and
geometry of one beamtime; `-p NAME` loads it. Profiles are searched in `BEXA_PROFILE_DIR`
(folders separated by `:` on Linux, `;` on Windows), in `configs/beamtimes` under the current
folder, then in the repository's `configs/beamtimes`; the first file of a name wins. A format
spec (`configs/formats/<name>.yaml`) describes one file layout.

#### `bexa profile list`
Python: `bexa.cli.main.profile_list`

Prints every profile found (name and file), then every format spec in `BEXA_FORMAT_DIR`,
`configs/formats` under the current folder and the repository's `configs/formats`. No options.

```bash
bexa profile list
```

#### `bexa profile new`
Python: `bexa.cli.main.profile_new`

Writes a profile skeleton (`bexa.config.write_profile_skeleton`) with `name`, `format`, `root`,
a TODO `processed_root`, `path_aliases`, a sample `SAMPLE` with a TODO dataset, `detector`,
`geometry`, `defaults` (float32, downsample `[1, 1, 4, 4]`, device auto), `rois` and
`format_overrides`. The format name is not checked. An existing file stops the command with
`FileExistsError` unless `--force` is given.

- `NAME` (argument, required): the name `-p` will take, e.g. `hc6293_id03_2026-01`.
- `--format TEXT` (required): the beamtime's format spec, e.g. `esrf_id03_bliss_2026` or
  `pal_xfel_points_2025_09` (see `bexa profile list`).
- `--root PATH`: data root at the beamline, written as `root`; a TODO when left out.
- `--out PATH`: output file; default `configs/beamtimes/<NAME>.yaml` of the repository,
  whatever the current folder, so the default writes into the checkout.
- `--force` / `--no-force` (default `--no-force`): overwrite an existing file.

```bash
bexa profile new hc6293 --format esrf_id03_bliss_2026 --root /data/visitor/hc6293/id03/20260119/RAW_DATA
bexa profile new ue_250913_FXS --format pal_xfel_points_2025_09 --out ~/bexa_profiles/ue_250913_FXS.yaml
```

#### `bexa profile sniff`
Python: `bexa.cli.main.profile_sniff`

Drafts a format spec for a layout no spec matches (`bexa.io.formats.write_draft`). From the HDF5
files at PATH (the file, or up to 20 `*.h5` files in the folder and its sub-folders) it takes
the largest 3-D datasets as frame candidates and the 1-D datasets as long as a frame count as
motor candidates, guesses the engine (`hdf5_stack` for BLISS `N.1` entries, `spec_h5` for a
`run` group, `hdf5_points` for pandas tables) and writes YAML with TODO entries for the
description, energy source, detector and convention. Parent folders are created, an existing
file is replaced. Fill in the TODOs and move the file to `configs/formats/` (`docs/formats.md`).

With `--recursive` it walks the whole tree under PATH instead (`bexa.io.survey.survey`):
it prints which spec recognises every dataset it finds, and which data nothing recognises with
the closest spec, then writes one draft per unknown HDF5 layout into the folder OUT
(`bexa.io.survey.write_drafts`), named `TODO_<folder>.yaml` with the paths it covers under
`seen_in`. Datasets of a known layout need no draft: one spec covers them all.

- `PATH` (argument, required, must exist): a file or folder of the unknown layout; with
  `--recursive`, the top of the tree to walk.
- `-o` / `--out PATH` (required): the draft to write; with `--recursive`, the folder of drafts.
- `--name TEXT`: spec name; default `TODO_<last part of PATH>`. Not used with `--recursive`.
- `-r` / `--recursive`: walk PATH and report every dataset, drafting one spec per unknown layout.
- `--depth INTEGER` (default 4): with `--recursive`, how many folder levels below PATH to walk.

```bash
bexa profile sniff /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM -o configs/formats/esrf_id03_bliss_2027.yaml --name esrf_id03_bliss_2027
bexa profile sniff /data/visitor/ma7352/id03/20260928/RAW_DATA -o drafts --recursive
```

### bexa cache: the result cache

Previews are cached on disk, and so are reductions of scans opened through a profile with a
`processed_root` or while `BEXA_CACHE_DIR` is set: one HDF5 file per result, named after a hash
of the source files and parameters, so a repeated call reads the result back. These commands
use `BEXA_CACHE_DIR`, or `~/.cache/bexa` when it is not set. Without `BEXA_CACHE_DIR`, scans
opened with `-p` cache in `<processed_root>/bexa_cache` (when writable) instead; point
`BEXA_CACHE_DIR` there to list or clear that cache.

#### `bexa cache path`
Python: `bexa.cli.main.cache_path`

Prints the cache folder these commands use. No options.

```bash
bexa cache path
```

#### `bexa cache ls`
Python: `bexa.cli.main.cache_ls`

Lists the cached results, one line each with key (a 40-character hash), size in MB and file, or
`cache is empty`. No options.

```bash
BEXA_CACHE_DIR=/data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa/bexa_cache bexa cache ls
```

#### `bexa cache clear`
Python: `bexa.cli.main.cache_clear`

Deletes every cached `.h5` file and the sub-folders left empty, and prints how many files were
removed; the next preview or reduction recomputes. No options.

```bash
bexa cache clear
```

### bexa scan: frame-stack scans

`bexa scan list` lists the scans of a dataset folder; the four other commands open a scan, or a
series of scans, through a profile or a path (`bexa.cli.scan_cmd.open_from_options`) and work
on every layout bexa reads as frames: ESRF BLISS datasets, PAL-XFEL point runs (dims `frame,
y, x`), plain stacks. Shared options:

- `PATH` (argument, optional): dataset folder, master file or `scanNNNN` folder (which opens
  that scan); ignored with `-p`.
- `-p` / `--profile TEXT`: beamtime profile (name or YAML file), with `-s`. `BEXA_PROFILE` is not
  read here: give `-p` even when it is set.
- `-s` / `--sample TEXT`: a sample of the profile.
- `--scan TEXT`: `7`, a range `20-40` (both ends included) or a list `2,3,5`. Several scans open
  as one series on a leading `energy` dim when each has its own energy, else on `scan`. Without
  it a dataset opens its only scan, or all its scans as a series.
- `--detector TEXT`: detector when the layout has several; default from the sample, the profile
  or the spec.

`--roi` takes `dim=low:high` pairs separated by commas, as `bexa.ROI.parse` reads them: `y` and
`x` are full-resolution pixel indices (stop excluded, clipped to the frame), motor dims take
motor values (both ends included, `chi=-0.2:0.2`), and a missing bound runs to the edge
(`x=800:`).

#### `bexa scan list`
Python: `bexa.cli.scan_cmd.scan_list`

Prints the table of a dataset folder (`Dataset.table`, compact by default): one row per scan
with its status (`complete`, `running`, `aborted`, `empty`, `unreadable`), type, motors, shape,
ranges, frames, energy and the positioners that change, then one line per scan that could not
be read and why. A scan still being written never hides the others.

- `PATH` (argument): dataset folder or master file.
- `--status TEXT`: `any` (default), `finished`, `complete`, or one status word: the rows to keep.
- `--full`: every positioner column, not the compact table.

```bash
bexa scan list /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM
bexa scan list /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM --status running
```

#### `bexa scan info`
Python: `bexa.cli.scan_cmd.scan_info`

Prints `scan.info()` from the metadata alone: source and format spec, scan type and title, dims
and shape, frames, points and range of each motor, energy, fixed positioners (`samz`, `ccmth`,
...), status, frame shape and stored dtype, MB per frame and GB for the stack as float32, files,
and the detector geometry.

- `PATH`, `-p` / `--profile`, `-s` / `--sample`, `--scan`, `--detector`: the scan, as above.

```bash
bexa scan info -p hc6293 -s WS2G_100 --scan 7
bexa scan info -p hc6293 -s WS2G_100 --scan 20-40          # dims ('energy', 'mu', 'y', 'x')
bexa scan info /data/visitor/hc6293/id03/20260119/RAW_DATA/WS2G_100/WS2G_100_DFXM/scan0007
```

#### `bexa scan preview`
Python: `bexa.cli.scan_cmd.scan_preview`

Builds the block-averaged preview volume `(motors..., y, x)` in one pass (`Scan.preview`) and
prints its dims and shape. The volume is cached, so the same scan, ROI and factors come back
from disk. Optionally saves it and shows `bexa.viz.volume.projections_panel`.

- `PATH`, `-p` / `--profile`, `-s` / `--sample`, `--scan`, `--detector`: the scan.
- `--downsample TEXT` (default `8`): one integer for both pixel axes (8 x 8 blocks), two factors
  `y,x`, one factor per dim, motors first (`1,1,4,4`; a motor factor keeps every n-th point), or
  factors by name (`mu=2,y=4,x=4`).
- `--roi TEXT`: ROI text.
- `--apply-log` / `--no-apply-log` (default `--no-apply-log`): store log10(1 + I).
- `-o` / `--out PATH`: also save the volume, as zarr for a `.zarr` name, else HDF5 (`bexa.save`).
- `--plot` / `--no-plot` (default `--no-plot`): show the projections in a window and wait until
  it is closed (nothing appears with `--headless`).

```bash
bexa scan preview -p hc6293 -s WS2G_100 --scan 7 --downsample 1,1,4,4 --plot
bexa scan preview -p hc6293 -s WS2G_100 --scan 7 --roi y=700:1100,x=800:1200 --downsample 2 --apply-log -o scan7_preview.h5
```

#### `bexa scan reduce`
Python: `bexa.cli.scan_cmd.scan_reduce`

Runs the chosen accumulators over the scan in one streaming pass with a progress bar
(`Scan.reduce`), saves all results as one dataset with provenance attributes (`bexa.save`) and
prints the variable names and the file. With `-p`, results are also cached when the profile has
a `processed_root` or `BEXA_CACHE_DIR` is set.

- `PATH`, `-p` / `--profile`, `-s` / `--sample`, `--scan`, `--detector`: the scan.
- `--acc TEXT` (default `sum,com`): comma list of accumulators and the variables they give:
  `sum`, `mean`, `max`, `min` the image of that name `(y, x)`; `com` the per-pixel centre of
  mass over every motor dim (`bexa.acc.MotorCOM`): `com_<motor>`, `width_<motor>`, `total`;
  `projections` `xy`, `grid`, `<motor>_y`, `<motor>_x`, `<motor>_<motor>`; `preview` the volume
  `preview`; `stats` `frame_sum`, `frame_mean`, `frame_max`, `frame_com_y`, `frame_com_x` per
  motor point; `argmax` `argmax_<last motor>` (motor value of the brightest frame per pixel) and
  `peak`. An unknown name stops the command with the list of choices.
- `--roi TEXT`: ROI text; the whole pass is restricted to it.
- `--downsample TEXT` (default: none, full resolution): as for `bexa scan preview`, applied to
  every accumulator.
- `--sigma FLOAT` (default 3.0): Gaussian smoothing, in pixels, of every frame before it weights
  the centre of mass; 0 turns it off.
- `-o` / `--out PATH` (required): output file, `.h5` or `.zarr`.

```bash
bexa scan reduce -p hc6293 -s WS2G_100 --scan 7 --acc sum,max,com,projections,stats --roi y=700:1100,x=800:1200 -o scan7_reduced.h5
bexa scan reduce -p hc6293 -s WS2G_100 --scan 7 --acc com --roi y=700:1100,x=800:1200,chi=-0.2:0.2 --sigma 2 --downsample 2 -o scan7_com.zarr
```

#### `bexa scan report`
Python: `bexa.cli.scan_cmd.scan_report`

Writes the standard DFXM figure set and the arrays behind it
(`bexa.pipelines.dfxm_report.report`) from one pass of Sum, Max, Projections, Preview and
MotorCOM over every motor: `<prefix>_sum.png`, `_max.png`, `_projections.png`, `_tiles.png`
(nine preview frames along the first motor), `_com_<motor>.png`, `_width_<motor>.png`,
`_bivariate.png` (two or more motors) and `_results.h5` with every array; prints each name and
file. The prefix is the scan name with `/` as `_` (`hc6293_WS2G_100_WS2G_100_DFXM` through a
profile) and has no scan number, so give each scan its own folder. Scalebars come from the
profile's `geometry.effective_pixel_nm`. Always draws with Agg.

- `PATH`, `-p` / `--profile`, `-s` / `--sample`, `--scan`, `--detector`: the scan.
- `-o` / `--out PATH` (required): output folder, created when missing.
- `--roi TEXT`: ROI text.
- `--downsample TEXT` (default `4`): as for `bexa scan preview`, applied to the maps too.
- `--sigma FLOAT` (default 3.0): smoothing of the centre-of-mass weights.
- `--apply-log` / `--no-apply-log` (default `--no-apply-log`): log10(1 + I) in the preview, so
  in the tiles.

```bash
bexa --headless scan report -p hc6293 -s WS2G_100 --scan 7 -o /data/visitor/hc6293/id03/20260119/PROCESSED_DATA/bexa/scan7
bexa scan report -p hc6293 -s WS2G_100 --scan 7 --roi y=700:1100,x=800:1200 --downsample 2 --sigma 2 -o figures/scan7_roi
```

### bexa cube: laser on/off cubes of XFEL runs

A cube is an `xarray.Dataset` with `frames(laser, <axis>, y, x)` (mean frame per laser state,
`off` and `on`, and motor point), `signal(laser, <axis>)` (mean ROI statistic / I0),
`n_shots(laser, <axis>)`, the motor medians as coordinates and the attributes `scan_axis`, `run`
and `format`. With a profile the cube of run N is `<processed_root>/data/runN.h5` and figures go
to `<processed_root>/outputs`. The commands read bexa cubes, legacy `runN.h5` files and LCLS
cube folders. `--roi` takes the ROI text of the scan commands (`y=400:550,x=200:375`) or the
legacy `r0:r1,c0:c1`, rows then columns (`400:550,200:375`).

#### `bexa cube build`
Python: `bexa.cli.cube_cmd.cube_build`

Reduces every motor point of a PAL-XFEL point-file run to one mean frame per laser state, in
parallel processes (`bexa.pipelines.xfel_cube.reduce_run` with a profile, `build_cube` with a
path): frames cut to the ROI, non-positive pixels set to 1e-4, shots with a NaN laser flag
dropped, the readbacks delay, phi, chi, th, tth, laser_v and laser_h taken as medians. The scan
axis is the first readback of the spec's priority list that varies (`delay` first), else
`index`. Saves the cube (bexa HDF5, zarr for a `.zarr` name, or the legacy layout) and prints its
sizes, axis and file.

- `PATH` (argument, optional): the run's `type=measurement/run=NNN/scan=NNN` folder (or the
  `run=NNN` folder with `--scan`); needs `-o`.
- `-p` / `--profile TEXT`: beamtime profile, with `--run`.
- `--run INTEGER`: run number, with `--profile`.
- `--scan INTEGER` (default 1): scan of the run.
- `--roi TEXT`: ROI in either form; the cube keeps only that window.
- `--workers INTEGER`: processes; default all usable cores but one (the SLURM allocation on a
  cluster); 1 runs in the main process.
- `-o` / `--out PATH`: cube file; default `<processed_root>/data/runN.h5` with a profile.
- `--legacy` / `--no-legacy` (default `--no-legacy`): the `runN.h5` layout of
  `Aaron_allscan_cube_parallel.py` (`delays`, `jungfrau_on`, `jungfrau_off`, `signals_on`, ...).

```bash
bexa cube build -p ue_250913_FXS --run 42 --workers 16
bexa cube build /path/to/ue_250913_FXS/raw/type=measurement/run=042/scan=001 --roi 400:550,200:375 -o run42_peak.h5
```

#### `bexa cube combine`
Python: `bexa.cli.cube_cmd.cube_combine`

Averages the cubes of several runs onto one scan axis (`bexa.analysis.pump_probe.combine_runs`,
replacing `Scan_Combiner.py`): the cubes are joined along the axis, its values rounded to
`--decimals`, and every variable and numeric coordinate averaged over the points sharing a
rounded value. Only what every run has is kept; frames must have the same shape (same ROI).
Saves the result and prints the number of cubes, the axis and the file.

- `FILES...` (arguments, optional): cube files.
- `-p` / `--profile TEXT` with `--runs TEXT`: instead of FILES, `<processed_root>/data/runN.h5`
  of these runs, a range `79-88` (both ends included) or a list `79,80,85`; `--runs` is ignored
  without `-p`.
- `--axis TEXT`: axis to bin on; default the first cube's `scan_axis` (`delays` reads as
  `delay`).
- `--decimals INTEGER` (default 2): rounding of the axis values.
- `-o` / `--out PATH`: output file; with a profile the default is
  `<processed_root>/data/run<first>-<last>Combined.h5`, which `bexa auto` processes like a run;
  required otherwise.

```bash
bexa cube combine -p ue_250913_FXS --runs 79-88 --axis delay
bexa cube combine run79.h5 run80.h5 run85.h5 --decimals 1 -o run79_80_85.h5
```

#### `bexa cube plot`
Python: `bexa.cli.cube_cmd.cube_plot`

Writes the legacy figure set of a cube (`bexa.pipelines.cube_report.figure_set`) as
`<label>_ROI_<r0>_<r1>_<c0>_<c1>_<kind>.png`: `image` (ROI summed over the axis, laser on),
`diffplot` (summed frame with the ROI and background boxes; ROI intensity on, off and on minus
off against the axis), `COMplot` (ROI centre of mass in x and y), `FWHMplot` (ROI variance),
`skewplot` (ROI skewness); prints each kind and file. The background box drawn is y 10:30,
x 10:30; the profile's `background_box` is not used here.

- `CUBE` (argument, optional): cube file or LCLS folder.
- `-p` / `--profile TEXT` with `--run TEXT`: instead of CUBE, `<processed_root>/data/runN.h5`;
  `--run` is a number (`42`) or a file name (`run79-88Combined`).
- `--axis TEXT`: axis to plot against; default the cube's `scan_axis`.
- `--roi TEXT`: ROI in either form; default the whole frame.
- `-o` / `--out PATH`: output folder; default `<processed_root>/outputs` with a profile, else an
  `outputs` folder beside the cube.
- `--label TEXT`: file-name prefix; default `runN` with a profile, else the cube's file name.
- `--show` / `--no-show` (default `--no-show`): meant to open the figures; in this version they
  are closed once saved, so nothing appears.

```bash
bexa --headless cube plot -p ue_250913_FXS --run 42 --roi 400:550,200:375
bexa --headless cube plot run42.h5 --roi y=400:550,x=200:375 --label run42_peak -o outputs/
```

#### `bexa cube animate`
Python: `bexa.cli.cube_cmd.cube_animate`

Writes laser-on and laser-off GIFs of the ROI along the scan axis
(`bexa.pipelines.cube_report.animations`), `<label>_animation_on.gif` and
`<label>_animation_off.gif` (no ROI in the name), and prints each kind and file. Every frame is
divided by the mean of the background box y 0:20, x 0:20 (for laser off a robust mean, with
values above the 90th and below the 10th percentile replaced by the median) and gets its own
colour range.

- `CUBE`, `-p` / `--profile`, `--run`, `--axis`, `--roi`, `-o` / `--out`, `--label`: as for
  `bexa cube plot`.
- `--fps INTEGER` (default 5): frames per second.

```bash
bexa --headless cube animate -p ue_250913_FXS --run 42 --roi 400:550,200:375 --fps 10
```

#### `bexa cube info`
Python: `bexa.cli.cube_cmd.cube_info`

Loads a cube (`bexa.load`) and prints its scan axis, dims, data variables, coordinates and the
attributes `run`, `format`, `bexa_version` and `created` when present (for legacy and LCLS cubes
the last two describe the loading, not the file).

- `CUBE` (argument, required): cube file, or LCLS folder (`RunNNNN_onStk.npy`,
  `RunNNNN_offStk.npy`, `RunNNNN_stats.csv`).

```bash
bexa cube info /path/to/ue_250913_FXS/reduced/data/run42.h5
```

### bexa auto: the beamtime watcher

#### `bexa auto`
Python: `bexa.cli.auto_cmd.register` (the command function `auto` is defined inside it)

Processes every run with raw data but no outputs (`bexa.pipelines.auto_process.run_once`, or
`watch` with `--watch`; the port of `scan_auto_process.py`). Runs are the `run=NNN` folders of
the raw folder plus the `*Combined*.h5` cubes of the cube folder; a run is done when the
outputs folder holds a file starting with `runN_`, and runs on the skip list are left out. For
each run the watcher builds the cube unless `<cube folder>/runN.h5` exists, then writes the
`bexa cube plot` figures for every ROI (the run's ROIs in the ROI table, else the profile's
`rois`, else the whole frame) and the `bexa cube animate` GIFs (their names carry no ROI, so
with several ROIs only the last one's remain). A failing run is retried once, then put on the
skip list. Every run is logged to `<outputs>/logs/bexa_auto.csv` and `<outputs>/logs/runN.json`,
and the ROI table is written back as YAML with the axis used (a CSV table as a `.yaml` beside
it). Prints one line per run (name, status `ok`, `failed` or `pending`, seconds, error) or
`nothing to process`. Always uses Agg.

Folders come from the profile unless given: raw data `<root or raw_root>/type=raw`, cubes
`<processed_root>/data`, outputs `auto_process.outputs` or `<processed_root>/outputs`, ROI table
`auto_process.roi_table` or `<processed_root>/ROI_dict.csv` when it exists, skip list
`auto_process.skip_list` or `<processed_root>/Skip_dict.csv`. Relative output, ROI-table and
skip-list paths, in the profile or the options, are relative to `processed_root`; relative
`--raw-dir` and `--cube-dir` to the current folder. The ROI table is the legacy `ROI_dict.csv`
(a column per run: the axis name, then `r0 r1 c0 c1` quadruples) or YAML
(`run42: {axis: delays, rois: {peak: [400, 550, 200, 375]}}`).

- `-p` / `--profile TEXT` (required): beamtime profile.
- `--raw-dir PATH`: the `type=raw` folder.
- `--cube-dir PATH`: folder of the `runN.h5` cubes.
- `--out-dir PATH`: folder of the figures, GIFs and logs.
- `--roi-table PATH`: `ROI_dict.csv` or a YAML table.
- `--skip-list PATH`: `Skip_dict.csv`.
- `--runs TEXT`: only these runs, `10-20` or `12,15` (combined cubes are then left out).
- `--scan INTEGER` (default 1): scan of each run.
- `--workers INTEGER`: processes of the cube builder; default all usable cores but one.
- `--watch` / `--no-watch` (default `--no-watch`): poll again every `--interval` seconds until
  Ctrl+C. The per-run lines are then never printed; follow the log and `bexa_auto.csv`.
- `--interval FLOAT` (default 60.0): seconds between polls; the profile's
  `auto_process.interval_s` is not read.
- `--dry-run` / `--no-dry-run` (default `--no-dry-run`): only list the pending runs; nothing is
  written.
- `--legacy` / `--no-legacy` (default `--no-legacy`): build new cubes in the legacy layout.
- `--no-animations` / `--no-no-animations` (default: GIFs written): skip the GIFs.

```bash
bexa auto -p ue_250913_FXS --dry-run
bexa auto -p ue_250913_FXS --runs 40-45 --workers 16
bexa auto -p ue_250913_FXS --watch --interval 60
```

### bexa crystal: crystallography

These commands need no data. `--lattice` (Angstrom, deg) is `diamond` (cubic, a = 3.5668),
`silicon` (cubic, a = 5.43102), `cubic:a`, `tetragonal:a,c`, `hexagonal:a,c` (gamma = 120),
`orthorhombic:a,b,c`, `monoclinic:a,b,c,beta`, or numbers alone: `a` (cubic), `a,b,c`
(orthorhombic), `a,b,c,alpha,beta,gamma`. Miller indices are three integers after the option
(`--hkl 2 2 0`, `--hkl -2 0 -2`). `--bravais` (`cubic`, `tetragonal`, `orthorhombic`,
`hexagonal`, `trigonal`, `monoclinic`, `triclinic`) sets the lattice system of the equivalent
reflections; it is guessed from the cell by default. The diffractometer has the sample circles
`mu`, `chi`, `phi`, `omega` and the detector circles `two_theta`, `eta`; `--oop` is the crystal
direction along the lab vertical and `--zone` the one horizontal and perpendicular to the beam,
at zero motor angles. `--convention` is `pal_fourc` (eta 0 vertical, limits -20 to 20 deg) or
`sacla` (eta from the -y axis, anticlockwise looking downstream, 0 to 360 deg); `--eta-offset`
(deg) is subtracted from every computed eta.

#### `bexa crystal dspacing`
Python: `bexa.cli.crystal_cmd.crystal_dspacing`

Prints d (Angstrom), |Q| = 2 pi / d (1/Angstrom) and the lattice system of a reflection, with
`--energy` its Bragg angle theta and 2theta (deg), and the number of equivalent reflections.

- `--lattice TEXT` (required): the lattice.
- `--hkl H K L` (required): the reflection.
- `--energy FLOAT`: energy in keV.
- `--bravais TEXT`: lattice system.

```bash
bexa crystal dspacing --lattice silicon --hkl 1 1 1 --energy 17     # d = 3.13560 A, 2theta = 13.3568 deg
bexa crystal dspacing --lattice hexagonal:3.155,12.35 --hkl 0 0 6 --energy 17
```

#### `bexa crystal solve`
Python: `bexa.cli.crystal_cmd.crystal_solve`

Motor angles that bring a reflection and its equivalents onto the detector
(`bexa.crystal.diffractometer.create_diffractometer`, then `solve_for_peak_family`). Prints the
motor limits, then a table (hkl, sample angles, 2theta, eta, d) with the fewest moved motors
first, then the least travel; `*` marks a motor moved from its default (the fixed value, the
middle of a range, 0 for a free motor). Default limits: mu -10 to 90, chi -5 to 95, phi and
omega -180 to 180, two_theta 0 to 170, eta by convention. Takes about 15 s.

- `--lattice TEXT` (required), `--energy FLOAT` (required, keV), `--hkl H K L` (required).
- `--oop H K L` (required): out-of-plane direction.
- `--zone H K L` (required): zone axis.
- `--bravais TEXT`: lattice system.
- `--convention TEXT` (default `pal_fourc`): `pal_fourc` or `sacla`.
- `--eta-offset FLOAT` (default 0.0): eta zero offset in deg.
- `--range NAME=LO:HI` (repeatable): limit a motor, e.g. `--range mu=-45:-10`.
- `--fixed NAME=VALUE` (repeatable): fix a motor, e.g. `--fixed omega=35.5`.
- `--no-equivalents` / `--no-no-equivalents` (default: equivalents solved): only the hkl given.
- `--max-display INTEGER` (default 20): rows shown; `... and N more` for the rest.

```bash
bexa crystal solve --lattice diamond --energy 11.1 --hkl 2 2 0 --oop 1 0 0 --zone 0 1 1
bexa crystal solve --lattice diamond --energy 11.1 --hkl 2 2 0 --oop 1 0 0 --zone 0 1 1 --convention sacla --eta-offset -180 --range mu=-45:-10 --fixed omega=35.5
```

#### `bexa crystal zone-axis`
Python: `bexa.cli.crystal_cmd.crystal_zone_axis`

Zone axes consistent with one indexed reflection seen at known angles
(`bexa.crystal.zone_axis.solve_zone_axis`). Every integer direction up to `--search-range`
(except those along `--oop`) is tried; each equivalent of `--hkl`, placed with the sample
angles, is compared with the Q measured from two_theta and eta, the residual being the smallest
|Q_predicted - Q_observed| (1/Angstrom). Axes within ten times the best residual (at least 1)
are kept, largest axis d-spacing first, then smallest residual; the table shows rank, zone axis,
d, residual and matched reflection, or `No valid zone axes found.`. With `--tolerances` only
axes whose angles, within those tolerances, reproduce the reflection remain, with the fitted
angles printed.

- `--lattice TEXT` (required), `--energy FLOAT` (required, keV).
- `--hkl H K L` (required): the observed reflection.
- `--oop H K L` (required): out-of-plane direction.
- `--angles TEXT` (required): `mu=..,chi=..,phi=..,omega=..,two_theta=..,eta=..` in deg;
  `two_theta` and `eta` are needed, missing sample angles are 0.
- `--tolerances TEXT`: the same form, the allowed change of each angle in deg (180 when
  missing; two_theta is only checked).
- `--bravais TEXT`; `--convention TEXT` (default `pal_fourc`); `--eta-offset FLOAT` (default 0.0).
- `--search-range INTEGER` (default 10): largest index of the trial axes.
- `--top INTEGER` (default 5): axes shown.

```bash
bexa crystal zone-axis --lattice cubic:3.56 --energy 11 --hkl -2 0 -2 --oop 1 1 0 --angles mu=-38,chi=0,phi=0,omega=0,two_theta=53.2,eta=-67.5 --eta-offset -90
bexa crystal zone-axis --lattice cubic:3.56 --energy 11 --hkl -2 0 -2 --oop 1 1 0 --angles mu=-38,chi=0,phi=0,omega=0,two_theta=53.2,eta=-67.5 --eta-offset -90 --tolerances mu=10,chi=10,phi=0,omega=10,two_theta=10,eta=10 --top 3
```

#### `bexa crystal omega-search`
Python: `bexa.cli.crystal_cmd.crystal_omega_search`

Scans omega with phi = chi = 0 for the settings where two reflections of a family come closest
in mu while mu, eta and two_theta stay in their windows, so that one mu scan reaches both
(`bexa.crystal.omega_search.find_omega_offsets`, the port of `omega_finder.py`). Prints a row per
candidate (rank, omega offset, omega, dmu in deg, both reflections with mu and eta, the eta
difference and, with `--verify`, the full solver's smallest gap and number of solutions as
`gap/n`), then `Recommended: omega_offset = ...`, or `No candidate omega found.`. Windows
default to mu -45 to -10, eta 0 to 360 and two_theta 0 to 170.

- `--lattice TEXT` (required), `--energy FLOAT` (required, keV).
- `--family H K L` (required): the reflection family.
- `--oop H K L` (required), `--zone H K L` (required).
- `--range NAME=LO:HI` (repeatable): window of `mu`, `eta` or `two_theta`, also a motor limit.
- `--bravais TEXT`; `--convention TEXT` (default `sacla`, unlike `solve`); `--eta-offset FLOAT`
  (default 0.0).
- `--omega-scan LO:HI` (default `-180:180`): omega range in deg.
- `--step FLOAT` (default 0.01): omega step in deg; minima are refined on a finer grid.
- `--top INTEGER` (default 5): candidates shown, smallest gap first (omegas within 0.3 deg count
  once).
- `--verify` / `--no-verify` (default `--no-verify`): re-solve each candidate with the full
  solver at its omega (slow).

```bash
bexa crystal omega-search --lattice diamond --energy 11.1 --family 3 1 1 --oop 1 1 0 --zone 0 0 1 --eta-offset -180 --range mu=-45:-10 --range eta=35:100 --range two_theta=20:70
```

#### `bexa crystal reflections`
Python: `bexa.cli.crystal_cmd.crystal_reflections`

Lists every reflection with indices from -N to N (N = `--max-hkl`) that the centering allows
and Bragg's law reaches at the energy, largest d first, with d (Angstrom) and 2theta (deg); each
signed reflection gets a line. Only centering conditions apply, not glide or screw extinctions:
diamond with `--centering F` still lists 200.

- `--lattice TEXT` (required), `--energy FLOAT` (required, keV).
- `--max-hkl INTEGER` (default 3): largest absolute index.
- `--centering TEXT` (default `P`, no extinctions): `P`, `I`, `F`, `A`, `B`, `C` or `R`.
- `--bravais TEXT`: accepted as elsewhere; it does not change the list.

```bash
bexa crystal reflections --lattice diamond --energy 11.1 --max-hkl 3 --centering F
```

### bexa optics: X-ray optics estimates

#### `bexa optics convert`
Python: `bexa.cli.optics_cmd.optics_convert`

For an energy or a wavelength, prints the energy (keV), wavelength (Angstrom), wavevector
k = 2 pi / lambda (1/Angstrom) and the Bragg angle (deg) of the monochromator reflections Si111,
Si220, Si311, Si333, Ge111, Ge220 and C111, each left out below its cutoff
(`bexa.optics.conversions.summary`). With both options the energy is used; with neither the
command stops with `ValueError`.

- `--energy FLOAT`: energy in keV.
- `--wavelength FLOAT`: wavelength in Angstrom.

```bash
bexa optics convert --energy 17            # wavelength_A: 0.729319, theta_Si111_deg: 6.67841, ...
bexa optics convert --wavelength 0.73
```

#### `bexa optics transmission`
Python: `bexa.cli.optics_cmd.optics_transmission`

Transmission exp(-mu t) through a thickness of a material and the 1/e attenuation length, at one
or several energies (`bexa.optics.attenuation.transmission` and `attenuation_length_um`; xraydb
Elam tables, needs xraydb). Prints material, formula, density and thickness, then per energy the
transmission in percent and the 1/e length in um.

- `--material TEXT` (required): an element (`Si`), a known name (`diamond`, `glassy carbon`,
  `LuAG`, `YAG`, `GAGG`, `LYSO`, `kapton`, `mylar`, `water`, `air`, `sapphire`, `quartz`,
  `WTe2`, `WS2`, `MoS2`; any case), a material of xraydb's table, or a formula with `--density`.
- `--thickness TEXT` (required): a number with unit `nm`, `um`, `mm`, `cm` or `m` (`500um`,
  `0.3mm`, `2cm`); a bare number is um.
- `--energy TEXT` (required): keV, one value `17`, a list `9.7,17`, or a range `lo:hi:step`
  including `hi` (`10:30:5`; step 1 when left out).
- `--density FLOAT`: g/cm3, replacing the table value.

```bash
bexa optics transmission --material Si --thickness 500um --energy 10:30:5   # 1.9346 % at 10 keV ... 84.5966 % at 30 keV
bexa optics transmission --material Fe2O3 --density 5.24 --thickness 2um --energy 17
```

#### `bexa optics photons`
Python: `bexa.cli.optics_cmd.optics_photons`

Photons in one pulse, pulse energy over photon energy
(`bexa.optics.conversions.photons_from_pulse_energy`).

- `--pulse-energy FLOAT` (required): pulse energy in J.
- `--energy FLOAT` (required): photon energy in keV.

```bash
bexa optics photons --pulse-energy 5e-6 --energy 16      # 5e-06 J at 16 keV = 1.950e+09 photons
```

#### `bexa optics magnifier`
Python: `bexa.cli.optics_cmd.optics_magnifier`

Best division of the magnification M between the two asymmetric silicon 220 crystals of one
Bragg magnifier pair (`bexa.optics.bragg_magnifier.optimal_split`, the port of
`Bragg_Mag_Efficiency_Calcs.py`): split ratios s from 0 to 1 are tried (M1 = M^s,
M2 = M^(1-s)), each rated from dynamical rocking curves integrated over the divergence and the
energy spread. Prints the best split with M1 and M2, the efficiency in percent, the angular
offset between the crystals in arcsec and the two miscuts in deg (`miscuts_from_split`).

- `--mag FLOAT` (required): total magnification of the pair.
- `--energy FLOAT` (default 9251.0): photon energy in eV (not keV).
- `--de FLOAT` (default 0.5): energy spread FWHM in eV.
- `--divergence FLOAT` (default 1.5): beam divergence sigma in urad.
- `--splits INTEGER` (default 25): split ratios tried.
- `--n-energies INTEGER` (default 11): energies sampled across the spread.
- `--points INTEGER` (default 2001): points of the rocking-angle grid, -100 to 100 arcsec.

```bash
bexa optics magnifier --mag 20         # best split 0.333 (M1 = 2.71, M2 = 7.37), efficiency 0.3221 %
```

### bexa bench: timings and regressions

#### `bexa bench`
Python: `bexa.cli.bench_cmd.register` (the command function `bench` is defined inside it)

Times a typical session on a synthetic ESRF scan in a temporary folder
(`bexa.pipelines.benchmark.run_suite`): metadata (`info_s`), a full read (`read_s`), a preview
(`preview_s`), Sum, Max and MotorCOM in one pass (`reduce_s`) and the DFXM report on the CPU
(`report_s`), plus read MB/s, frames/s, compute-to-read ratio, peak RAM and GPU memory (MB).
Then it compares with `<baselines>/<host>-<device>.json`, lists the timings slower by more than
the tolerance, and exits with code 1 on a regression (unless `--save`). The quick set is 48
frames of 48 x 64 (seconds); `--full` is 32 frames of 2160 x 2560 (about 15 s, 6 GB of RAM).
The baseline name uses the device bexa picks itself, not `--device`, and one file serves quick
and full runs, so compare like with like.

- `--full` / `--no-full` (default `--no-full`): realistic frame size.
- `--device TEXT` (default `auto`): device of the preview and the reductions.
- `--real PATH`: also time a real frame stack (`.npy`, tiff, `file.h5::/path`), read with the
  generic reader.
- `--baselines PATH` (default `benchmarks/baselines`, relative to the current folder): folder of
  the baseline JSON files; run from the checkout or give the path.
- `--save` / `--no-save` (default `--no-save`): store this run as the machine's baseline (inside
  the checkout by default).
- `--tolerance FLOAT` (default 0.2): allowed slowdown as a fraction.
- `--no-plots` / `--no-no-plots` (default: the report is timed): skip the report timing.

```bash
bexa --headless bench
bexa --headless bench --full --no-plots --real scan7_frames.npy --baselines ~/bexa/benchmarks/baselines
```

### Python helpers of the command line

The command functions take Typer option objects as defaults, so run them through the
application: `bexa.cli.main.app` is the Typer application, with the groups
`bexa.cli.main.profile_app` and `cache_app`, `bexa.cli.scan_cmd.scan_app`,
`bexa.cli.cube_cmd.cube_app`, `bexa.cli.crystal_cmd.crystal_app` and
`bexa.cli.optics_cmd.optics_app` attached at import. `CliRunner().invoke(app, [...])` from
`typer.testing` runs a command from Python and returns its exit code and output, as the tests
do. The parsers below turn command-line text into objects of the Python API.

```python
from typer.testing import CliRunner
from bexa.cli.main import app
result = CliRunner().invoke(app, ["crystal", "dspacing", "--lattice", "silicon", "--hkl", "1", "1", "1"])
result.exit_code, result.output      # 0, '(1, 1, 1): d = 3.13560 A, |Q| = 2.00382 1/A, cubic\n  8 equivalent reflections\n'
```

#### `parse_roi`
`bexa.cli.scan_cmd.parse_roi(text)`

ROI text to a `bexa.ROI` through `ROI.parse` (`y`, `x` as integer pixel indices, motor dims as
floats, a missing bound open); empty text or `None` gives `None`, a part without `=` or `:`
raises `ValueError`.

```python
from bexa.cli.scan_cmd import parse_roi
roi = parse_roi("y=700:1100,x=800:1200,chi=-0.2:0.2")   # ROI(y=(700, 1100), x=(800, 1200), chi=(-0.2, 0.2))
roi.get("chi"), parse_roi("")                          # (-0.2, 0.2), None
```

#### `parse_downsample`
`bexa.cli.scan_cmd.parse_downsample(text)`

`--downsample` text to what `Scan.preview` and `Scan.reduce` take: one number gives an int, a
comma list a tuple (`y,x`, or one factor per dim, motors first), `name=factor` pairs a dict by
name; empty text or `None` gives `None` (full resolution).

```python
from bexa.cli.scan_cmd import parse_downsample
parse_downsample("4"), parse_downsample("4,4"), parse_downsample("1,1,4,4")   # 4, (4, 4), (1, 1, 4, 4)
parse_downsample("mu=2,y=4,x=4")                 # {'mu': 2, 'y': 4, 'x': 4}
```

#### `parse_scan`
`bexa.cli.scan_cmd.parse_scan(text)`

`--scan` text to the `scan=` argument of `bexa.open`: `"7"` gives 7, `"20-40"` the range
`(20, 40)` (both ends included when opened), `"2,3,5"` the list `[2, 3, 5]`; empty text or
`None` gives `None`.

```python
from bexa.cli.scan_cmd import parse_scan
parse_scan("7"), parse_scan("20-40"), parse_scan("2,3,5")   # 7, (20, 40), [2, 3, 5]
```

#### `open_from_options`
`bexa.cli.scan_cmd.open_from_options(path, profile, sample, scan, detector)`

Opens the scan the `bexa scan` options describe and returns a `bexa.Scan` (a context manager
that closes its files). With `profile` it calls `bexa.core.scan.open_profile` with the sample,
`parse_scan(scan)` and the detector, and ignores `path`; otherwise
`bexa.core.scan.open(path, scan=..., detector=...)`. Raises `typer.BadParameter` without either.

```python
from bexa.cli.scan_cmd import open_from_options
with open_from_options(None, "hc6293", "WS2G_100", "7", None) as scan:
    print(scan.dims)                                   # the motor dims, then ('y', 'x')
```

#### `parse_cube_roi`
`bexa.cli.cube_cmd.parse_cube_roi(text)`

The `--roi` of the cube commands: text with `=` goes to `parse_roi`, the legacy
`r0:r1,c0:c1` (rows, then columns) becomes `ROI(y=(r0, r1), x=(c0, c1))`; empty text or `None`
gives `None`.

```python
from bexa.cli.cube_cmd import parse_cube_roi
parse_cube_roi("400:550,200:375").get("x"), parse_cube_roi("y=400:550,x=200:375").get("y")   # (200, 375), (400, 550)
```

#### `parse_runs`
`bexa.cli.cube_cmd.parse_runs(text)`

Run numbers from a range `"79-88"` (both ends included) or a list `"79,80,85"`, as
`bexa cube combine --runs` and `bexa auto --runs` read them.

```python
from bexa.cli.cube_cmd import parse_runs
parse_runs("79-88"), parse_runs("79,80,85")   # [79, 80, 81, 82, 83, 84, 85, 86, 87, 88], [79, 80, 85]
```

#### `parse_lattice`
`bexa.cli.crystal_cmd.parse_lattice(text)`

A `bexa.crystal.lattice.LatticeParameters` (Angstrom, deg) from a `--lattice` form: `diamond`,
`silicon`, `kind:numbers` with `kind` a constructor of `LatticeParameters` (`cubic`,
`tetragonal`, `hexagonal`, `orthorhombic`, `monoclinic`), or 1, 3 or 6 bare numbers (cubic,
orthorhombic, full cell). Other counts raise `typer.BadParameter` quoting `LATTICE_HELP`.

```python
from bexa.cli.crystal_cmd import parse_lattice
parse_lattice("diamond")        # LatticeParameters(a=3.5668, b=3.5668, c=3.5668, alpha=90.0, beta=90.0, gamma=90.0)
parse_lattice("hexagonal:3.155,12.35").gamma, parse_lattice("monoclinic:5.1,5.2,5.3,99").beta   # 120.0, 99.0
```

#### `parse_ranges`
`bexa.cli.crystal_cmd.parse_ranges(items)`

The repeated `--range` values, `name=lo:hi` strings, to a dict of name to `(lo, hi)` floats;
`None` gives an empty dict.

```python
from bexa.cli.crystal_cmd import parse_ranges
parse_ranges(["mu=-45:-10", "eta=35:100"])             # {'mu': (-45.0, -10.0), 'eta': (35.0, 100.0)}
```

#### `parse_angles`
`bexa.cli.crystal_cmd.parse_angles(text)`

Text `name=value,...` (the `--angles` and `--tolerances` values, and the joined `--fixed`
values) to a dict of floats; `None` or empty text gives an empty dict.

```python
from bexa.cli.crystal_cmd import parse_angles
parse_angles("mu=-38,chi=0,two_theta=53.2,eta=-67.5")  # {'mu': -38.0, 'chi': 0.0, 'two_theta': 53.2, 'eta': -67.5}
```

#### `parse_energies`
`bexa.cli.optics_cmd.parse_energies(text)`

The `--energy` of `bexa optics transmission` as a numpy array in keV: one value `"17"`, a list
`"9.7,17"`, or a range `"lo:hi:step"` that includes `hi` (step 1 when left out).

```python
from bexa.cli.optics_cmd import parse_energies
parse_energies("10:30:5"), parse_energies("9.7,17")   # array([10., 15., 20., 25., 30.]), array([ 9.7, 17. ])
```

#### `auto_cmd.register`
`bexa.cli.auto_cmd.register(app)`

Adds the `auto` command to a Typer application and returns `None`; `bexa.cli.main` calls it on
its `app` at import.

```python
import typer
from bexa.cli import auto_cmd
app = typer.Typer()
auto_cmd.register(app)                                 # app now has the "auto" command
```

#### `bench_cmd.register`
`bexa.cli.bench_cmd.register(app)`

Adds the `bench` command to a Typer application in the same way and returns `None`.

```python
from bexa.cli import bench_cmd
bench_cmd.register(app)
sorted(c.name for c in app.registered_commands)        # ['auto', 'bench']
```

#### `CliState`
`bexa.cli.main.CliState(device="auto", memory_fraction=None, headless=False)`

Dataclass of the global options that the top-level callback stores on the Typer context
(`ctx.obj`): `device`, `memory_fraction` (`None` when not given) and `headless`. A command with a
`ctx: typer.Context` parameter reads them as `ctx.obj.device`; the built-in commands use the
environment variables instead.

```python
from bexa.cli.main import CliState
CliState(device="cpu", headless=True)                  # CliState(device='cpu', memory_fraction=None, headless=True)
```

Help texts shared by several options:

- `bexa.cli.crystal_cmd.LATTICE_HELP`: help of every `--lattice`, also quoted by `parse_lattice`
  errors: `"diamond | silicon | cubic:3.56 | tetragonal:a,c | hexagonal:a,c | orthorhombic:a,b,c | a,b,c,alpha,beta,gamma"`.
- `bexa.cli.crystal_cmd.HKL_HELP`: help of `--hkl`: `"Three Miller indices, e.g. --hkl 3 1 1"`.
- `bexa.cli.cube_cmd.ROI_HELP`: help of the cube `--roi`:
  `"y=400:550,x=200:375 or the legacy r0:r1,c0:c1 (rows then columns)."`.
- `bexa.cli.scan_cmd.PATH_HELP`: help of the scan `PATH`:
  `"Dataset folder, master file or scanNNNN folder (or use --profile/--sample)."`.

```python
from bexa.cli.crystal_cmd import HKL_HELP, LATTICE_HELP
from bexa.cli.cube_cmd import ROI_HELP
from bexa.cli.scan_cmd import PATH_HELP
print(LATTICE_HELP)            # diamond | silicon | cubic:3.56 | tetragonal:a,c | hexagonal:a,c | ...
```
