# Neural Screened Diffusion on Surfaces

A neural field `u(x)` that solves a screened diffusion (screened Poisson) problem
with soft Dirichlet sources on a triangle-mesh surface, without a surface
parameterization: the network is defined in a narrow band around the surface and
all surface differential quantities come from a sparse signed distance field
(SDF). A cotangent-Laplacian FEM solver for the same problem is included as the
reference for quantitative evaluation.

Each experiment is one bash script in the repository root:

| Script | Experiment |
|---|---|
| `run_single.sh` | optimization on a mesh for **given sources and Dirichlet values**, compared with FEM |
| `run_generalization.sh` | training of one network that **generalizes over the Dirichlet values** of the sources |
| `run_evaluation.sh` | **quantitative evaluation** of that generalization against FEM |
| `run_sdf_sweep.sh` | **SDF-resolution study** on the fandisk at two mesh resolutions: several SDF resolutions with their voxel spacing |
| `run_thin_star.sh` | single training on a thin star-shaped plate |

## Problem

On a closed surface `M` with `m` sources `S_k` (points or curves) carrying target
values `a_k`, the field minimizes

```
E(u) = lambda_dir  * mean_M |grad_M u|^2          (smoothness)
     + lambda_mass * mean_M u^2                    (screening)
     + (1/m) * sum_k w_k * mean_{S_k} (u - a_k)^2  (sources)
```

`mean_M` is the area average and `w_k` the weight of source `k`. The sources are
soft constraints: with `w_k = 1` the field is pulled towards `a_k` but does not
reach it; raise the weights (`LAMBDA_DATA` in the scripts) for harder
enforcement. The screening length is of
the order of `sqrt(lambda_dir / lambda_mass)` in normalized units.

* **Network** (`Surfacescreendiffusion.py`): a sine MLP `u(x, n(x), g)` taking a
  positional encoding of `x`, the SDF normal `n(x)` and a code `g` (the
  Dirichlet values when they are generalized). The loss is a Monte-Carlo
  estimate of `E` plus a closest-point regularizer weighted by `lambda_cpm`,
  which keeps `u` constant along the normal inside the band.
* **FEM reference** (`fem_screen.py`): the linear system
  `(lambda_dir K + lambda_mass M + (|M|/m) sum_k w_k B_k^T W_k B_k) u = rhs`
  with the cotangent stiffness matrix `K` and the lumped mass matrix `M`.

## Installation

Training needs **Linux and an NVIDIA GPU**: the sparse SDF grid is built with
[fVDB](https://github.com/openvdb/fvdb-core), which has no CPU or Windows build.
The code was developed with Python 3.10.

```bash
conda create -n surface_screened python=3.10 -y
conda activate surface_screened
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `torch` (CUDA), `fvdb-core`, `fcpw` | training, and every script that rebuilds the SDF |
| `numpy`, `scipy` | everywhere |
| `libigl` | FEM reference (`fem_screen.py`) and the evaluation |
| `matplotlib`, optionally `plotly` + `kaleido` | figures; Matplotlib is used when plotly is missing or cannot export |
| `polyscope` (optional) | interactive viewers, on a machine with a display |

Notes:

* **fVDB and PyTorch must match.** fVDB wheels are built for one PyTorch/CUDA
  pair (the tag `+pt210.cu128` means PyTorch 2.10.0 with CUDA 12.8).
  `requirements.txt` lists one such pair; change both lines together for another
  CUDA version, following the
  [fVDB installation guide](https://fvdb-core.readthedocs.io/latest/installation.html).
* **fVDB version.** `SparseGridFDM_SDF.py` uses the `GridBatch` API of
  `fvdb-core` 0.3 - 0.4. Later releases changed the `GridBatch` constructor; see
  the comment at the `from_mesh` call in that file.
* **fcpw runs on the CPU.** Every SDF build prints
  `[fcpw] GPU path failed ... falling back to CPU scene`. This is expected with
  the `pip` wheel and is not an error.
* The FEM solver, the metric and split scripts and the plots also run on a
  machine without a GPU.

## Running the experiments

Every script has a `parameters` block at the top (mesh, sources, SDF
resolution, epochs, loss weights, output folder) followed by the Python
commands it runs. Edit the block, then run the script from the repository root:

```bash
bash run_single.sh
CUDA_VISIBLE_DEVICES=1 bash run_single.sh        # choose the GPU
```

Results are written below `results/`.

`LAMBDA_DATA` in the parameters block is the weight `w_k` of the sources. Each
script writes a copy of the source specification in which every source has
this weight (`constraints_lambda_data.json` in the output folder) and gives
that copy to both the network and FEM.

### `run_single.sh`: given sources and Dirichlet values

Trains the network for one mesh and one source specification, solves the same
problem with FEM and compares the two fields at the mesh vertices. As shipped it
runs the fandisk; the parameters block lists the settings for the other meshes
in `data/`. To use your own mesh, set `MESH` and `SOURCES` (see
[Source specification](#source-specification)).

```bash
bash run_single.sh
SDF_RES=300 FD_HALO=8 bash run_single.sh     # another SDF resolution
SKIP_TRAIN=1 bash run_single.sh              # only FEM + error
```

Output in `results/single/<name>_res<SDF_RES>_halo<FD_HALO>/`:

| File | Content |
|---|---|
| `constraints_lambda_data.json` | the sources with the weight `LAMBDA_DATA` |
| `model.pt` | network checkpoint |
| `diffusion.npz` | network field at the mesh vertices (`u_vertex`), the normalized mesh, the sources |
| `fem_diffusion.npz` | FEM field, same layout |
| `nn_vs_fem_mse.txt` | error of the network, see below |

With `e = u_nn - u_fem` over the mesh vertices:

| Metric | Definition |
|---|---|
| `mse`, `rmse` | `mean(e^2)` and its square root |
| `rel_l2` | `sqrt(sum(e^2) / sum(u_fem^2))` |
| `mean_abs_error`, `max_abs_error` | `mean(abs(e))`, `max(abs(e))` |

`run_thin_star.sh` is the same experiment with the parameters of the thin star
(`data/99276_thin_star.obj`, one curve source). The plate is thin, so its two
sides are close in space but far apart along the surface.

### `run_generalization.sh`: generalization over the Dirichlet values

The sources keep their geometry; the value of every source listed in
`SOURCE_INDICES` becomes an input of the network. At each epoch one vector of
values is drawn uniformly from `[G_MIN, G_MAX]`, so after training the network
returns the solution for any values in that range without retraining. As
shipped it trains the `dc` model (22,638 vertices; three geodesic circles and
two polylines as sources, values in `[-3, 2]`, 30,000 epochs); the parameters
block gives the settings for the bunny.

Output in `results/generalization/dc/`: `constraints_lambda_data.json`,
`model.pt`, `diffusion.npz` (the field for the mid-range values) and
`train_codes.txt` (the values used at every epoch).

### `run_evaluation.sh`: quantitative evaluation

Evaluates the network trained by `run_generalization.sh` for many vectors of
Dirichlet values, each against its own FEM solve
(`evaluate_dirichlet_generalization.py`). Three sets of vectors are built by
`prepare_dirichlet_splits.py`:

* `train` - vectors that were used during training,
* `interp` - unseen vectors inside the training range `[G_MIN, G_MAX]`,
* `extrap` - vectors from the larger range `[EXTRAP_MIN, EXTRAP_MAX]` with at
  least one value outside the training range.

The first lines of its parameters block must be the same as in
`run_generalization.sh`: the SDF normal is an input of the network and the SDF
settings are not stored in the checkpoint. The sources are read from the
`constraints_lambda_data.json` that the training wrote. Output in
`results/generalization/dc/eval/`:

| File | Content |
|---|---|
| `codes_<set>.txt` | the vectors of a set |
| `metrics_<set>.txt` | `mse` and `rel_l2` per vector; minimum, mean and maximum in the header |
| `mse_<set>.png`, `rel_l2_<set>.png` | per-vector errors |

Every vector costs one FEM assembly and factorization, so the run time grows
linearly with `NUM_TRAIN`, `NUM_INTERP` and `NUM_EXTRAP`.

`export_split_samples.py` and `export_worst_rel_l2.py` save the network and FEM
fields for a few vectors of each set, for figures; their docstrings show the
commands.

### `run_sdf_sweep.sh`: SDF-resolution study on the fandisk

The fandisk is provided at two mesh resolutions with the same shape and the same
normalized frame: `data/fandisk.obj` (7,229 vertices) and `data/fandisk_5w.obj`
(28,617 vertices). For each mesh the script solves the FEM reference once and
then, for each `SDF_RES,FD_HALO` pair, trains the network on that SDF grid and
compares it with FEM.

| Mesh | `SDF_RES,FD_HALO` pairs |
|---|---|
| `fandisk.obj` | 20,3 30,3 40,3 50,3 60,3 70,4 90,4 100,4 150,5 200,6 250,7 300,8 |
| `fandisk_5w.obj` | 20,3 35,3 50,3 200,6 250,7 300,8 400,2 |

Output in `results/sdf_sweep/fandisk/` and `results/sdf_sweep/fandisk_5w/`:

| File | Content |
|---|---|
| `res<SDF_RES>_halo<FD_HALO>/` | checkpoint, field and error of one pair |
| `summary.txt` | one block per pair: `SDF_RES,FD_HALO`, the voxel spacing of the SDF grid and the five error metrics |
| `summary_rel_l2.png` | `rel_l2` against `SDF_RES` |

A pair whose folder already holds a trained field is not trained again, so an
interrupted study continues where it stopped.

### Short test run

To check an installation without waiting for full runs, set `EPOCHS` to a few
hundred in the parameters block (and, for `run_sdf_sweep.sh`, keep one or two
pairs per mesh). The errors of such runs are large; they only show that every
step works.

## SDF resolution and voxel spacing

The SDF is stored on a sparse lattice around the surface
(`SparseGridFDM_SDF.py`). Three settings define it:

* `SDF_RES` - number of lattice nodes along the longest axis of the padded
  bounding box. The voxel spacing is
  `voxel_size = (1 + 2 * SDF_PADDING) * L_max / (SDF_RES - 1)`,
  where `L_max` is the longest bounding-box edge of the normalized mesh.
* `SDF_PADDING` - padding of the bounding box as a fraction of `L_max`
  (0.15 in all experiments). The grid is sparse, so the padding adds no voxels;
  it only enters the spacing.
* `FD_HALO` - number of voxel layers kept on every side of the voxels that
  intersect the surface (at least 2). The band therefore reaches about
  `FD_HALO * voxel_size` from the surface, and queries outside the band return
  0. The narrow-band samples of the training lie at a distance of the order of
  `BAND_WIDTH` (0.01 in all experiments) from the surface, so when `SDF_RES` is
  raised, raise `FD_HALO` with it to keep a band of comparable width.

`sdf_spacing.py` evaluates the formula for a mesh without building the grid:

```bash
python sdf_spacing.py data/fandisk.obj --pairs 100,4 200,6 300,8
```

For both fandisk meshes `L_max = 1.377`, so a given `SDF_RES` gives the same
spacing on both; what differs is the spacing relative to the mesh edges (mean
edge length 0.0261 for `fandisk.obj`, 0.0131 for `fandisk_5w.obj`, in normalized
units).

| `SDF_RES` | `FD_HALO` | voxel size | band (`FD_HALO` x voxel) | voxel / edge, `fandisk` | voxel / edge, `fandisk_5w` |
|---:|---:|---:|---:|---:|---:|
| 20 | 3 | 0.0942 | 0.283 | 3.61 | 7.19 |
| 30 | 3 | 0.0617 | 0.185 | 2.37 | |
| 35 | 3 | 0.0527 | 0.158 | | 4.02 |
| 40 | 3 | 0.0459 | 0.138 | 1.76 | |
| 50 | 3 | 0.0365 | 0.110 | 1.40 | 2.79 |
| 60 | 3 | 0.0303 | 0.091 | 1.16 | |
| 70 | 4 | 0.0259 | 0.104 | 0.99 | |
| 90 | 4 | 0.0201 | 0.080 | 0.77 | |
| 100 | 4 | 0.0181 | 0.072 | 0.69 | |
| 150 | 5 | 0.0120 | 0.060 | 0.46 | |
| 200 | 6 | 0.00900 | 0.054 | 0.34 | 0.69 |
| 250 | 7 | 0.00719 | 0.050 | 0.28 | 0.55 |
| 300 | 8 | 0.00599 | 0.048 | 0.23 | 0.46 |
| 400 | 2 | 0.00449 | 0.009 | | 0.34 |

The SDF settings are not stored in `model.pt`. Any script that evaluates a
checkpoint must be given the `--sdf-res`, `--sdf-padding` and `--fd-halo` of the
training run, because the SDF normal is an input of the network.

## Source specification

A source specification is a JSON list; `data/*.json` are examples. Every source
has a `type`, a target `value` and a `weight` (default 1).

| `type` | Fields | Source |
|---|---|---|
| `vertex` | `ids` (list of 0-based vertex indices), `patch_radius` (default 0) | the listed vertices, plus all vertices within `patch_radius` of them |
| `disk_curve` | `seed` (vertex index), `radius` | the closed curve at distance `radius` from the seed vertex |
| `curve` | `points` (list of `[x, y, z]`), `closed` (default false) | a polyline, projected onto the surface |

```json
[
  {"type": "vertex", "ids": [1700], "value": 2.0, "weight": 1.0, "patch_radius": 0.02},
  {"type": "disk_curve", "seed": 20628, "radius": 0.2, "value": -1.0, "weight": 1.0}
]
```

`make_screen_sources.py` writes such a file from the command line:

```bash
python make_screen_sources.py -o data/my_sources.json \
    --vertex-ids 1101 --vertex-value 1.0 \
    --vertex-ids 1589 --vertex-value -1.0
```

Conventions:

* **Normalized coordinates.** Every mesh is translated so that its bounding-box
  center is at the origin and scaled so that its farthest vertex has norm 1.
  Curve points, radii, `BAND_WIDTH` and voxel sizes are in these units.
* **Distances along the surface.** `radius` and `patch_radius` are distances in
  the edge graph of the mesh in which every interior edge is counted twice, so
  on a closed mesh they correspond to twice the edge-path length. The network
  and the FEM reference use the same convention. With `patch_radius: 0.02`, as
  in the provided vertex specifications, each source is a single vertex.
* **Generalized values.** Which sources have a generalized value is chosen with
  `SOURCE_INDICES` in `run_generalization.sh` and `run_evaluation.sh`
  (zero-based indices into the list, ascending), not in the JSON file.

## Reference results

Errors of the network against the FEM reference at the mesh vertices, from the
runs of the authors. Training is stochastic, and the error does not decrease
monotonically with the SDF resolution once the grid is fine enough, so expect
numbers of the same size rather than identical digits.

**Given sources and Dirichlet values** (`run_single.sh`)

| Mesh | `SDF_RES` | `FD_HALO` | Epochs | RMSE | rel_l2 | max abs error |
|---|---:|---:|---:|---:|---:|---:|
| `Bunny.obj` | 300 | 6 | 10,000 | 0.00371 | 0.0181 | 0.0243 |
| `fandisk.obj` | 200 | 6 | 10,000 | 0.00634 | 0.0518 | 0.0489 |

**SDF-resolution study** (`run_sdf_sweep.sh`), 10,000 epochs per run

`fandisk.obj` (7,229 vertices):

| `SDF_RES` | `FD_HALO` | voxel size | RMSE | rel_l2 | max abs error |
|---:|---:|---:|---:|---:|---:|
| 20 | 3 | 0.0942 | 0.201 | 1.641 | 0.849 |
| 30 | 3 | 0.0617 | 0.0994 | 0.812 | 0.360 |
| 40 | 3 | 0.0459 | 0.0564 | 0.461 | 0.241 |
| 50 | 3 | 0.0365 | 0.0172 | 0.140 | 0.0869 |
| 60 | 3 | 0.0303 | 0.00631 | 0.0516 | 0.0480 |
| 70 | 4 | 0.0259 | 0.00608 | 0.0497 | 0.0475 |
| 90 | 4 | 0.0201 | 0.00514 | 0.0420 | 0.0483 |
| 100 | 4 | 0.0181 | 0.00441 | 0.0360 | 0.0386 |
| 150 | 5 | 0.0120 | 0.00551 | 0.0450 | 0.0500 |
| 200 | 6 | 0.00900 | 0.00634 | 0.0518 | 0.0489 |
| 250 | 7 | 0.00719 | 0.00508 | 0.0415 | 0.0460 |
| 300 | 8 | 0.00599 | 0.00691 | 0.0565 | 0.0468 |

`fandisk_5w.obj` (28,617 vertices):

| `SDF_RES` | `FD_HALO` | voxel size | RMSE | rel_l2 | max abs error |
|---:|---:|---:|---:|---:|---:|
| 20 | 3 | 0.0942 | 0.164 | 1.502 | 0.692 |
| 35 | 3 | 0.0527 | 0.0677 | 0.619 | 0.295 |
| 50 | 3 | 0.0365 | 0.0234 | 0.214 | 0.127 |
| 200 | 6 | 0.00900 | 0.00854 | 0.0781 | 0.0790 |
| 250 | 7 | 0.00719 | 0.00839 | 0.0767 | 0.0731 |
| 300 | 8 | 0.00599 | 0.00874 | 0.0799 | 0.0658 |
| 400 | 2 | 0.00449 | 0.00737 | 0.0674 | 0.0650 |

**Generalization over the Dirichlet values** (`run_generalization.sh`, then
`run_evaluation.sh`), `dc` model, 2,000 vectors per set

| Set | Vectors of values | MSE mean | rel_l2 mean | rel_l2 median | rel_l2 max |
|---|---|---:|---:|---:|---:|
| `train` | seen during training, in `[-3, 2]^5` | 7.95e-05 | 0.0224 | 0.0199 | 0.108 |
| `interp` | unseen, in `[-3, 2]^5` | 7.86e-05 | 0.0226 | 0.0200 | 0.105 |
| `extrap` | in `[-3.5, 2.5]^5`, outside `[-3, 2]^5` | 4.00e-04 | 0.0359 | 0.0342 | 0.137 |

## Files

```
data/                                   meshes (.obj) and source specifications (.json)

train_screen.py                         training CLI
Surfacescreendiffusion.py               network, losses, source sampling
SparseGridFDM_SDF.py                    sparse narrow-band SDF with finite differences
Mesh_geometry.py                        mesh loading, normalization, surface sampling
fem_screen.py                           FEM reference solver
make_screen_sources.py                  write a source specification from the command line
sdf_spacing.py                          voxel spacing of the SDF grid for given settings

compare_nn_fem_mse.py                   error of one run against FEM
evaluate_dirichlet_generalization.py    error of a generalizing network, one FEM solve per vector
prepare_dirichlet_splits.py             train / interp / extrap vectors for the evaluation
plot_generalization_mse.py              scatter plots of the per-vector errors
plot_sdf_sweep.py                       error versus SDF resolution
export_split_samples.py, export_worst_rel_l2.py   network and FEM fields for figures
vis_compare_nn_fem.py                   polyscope viewer of the error map
```

Every Python file starts with a docstring that describes its options and shows
example commands.

## Viewing results

On a machine with a display and `polyscope` installed:

```bash
python Surfacescreendiffusion.py --vis results/single/fandisk_res200_halo6/diffusion.npz
python fem_screen.py --vis results/single/fandisk_res200_halo6/fem_diffusion.npz
python vis_compare_nn_fem.py results/single/fandisk_res200_halo6/diffusion.npz \
                             results/single/fandisk_res200_halo6/fem_diffusion.npz
```

`diffusion.npz` and `fem_diffusion.npz` are plain NumPy archives
(`mesh_vertices`, `mesh_faces`, `u_vertex`, the source geometry and values) and
can be loaded with `numpy.load` in any other tool.

## Notes

* **Checkpoints** are read with `torch.load(..., weights_only=False)`, which
  unpickles arbitrary objects. Only load checkpoints you trust.
* **Memory.** The grid uses a dense index table over the bounding box, which
  grows with `SDF_RES^3`, and the build visits `(2 * FD_HALO + 1)^3` neighbors
  per surface voxel. Large `SDF_RES` and `FD_HALO` together need considerably
  more GPU memory.
* **Meshes** should be closed, consistently oriented triangle meshes in OBJ
  format; only `v` and `f` lines are read.
