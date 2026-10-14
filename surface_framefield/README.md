# Neural Frame Field Design on Surfaces

A smooth, feature-aligned frame field on a triangle-mesh surface, represented
by a neural field. A network `theta(x, n(x))` gives, at every surface point,
the angle by which a reference tangent frame is rotated about the normal
`n(x)`; the resulting frame `(e1, e2, n)` is compared through its degree-4
spherical-harmonic coefficients `q`, so the field is defined up to the
symmetries of a cross. The network is defined in a narrow band around the
surface; the normal, the curvature directions and the closest-point map come
from a sparse signed distance field (SDF).

The training minimizes a smoothness energy of `q` over the surface, an
alignment with the principal-curvature directions (weighted by how anisotropic
the curvature is), an alignment on the feature curves (sharp edges) of the
mesh, and a closest-point term that keeps the field constant along the normal.

## Installation

Training needs **Linux and an NVIDIA GPU**: the sparse SDF grid is built with
[fVDB](https://github.com/openvdb/fvdb-core). The code was developed with
Python 3.10.

```bash
conda create -n surface_framefield python=3.10 -y
conda activate surface_framefield
pip install -r requirements.txt
```

`requirements.txt` lists a matching PyTorch / fVDB pair; change both lines
together for another CUDA version. `SparseGridFDM_SDF.py` uses the `GridBatch`
API of `fvdb-core` 0.3 - 0.4. Every SDF build prints
`[fcpw] GPU path failed ... falling back to CPU scene`; this is expected with
the `pip` wheel of fcpw and is not an error.

## Training

`run_train_frame_field.sh` has a `parameters` block at the top (mesh and
output folder, SDF grid, network, learning rate, number of epochs, sample
counts, loss weights, feature curves, schedules) followed by the Python
command it runs. Edit the block, then run the script from this folder:

```bash
bash run_train_frame_field.sh
CUDA_VISIBLE_DEVICES=1 bash run_train_frame_field.sh        # choose the GPU
```

As shipped it trains on `data/Bunny.obj` for 10,000 epochs; `MODEL_NAME`
selects another mesh of `data/`.

Output in `results/Bunny/`:

| File | Content |
|---|---|
| `Bunny.npz` | the frame field at the face centers (see below) |
| `Bunny.obj`, `Bunny_crossfield.txt` | normalized mesh and per-face frame directions for parameterization, exported automatically after training |
| `model.pt` | network checkpoint |
| `frame_field.png`, `frame_field.npz` | a plot of the field at random surface samples, and its data |
| `checkpoints/` | intermediate checkpoints |

Main arrays of the face-center file:

| Array | Content |
|---|---|
| `mesh_vertices`, `mesh_faces` | the normalized mesh |
| `face_centers`, `face_normals` | where the field is sampled, and the SDF normals there |
| `e1`, `e2` | the two tangent directions of the frame at every face, `(F, 3)` each, in the plane of the face |
| `theta` | the rotation angle given by the network |
| `q` | spherical-harmonic coefficients of the frame, `(F, 9)` |
| `feat_curve_pts`, `feat_curve_tangent` | samples on the feature curves and their tangents (with `EXTRACT_FEATURE_CURVES=1`) |

The file also holds the reference frame and per-face diagnostics of the
smoothness and alignment terms.

The mesh is normalized on loading: its bounding-box center is moved to the
origin and it is scaled so that its farthest vertex has norm 1.

## Parameterization and quad extraction

The CMake build compiles the libigl parameterization tool and the
[libQEx](https://github.com/hcebke/libQEx) submodule for quad mesh extraction.
From this directory:

```bash
./build.sh
```

The script initializes the libQEx submodule and configures and builds both tools.
It defaults to a Release build in `build/` with four parallel jobs. Override
`BUILD_DIR`, `BUILD_TYPE`, or `JOBS` as needed; additional arguments are passed
to CMake, for example `JOBS=2 ./build.sh -DBUILD_QEX=OFF`.

On Linux, extract a quad mesh from an OBJ containing per-face UVs with:

```bash
build/cmdline_tool mesh_with_uv.obj quad_mesh.obj
```

See [the parameterization README](parameterization_libigl/README.md) for input
formats, dependencies, and build options. libQEx is distributed under GPLv3;
its README includes citation information.

## Files

```
data/                       meshes (.obj)
run_train_frame_field.sh    training script
train_frame_field.py        training CLI
SurfaceFrameFieldSH.py      network, losses, feature curves, export
model_pytorch.py            Siren network
sh_pytorch.py               spherical-harmonic representation of frames
common_pytorch.py           small tensor helpers
SparseGridFDM_SDF.py        sparse narrow-band SDF with finite differences
Mesh_geometry.py            mesh loading, normalization, surface sampling
```

## Notes

* **Checkpoints** are read with `torch.load`, which unpickles arbitrary
  objects. Only load checkpoints you trust.
* The SDF settings (`SDF_RES`, `SDF_PADDING`, `FD_HALO`) are not stored in
  `model.pt`. A script that reloads a checkpoint has to rebuild the SDF with
  the settings of the training run.
* **Meshes** should be closed, consistently oriented triangle meshes in OBJ
  format; only `v` and `f` lines are read.
