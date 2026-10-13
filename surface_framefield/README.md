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

`run_train_frame_field.sh` has a `parameters` block at the top (mesh, SDF
resolution, feature curves, schedules, output folder) followed by the Python
command it runs. Edit the block, then run the script from this folder:

```bash
bash run_train_frame_field.sh
CUDA_VISIBLE_DEVICES=1 bash run_train_frame_field.sh        # choose the GPU
```

As shipped it trains on `data/rockerarm.obj`; `MODEL_NAME` selects another
mesh of `data/`. The number of epochs (10,000), the sample counts and the
loss weights are set in `train_frame_field.py`, in the call of
`trainer.train(...)`.

Output in `results/rockerarm/`:

| File | Content |
|---|---|
| `rockerarm.npz` | the frame field at the face centers (see below) |
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
* The SDF settings (resolution `SDF_RES`; padding 0.15 and a halo of 4 voxels,
  set in `train_frame_field.py`) are not stored in `model.pt`. A script that
  reloads a checkpoint has to rebuild the SDF with the settings of the
  training run.
* **Meshes** should be closed, consistently oriented triangle meshes in OBJ
  format; only `v` and `f` lines are read.

## Citation

To be added.

## License

To be added. The meshes in `data/` are included to reproduce the experiments;
check the terms of the original models before redistributing them.
