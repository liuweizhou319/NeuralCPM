# Neural Vector Field Design on Surfaces

A tangent vector field on a triangle-mesh surface, designed from a few anchor
vectors and represented by two neural scalar potentials (a Hodge
decomposition):

```
u = grad_S phi + n x grad_S psi
```

`phi` and `psi` are networks `Phi(x, n(x), theta)` and `Psi(x, n(x), theta)`
defined in a narrow band around the surface; the normal `n(x)` and all surface
differential quantities come from a sparse signed distance field (SDF). The
field follows the anchors and is as smooth as possible elsewhere.

| Script | Experiment |
|---|---|
| `run_single.sh` | training of a **single model** for given anchors at one fixed angle |
| `run_generalization.sh` | **angle generalization** for given anchors: one model for every rotation angle of the anchors |

## Installation

Training needs **Linux and an NVIDIA GPU**: the sparse SDF grid is built with
[fVDB](https://github.com/openvdb/fvdb-core). The code was developed with
Python 3.10.

```bash
conda create -n surface_vector python=3.10 -y
conda activate surface_vector
pip install -r requirements.txt
```

`requirements.txt` lists a matching PyTorch / fVDB pair; change both lines
together for another CUDA version. `SparseGridFDM_SDF.py` uses the `GridBatch`
API of `fvdb-core` 0.3 - 0.4. Every SDF build prints
`[fcpw] GPU path failed ... falling back to CPU scene`; this is expected with
the `pip` wheel of fcpw and is not an error.

## Running the experiments

Each script has a `parameters` block at the top (mesh, anchors, SDF
resolution, epochs, loss weights, output folder) followed by the Python
command it runs. Edit the block, then run the script from this folder:

```bash
bash run_single.sh
bash run_generalization.sh
CUDA_VISIBLE_DEVICES=1 bash run_single.sh        # choose the GPU
```

### `run_single.sh`: one model, fixed angle

Trains the two potentials for the anchors in `CONSTRAINTS`, rotated by the
fixed angle `THETA` (radians, 0 = as given), and exports the field. As shipped
it uses `data/78481root_5w.obj` with five anchors.

Output in `results/single/78481root_5w/`: `model.pt`, `vecfield_000.npz` and
`checkpoints/` (intermediate checkpoints).

### `run_generalization.sh`: angle generalization

The anchors keep their faces; their vectors are rotated by an angle `theta` in
the tangent plane. During training `theta` is drawn uniformly from
`[0, 2*pi)` at every epoch and given to the networks, so after training one
model returns the field for any angle without retraining. As shipped it uses
`data/botijo.obj` with nine anchors.

Output in `results/generalization/botijo/`: `model.pt` and
`vecfield_000.npz ... vecfield_127.npz`, the field for `EXPORT_K = 128` equally
spaced angles.

## Anchors

The anchors are an `.npz` file with two arrays:

| Array | Shape | Content |
|---|---|---|
| `face_ids` | `(N,)` int | face indices, 0-based |
| `vectors` | `(N, 3)` float | target vector on each face |

Only the directions matter: the training normalizes every vector, scales it to
twice the average edge length of the normalized mesh and projects it onto the
tangent plane. `make_vecfield_constraints.py` writes such a file:

```bash
python make_vecfield_constraints.py data/37384.obj -o data/my_anchors.npz --faces 0,5,100 --random
python make_vecfield_constraints.py data/37384.obj -o data/my_anchors.npz --faces 0,5,100 --vector 1 0 0
```

The mesh is normalized on loading: its bounding-box center is moved to the
origin and it is scaled so that its farthest vertex has norm 1.

## Result files

Every `vecfield_XXX.npz` holds one field, sampled at the face centers:

| Array | Content |
|---|---|
| `mesh_vertices`, `mesh_faces` | the normalized mesh |
| `face_centers`, `face_normals` | where the field is sampled, and the SDF normals there |
| `u` | the vector field, `(F, 3)` |
| `u_pot`, `u_stream` | its two parts `grad_S phi` and `n x grad_S psi` |
| `phi`, `psi` | the two potentials |
| `theta` | rotation angle of the anchors for this file |
| `constraint_face_ids` | anchor faces |
| `constraint_vectors_original`, `constraint_vectors_rotated` | anchor vectors as given, and rotated by `theta` |

On a machine with a display and `polyscope` installed:

```bash
python SurfaceVectorFieldDesign.py --vis results/single/78481root_5w/vecfield_000.npz
```

## FEM reference

`fem_vector_field_design_reference.py` solves the corresponding FEM problem
(smoothest vector field with the anchors as constraints) and writes an npz
with the same arrays, so the same viewer can show it:

```bash
python fem_vector_field_design_reference.py data/78481root_5w.obj \
    --constraints data/78481root_5w.contraints_k5.npz \
    -o results/single/78481root_5w/fem_vecfield.npz
```

The FEM solver takes the anchor vectors as they are in the file, whereas the
training rescales them to twice the average edge length. The two fields
therefore agree in direction; their magnitudes agree only if the vectors in
the file already have that length.

## Files

```
data/                                  meshes (.obj) and anchors (.npz)
train_vecfield.py                      training CLI
SurfaceVectorFieldDesign.py            networks, losses, export, viewer
SparseGridFDM_SDF.py                   sparse narrow-band SDF with finite differences
Mesh_geometry.py                       mesh loading, normalization, surface sampling
make_vecfield_constraints.py           write an anchor file from the command line
fem_vector_field_design_reference.py   FEM reference solver
```

## Notes

* **Checkpoints** are read with `torch.load(..., weights_only=False)`, which
  unpickles arbitrary objects. Only load checkpoints you trust.
* The SDF settings (`BSP_RES`, `PADDING`, `FD_HALO`) are not stored in
  `model.pt`. A script that reloads a checkpoint has to rebuild the SDF with
  the settings of the training run, because the SDF normal is an input of the
  networks.
* **Meshes** should be closed, consistently oriented triangle meshes in OBJ
  format; only `v` and `f` lines are read.

## Citation

To be added.

## License

To be added. The meshes in `data/` are included to reproduce the experiments;
check the terms of the original models before redistributing them.
