# Neural Divergence-Free Eigenbasis on Surfaces: Generalization over a Shape Space

A basis of divergence-free tangent vector fields (the lowest eigenmodes of the
vector Laplacian), learned by **one model for a whole family of shapes**. Each
basis function is given by a neural stream function

```
psi_i = Psi_i(x, code)       u_i = n x grad_S psi_i
```

where `code` in `[0, 1]` is the position of the shape in the family. The
networks are shared by all shapes, so after training the basis is available
for every shape of the family, and it changes continuously with the shape.

The single-shape version of this code is in `surface_divfreeBasis`. The
differences are: no signed distance field is built (the normals are the mesh
normals), the closest-point term uses samples on the surface, and every
training iteration uses one shape picked at random.

## Installation

The training needs PyTorch (a CUDA GPU is strongly recommended), NumPy and
Matplotlib. The code was developed with Python 3.10.

```bash
conda create -n surface_dfbasis_gen python=3.10 -y
conda activate surface_dfbasis_gen
pip install -r requirements.txt
```

The advection scripts additionally need `fcpw` (closest-point queries), and
the viewers need `polyscope` and a display.

## Training

`run_train_basis_time.sh` has a `parameters` block at the top (mesh sequence,
number of basis functions, iterations, loss weights, output folder) followed
by the Python command it runs. Edit the block, then run the script from this
folder:

```bash
bash run_train_basis_time.sh
CUDA_VISIBLE_DEVICES=1 bash run_train_basis_time.sh        # choose the GPU
```

**Shape family.** A folder of `.obj` files, taken in the order of their file
names. All files must have the same vertices and faces (only the vertex
positions change). The file number `k` of `n` files gets the code
`k / (n - 1)`. As shipped the script uses `data/hand_2_top60`, a sequence of a
deforming hand, and trains `K = 50` basis functions.

**Normalization.** With `NORMALIZE="shared"` all meshes are transformed by one
common center and scale (computed over the whole sequence), so the shapes stay
in one coordinate frame. The meshes are never normalized one by one.

Output in `results/hand_2_top60/`:

| File | Content |
|---|---|
| `model.pt` | checkpoint of the `K` networks |
| `frame_0000/`, `frame_0001/`, ... | one folder per mesh of the sequence |
| `frame_XXXX/basis_000.npz` ... | one file per mode on that mesh, sampled at the face centers |
| `frame_XXXX/meta.npz` | index and code of the mesh |
| `checkpoints/` | intermediate checkpoints |

Every `basis_XXX.npz` holds `mesh_vertices`, `mesh_faces`, `face_centers`,
`face_normals`, the basis vector field `u` `(F, 3)`, its stream function `psi`
and its vorticity `omega`.

On a machine with a display and `polyscope` installed:

```bash
python SurfaceDFVecLaplacianBasis.py --vis results/hand_2_top60/frame_0000/basis_000.npz
```

### Analytic shape families

`train_basis_time.py` can also generate its shapes from a formula
(`surfaces.py`): a wavy torus or a radial icosphere, sampled at `--num-frames`
values of the time `t`:

```bash
python train_basis_time.py --source torus     --num-frames 8 -k 10 --out-dir results/torus
python train_basis_time.py --source icosphere --num-frames 8 -k 10 --out-dir results/icosphere
```

## Advection on the moving shape

`run_advection_time.sh` uses the trained basis to advect a density while the
shape goes through the sequence (`surface_basis_advection_time.py`), and
`vis_advection_time.py` plays the saved result:

```bash
bash run_advection_time.sh
python vis_advection_time.py results/hand_2_top60/advection_time.npz
```

Use the same mesh sequence and the same `NORMALIZE` as for training.

## Files

```
data/hand_2_top60/                 sequence of meshes (.obj)
run_train_basis_time.sh            training script
run_advection_time.sh              advection script
train_basis_time.py                training CLI for a family of shapes
SurfaceDFVecLaplacianBasis.py      networks, losses, export, viewer
surfaces.py                        analytic shape families (torus, icosphere)
Mesh_geometry.py                   mesh loading and surface sampling
surface_basis_advection_time.py    density advection on a sequence of shapes
surface_basis_advection.py         geometry helpers used by the advection
SparseGridFDM_SDF.py               fcpw helpers used by the advection
vis_advection_time.py              polyscope player for an advection result
```

## Notes

* **Checkpoints** are read with `torch.load(..., weights_only=False)`, which
  unpickles arbitrary objects. Only load checkpoints you trust.
* **Meshes** should be closed, consistently oriented triangle meshes in OBJ
  format; only `v` and `f` lines are read.

## Citation

To be added.

## License

To be added. The meshes in `data/` are included to reproduce the experiments;
check the terms of the original models before redistributing them.
