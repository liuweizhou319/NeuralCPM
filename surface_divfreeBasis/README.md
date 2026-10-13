# Neural Divergence-Free Eigenbasis on Surfaces

A basis of divergence-free tangent vector fields on a triangle-mesh surface:
the lowest eigenmodes of the vector Laplacian, each represented by a neural
stream function

```
psi_i = Psi_i(x, n(x), code)       u_i = n x grad_S psi_i
```

so every basis field `u_i` is tangent and divergence-free by construction. The
networks are defined in a narrow band around the surface; the normal `n(x)`
and all surface differential quantities come from a sparse signed distance
field (SDF). This folder trains a **single model for one shape**.

## Installation

Training needs **Linux and an NVIDIA GPU**: the sparse SDF grid is built with
[fVDB](https://github.com/openvdb/fvdb-core). The code was developed with
Python 3.10.

```bash
conda create -n surface_dfbasis python=3.10 -y
conda activate surface_dfbasis
pip install -r requirements.txt
```

`requirements.txt` lists a matching PyTorch / fVDB pair; change both lines
together for another CUDA version. `SparseGridFDM_SDF.py` uses the `GridBatch`
API of `fvdb-core` 0.3 - 0.4. Every SDF build prints
`[fcpw] GPU path failed ... falling back to CPU scene`; this is expected with
the `pip` wheel of fcpw and is not an error.

## Training

`run_train_basis.sh` has a `parameters` block at the top (mesh, number of
basis functions, SDF resolution, epochs, loss weights, output folder) followed
by the Python command it runs. Edit the block, then run the script from this
folder:

```bash
bash run_train_basis.sh
CUDA_VISIBLE_DEVICES=1 bash run_train_basis.sh        # choose the GPU
```

As shipped it trains `K = 60` basis functions on `data/Bunny.obj`. The modes
are trained together: at every epoch each mode minimizes its Rayleigh quotient
`int (Lap_S psi)^2 / int |grad_S psi|^2` after being made orthogonal to the
previous modes.

Output in `results/Bunny/`:

| File | Content |
|---|---|
| `model.pt` | checkpoint of the `K` networks |
| `basis_000.npz` ... | one file per mode, sampled at the face centers |
| `checkpoints/` | intermediate checkpoints |

Every `basis_XXX.npz` holds:

| Array | Content |
|---|---|
| `mesh_vertices`, `mesh_faces` | the normalized mesh |
| `face_centers`, `face_normals` | where the mode is sampled, and the SDF normals there |
| `u` | the basis vector field, `(F, 3)` |
| `psi` | its stream function |
| `omega` | its vorticity `-Lap_S psi` |

The mesh is normalized on loading: its bounding-box center is moved to the
origin and it is scaled so that its farthest vertex has norm 1.

On a machine with a display and `polyscope` installed:

```bash
python SurfaceDFVecLaplacianBasis.py --vis results/Bunny/basis_000.npz
```

## Advection on the surface

`surface_basis_advection.py` uses a trained basis to advect a density on the
surface (semi-Lagrangian, velocity expressed in the basis), and
`vis_advection.py` plays the saved result:

```bash
python surface_basis_advection.py data/Bunny.obj results/Bunny/model.pt \
    --center 0.0,0.5,0.0 --speed 1.5 --n-steps 1500 --dt 0.005 \
    --out results/Bunny/advection.npz
python vis_advection.py results/Bunny/advection.npz
```

It rebuilds the SDF, so pass the `--bsp-res` and `--padding` of the training
run.

## Files

```
data/                           meshes (.obj)
run_train_basis.sh              training script
train_basis.py                  training CLI
SurfaceDFVecLaplacianBasis.py   networks, losses, export, viewer
SparseGridFDM_SDF.py            sparse narrow-band SDF with finite differences
Mesh_geometry.py                mesh loading, normalization, surface sampling
surface_basis_advection.py      density advection with a trained basis
vis_advection.py                polyscope player for an advection result
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
