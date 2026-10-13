# Visualization

Interactive viewers for the results of the project folders of this repository,
written with [polyscope](https://polyscope.run/py/). A viewer only reads the
`.npz` files that a run script has written: it needs neither a GPU nor the
training packages, so results computed on a remote machine can be copied to a
laptop and looked at there.

## Installation

```bash
pip install numpy polyscope
```

The scripts use the polyscope 2 API. polyscope opens a window, so a machine
with a display is needed.

## Usage

Every script takes one argument: the result file, or the result folder, to
show.

```bash
cd visualization
python vis_screened_npz.py ../surface_screened/results/single/thin_star_res150_halo2/diffusion.npz
```

| Script | Shows | Argument | Written by |
|---|---|---|---|
| `vis_screened_npz.py` | screened diffusion: the field and its constraints | `diffusion.npz` or `fem_diffusion.npz` | `surface_screened/run_thin_star.sh`, `run_single.sh` |
| `vis_vecfield_npz.py` | vector field design: one field and its anchors | `vecfield_000.npz` | `surfacevector/run_single.sh` |
| `vis_vecfield_angles.py` | vector field design: the field for every angle of the anchors | folder with `vecfield_*.npz` | `surfacevector/run_generalization.sh` |
| `vis_dfbasis.py` | divergence-free basis on one shape | folder with `basis_*.npz` | `surface_divfreeBasis/run_train_basis.sh` |
| `vis_dfbasis_generalization.py` | divergence-free basis on a family of shapes | folder with `frame_*/` | `surface_divfreeBasis_generalization/run_train_basis_time.sh` |
| `vis_framefield_npz.py` | frame field | `<mesh name>.npz` | `surface_framefield/run_train_frame_field.sh` |

The run scripts write into `results/` of their own folder. The commands below
are run from this folder and use the output names of the run scripts as
shipped.

Without an argument, a script shows the result of its run script as shipped
(for the screened diffusion: of `run_thin_star.sh`). The path is `DEFAULT_NPZ`
or `DEFAULT_DIR` at the top of the script. For example, after
`bash run_train_basis.sh` in `surface_divfreeBasis/`:

```bash
python vis_dfbasis.py
```

In the polyscope window, the panel on the left lists the mesh with its
quantities; a checkbox shows or hides a quantity. A mesh shows one scalar at a
time. The sliders of a viewer are in the panel on the right.

## Screened diffusion

```bash
python vis_screened_npz.py ../surface_screened/results/single/thin_star_res150_halo2/diffusion.npz
python vis_screened_npz.py ../surface_screened/results/single/thin_star_res150_halo2/fem_diffusion.npz
```

`diffusion.npz` is the result of the network and `fem_diffusion.npz` the FEM
reference; `run_thin_star.sh` and `run_single.sh` write both into
`results/single/<name>/`.

| Shown | Content |
|---|---|
| `u` | the field on the mesh, with the color range of the constraints |
| `u (autoscale)` | the same field with its own color range (off at the start) |
| `\|grad u\|` | norm of the surface gradient, only in `diffusion.npz` (off at the start) |
| `constraint points`, `constraint curves` | the point and curve sources, colored by their Dirichlet value |

## Vector field design

One field (`run_single.sh`):

```bash
python vis_vecfield_npz.py ../surfacevector/results/single/78481root_5w/vecfield_000.npz
```

| Shown | Content |
|---|---|
| `u` | the vector field at the face centers |
| `u_pot`, `u_stream` | its two parts (off at the start) |
| `\|u\|`, `phi`, `psi` | its norm and the two potentials (off at the start) |
| `anchors` | at every anchor face the `target` vector (red) and the vector of the `network` (blue) |

The error at the anchors is printed in the terminal. The FEM reference written
by `fem_vector_field_design_reference.py` holds the same arrays and is shown
in the same way.

Angle generalization (`run_generalization.sh`):

```bash
python vis_vecfield_angles.py ../surfacevector/results/generalization/botijo
```

The folder holds one file per rotation angle of the anchors
(`vecfield_000.npz ... vecfield_127.npz` as shipped). The slider
`angle index` selects the angle; the field `u` and the `target` and `network`
vectors of the anchors follow, and the angle is shown below the slider.

Both viewers draw the vectors with their true length.

## Divergence-free basis

One shape (`run_train_basis.sh`):

```bash
python vis_dfbasis.py ../surface_divfreeBasis/results/Bunny
python vis_dfbasis.py ../surface_divfreeBasis/results/Bunny --length 0.03
```

The folder holds one file per basis field (`basis_000.npz ...`). The slider
`basis index` selects the field.

| Shown | Content |
|---|---|
| `u` | the basis vector field at the face centers |
| `psi` | its stream function |
| `omega` | its vorticity (off at the start) |
| `\|u\|` | its norm (off at the start) |

A family of shapes (`run_train_basis_time.sh`):

```bash
python vis_dfbasis_generalization.py ../surface_divfreeBasis_generalization/results/hand_2_top60
python vis_dfbasis_generalization.py ../surface_divfreeBasis_generalization/results/hand_2_top60 --length 0.03
```

The folder holds one subfolder per shape (`frame_0000/ ...`), each with the
files of the basis fields and `meta.npz`. The slider `shape index` selects the
shape and the slider `basis index` the field, so one basis field can be
followed over the shapes; the shape code is shown below the sliders. The
quantities are the same as for one shape.

## Frame field

```bash
python vis_framefield_npz.py ../surface_framefield/results/rockerarm/rockerarm.npz
python vis_framefield_npz.py ../surface_framefield/results/rockerarm/rockerarm.npz --length 0.008
```

The argument is the face-center file of the run, `<MODEL_NAME>.npz` in the
output folder. `frame_field.npz` of the same folder holds random surface
samples without the mesh and cannot be shown with this script.

| Shown | Content |
|---|---|
| `cross` | the frame at every face, drawn as its four directions |
| `feature curves` | midpoints of the sharp edges, if the run extracted them; `tangent`: directions of these edges (off at the start) |
| `e1`, `e2` | the two axes of the frame (off at the start) |
| `t_feat` | direction of minimal curvature that the field is aligned with (off at the start) |
| `theta`, `w_curv`, `feat_align`, `feat_align_weighted`, `smooth_*` | per-face scalars of the file: rotation angle, curvature weight, alignment and smoothness terms (off at the start) |

## Length of the vectors

`vis_dfbasis.py`, `vis_dfbasis_generalization.py` and `vis_framefield_npz.py`
have the option `--length`: the length of the longest vector, relative to the
size of the scene. The default is 0.02, and 0.004 for the frame field, where a
cross is drawn at every face. The `Length` slider of a vector quantity in the
polyscope window changes it interactively; in the viewers with a slider it
goes back to `--length` when another field is selected.

## Other viewers

Some project folders have their own polyscope scripts for results that are
not covered here: `surface_screened/vis_compare_nn_fem.py` (error between the
network and the FEM reference), `surface_divfreeBasis/vis_advection.py` and
`surface_divfreeBasis_generalization/vis_advection_time.py` (advection on the
surface). They are described in the README of these folders.
