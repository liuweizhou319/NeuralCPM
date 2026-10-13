# Neural Field Processing on Surfaces

This repository contains the code for our SIGGRAPH Asia 2026 conference paper, [**Neural Field Processing on Surfaces**](https://jiongchen.github.io/files/neuralcpm-paper.pdf).

## Experiments

Each directory contains a self-contained experiment with its own dependencies,
example data, scripts, and detailed README.

| Directory | Task | Entry points |
|---|---|---|
| [surface_screened](surface_screened/README.md) | Screened diffusion, Dirichlet-value generalization, FEM comparisons, and SDF-resolution studies | `run_single.sh`, `run_generalization.sh`, `run_evaluation.sh` |
| [surfacevector](surfacevector/README.md) | Tangent vector field design from anchors, including generalization over anchor rotation | `run_single.sh`, `run_generalization.sh` |
| [surface_divfreeBasis](surface_divfreeBasis/README.md) | Divergence-free vector Laplacian eigenbasis on a fixed shape and density advection | `run_train_basis.sh`, `surface_basis_advection.py` |
| [surface_divfreeBasis_generalization](surface_divfreeBasis_generalization/README.md) | Eigenbasis generalization across a shape family and advection on deforming surfaces | `run_train_basis_time.sh`, `run_advection_time.sh` |
| [surface_framefield](surface_framefield/README.md) | Smooth, feature-aligned cross/frame field design | `run_train_frame_field.sh` |

## Contact

For questions about the paper or code, please contact Weizhou Liu at [weizhou.liu@inria.fr](mailto:weizhou.liu@inria.fr).
