"""
train_basis.py - Train divergence-free vector Laplacian eigenbasis
===================================================================

Two-stage pipeline:
  Stage 1: Build FDM SDF oracle (sparse grid + finite differences)
  Stage 2: Learn k div-free eigenbases via stream functions psi_i
           with SDF chain-rule surface operators + adaptive CPM sampling.

Usage:
    python train_basis.py data/Bunny.obj -k 10 --out-dir results/Bunny
    See run_train_basis.sh for a full list of options.

Output in --out-dir: model.pt and basis_000.npz ... basis_{k-1}.npz (one file
per mode, sampled at the face centres).  Requires Linux, CUDA, fvdb and fcpw.
"""

import argparse
import os
import torch

from Mesh_geometry import MeshGeometry
from SparseGridFDM_SDF import SparseGridFDM_SDF
from SurfaceDFVecLaplacianBasis import SurfaceDFVecLaplacianBasis


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Train divergence-free vector Laplacian eigenbasis (SDF + neural psi)",
    )
    p.add_argument(
        "mesh",
        nargs="?",
        default="data/Bunny.obj",
        help="Path to input mesh .obj (default: data/Bunny.obj)",
    )
    p.add_argument("-k", "--num-basis", type=int, default=10, help="Number of eigenmodes")
    p.add_argument("--num-epochs", type=int, default=4000)
    p.add_argument("--bsp-res", type=int, default=500, help="SDF grid resolution (Stage 1)")
    p.add_argument("--padding", type=float, default=0.15, help="SDF bounding-box padding (fractional)")
    p.add_argument("--fd-halo", type=int, default=2, help="Finite-difference halo for SDF")
    p.add_argument("--no-normalize", action="store_true", help="Do not normalize mesh to unit box")

    p.add_argument("--input-size", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--num-surface", type=int, default=10000)
    p.add_argument("--num-narrowband", type=int, default=10000)
    p.add_argument("--code-value", type=float, default=0.5)
    p.add_argument("--lambda-ortho", type=float, default=1e3)
    p.add_argument("--lambda-norm", type=float, default=1000.0)
    p.add_argument("--lambda-mean", type=float, default=0.0)
    p.add_argument("--lambda-cpm", type=float, default=1000.0)
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--band-width", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-dir", type=str, default="checkpts_basis")
    p.add_argument("--out-dir", type=str, default="Bunny_basis_results", help="Export NPZ + model.pt")
    p.add_argument("--export-batch-size", type=int, default=4096)
    return p


def main():
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mesh = MeshGeometry(
        args.mesh, normalize=not args.no_normalize, device=device
    )

    # ================================================================
    # Stage 1: FDM SDF oracle
    # ================================================================
    print("\n" + "=" * 60)
    print(f"Stage 1: Building FDM SDF Oracle (res={args.bsp_res})")
    print("=" * 60)
    sdf = SparseGridFDM_SDF.from_mesh(
        mesh,
        resolution=args.bsp_res,
        padding=args.padding,
        fd_halo=args.fd_halo,
        device=device,
    )

    # ================================================================
    # Stage 2: Divergence-free vector Laplacian eigenbasis
    # ================================================================
    print("\n" + "=" * 60)
    print(f"Stage 2: Learning {args.num_basis} DF Vector Laplacian Eigenbases")
    print("  Using FDM SDF geometry oracle (chain-rule surface operators)")
    print("=" * 60)

    basis = SurfaceDFVecLaplacianBasis(
        num_basis=args.num_basis,
        input_size=args.input_size,
        hidden=args.hidden,
        lr=args.lr,
        device=device,
        sdf=sdf,
    )

    basis.train(
        mesh=mesh,
        num_epochs=args.num_epochs,
        num_surface=args.num_surface,
        num_narrowband=args.num_narrowband,
        code_value=args.code_value,
        lambda_ortho=args.lambda_ortho,
        lambda_norm=args.lambda_norm,
        lambda_mean=args.lambda_mean,
        lambda_cpm=args.lambda_cpm,
        ridge=args.ridge,
        eps=args.eps,
        band_width=args.band_width,
        clip_grad=args.clip_grad,
        log_every=args.log_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
    )

    out_dir = args.out_dir
    basis.export(
        mesh, out_dir=out_dir, code_value=args.code_value, batch_size=args.export_batch_size
    )
    basis.save(os.path.join(out_dir, "model.pt"))

    print(f"\nDone! Visualize with:")
    print(f"  python SurfaceDFVecLaplacianBasis.py --vis {out_dir}/basis_000.npz")


if __name__ == "__main__":
    main()
