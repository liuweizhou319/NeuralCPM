"""
Train a Hodge surface vector field  u = grad_S phi + n x grad_S psi  on a mesh.

Mesh: the positional argument, an .obj file.
Constraints (anchors): --constraints path.npz with two arrays
    face_ids : (N,) int      face indices (0-based)
    vectors  : (N,3) float   target vector on each face
The vectors are normalised and scaled to 2 * (average edge length), then
projected onto the tangent plane.  make_vecfield_constraints.py writes such a
file.  Without --constraints, --n-random-constraints random anchors are used.

Two modes:
    --theta T     fixed angle: the anchors are rotated by T (radians) in their
                  tangent planes and one field is trained and exported
    (no --theta)  angle generalization: the rotation angle is sampled in
                  [0, 2*pi) at every epoch and given to the network; after
                  training, -k K fields are exported for K equally spaced angles

Output in --out-dir:
    model.pt                        checkpoint
    vecfield_000.npz ... (K files)  the field at the face centres, the anchors
                                    and the mesh (one file in fixed-angle mode)

Examples (see run_single.sh and run_generalization.sh):

    python train_vecfield.py data/78481root_5w.obj \\
        --constraints data/78481root_5w_constraints_v3.npz \\
        --theta 0 --bsp-res 500 --no-profile --out-dir results/single/78481root_5w

    python train_vecfield.py data/botijo.obj \\
        --constraints data/botijo_constraints_v0.npz \\
        -k 128 --bsp-res 500 --no-profile --out-dir results/generalization/botijo

Requires Linux, CUDA, fvdb and fcpw (the SDF grid is built with fvdb).
"""

import argparse
import os
import numpy as np
import torch

from Mesh_geometry import MeshGeometry
from SparseGridFDM_SDF import SparseGridFDM_SDF
from SurfaceVectorFieldDesign import SurfaceVectorFieldDesign


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Hodge surface vector field design (SDF + neural phi, psi)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "mesh",
        nargs="?",
        default="Bunny.obj",
        help="Path to mesh .obj (default: Bunny.obj)",
    )
    p.add_argument(
        "--constraints",
        metavar="PATH.npz",
        type=str,
        default=None,
        help="Optional NumPy archive: arrays 'face_ids' (int) and 'vectors' (N,3) tangent targets",
    )
    p.add_argument(
        "-k",
        "--export-k",
        type=int,
        default=1,
        dest="export_k",
        help="Number of theta angles in export (ignored if --theta is set on a field w/ fixed angle)",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=None,
        help="If set, fix rotation angle theta in training / export",
    )
    p.add_argument("--bsp-res", type=int, default=128, help="SDF grid resolution")
    p.add_argument("--padding", type=float, default=0.15)
    p.add_argument("--fd-halo", type=int, default=2)
    p.add_argument("--no-normalize", action="store_true")

    p.add_argument("--input-size", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--n-random-constraints", type=int, default=10)
    p.add_argument("--num-epochs", type=int, default=3000)
    p.add_argument("--num-surface", type=int, default=10000)
    p.add_argument("--num-narrowband", type=int, default=10000)
    p.add_argument("--band-width", type=float, default=0.01)
    p.add_argument("--lambda-data", type=float, default=100.0)
    p.add_argument("--lambda-smooth-phi", type=float, default=1.0)
    p.add_argument("--lambda-smooth-psi", type=float, default=1.0)
    p.add_argument("--lambda-cpm", type=float, default=1000.0)
    p.add_argument("--lambda-mean", type=float, default=5.0)
    p.add_argument("--lambda-orth", type=float, default=1.0)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-dir", type=str, default="checkpts_vecfield")
    p.add_argument("--out-dir", type=str, default="vecfield_results")
    p.add_argument("--export-batch-size", type=int, default=4096)
    p.add_argument(
        "--no-profile", dest="profile", action="store_false",
        help="Disable training step profiling",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mesh = MeshGeometry(
        args.mesh, normalize=not args.no_normalize, device=device
    )

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

    print("\n" + "=" * 60)
    print("Stage 2: Learning Hodge Vector Field")
    print("  Using FDM SDF geometry oracle (chain-rule surface operators)")
    print("=" * 60)

    face_ids, vectors = None, None
    if args.constraints and os.path.exists(args.constraints):
        c = np.load(args.constraints)
        face_ids = c["face_ids"]
        vectors = c["vectors"]
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12) * (2.0 * mesh.avg_edge)
        print(f"Scaled constraint vectors by 2*avg_edge = {2.0 * mesh.avg_edge:.6g}")
        print(f"Loaded {len(face_ids)} constraints from {args.constraints}")
        print("mesh.avg_edge =", mesh.avg_edge)
        print("target norms =", np.linalg.norm(vectors, axis=1))
        print("target / avg_edge =", np.linalg.norm(vectors, axis=1) / mesh.avg_edge)
    elif args.constraints:
        print(f"Warning: --constraints {args.constraints} not found, using random")

    designer = SurfaceVectorFieldDesign(
        input_size=args.input_size,
        hidden=args.hidden,
        lr=args.lr,
        device=device,
        sdf=sdf,
    )

    designer.train(
        mesh=mesh,
        constraint_face_ids=face_ids,
        constraint_vectors=vectors,
        n_random_constraints=args.n_random_constraints,
        num_epochs=args.num_epochs,
        num_surface=args.num_surface,
        num_narrowband=args.num_narrowband,
        band_width=args.band_width,
        lambda_data=args.lambda_data,
        lambda_smooth_phi=args.lambda_smooth_phi,
        lambda_smooth_psi=args.lambda_smooth_psi,
        lambda_cpm=args.lambda_cpm,
        lambda_mean=args.lambda_mean,
        lambda_orth=args.lambda_orth,
        clip_grad=args.clip_grad,
        log_every=args.log_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        fixed_theta=args.theta,
        profile=args.profile,
    )

    out_dir = args.out_dir
    designer.export_k(
        mesh, out_dir=out_dir, k=args.export_k, batch_size=args.export_batch_size
    )
    designer.save(os.path.join(out_dir, "model.pt"))

    print(f"\nDone! Visualize with:")
    print(f"  python SurfaceVectorFieldDesign.py --vis {out_dir}/vecfield_000.npz")


if __name__ == "__main__":
    main()
