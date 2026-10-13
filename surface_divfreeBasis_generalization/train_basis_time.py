"""
train_basis_time.py - Generalize the div-free vector Laplacian basis to a
time-varying surface family.

Sources
-------
  --source torus       Wavy torus       (analytical)
  --source icosphere   Radial icosphere (analytical)
  --source obj-seq     Numbered .obj files in a folder

Differences from the single-shape training (train_basis.py of the
single-model code)
-------------------------------------------------------------------
  * No SDF (per-frame SDF is too heavy).
  * CPM loss uses on-surface (point, matching normal) pairs from
    mesh.sample_surface - no narrow band.
  * Normalized time t is fed to the network's 1-D `code` input, so the
    learned psi_k(x, n, t) is shared across all frames.
  * Optional `--normalize shared`: a SINGLE global (center, scale) computed
    over all frames is applied once. Per-frame normalize is intentionally
    not offered: each frame would otherwise use its own (center_t, scale_t),
    which silently cancels rigid motion / scale drift in normalized coords
    and breaks time correspondence. Shared normalize is just a global
    similarity transform - fine to invert at the end.

Usage
-----
    python train_basis_time.py --source torus     --num-frames 8 -k 10
    python train_basis_time.py --source icosphere --num-frames 8 -k 10
    python train_basis_time.py --source obj-seq   --obj-dir data/hand_2_top60 -k 10 --normalize shared
    See run_train_basis_time.sh for a full list of options.

Output in --out-dir: model.pt and one folder per frame, frame_0000/ ...,
each with basis_000.npz ... (one file per mode) and meta.npz (t, code).

    python SurfaceDFVecLaplacianBasis.py --vis results/hand_2_top60/frame_0000/basis_000.npz
"""

import argparse
import glob
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from Mesh_geometry import MeshGeometry, load_obj
from SurfaceDFVecLaplacianBasis import SurfaceDFVecLaplacianBasis
from surfaces import wavy_torus, radial_icosphere


class SurfaceDFVecLaplacianBasisTime(SurfaceDFVecLaplacianBasis):
    """Solution-2: keep [x, n, code] input shape, but zero the n channel
    on input. Tangent projection, u = n x gradpsi, and CPM dot-product still
    use the REAL n inside _surface_ops / _cpm_loss - only the network's
    *input* sees a constant n.

    Why: detaching n alone (parent's no-SDF path) only suppresses J_n^T d_n Psi
    in autograd; the network can still modulate psi-amplitude through n
    (e.g. psi(x,n,t) = a(n,t).f(x,t), where a(n) is "free"). Zeroing n on
    input forces psi = psi(x, code). To switch to a real SDF chain-rule
    variant later, just delete the zero line.
    """
    def _make_input(self, pts_leaf, normals, code_value):
        N = pts_leaf.shape[0]
        code = torch.full((N, 1), float(code_value),
                          device=self.device, dtype=pts_leaf.dtype)
        n_const = torch.zeros_like(normals)        # <-- Solution 2
        return torch.cat([pts_leaf, n_const, code], dim=1)

# ============================================================
# Frame setup
# ============================================================

def write_obj(path, V, F):
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in F:
            fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def build_frames(args, device):
    """Return list of (mesh, code, t, path) tuples."""
    times, src_paths = [], []

    # ---- Generate / collect raw OBJ paths ----
    if args.source in ("torus", "icosphere"):
        ts = np.linspace(args.t_min, args.t_max, args.num_frames)
        work_dir = tempfile.mkdtemp(prefix=f"{args.source}_")
        print(f"  (analytical frames -> {work_dir})")
        for k, t in enumerate(ts):
            if args.source == "torus":
                V, F = wavy_torus(t, nu=args.torus_nu, nv=args.torus_nv)
            else:
                V, F = radial_icosphere(t, subdiv=args.ico_subdiv)
            p = os.path.join(work_dir, f"frame_{k:04d}.obj")
            write_obj(p, V, F)
            times.append(float(t))
            src_paths.append(p)

    elif args.source == "obj-seq":
        files = sorted(glob.glob(os.path.join(args.obj_dir, args.obj_pattern)))
        if not files:
            raise SystemExit(
                f"No .obj matching '{args.obj_pattern}' in {args.obj_dir}"
            )
        for k, p in enumerate(files):
            src_paths.append(p)
            times.append(float(k))

    else:
        raise SystemExit(f"unknown source: {args.source}")

    # ---- Optional shared normalization (single global transform) ----
    if args.normalize == "shared":
        all_v = np.concatenate([load_obj(p)[0] for p in src_paths], axis=0)
        center = (all_v.max(axis=0) + all_v.min(axis=0)) / 2
        scale = float(np.sqrt(((all_v - center) ** 2).sum(axis=1)).max())
        print(f"  Shared normalize: center={center}, scale={scale:.4f}")
        norm_dir = tempfile.mkdtemp(prefix="shared_norm_")
        new_paths = []
        for k, p in enumerate(src_paths):
            V, F = load_obj(p)
            V = (V - center) / scale
            dst = os.path.join(norm_dir, f"frame_{k:04d}.obj")
            write_obj(dst, V, F)
            new_paths.append(dst)
        src_paths = new_paths

    # ---- Time -> code in [0, 1] ----
    times = np.array(times, dtype=float)
    if times.max() > times.min():
        codes = (times - times.min()) / (times.max() - times.min())
    else:
        codes = np.zeros_like(times)

    # ---- Build MeshGeometry (never per-frame normalize) ----
    frames = []
    for path, t, c in zip(src_paths, times, codes):
        mesh = MeshGeometry(path, normalize=False, device=device)
        frames.append((mesh, float(c), float(t), path))
        print(f"  frame {len(frames)-1:03d}: t={t:.4f}  code={c:.4f}  "
              f"({os.path.basename(path)})")
    return frames


# ============================================================
# Training: random-frame SGD with surface-only CPM
# ============================================================

def train_time_varying(basis, frames, args):
    """
    At each iteration: pick a random frame, sample on-surface (point, normal)
    pairs (used for both forward_surface and CPM), step each of the k bases
    once with sequential deflation. No narrow band, no SDF.
    """
    dev = basis.device
    eps = args.eps
    n_frames = len(frames)
    rng = np.random.default_rng(0)

    for m in basis.models:
        m.train()

    for it in range(1, args.num_epochs + 1):
        idx = int(rng.integers(0, n_frames))
        mesh, code, t, _ = frames[idx]

        # On-surface samples (matching normals) for the eigenproblem
        surf_pts, surf_normals = mesh.sample_surface(args.num_surface, device=dev)
        surf_normals = surf_normals.float().detach()

        # Separate on-surface sample for CPM loss (matched (x, n) pairs)
        cpm_pts, cpm_normals = mesh.sample_surface(args.num_cpm, device=dev)
        cpm_normals = cpm_normals.float().detach()

        U_prev, G_prev = None, None
        ls = {"energy": 0., "ortho": 0., "norm": 0., "mean": 0., "cpm": 0.}

        for k_idx in range(basis.k):
            # Forward + CPM
            u, omega, psi, _ = basis.forward_surface(
                basis.models[k_idx], surf_pts, code, surf_normals
            )
            cpm_vals = basis._cpm_loss(
                basis.models[k_idx], cpm_pts, cpm_normals, code
            )

            u_vec = u.reshape(-1, 1)
            omega_vec = omega.reshape(-1, 1)

            u_proj, reconstruct, q = basis.project_with_basis(
                u_vec, U_prev, args.ridge
            )
            omega_proj = omega_vec if U_prev is None else omega_vec - G_prev @ q

            u_proj_energy = (u_proj ** 2).mean()
            scale_norm = torch.sqrt(u_proj_energy + eps)
            omega_unit = omega_proj / scale_norm

            loss_energy = (omega_unit ** 2).mean()
            loss_ortho = (reconstruct ** 2).mean() / ((u_vec ** 2).mean() + eps)
            loss_norm = torch.log(u_proj_energy + 1e-8).pow(2)
            loss_mean = psi.mean().pow(2)
            loss_cpm = cpm_vals.mean()

            loss = (loss_energy
                    + args.lambda_ortho * loss_ortho
                    + args.lambda_norm * loss_norm
                    + args.lambda_mean * loss_mean
                    + args.lambda_cpm * loss_cpm)

            basis.optimizers[k_idx].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                basis.models[k_idx].parameters(), args.clip_grad
            )
            basis.optimizers[k_idx].step()

            # Refresh deflation buffer (re-eval after step, then detach)
            u2, omega2, _, _ = basis.forward_surface(
                basis.models[k_idx], surf_pts, code, surf_normals
            )
            u2_vec = u2.reshape(-1, 1).detach()
            omega2_vec = omega2.reshape(-1, 1).detach()
            u2_proj, _, q2 = basis.project_with_basis(
                u2_vec, U_prev, args.ridge
            )
            omega2_proj = (omega2_vec if U_prev is None
                           else omega2_vec - G_prev @ q2)
            scale2 = torch.sqrt((u2_proj ** 2).mean() + eps)
            u2_unit = (u2_proj / scale2).detach()
            omega2_unit = (omega2_proj / scale2).detach()
            if U_prev is None:
                U_prev, G_prev = u2_unit, omega2_unit
            else:
                U_prev = torch.cat([U_prev, u2_unit], dim=1)
                G_prev = torch.cat([G_prev, omega2_unit], dim=1)

            ls["energy"] += float(loss_energy.detach())
            ls["ortho"]  += float(loss_ortho.detach())
            ls["norm"]   += float(loss_norm.detach())
            ls["mean"]   += float(loss_mean.detach())
            ls["cpm"]    += float(loss_cpm.detach())

        if it == 1 or it % args.log_every == 0 or it == args.num_epochs:
            print(f"  [{it}/{args.num_epochs}]  frame={idx:03d}  t={t:.3f}  "
                  f"E={ls['energy']:.4f}  ortho={ls['ortho']:.6f}  "
                  f"norm={ls['norm']:.4f}  cpm={ls['cpm']:.6f}")

        if args.save_dir and args.save_every and it % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt = os.path.join(args.save_dir, f"basis_iter_{it}.pt")
            basis.save(ckpt)


# ============================================================
# Export every frame
# ============================================================

def export_all_frames(basis, frames, out_dir, batch_size=4096):
    os.makedirs(out_dir, exist_ok=True)
    for k, (mesh, code, t, src) in enumerate(frames):
        sub = os.path.join(out_dir, f"frame_{k:04d}")
        basis.export(mesh, out_dir=sub, code_value=code, batch_size=batch_size)
        np.savez(os.path.join(sub, "meta.npz"),
                 t=t, code=code, frame_idx=k, source=str(src))
    print(f"Exported {len(frames)} frames to {out_dir}/")


# ============================================================
# CLI
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Train div-free vector Laplacian basis across a "
                    "time-varying surface family (mesh-only, surface-CPM, "
                    "no SDF)."
    )
    # Source
    p.add_argument("--source", required=True,
                   choices=["torus", "icosphere", "obj-seq"])
    p.add_argument("--num-frames", type=int, default=8,
                   help="(torus/icosphere) number of t samples")
    p.add_argument("--t-min", type=float, default=0.0)
    p.add_argument("--t-max", type=float, default=2 * np.pi)
    p.add_argument("--torus-nu", type=int, default=128)
    p.add_argument("--torus-nv", type=int, default=64)
    p.add_argument("--ico-subdiv", type=int, default=4)
    p.add_argument("--obj-dir", type=str, help="(obj-seq) folder of .obj files")
    p.add_argument("--obj-pattern", type=str, default="*.obj")

    # Normalization
    p.add_argument("--normalize", choices=["none", "shared"], default="none",
                   help="'none': use frames as-is. "
                        "'shared': single global (center, scale) across ALL "
                        "frames - preserves time correspondence.")

    # Basis / training
    p.add_argument("-k", "--num-basis", type=int, default=10)
    p.add_argument("--num-epochs", type=int, default=4000,
                   help="total outer iterations (one random frame per iter)")
    p.add_argument("--input-size", type=int, default=3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-surface", type=int, default=10000)
    p.add_argument("--num-cpm", type=int, default=10000,
                   help="on-surface samples used for CPM loss")
    p.add_argument("--lambda-ortho", type=float, default=1e3)
    p.add_argument("--lambda-norm", type=float, default=1000.0)
    p.add_argument("--lambda-mean", type=float, default=0.0)
    p.add_argument("--lambda-cpm", type=float, default=1000.0)
    p.add_argument("--ridge", type=float, default=1e-8)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--clip-grad", type=float, default=1.0)

    # Logging / IO
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--save-dir", type=str, default="checkpts_time_basis")
    p.add_argument("--out-dir", type=str, default="time_basis_results")
    p.add_argument("--export-batch-size", type=int, default=4096)
    return p


def main():
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.source == "obj-seq" and not args.obj_dir:
        raise SystemExit("--obj-dir is required for --source obj-seq")

    print(f"\nBuilding frames (source={args.source}, normalize={args.normalize}) ...")
    frames = build_frames(args, device)

    print("\n" + "=" * 60)
    print(f"Training {args.num_basis} bases over {len(frames)} frames "
          f"({args.num_epochs} iters, surface-CPM, no SDF)")
    print("=" * 60)

    basis = SurfaceDFVecLaplacianBasisTime(
        num_basis=args.num_basis,
        input_size=args.input_size,
        hidden=args.hidden,
        lr=args.lr,
        device=device,
        sdf=None,
    )

    train_time_varying(basis, frames, args)

    print("\nExporting per-frame NPZ ...")
    export_all_frames(basis, frames, args.out_dir, args.export_batch_size)
    basis.save(os.path.join(args.out_dir, "model.pt"))

    print(f"\nDone! Visualize a frame with:")
    print(f"  python SurfaceDFVecLaplacianBasis.py --vis "
          f"{args.out_dir}/frame_0000/basis_000.npz")


if __name__ == "__main__":
    main()
