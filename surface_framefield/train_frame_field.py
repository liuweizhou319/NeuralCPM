"""
Train dual-potential frame field with fixed sparse-grid SDF oracle
===================================================================

Usage:
    python train_frame_field.py data/rockerarm.obj --sdf-res 500 --extract-feature-curves \\
        --out-dir results/rockerarm
    See run_train_frame_field.sh for the options used in the experiments.

Output in --out-dir (default: frame_field_results): the face-centre field
<mesh name>.npz (or --output-npz), frame_field.png / frame_field.npz (a plot
of random surface samples), model.pt, and checkpoints/.

Pipeline:
    Step 0: Build SparseGridFDM_SDF oracle (fixed geometry)
    Step 1: Extract feature curves via libigl
    Step 2: Train dual potentials u(x,n), v(x,n)

CLI: ``--dual-surface-smooth`` - random surface samples use kNN SH smoothness; all face barycenters
use SH Dirichlet along edge-adjacent face pairs (see SurfaceFrameFieldSH.train).

CLI: ``--surface-placement per_face_adj`` (default) - one random point **per triangle** each epoch;
surface smoothness uses edge-adjacent face pairs (sparse, O(|F|) pairs), not Euclidean kNN.
``--surface-placement random`` - ``--num-surface`` random points, Euclidean kNN smoothness.

CLI: ``--curv-weight-mode scale01|bottom_floor`` - rescale ``|||kappa_1|-|kappa_2|||`` to feature weights in
``[0,1]`` (needs ``--feat-align-weight abs_abs_diff``). ``scale01`` = min-max per batch;
``bottom_floor`` uses ``--curv-weight-floor`` and ``--curv-bottom-q`` (same as ``test_crossfield``).

CLI: ``--lr-schedule fixed`` (default) - constant ``--lr-fixed`` (default ``5e-5``), no scheduler.
``--lr-schedule cosine`` - ``CosineAnnealingLR`` from ``--lr`` (default ``1e-3``) to ``1e-6``.

CLI: ``--cpm-schedule ramp`` (default) - weight ``0`` for the first ``--cpm-ramp-hold-frac``
fraction of training (default ``0.2``), then linear from ``--lambda-cpm-ramp-start``
(default ``0``) to ``--lambda-cpm`` over the rest. ``--cpm-schedule fixed`` - ``--lambda-cpm``
constant.

CLI: grid, network, sampling and loss weights (the defaults are the values of the experiments):
``--sdf-padding 0.15``, ``--fd-halo 4``; ``--hidden 128``, ``--n-layers 4``;
``--num-epochs 10000``, ``--num-narrowband 30000``, ``--band-width 0.015``, ``--clip-grad 1.0``,
``--log-every 50``, ``--save-every 2000``; ``--num-surface 30000`` and ``--knn-k 6`` (both used
only with ``--surface-placement random``); ``--lambda-smooth 100``, ``--lambda-feat 100``,
``--lambda-feat-edge 100``, ``--lambda-cpm 100``.

CLI: ``--output-npz`` / ``--export-npz`` - same option (alias): filename or path for
``evaluate_mesh_face_centers`` output. Default is ``<mesh_basename>.npz`` under
``--out-dir`` (e.g. ``./data/gargoyle.obj`` -> ``<out-dir>/gargoyle.npz``).
If both flags appear, ``--output-npz`` wins. A basename gets ``.npz`` appended if missing;
absolute paths or paths containing ``/`` are written as-is (parent dirs created).

CLI: ``--extract-feature-curves`` - if present, run libigl sharp-edge feature sampling
(``extract_feature_curves``). If omitted, no extraction and the extra ``L_feat`` on feature
points is skipped (empty point set).

CLI: ``--feat-weight-schedule fixed|anneal`` - ``fixed``: ``lambda_feat_eff = lambda_feat`` always;
``anneal`` (default): piecewise ramp / plateau / cosine decay on ``lambda_feat`` (see
``lambda_feat_schedule_ratio``). Optional ``--feat-schedule-plateau-start`` and
``--feat-schedule-plateau-end`` (defaults ``0.2``, ``0.8``; must satisfy ``0 < start < end < 1``).

Losses (3 only):
    L_smooth : SH Dirichlet via Hutchinson autograd
    L_feat   : symmetry-aware octahedral alignment on feature curves
    L_rank   : anti-collapse (min singular value)
"""

import os
import sys

import numpy as np
import torch

from Mesh_geometry import MeshGeometry
from SparseGridFDM_SDF import SparseGridFDM_SDF
from SurfaceFrameFieldSH import SurfaceFrameFieldSH, extract_feature_curves


def arg_value(flag, default, cast=float):
    """Value given after ``flag`` on the command line, or ``default``."""
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mesh_path = 'Bunny.obj'
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        mesh_path = sys.argv[1]

    mesh_basename = os.path.splitext(os.path.basename(mesh_path))[0]
    # Face-center NPZ: default = mesh stem + .npz; override with --output-npz or --export-npz
    export_npz_arg = mesh_basename + ".npz"
    if "--export-npz" in sys.argv:
        export_npz_arg = sys.argv[sys.argv.index("--export-npz") + 1]
    if "--output-npz" in sys.argv:
        export_npz_arg = sys.argv[sys.argv.index("--output-npz") + 1]

    # Output folder: face-centre npz, plot, model.pt and checkpoints/
    out_dir = 'frame_field_results'
    if '--out-dir' in sys.argv:
        out_dir = sys.argv[sys.argv.index('--out-dir') + 1]

    feat_angle = 30.0
    if '--feat-angle' in sys.argv:
        feat_angle = float(sys.argv[sys.argv.index('--feat-angle') + 1])

    sdf_res = 128
    if '--sdf-res' in sys.argv:
        sdf_res = int(sys.argv[sys.argv.index('--sdf-res') + 1])

    smooth_p = 2.0
    if '--smooth-p' in sys.argv:
        smooth_p = float(sys.argv[sys.argv.index('--smooth-p') + 1])

    # L_feat weight: k1_minus_k2 = |kappa1-kappa2|, abs_abs_diff = ||kappa1|-|kappa2||
    feat_align_weight = 'abs_abs_diff'
    if '--feat-align-weight' in sys.argv:
        feat_align_weight = sys.argv[sys.argv.index('--feat-align-weight') + 1]
    if feat_align_weight not in ('k1_minus_k2', 'abs_abs_diff'):
        raise SystemExit(
            "--feat-align-weight must be k1_minus_k2 or abs_abs_diff "
            f"(got {feat_align_weight!r})")

    dual_surface_smooth = '--dual-surface-smooth' in sys.argv

    surface_placement = 'per_face_adj'
    if '--surface-placement' in sys.argv:
        surface_placement = sys.argv[sys.argv.index('--surface-placement') + 1]
        if surface_placement not in ('random', 'per_face_adj'):
            raise SystemExit(
                "--surface-placement must be random or per_face_adj "
                f"(got {surface_placement!r})")

    curv_weight_mode = None
    if "--curv-weight-mode" in sys.argv:
        curv_weight_mode = sys.argv[sys.argv.index("--curv-weight-mode") + 1].lower()
        if curv_weight_mode in ("none", "off"):
            curv_weight_mode = None
        elif curv_weight_mode not in ("scale01", "bottom_floor"):
            raise SystemExit(
                "--curv-weight-mode must be none, off, scale01, or bottom_floor "
                f"(got {curv_weight_mode!r})")
    curv_weight_floor = 0.001
    if "--curv-weight-floor" in sys.argv:
        curv_weight_floor = float(sys.argv[sys.argv.index("--curv-weight-floor") + 1])
    curv_bottom_q = 0.2
    if "--curv-bottom-q" in sys.argv:
        curv_bottom_q = float(sys.argv[sys.argv.index("--curv-bottom-q") + 1])
    if curv_weight_mode is not None:
        if feat_align_weight != "abs_abs_diff":
            raise SystemExit(
                "--curv-weight-mode requires --feat-align-weight abs_abs_diff")
        if curv_weight_mode == "bottom_floor":
            if not (0.0 < curv_bottom_q < 1.0):
                raise SystemExit("--curv-bottom-q must be strictly between 0 and 1.")
            if not (0.0 < curv_weight_floor < 1.0):
                raise SystemExit("--curv-weight-floor must be strictly between 0 and 1.")

    lr_schedule = "fixed"
    if "--lr-schedule" in sys.argv:
        lr_schedule = sys.argv[sys.argv.index("--lr-schedule") + 1].lower()
        if lr_schedule not in ("cosine", "fixed"):
            raise SystemExit(
                "--lr-schedule must be cosine or fixed "
                f"(got {lr_schedule!r})")

    cpm_schedule = "ramp"
    if "--cpm-schedule" in sys.argv:
        cpm_schedule = sys.argv[sys.argv.index("--cpm-schedule") + 1].lower()
        if cpm_schedule not in ("fixed", "ramp"):
            raise SystemExit(
                "--cpm-schedule must be fixed or ramp "
                f"(got {cpm_schedule!r})")
    lambda_cpm_ramp_start = 0.0
    if "--lambda-cpm-ramp-start" in sys.argv:
        lambda_cpm_ramp_start = float(
            sys.argv[sys.argv.index("--lambda-cpm-ramp-start") + 1])
    cpm_ramp_hold_frac = 0.2
    if "--cpm-ramp-hold-frac" in sys.argv:
        cpm_ramp_hold_frac = float(
            sys.argv[sys.argv.index("--cpm-ramp-hold-frac") + 1])
        if not (0.0 <= cpm_ramp_hold_frac < 1.0):
            raise SystemExit("--cpm-ramp-hold-frac must be in [0, 1).")

    ref_mode = 'principal'
    if '--ref-mode' in sys.argv:
        ref_mode = sys.argv[sys.argv.index('--ref-mode') + 1]
        assert ref_mode in ('normal', 'principal', 'axis_swap'), \
            f"ref_mode must be 'normal', 'principal', or 'axis_swap', got {ref_mode}"
    print(f"Using reference mode: {ref_mode}")

    extract_feature_curves_flag = '--extract-feature-curves' in sys.argv

    feat_weight_schedule = "anneal"
    if "--feat-weight-schedule" in sys.argv:
        feat_weight_schedule = sys.argv[
            sys.argv.index("--feat-weight-schedule") + 1].lower()
    if feat_weight_schedule not in ("fixed", "anneal"):
        raise SystemExit(
            "--feat-weight-schedule must be fixed or anneal "
            f"(got {feat_weight_schedule!r})")

    feat_edge_target = "mincurv"
    if "--feat-edge-target" in sys.argv:
        feat_edge_target = sys.argv[
            sys.argv.index("--feat-edge-target") + 1].lower()
    if feat_edge_target not in ("mincurv", "tangent"):
        raise SystemExit(
            "--feat-edge-target must be mincurv or tangent "
            f"(got {feat_edge_target!r})")

    feat_schedule_plateau_start = 0.2
    if "--feat-schedule-plateau-start" in sys.argv:
        feat_schedule_plateau_start = float(
            sys.argv[sys.argv.index("--feat-schedule-plateau-start") + 1])
    feat_schedule_plateau_end = 0.8
    if "--feat-schedule-plateau-end" in sys.argv:
        feat_schedule_plateau_end = float(
            sys.argv[sys.argv.index("--feat-schedule-plateau-end") + 1])
    if not (0.0 < feat_schedule_plateau_start < feat_schedule_plateau_end < 1.0):
        raise SystemExit(
            "--feat-schedule-plateau-start / --feat-schedule-plateau-end must satisfy "
            "0 < start < end < 1.")

    # Grid, network, sampling and loss weights (defaults: the values of the experiments)
    sdf_padding = arg_value('--sdf-padding', 0.15)
    fd_halo = arg_value('--fd-halo', 4, int)
    hidden = arg_value('--hidden', 128, int)
    n_layers = arg_value('--n-layers', 4, int)
    lr = arg_value('--lr', 1e-3)
    lr_fixed = arg_value('--lr-fixed', 5e-5)
    num_epochs = arg_value('--num-epochs', 10000, int)
    num_surface = arg_value('--num-surface', 30000, int)
    num_narrowband = arg_value('--num-narrowband', 30000, int)
    band_width = arg_value('--band-width', 0.015)
    knn_k = arg_value('--knn-k', 6, int)
    clip_grad = arg_value('--clip-grad', 1.0)
    log_every = arg_value('--log-every', 50, int)
    save_every = arg_value('--save-every', 2000, int)
    lambda_smooth = arg_value('--lambda-smooth', 100.0)
    lambda_feat = arg_value('--lambda-feat', 100.0)
    lambda_feat_edge = arg_value('--lambda-feat-edge', 100.0)
    lambda_cpm = arg_value('--lambda-cpm', 100.0)

    # ================================================================
    # Step 0: Load mesh & build fixed oracle
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 0: Build sparse-grid SDF oracle")
    print("=" * 60)

    mesh = MeshGeometry(mesh_path, normalize=True, device=device)

    oracle = SparseGridFDM_SDF.from_mesh(
        mesh, resolution=sdf_res, padding=sdf_padding, fd_halo=fd_halo, device=device)

    # ================================================================
    # Step 1: Extract feature curves (optional)
    # ================================================================
    if extract_feature_curves_flag:
        print("\n" + "=" * 60)
        print(f"Step 1: Extract feature curves (threshold={feat_angle} deg)")
        print("=" * 60)
        feat_pts, feat_tangents, _, feat_face_pairs = extract_feature_curves(
            mesh.vertices_np, mesh.faces_np,
            mesh.face_normals_np, angle_deg=feat_angle)
        print(f"  Got {len(feat_pts)} feature sample points")
    else:
        print("\n" + "=" * 60)
        print("Step 1: Feature curves skipped (pass --extract-feature-curves to enable)")
        print("=" * 60)
        feat_pts = np.zeros((0, 3), dtype=np.float32)
        feat_tangents = np.zeros((0, 3), dtype=np.float32)
        feat_face_pairs = np.zeros((0, 2), dtype=np.int64)

    # ================================================================
    # Step 2: Train frame field
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 2: Train frame field")
    print("=" * 60)

    trainer = SurfaceFrameFieldSH(
        sdf_oracle=oracle,
        hidden=hidden,
        n_layers=n_layers,
        omega_0=30.0,
        lr=lr,
        device=device,
    )

    trainer.train(
        mesh=mesh,
        feat_pts_np=feat_pts,
        feat_tangents_np=feat_tangents,
        feat_face_pairs_np=feat_face_pairs,
        num_epochs=num_epochs,
        num_surface=num_surface,
        num_narrowband=num_narrowband,
        lambda_smooth=lambda_smooth,
        lambda_feat=lambda_feat,
        lambda_feat_edge=lambda_feat_edge,
        lambda_cpm=lambda_cpm,
        cpm_schedule=cpm_schedule,
        lambda_cpm_ramp_start=lambda_cpm_ramp_start,
        cpm_ramp_hold_frac=cpm_ramp_hold_frac,
        lambda_amp=0.0,
        tau_amp=0.0,
        band_width=band_width,
        knn_k=knn_k,
        smooth_p=smooth_p,
        clip_grad=clip_grad,
        use_w_curv=True,
        feat_align_weight=feat_align_weight,
        curv_weight_mode=curv_weight_mode,
        curv_weight_floor=curv_weight_floor,
        curv_bottom_quantile=curv_bottom_q,
        log_every=log_every,
        save_every=save_every,
        save_dir=os.path.join(out_dir, 'checkpoints'),
        dual_surface_smooth=dual_surface_smooth,
        surface_placement=surface_placement,
        lr_schedule=lr_schedule,
        fixed_lr=lr_fixed,
        ref_mode=ref_mode,
        feat_weight_schedule=feat_weight_schedule,
        feat_edge_target=feat_edge_target,
        feat_schedule_plateau_start=feat_schedule_plateau_start,
        feat_schedule_plateau_end=feat_schedule_plateau_end,
    )

    # ================================================================
    # Export
    # ================================================================
    os.makedirs(out_dir, exist_ok=True)

    export_npz = export_npz_arg.strip()
    if not export_npz.lower().endswith(".npz"):
        export_npz += ".npz"
    if os.path.isabs(export_npz) or os.path.dirname(export_npz):
        mesh_npz_path = export_npz
        os.makedirs(os.path.dirname(os.path.abspath(mesh_npz_path)) or ".", exist_ok=True)
    else:
        mesh_npz_path = os.path.join(out_dir, export_npz)

    trainer.visualize(
        sample_surface_fn=mesh.sample_surface,
        N=5000,
        save_path=os.path.join(out_dir, 'frame_field.png'),
        ref_mode=ref_mode,
    )

    trainer.evaluate_mesh_face_centers(
        mesh,
        save_path=mesh_npz_path,
        ref_mode=ref_mode,
        feat_align_weight=feat_align_weight,
        curv_weight_mode=curv_weight_mode,
        curv_weight_floor=curv_weight_floor,
        curv_bottom_quantile=curv_bottom_q,
        include_feat_curves=extract_feature_curves_flag,
        feat_curve_angle_deg=feat_angle,
        feat_edge_target=feat_edge_target,
    )
    print(f"  Face-center field npz -> {mesh_npz_path}")

    trainer.save(os.path.join(out_dir, 'model.pt'))
    print("Done!")


if __name__ == '__main__':
    main()
