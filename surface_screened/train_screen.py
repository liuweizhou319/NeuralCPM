r"""
train_screen.py - train a neural screened-diffusion field with several
Dirichlet sources on a triangle mesh.

Role in the pipeline
--------------------
This is the training entry point.  In this order it

  1. seeds NumPy and PyTorch (--seed);
  2. loads the OBJ mesh and normalizes it (MeshGeometry);
  3. composes the list of Dirichlet sources from --source-spec and from the
     inline shortcuts;
  4. builds the sparse-grid SDF oracle of the surface
     (SparseGridFDM_SDF.from_mesh);
  5. trains the network u (Surfacescreendiffusion.train) by minimizing

         L_src + lambda_dir * L_dir + lambda_mass * L_mass
               + lambda_cpm * L_cpm + lambda_range * L_range

     where L_dir is the mean of |grad u|^2 and L_mass the mean of u^2 over
     surface samples, L_src pulls u towards the target value a_i on every
     source (weighted by lambda_i, averaged over the sources), L_cpm is the
     closest-point (CPM) term on narrow-band and surface samples, and
     L_range is an optional soft clamp (off by default);
  6. evaluates the field at the mesh vertices and saves it together with the
     checkpoint.

fem_screen.py solves the same problem with finite elements from the same mesh
and the same source spec; it is the reference the network is compared with.

Source spec (--source-spec)
---------------------------
A JSON file containing a list of source objects.  Each object describes one
Dirichlet source, i.e. a set of surface points on which u is pulled towards a
target value:

    [
      { "type": "vertex",     "value": 1.0,  "weight": 1.0,
        "ids": [100, 250],    "patch_radius": 0.0 },

      { "type": "disk_curve", "value": -1.0, "weight": 2.0,
        "seed": 9876, "radius": 0.2 },

      { "type": "curve",      "value": 0.5,  "weight": 1.0,
        "points": [[0.1, 0.0, 0.3], [0.2, 0.0, 0.3], [0.3, 0.1, 0.3]],
        "closed": false }
    ]

Fields of every source:

    type              "vertex", "disk_curve" or "curve".  Required.
    value             float, target value a_i.  Default 1.0 (fem_screen.py
                      requires the field, so always write it).
    weight            float, lambda_i in L_src.  Default 1.0.
    value_generalize  bool, default false.  If true, the target of this
                      source is not `value` but the source's own component
                      of the conditioning code (see "Dirichlet-value
                      generalization" below).
    value_range       [min, max] with min < max, default [-2, 2].  Only read
                      when value_generalize is true: the interval the target
                      is drawn from.  All value-generalized sources must use
                      the same range.

Fields of type "vertex" (a set of mesh vertices):

    ids               list of zero-based vertex indices.  Required, not
                      empty.
    patch_radius      float, default 0.0.  If > 0, the source is enlarged to
                      every vertex whose graph distance to one of `ids` is
                      <= patch_radius.

Fields of type "disk_curve" (the iso-contour "graph distance to the seed
vertex = radius", i.e. the rim of a disk around the seed, not its interior):

    seed              zero-based index of the seed vertex.  Required.
    radius            float, default 0.1 (fem_screen.py requires the field).
    generalize        bool, default false.  Legacy disk-radius
                      generalization: the radius is not `radius` but a
                      conditioning code drawn from r_range every epoch.
    r_range           [r_min, r_max] with r_min <= r_max.  Required when
                      generalize is true; all such disks must use the same
                      range.

Fields of type "curve" (a polyline):

    points            [[x, y, z], ...], at least 2 points, given in the
                      normalized mesh frame.
    points_npy        alternative to `points`: path of a .npy file holding a
                      (K, 3) array.  It is loaded with numpy.load relative to
                      the current working directory and replaces `points`.
                      fem_screen.py does the same;
                      evaluate_dirichlet_generalization.py resolves the path
                      relative to the JSON file, and export_split_samples.py
                      / export_worst_rel_l2.py need inline `points`.
    closed            bool, default false.  If true, the last point is joined
                      to the first one.

Units.  The mesh is always normalized after loading: it is shifted by its
bounding-box centre and divided by its largest vertex norm.  Curve points,
radius, patch_radius, r_range and --band-width are expressed in that frame.
radius and patch_radius are graph (Dijkstra) distances on the mesh edge
graph, in which every interior edge is counted twice (a boundary edge once),
so on a closed mesh they measure 2x the length of the shortest edge path.
fem_screen.py uses the same convention,
which is why one source spec describes the same sources for the network and
for the FEM reference.  During training all source samples are projected
onto the surface with the closest-point map of the SDF.

Inline shortcuts
----------------
Sources can also be given on the command line.  Every --vertex-ids,
--disk-seed and --curve-npy adds one source; the companion flags
(--vertex-value, --vertex-weight, --vertex-patch-radius, --disk-radius,
--disk-value, --disk-weight, --disk-generalize, --curve-value,
--curve-weight, --curve-closed) are matched to them by order of appearance
and fall back to the defaults listed above when missing:

    --vertex-ids 100,250 --vertex-value 1.0
    --disk-seed 9876 --disk-radius 0.2 --disk-value -1.0

The final source list is: the JSON entries, then the --vertex-ids sources,
then the --disk-seed sources, then the --curve-npy sources.  The indices of
--value-generalize-source refer to this list.  make_screen_sources.py writes
the same shortcuts to a source-spec JSON instead.

Dirichlet-value generalization
------------------------------
--value-generalize-source INDEX (repeatable) marks source INDEX of the final
source list as value-generalized and sets its value_range to
[--value-min, --value-max]; for training this is the same as writing
value_generalize / value_range in the JSON.  The network receives one extra
input (code component) per marked source; components are assigned in
ascending source index.  Every epoch each component is drawn independently
and uniformly from the range and used as the Dirichlet value of its source,
so a single network is trained for all value combinations in the range.

Give the indices in ASCENDING order: training does not depend on the order of
the flags, but evaluate_dirichlet_generalization.py expects the same indices
in source order.

--export-value VALUE (repeated once per component, in component order)
chooses the code at which diffusion.npz is exported; the default is the
midpoint of the range for every component.  --train-codes-out PATH writes the
code vector of every epoch to a TSV file.

Disk-radius generalization (legacy)
-----------------------------------
A disk_curve source with generalize = true in the JSON, or an inline
--disk-seed source with --disk-generalize true and --r-min / --r-max, has a
random radius: every epoch one scalar is drawn uniformly from r_range, shared
by all such disks, and fed to the network as conditioning code.  --export-r
chooses the radius at which diffusion.npz is exported (default: midpoint of
r_range).  This mode cannot be combined with Dirichlet-value generalization.

Outputs
-------
    <out-dir>/model.pt        checkpoint (Surfacescreendiffusion.save):
                              network weights, optimizer state, the resolved
                              source list and the generalization metadata.
    <out-dir>/diffusion.npz   normalized mesh_vertices, mesh_faces, the field
                              u_vertex, grad_mag and the source geometry
                              (Surfacescreendiffusion.export).
    <out-dir>/diffusion_epoch_<epoch>.pt
                              intermediate checkpoints, written every N
                              epochs with --save-every N.

The SDF settings --sdf-res, --sdf-padding and --fd-halo are NOT stored in the
checkpoint.  The evaluation and export tools rebuild the SDF, so the values
used for training must be repeated on their command lines.

Requirements
------------
Linux, a CUDA GPU, PyTorch, fvdb and fcpw (the SDF oracle has no CPU
fallback), plus numpy and scipy.

Usage (from the repository root)
--------------------------------
Fixed Dirichlet values:

    python train_screen.py data/fandisk.obj \
        --source-spec data/fandisk_4v.json \
        --sdf-res 200 --fd-halo 6 \
        --out-dir results/single/fandisk

Dirichlet-value generalization over the four sources of the Bunny spec, with
values in [-2, 2]:

    python train_screen.py data/Bunny.obj \
        --source-spec data/Bunny_2.json \
        --value-generalize-source 0 --value-generalize-source 1 \
        --value-generalize-source 2 --value-generalize-source 3 \
        --value-min -2 --value-max 2 \
        --train-codes-out results/generalization/bunny/train_codes.txt \
        --out-dir results/generalization/bunny

run_single.sh and run_generalization.sh run these two kinds of training with
the hyper-parameters used in the experiments.
"""

import os
import json
import argparse
import numpy as np
import torch

from Mesh_geometry import MeshGeometry
from SparseGridFDM_SDF import SparseGridFDM_SDF
from Surfacescreendiffusion import Surfacescreendiffusion


def _parse_int_list(s):
    """Parse a comma-separated list of integers ('100,250' -> [100, 250])."""
    return [int(x) for x in s.split(',') if x.strip()]


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    """Define and parse the command line (see the module docstring)."""
    p = argparse.ArgumentParser()
    p.add_argument('mesh', type=str, help='mesh .obj path')

    # ---- main source mechanism: JSON file ----
    p.add_argument('--source-spec', type=str, default=None,
                   help='JSON file with a list of source dicts')

    # ---- inline shortcuts (action=append, repeatable) ----
    p.add_argument('--vertex-ids', action='append', default=[],
                   help="comma-separated IDs for one 'vertex' source; "
                        "repeat for multiple sources")
    p.add_argument('--vertex-value', action='append', default=[],
                   type=float, help='value for the matching --vertex-ids')
    p.add_argument('--vertex-weight', action='append', default=[],
                   type=float, help='weight for the matching --vertex-ids')
    p.add_argument('--vertex-patch-radius', action='append', default=[],
                   type=float,
                   help='Dijkstra patch radius (mesh-units) to fatten each '
                        'vertex source; 0 = bare vertex')

    p.add_argument('--disk-seed', action='append', default=[], type=int,
                   help='seed vertex ID for one disk_curve source; repeatable')
    p.add_argument('--disk-radius', action='append', default=[], type=float)
    p.add_argument('--disk-value',  action='append', default=[], type=float)
    p.add_argument('--disk-weight', action='append', default=[], type=float)
    p.add_argument('--disk-generalize', action='append', default=[],
                   help='"true"/"false" per disk source; defaults all false')

    p.add_argument('--curve-npy', action='append', default=[], type=str,
                   help='path to (K,3) .npy file for one curve source; repeatable')
    p.add_argument('--curve-value',  action='append', default=[], type=float)
    p.add_argument('--curve-weight', action='append', default=[], type=float)
    p.add_argument('--curve-closed', action='append', default=[],
                   help='"true"/"false" per curve source')

    # ---- legacy disk-radius generalization: --r-min/--r-max give the shared
    #      r_range of the inline --disk-seed sources that have
    #      --disk-generalize true (disks from the JSON carry their own
    #      r_range) ----
    p.add_argument('--r-min', type=float, default=None)
    p.add_argument('--r-max', type=float, default=None)
    p.add_argument('--export-r', type=float, default=None,
                   help='radius used for the final export when '
                        'generalization is ON (default: mid of r_range)')

    # ---- Dirichlet-value generalization (any source type) ----
    p.add_argument('--value-generalize-source', action='append', default=[],
                   type=int, metavar='INDEX',
                   help='zero-based index in the final combined source list; '
                        'repeat to give multiple sources independent code '
                        'components')
    p.add_argument('--value-min', type=float, default=-2.0,
                   help='minimum generalized Dirichlet value (default: -2)')
    p.add_argument('--value-max', type=float, default=2.0,
                   help='maximum generalized Dirichlet value (default: 2)')
    p.add_argument('--export-value', action='append', type=float, default=None,
                   metavar='VALUE',
                   help='one code component used for final export in value '
                        'generalization mode; repeat in generalized-source '
                        'order (default: all range midpoints)')
    p.add_argument(
        '--train-codes-out', type=str, default=None,
        help='if set, write every epoch Dirichlet code vector to this TSV',
    )

    # ---- SDF oracle (not stored in the checkpoint: repeat the same values
    #      when evaluating) ----
    p.add_argument('--sdf-res',     type=int,   default=400)
    p.add_argument('--sdf-padding', type=float, default=0.15)
    p.add_argument('--fd-halo',     type=int,   default=2)

    # ---- model & training ----
    p.add_argument('--hidden',         type=int,   default=128)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--epochs',         type=int,   default=6000)
    p.add_argument('--num-surface',    type=int,   default=30000)
    p.add_argument('--num-narrowband', type=int,   default=20000)
    p.add_argument('--num-per-source', type=int,   default=256,
                   help='samples per source per epoch')
    p.add_argument('--band-width',     type=float, default=0.01)

    # ---- loss weights ----
    p.add_argument('--lambda-dir',   type=float, default=1.0)
    p.add_argument('--lambda-mass',  type=float, default=1.0)
    p.add_argument('--lambda-cpm',   type=float, default=1.0)
    p.add_argument('--no-cpm', action='store_true',
                   help='ablation: disable the CPM loss term L_cpm (the '
                        'other loss terms are computed as usual)')
    p.add_argument('--no-normal-lift', action='store_true',
                   help='ablation: feed a constant normal (zeros) to the '
                        'network instead of n(x)')
    p.add_argument('--lambda-range', type=float, default=0.0,
                   help='soft clamp u in [v_min, v_max] (auto-derived '
                        'from source values); 0 disables')

    # ---- I/O ----
    p.add_argument('--out-dir',    type=str, default='screen_results',
                   help='output folder for model.pt and diffusion.npz '
                        '(default: screen_results)')
    p.add_argument('--save-every', type=int, default=0)
    p.add_argument('--log-every',  type=int, default=100)
    p.add_argument('--seed',       type=int, default=0)
    return p.parse_args()


def _str_to_bool(x, default=False):
    """Interpret '1' / 'true' / 't' / 'yes' / 'y' (any case) as True."""
    if x is None:
        return default
    return str(x).lower() in ('1', 'true', 't', 'yes', 'y')


def build_sources(args):
    """Compose the source list from --source-spec JSON + inline CLI shortcuts.

    Order of the result: JSON entries, --vertex-ids sources, --disk-seed
    sources, --curve-npy sources.  The sources selected with
    --value-generalize-source are then marked as value-generalized with the
    range (--value-min, --value-max).
    """
    sources = []

    # 1) JSON spec
    if args.source_spec is not None:
        with open(args.source_spec, 'r') as f:
            json_sources = json.load(f)
        if not isinstance(json_sources, list):
            raise SystemExit("--source-spec JSON must be a list of dicts")
        for s in json_sources:
            s = dict(s)
            if 'points_npy' in s:
                s['points'] = np.load(s.pop('points_npy'))
            sources.append(s)

    # 2) inline vertex shortcuts
    n_v = len(args.vertex_ids)
    for k in range(n_v):
        ids = _parse_int_list(args.vertex_ids[k])
        val = args.vertex_value[k] if k < len(args.vertex_value) else 1.0
        wt  = args.vertex_weight[k] if k < len(args.vertex_weight) else 1.0
        pr  = args.vertex_patch_radius[k] if k < len(args.vertex_patch_radius) else 0.0
        sources.append({
            'type': 'vertex', 'ids': ids,
            'value': float(val), 'weight': float(wt),
            'patch_radius': float(pr),
        })

    # 3) inline disk shortcuts
    n_d = len(args.disk_seed)
    for k in range(n_d):
        seed = args.disk_seed[k]
        r    = args.disk_radius[k] if k < len(args.disk_radius) else 0.1
        val  = args.disk_value[k]  if k < len(args.disk_value)  else 1.0
        wt   = args.disk_weight[k] if k < len(args.disk_weight) else 1.0
        gen  = _str_to_bool(args.disk_generalize[k]
                            if k < len(args.disk_generalize) else None)
        d = {'type': 'disk_curve', 'seed': int(seed),
             'radius': float(r), 'value': float(val),
             'weight': float(wt), 'generalize': gen}
        if gen:
            if args.r_min is None or args.r_max is None:
                raise SystemExit("--disk-generalize true requires --r-min and --r-max")
            d['r_range'] = (float(args.r_min), float(args.r_max))
        sources.append(d)

    # 4) inline curve shortcuts
    n_c = len(args.curve_npy)
    for k in range(n_c):
        pts = np.load(args.curve_npy[k])
        val  = args.curve_value[k]  if k < len(args.curve_value)  else 1.0
        wt   = args.curve_weight[k] if k < len(args.curve_weight) else 1.0
        clo  = _str_to_bool(args.curve_closed[k]
                            if k < len(args.curve_closed) else None)
        sources.append({
            'type': 'curve', 'points': pts,
            'value': float(val), 'weight': float(wt), 'closed': clo,
        })

    if not sources:
        raise SystemExit(
            "No sources specified. Use --source-spec, or one of "
            "--vertex-ids / --disk-seed / --curve-npy.")

    # 5) Dirichlet-value generalization markers (indices into the list above)
    selected = list(args.value_generalize_source)
    duplicates = sorted({i for i in selected if selected.count(i) > 1})
    if duplicates:
        raise SystemExit(
            "--value-generalize-source contains duplicate index(es): "
            + ", ".join(map(str, duplicates)))
    invalid = [i for i in selected if i < 0 or i >= len(sources)]
    if invalid:
        raise SystemExit(
            "--value-generalize-source index out of range: "
            + ", ".join(map(str, invalid))
            + f"; valid range is 0..{len(sources) - 1}")
    if selected and args.value_min >= args.value_max:
        raise SystemExit(
            f"invalid value range [{args.value_min}, {args.value_max}]: "
            "--value-min must be < --value-max")
    for index in selected:
        sources[index]['value_generalize'] = True
        sources[index]['value_range'] = (
            float(args.value_min), float(args.value_max))

    return sources


def _generalization_mode(sources):
    """Return 'radius', 'value' or 'none'; exit if both modes are requested."""
    radius_on = any(s.get('type') == 'disk_curve'
                    and bool(s.get('generalize', False)) for s in sources)
    value_on = any(bool(s.get('value_generalize', False)) for s in sources)
    if radius_on and value_on:
        raise SystemExit(
            "disk-radius and Dirichlet-value generalization cannot be mixed: "
            "they use distinct conditioning semantics")
    return 'radius' if radius_on else ('value' if value_on else 'none')


def _value_generalize_indices(sources):
    """Indices of the value-generalized sources, in source-list order."""
    return [
        index for index, source in enumerate(sources)
        if bool(source.get('value_generalize', False))
    ]


def main():
    """Run one training: seed, mesh, sources, SDF, training, export."""
    args = parse_args()
    if args.mesh.lower().endswith(".json"):
        raise SystemExit(
            "Positional argument must be a triangle mesh (.obj), not a source JSON.\n"
            "Example:\n"
            f"  python train_screen.py <mesh.obj> --source-spec {args.mesh} "
            "--lambda-dir 1.0 --lambda-mass 1.0"
        )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Mesh ----
    mesh = MeshGeometry(args.mesh, normalize=True, device=device)

    # ---- Sources ----
    sources = build_sources(args)
    requested_mode = _generalization_mode(sources)
    value_source_indices = _value_generalize_indices(sources)
    code_size = len(value_source_indices) if requested_mode == 'value' else 1
    if args.export_r is not None and requested_mode != 'radius':
        raise SystemExit(
            "--export-r is only valid in disk-radius generalization mode")
    if args.export_value is not None and requested_mode != 'value':
        raise SystemExit(
            "--export-value is only valid in Dirichlet-value "
            "generalization mode")
    if (requested_mode == 'value' and args.export_value is not None
            and len(args.export_value) != code_size):
        raise SystemExit(
            f"value mode has {code_size} code components for source indices "
            f"{value_source_indices}; repeat --export-value exactly "
            f"{code_size} time(s)")
    print(f"Total sources: {len(sources)}")
    print(f"Generalization mode: {requested_mode}")
    print(f"Code size: {code_size}")
    if requested_mode == 'value':
        print(f"Value code source order: {value_source_indices}")

    # ---- Stage 1: SDF oracle ----
    print("\n" + "=" * 60)
    print(f"Stage 1: Building FDM SDF Oracle (res={args.sdf_res})")
    print("=" * 60)
    sdf = SparseGridFDM_SDF.from_mesh(
        mesh, resolution=args.sdf_res, padding=args.sdf_padding,
        fd_halo=args.fd_halo, device=device,
    )

    # ---- Stage 2: diffusion field ----
    print("\n" + "=" * 60)
    print("Stage 2: Multi-source Screened-Dirichlet Diffusion Field")
    print("  Using FDM SDF chain-rule for grad_M u")
    print("=" * 60)

    designer = Surfacescreendiffusion(
        input_size=3, hidden=args.hidden, lr=args.lr,
        device=device, sdf=sdf, code_size=code_size,
    )
    designer.train(
        mesh=mesh,
        sources=sources,
        num_epochs=args.epochs,
        num_surface=args.num_surface,
        num_narrowband=args.num_narrowband,
        num_per_source=args.num_per_source,
        band_width=args.band_width,
        lambda_dir=args.lambda_dir,
        lambda_mass=args.lambda_mass,
        lambda_cpm=args.lambda_cpm,
        use_cpm=not args.no_cpm,
        no_normal_lift=args.no_normal_lift,
        lambda_range=args.lambda_range,
        log_every=args.log_every,
        save_every=args.save_every,
        save_dir=args.out_dir,
        train_codes_path=args.train_codes_out,
    )

    # ---- diagnostic: does U use n? ----
    # Compare U(x, n) with U(x, -n) at the mesh vertices, at the code used
    # for the export.  This only prints; it does not affect the saved files.
    designer.model_u.eval()
    value_export_code = (None if args.export_value is None else
                         np.asarray(args.export_value, dtype=np.float64))
    if designer._generalization_mode == 'radius' and args.export_r is not None:
        diag_code = float(args.export_r)
    elif (designer._generalization_mode == 'value'
          and value_export_code is not None):
        diag_code = value_export_code
    elif designer._generalization_mode == 'value':
        midpoint = 0.5 * (designer._code_range[0]
                          + designer._code_range[1])
        diag_code = np.full(designer.code_size, midpoint, dtype=np.float64)
    elif designer._generalization_mode == 'radius':
        diag_code = 0.5 * (designer._code_range[0]
                           + designer._code_range[1])
    else:
        diag_code = 0.0

    with torch.no_grad():
        V = torch.from_numpy(mesh.vertices_np.astype("float32")).to(device)
        n = sdf.normal(V).detach()
        x_pos = designer._make_input(V,  n, diag_code)
        x_neg = designer._make_input(V, -n, diag_code)
        u_pos = designer.model_u(x_pos).squeeze(-1)
        u_neg = designer.model_u(x_neg).squeeze(-1)
        diff = (u_pos - u_neg).abs()
        scale = u_pos.abs().mean().clamp(min=1e-8)
        print("\n[normal-use diagnostic]")
        print("  code =", designer._format_code(diag_code))
        print("  mean |U(x,n)-U(x,-n)| =", diff.mean().item())
        print("  max  |U(x,n)-U(x,-n)| =", diff.max().item())
        print("  relative mean diff     =", (diff.mean() / scale).item())

    os.makedirs(args.out_dir, exist_ok=True)
    export_code = (args.export_r if designer._generalization_mode == 'radius'
                   else value_export_code
                   if designer._generalization_mode == 'value'
                   else None)
    designer.export(mesh, os.path.join(args.out_dir, 'diffusion.npz'),
                    code_value=export_code)
    designer.save(os.path.join(args.out_dir, 'model.pt'))

    print("\nDone!")
    print(f"  Visualize with:")
    print(f"    python Surfacescreendiffusion.py --vis "
          f"{args.out_dir}/diffusion.npz")


if __name__ == '__main__':
    main()
