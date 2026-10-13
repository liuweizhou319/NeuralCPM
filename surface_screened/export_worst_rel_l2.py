#!/usr/bin/env python3
r"""Export the worst-case (largest rel_l2) code of each split, network vs FEM.

Qualitative export of the Dirichlet-value generalization experiment.  For each
split (train, interp, extrap) the script reads <folder>/metrics_<split>.txt,
the metrics TSV written by evaluate_dirichlet_generalization.py, takes the row
with the largest rel_l2, re-evaluates the trained network at all mesh vertices
for that code vector, solves the FEM reference (fem_screen.solve_fem_screened)
for the same boundary values and writes

    <out-dir>/nn_fem_<split>_maxrel.npz

where --out-dir defaults to <folder>/exports_worst_rel_l2.

--folder must contain metrics_train.txt, metrics_interp.txt and
metrics_extrap.txt.  Blank lines and '#' metadata lines are skipped; the
columns sample_index, rel_l2 and source_<i>_value are used, the code vector
being the source_<i>_value columns in header order.

The following must match the training run (nothing here checks it; see also
export_split_samples.py, whose helpers are reused):

  * --value-generalize-source: repeat the flag once per value-generalized
    source and pass the indices in ASCENDING order, the same set as in
    training and evaluation.
  * --source-spec: the SAME source JSON that was used for training;
    a JSON list of source dicts with
    inline data.
  * --sdf-res / --sdf-padding / --fd-halo: the SAME SDF flags as in training,
    because the SDF normals are network inputs.  The defaults here are
    300 / 0.15 / 2 whereas train_screen.py defaults to --sdf-res 400.
  * --lambda-dir / --lambda-mass: same values as in training (they define
    the FEM reference).

Output npz (np.savez_compressed) for V vertices, F faces and d generalized
sources:

    mesh_vertices     float32 (V, 3)   normalized mesh coordinates
    mesh_faces        int64   (F, 3)
    split             str              'train', 'interp' or 'extrap'
    sample_index      int64   scalar   sample_index of the worst row
    rel_l2            float64 scalar   copied from the metrics file
    codes             float64 (d,)     the worst code vector
    source_indices    int64   (d,)     the --value-generalize-source list
    u_nn              float32 (V,)     network prediction at the vertices
    u_fem             float32 (V,)     FEM reference
    src_pts           float32 (P, 3)   positions of the vertex sources
    src_pts_value     float32 (P,)
    src_seg_p1        float32 (S, 3)   segments of the curve sources
    src_seg_p2        float32 (S, 3)
    src_seg_value     float32 (S,)
    constraints_json  object  ()       JSON string of the full source list
                                       with the code values substituted

constraints_json is a 0-d object array, so reading the file needs
np.load(path, allow_pickle=True); decode it with
json.loads(str(data["constraints_json"])).

Requires a CUDA GPU with fvdb (SDF construction) and libigl (FEM solve).

Usage (from the repository root):

    python export_worst_rel_l2.py \
        --folder results/generalization/dc/eval \
        --mesh data/dc.obj \
        --source-spec data/dc_constraints_v0.json \
        --checkpoint results/generalization/dc/model.pt \
        --value-generalize-source 0 --value-generalize-source 1 \
        --value-generalize-source 2 --value-generalize-source 3 \
        --value-generalize-source 4 \
        --sdf-res 300 --sdf-padding 0.15 --fd-halo 2

The SDF flags shown are those of run_generalization.sh;
always pass the values the model was trained with.  The three files are
written to results/generalization/dc/eval/exports_worst_rel_l2/.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from Mesh_geometry import MeshGeometry
from SparseGridFDM_SDF import SparseGridFDM_SDF
from Surfacescreendiffusion import Surfacescreendiffusion
from fem_screen import solve_fem_screened
from export_split_samples import (
    eval_nn,
    load_raw_sources,
    sources_with_values,
)


SPLITS = ("train", "interp", "extrap")


def max_rel_row(metrics_txt: Path):
    """Return (sample_index, rel_l2, code) of the row with the largest rel_l2."""
    with metrics_txt.open(encoding="utf-8-sig") as fh:
        lines = [
            ln for ln in fh
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    best = max(rows, key=lambda r: float(r["rel_l2"]))
    code_cols = [c for c in best if c.startswith("source_") and c.endswith("_value")]
    code = np.array([float(best[c]) for c in code_cols], dtype=np.float64)
    return int(best["sample_index"]), float(best["rel_l2"]), code


def main():
    """Command-line entry point: export the worst code of every split."""
    p = argparse.ArgumentParser(
        description=(
            "Export network and FEM vertex fields for the code with the "
            "largest rel_l2 in each of metrics_train.txt, metrics_interp.txt "
            "and metrics_extrap.txt."
        )
    )
    p.add_argument("--folder", type=Path, required=True,
                   help="evaluation folder containing "
                        "metrics_{train,interp,extrap}.txt")
    p.add_argument("--mesh", required=True,
                   help="mesh .obj path (the mesh used for training)")
    p.add_argument("--source-spec", type=Path, required=True,
                   help="source JSON (a list of source dicts); must be the "
                        "same file as in training")
    p.add_argument("--checkpoint", required=True,
                   help="trained model checkpoint (model.pt)")
    p.add_argument("--value-generalize-source", action="append", type=int,
                   required=True,
                   help="zero-based index of a value-generalized source; "
                        "repeat once per source, in ascending order and "
                        "matching training")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory "
                        "(default: <folder>/exports_worst_rel_l2)")
    p.add_argument("--lambda-dir", type=float, default=1.0,
                   help="Dirichlet-energy weight of the FEM reference; same "
                        "as in training (default: 1.0)")
    p.add_argument("--lambda-mass", type=float, default=1.0,
                   help="mass-term weight of the FEM reference; same as in "
                        "training (default: 1.0)")
    p.add_argument("--sdf-res", type=int, default=300,
                   help="SDF grid resolution; must equal the training value "
                        "(default: 300)")
    p.add_argument("--sdf-padding", type=float, default=0.15,
                   help="SDF grid padding; must equal the training value "
                        "(default: 0.15)")
    p.add_argument("--fd-halo", type=int, default=2,
                   help="finite-difference halo of the SDF grid; must equal "
                        "the training value (default: 2)")
    p.add_argument("--batch-size", type=int, default=8192,
                   help="vertices per network forward pass (default: 8192)")
    p.add_argument("--device", default="cuda",
                   help="torch device, or 'auto' (default: cuda); the SDF "
                        "construction needs CUDA")
    args = p.parse_args()

    selected = args.value_generalize_source
    folder = args.folder.resolve()
    out_dir = (
        args.out_dir.resolve() if args.out_dir is not None
        else folder / "exports_worst_rel_l2"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- find worst BC per split ----
    worst = {}
    for split in SPLITS:
        idx, rel, code = max_rel_row(folder / f"metrics_{split}.txt")
        worst[split] = (idx, rel, code)
        print(f"{split}: sample={idx}  rel_l2={rel:.8g}")
        print(f"  code={np.array2string(code, precision=8)}")

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    mesh = MeshGeometry(args.mesh, normalize=True, device=device)
    sdf = SparseGridFDM_SDF.from_mesh(
        mesh, resolution=args.sdf_res, padding=args.sdf_padding,
        fd_halo=args.fd_halo, device=device,
    )
    designer = Surfacescreendiffusion.load(
        args.checkpoint, device=device, sdf=sdf
    )
    raw = load_raw_sources(args.source_spec)
    V = np.asarray(mesh.vertices_np, dtype=np.float64)
    F = np.asarray(mesh.faces_np, dtype=np.int64)

    for split, (idx, rel, code) in worst.items():
        print(f"==> evaluate {split}")
        src = sources_with_values(raw, selected, code)
        u_nn = eval_nn(designer, mesh, code, args.batch_size)
        u_fem, info = solve_fem_screened(
            V, F, src,
            lambda_dir=args.lambda_dir,
            lambda_mass=args.lambda_mass,
            verbose=False,
        )
        u_fem = np.asarray(u_fem, dtype=np.float32)
        out = out_dir / f"nn_fem_{split}_maxrel.npz"
        np.savez_compressed(
            out,
            mesh_vertices=V.astype(np.float32),
            mesh_faces=F.astype(np.int64),
            split=np.asarray(split),
            sample_index=np.int64(idx),
            rel_l2=np.float64(rel),
            codes=code,
            source_indices=np.asarray(selected, dtype=np.int64),
            u_nn=u_nn,
            u_fem=u_fem,
            src_pts=np.asarray(info["src_pts"], dtype=np.float32),
            src_pts_value=np.asarray(info["src_pts_value"], dtype=np.float32),
            src_seg_p1=np.asarray(info["src_seg_p1"], dtype=np.float32),
            src_seg_p2=np.asarray(info["src_seg_p2"], dtype=np.float32),
            src_seg_value=np.asarray(info["src_seg_value"], dtype=np.float32),
            constraints_json=np.asarray(
                json.dumps(src, ensure_ascii=False), dtype=object
            ),
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
