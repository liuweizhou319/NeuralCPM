#!/usr/bin/env python3
r"""Export network and FEM vertex fields for a few codes of each split.

Qualitative export of the Dirichlet-value generalization experiment.  For each
of the three splits (train, interp, extrap) the script

  1. draws K code vectors at random, without replacement, from the split's
     codes file (codes_<split>.txt written by prepare_dirichlet_splits.py);
     K = min(--num-per-split, number of rows in the file),
  2. evaluates the trained network at all mesh vertices for each code,
  3. solves the FEM reference (fem_screen.solve_fem_screened) with the values
     of the generalized sources replaced by the code, and
  4. writes both fields plus the per-sample constraints to
     <out-dir>/nn_fem_<split><K>.npz (e.g. nn_fem_train10.npz).

A single numpy default_rng(--seed) is shared by the three splits and consumed
in the order train, interp, extrap, so the selection is reproducible for a
fixed seed and fixed codes files.  The picked rows keep their file order.

The following must match the training run (nothing here checks it):

  * --value-generalize-source: repeat the flag once per value-generalized
    source and pass the indices in ASCENDING order, the same set as in
    training.  Code component i belongs to the i-th generalized source in
    ascending source index.
  * --source-spec: pass the SAME source JSON that was used for training.
    It must be a JSON list of source
    dicts with inline data; "points_npy" references are not resolved here.
  * --sdf-res / --sdf-padding / --fd-halo: pass the SAME SDF flags as in
    training, because the SDF normals are network inputs.  The defaults here
    are 300 / 0.15 / 2 whereas train_screen.py defaults to --sdf-res 400.
  * --lambda-dir / --lambda-mass: same values as in training (they define
    the FEM reference).

Output npz (np.savez_compressed) for K codes, V vertices, F faces and d
generalized sources:

    mesh_vertices     float32 (V, 3)   normalized mesh coordinates
    mesh_faces        int64   (F, 3)
    split             str              'train', 'interp' or 'extrap'
    codes             float64 (K, d)
    source_indices    int64   (d,)     the --value-generalize-source list
    u_nn              float32 (K, V)   network prediction at the vertices
    u_fem             float32 (K, V)   FEM reference
    constraints_json  object  (K,)     JSON string of the full source list
                                       with the code values substituted
    lambda_dir        float32 scalar
    lambda_mass       float32 scalar

constraints_json is an object array, so reading the file needs
np.load(path, allow_pickle=True); decode sample i with
json.loads(str(data["constraints_json"][i])).

Requires a CUDA GPU with fvdb (SDF construction) and libigl (FEM solve).

Usage (from the repository root):

    python export_split_samples.py \
        --mesh data/dc.obj \
        --source-spec data/dc_constraints_v0.json \
        --checkpoint results/generalization/dc/model.pt \
        --codes-train results/generalization/dc/eval/codes_train.txt \
        --codes-interp results/generalization/dc/eval/codes_interp.txt \
        --codes-extrap results/generalization/dc/eval/codes_extrap.txt \
        --out-dir results/generalization/dc/eval/exports \
        --value-generalize-source 0 --value-generalize-source 1 \
        --value-generalize-source 2 --value-generalize-source 3 \
        --value-generalize-source 4 \
        --sdf-res 300 --sdf-padding 0.15 --fd-halo 2

The SDF flags shown are those of run_generalization.sh;
always pass the values the model was trained with.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from Mesh_geometry import MeshGeometry
from SparseGridFDM_SDF import SparseGridFDM_SDF
from Surfacescreendiffusion import Surfacescreendiffusion
from fem_screen import solve_fem_screened
from evaluate_dirichlet_generalization import _load_codes_file


def load_raw_sources(path: Path) -> list:
    """Load the source JSON as is.

    The top level must be a list (SystemExit otherwise); the entries are not
    validated and "points_npy" references are not resolved.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"constraints must be a JSON list: {path}")
    return data


def sources_with_values(raw: list, selected: list[int], code: np.ndarray) -> list:
    """Return a deep copy of ``raw`` with ``raw[selected[i]]["value"] = code[i]``."""
    out = copy.deepcopy(raw)
    for i, v in zip(selected, code):
        out[i]["value"] = float(v)
    return out


def eval_nn(designer, mesh, code: np.ndarray, batch: int) -> np.ndarray:
    """Evaluate the network at all mesh vertices for one code vector.

    Vertices are fed as float32 in batches of ``batch``.  When the model has
    an SDF the normals come from ``designer.sdf.normal`` (``_make_input``
    replaces them by zeros for a no-normal-lift checkpoint).  Otherwise the
    mesh vertex normals are passed, but ``_make_input`` ignores them for a
    model without SDF (network input [x | code]).  Returns a float32 array
    of shape (V,).
    """
    designer.model_u.eval()
    V = mesh.vertices_np.astype(np.float32)
    out = np.empty(len(V), dtype=np.float32)
    code = np.asarray(code, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(V), batch):
            e = min(s + batch, len(V))
            pts = torch.from_numpy(V[s:e]).to(designer.device)
            if designer.sdf is not None:
                n = designer.sdf.normal(pts).detach()
            else:
                n = torch.from_numpy(
                    mesh.vertex_normals_np[s:e].astype(np.float32)
                ).to(designer.device)
            x = designer._make_input(pts, n, code)
            out[s:e] = designer.model_u(x).squeeze(-1).cpu().numpy()
    return out


def pick_codes(path: Path, code_size: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``min(k, rows)`` code rows without replacement, kept in file order."""
    codes = _load_codes_file(path, code_size)
    k = min(k, len(codes))
    idx = rng.choice(len(codes), size=k, replace=False)
    idx.sort()
    return codes[idx]


def export_split(
    name: str,
    codes: np.ndarray,
    designer,
    mesh,
    raw_sources: list,
    selected: list[int],
    lambda_dir: float,
    lambda_mass: float,
    out_path: Path,
    batch: int,
) -> None:
    """Evaluate network and FEM for every code and write one npz for the split.

    The npz keys are listed in the module docstring.
    """
    V = np.asarray(mesh.vertices_np, dtype=np.float64)
    F = np.asarray(mesh.faces_np, dtype=np.int64)
    n = len(codes)
    u_nn = np.empty((n, len(V)), dtype=np.float32)
    u_fem = np.empty((n, len(V)), dtype=np.float32)
    constraints = []

    for i, code in enumerate(codes):
        print(f"[{name}] {i+1}/{n} code={np.array2string(code, precision=4)}")
        src = sources_with_values(raw_sources, selected, code)
        constraints.append(src)
        u_nn[i] = eval_nn(designer, mesh, code, batch)
        u, _ = solve_fem_screened(
            V, F, src, lambda_dir=lambda_dir, lambda_mass=lambda_mass, verbose=False
        )
        u_fem[i] = np.asarray(u, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mesh_vertices=V.astype(np.float32),
        mesh_faces=F.astype(np.int64),
        split=np.asarray(name),
        codes=codes.astype(np.float64),
        source_indices=np.asarray(selected, dtype=np.int64),
        u_nn=u_nn,
        u_fem=u_fem,
        constraints_json=np.asarray(
            [json.dumps(c, ensure_ascii=False) for c in constraints],
            dtype=object,
        ),
        lambda_dir=np.float32(lambda_dir),
        lambda_mass=np.float32(lambda_mass),
    )
    print(f"Wrote {out_path.resolve()}  ({n} samples)")


def main():
    """Command-line entry point: export train, interp and extrap samples."""
    p = argparse.ArgumentParser(
        description=(
            "Export network and FEM vertex fields for a few randomly picked "
            "codes of the train / interp / extrap splits of a "
            "Dirichlet-value generalization run."
        )
    )
    p.add_argument("--mesh", required=True,
                   help="mesh .obj path (the mesh used for training)")
    p.add_argument("--source-spec", required=True, type=Path,
                   help="source JSON (a list of source dicts); must be the "
                        "same file as in training")
    p.add_argument("--checkpoint", required=True,
                   help="trained model checkpoint (model.pt)")
    p.add_argument("--codes-train", required=True, type=Path,
                   help="codes file of the train split (codes_train.txt)")
    p.add_argument("--codes-interp", required=True, type=Path,
                   help="codes file of the interp split (codes_interp.txt)")
    p.add_argument("--codes-extrap", required=True, type=Path,
                   help="codes file of the extrap split (codes_extrap.txt)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="output directory for nn_fem_<split><K>.npz")
    p.add_argument("--value-generalize-source", action="append", type=int, required=True,
                   help="zero-based index of a value-generalized source; "
                        "repeat once per source, in ascending order and "
                        "matching training")
    p.add_argument("--num-per-split", type=int, default=10,
                   help="number of codes drawn per split (default: 10)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the random code selection (default: 0)")
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

    selected = list(args.value_generalize_source)
    code_size = len(selected)
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    rng = np.random.default_rng(args.seed)

    mesh = MeshGeometry(args.mesh, normalize=True, device=device)
    sdf = SparseGridFDM_SDF.from_mesh(
        mesh,
        resolution=args.sdf_res,
        padding=args.sdf_padding,
        fd_halo=args.fd_halo,
        device=device,
    )
    designer = Surfacescreendiffusion.load(
        args.checkpoint, device=device, sdf=sdf
    )
    raw = load_raw_sources(args.source_spec)

    # One generator for all splits, consumed in this fixed order.
    pairs = [
        ("train", args.codes_train),
        ("interp", args.codes_interp),
        ("extrap", args.codes_extrap),
    ]
    for name, path in pairs:
        codes = pick_codes(path, code_size, args.num_per_split, rng)
        export_split(
            name,
            codes,
            designer,
            mesh,
            raw,
            selected,
            args.lambda_dir,
            args.lambda_mass,
            args.out_dir / f"nn_fem_{name}{len(codes)}.npz",
            args.batch_size,
        )


if __name__ == "__main__":
    main()
