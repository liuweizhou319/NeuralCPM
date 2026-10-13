r"""Evaluate Dirichlet-value generalization against the FEM baseline.

Pipeline position
-----------------
``train_screen.py`` trains a single network that is conditioned on a code
vector ``g`` holding one Dirichlet value per generalized source (the sources
named with ``--value-generalize-source``).  This script measures how well
that network reproduces the FEM reference over many code vectors.  Its
output table is read by ``plot_generalization_mse.py`` and
``export_worst_rel_l2.py``; its code-file reader ``_load_codes_file`` is
imported by ``export_split_samples.py``.

What is compared
----------------
For every code vector ``g = (g_0, ..., g_{k-1})``, with ``k`` the number of
generalized sources, two fields are computed at the vertices of the
normalized mesh (``MeshGeometry(normalize=True)``):

* ``u_nn``: the network output for the input ``[x, n(x), g]`` at every vertex
  ``x`` (float32), where ``n`` is the normal chosen by ``--normal-source``;
* ``u_fem``: the solution of ``fem_screen.solve_fem_screened`` on the same
  vertices in the same order (cast to float64), where the ``value`` of the
  j-th generalized source is replaced by ``g_j`` and every other source
  keeps the value from the JSON file.

Every code vector is sent to ``solve_fem_screened`` directly.  No FEM
basis-response interpolation or linear combination is used.

Two modes
---------
Random codes (default)
    ``--num-samples`` code vectors are drawn with
    ``numpy.random.default_rng(--seed)``, every component independently and
    uniformly in ``[g_min, g_max]``.

Explicit codes (``--codes-file FILE``)
    The rows of FILE are evaluated in file order, one whitespace-separated
    code vector per row.  Blank lines, ``#`` lines and header lines are
    skipped and a leading integer epoch column is dropped, so both the
    trainer's ``--train-codes-out`` log and the ``codes_*.txt`` files of
    ``prepare_dirichlet_splits.py`` can be passed.  ``--num-samples`` and
    ``--seed`` are ignored in this mode (the seed is still copied to the
    output header).

Value range
-----------
``--g-min`` / ``--g-max`` are the sampling interval of the random mode.  In
both modes ``[g_min, g_max]`` is also a declared range that is checked
against the trained range (the checkpoint ``code_range``, the ``value_range``
saved with every generalized source, and a ``value_range`` given for a
generalized source in the JSON).  If it is not contained in the trained range
the run aborts, unless ``--allow-code-extrapolation`` is given, which turns
the error into a warning.  With ``--codes-file`` the rows themselves are NOT
checked against any range: there ``--g-min`` / ``--g-max`` /
``--allow-code-extrapolation`` only affect this declared-range check and the
``g_min`` / ``g_max`` entries of the output header.  When a codes file holds
values outside the trained range, declare the wider range and pass
``--allow-code-extrapolation`` so that the header describes the run.

Arguments that must match training
----------------------------------
Verified against the checkpoint (the run aborts on a mismatch):

* ``--value-generalize-source`` must be repeated once per generalized
  source, in ASCENDING source index (the order of the code components), and
  the resulting list must equal the generalized sources of the checkpoint.
* ``--source-spec`` must be the same JSON that was given to
  ``train_screen.py`` for training.  Number and type of
  the sources, the ``value`` and ``weight`` of every source and the source
  geometry are compared with the sources stored in the checkpoint.

NOT stored in the checkpoint and therefore NOT verified (a mismatch silently
produces wrong numbers), so pass the values used for training:

* ``--sdf-res``, ``--sdf-padding``, ``--fd-halo``: they define the SDF whose
  normals are a network input (used with ``--normal-source sdf``);
* ``--lambda-dir``, ``--lambda-mass``: the weights of the FEM reference,
  which must be the ones the network was trained with;
* ``--mesh``: the training mesh.

Normals
-------
``--normal-source sdf`` (default) builds ``SparseGridFDM_SDF`` once and
queries its normals at the mesh vertices, as in training; it needs a CUDA
device.  ``--normal-source mesh`` feeds the mesh vertex normals instead; it
does not need the SDF stack, but it is only a lower-cost approximation of
the input the network was trained with.  The checkpoint can override the
choice: zero normals are supplied if it stores ``no_normal_lift=True``, and
no normals at all if the network has no normal input.  The output header
records both the requested and the effective normal source.

Output
------
``--output`` is a UTF-8 text file (parent directories are created):

    # format: dirichlet_generalization_mse_v3
    # <key>: <JSON value>                  one line per metadata entry
    sample_index  source_<i>_value ...  mse  rel_l2
    <one row per code vector>

The ``# key: json`` metadata lines record, among others, the resolved
absolute paths of the mesh, the source spec and the checkpoint, the codes
file as given on the command line (not resolved; ``null`` in the random
mode), the mesh size, the source indices, ``g_min`` / ``g_max``, the seed,
the lambda weights, the normal source and the min / mean / max of both
metrics.  The table is tab-separated: ``sample_index`` is the 0-based
evaluation order, there is one ``source_<i>_value`` column per generalized
source ``<i>`` (in command-line order), and floats are written with 17
significant digits.

Metrics, both unweighted over all mesh vertices ``v`` (not area-weighted):

    mse    = mean_v (u_nn(v) - u_fem(v))^2
    rel_l2 = sqrt(sum_v (u_nn(v) - u_fem(v))^2 / sum_v u_fem(v)^2)

``rel_l2`` is written as ``nan`` when the FEM solution is identically zero.

Cost
----
``solve_fem_screened`` is called once per code vector, and the system is
re-assembled and re-factorized every time, so every code costs one FEM
assembly plus one sparse factorization.  The only thing reused between calls
is the cache of ``fem_screen.dijkstra_from``: the edge graph and the per-seed
graph distances needed by ``disk_curve`` sources and by ``vertex`` sources
with ``patch_radius > 0``.  The run time is linear in the number of codes;
``--progress-every`` prints the progress and an ETA.

Security
--------
The checkpoint is loaded with ``torch.load(weights_only=False)``, i.e. it is
unpickled and can execute arbitrary code.  Only evaluate checkpoints that
you trust.

Exit status: 0 on success, 2 after printing ``ERROR: <message>`` to stderr.

Examples (run from the repository root)
---------------------------------------
The second example evaluates the run trained by ``run_generalization.sh``; it
reads ``codes_interp.txt``, which ``prepare_dirichlet_splits.py`` writes
(``run_evaluation.sh`` runs both steps).

Random codes, Bunny with four generalized sources:

python evaluate_dirichlet_generalization.py \
    --mesh data/Bunny.obj \
    --source-spec data/Bunny_2.json \
    --checkpoint results/generalization/bunny/model.pt \
    --value-generalize-source 0 --value-generalize-source 1 \
    --value-generalize-source 2 --value-generalize-source 3 \
    --g-min -2 --g-max 2 --num-samples 1000 \
    --sdf-res 400 --fd-halo 2 \
    --output results/generalization/bunny/eval/metrics_random.txt

Explicit codes, dc with five generalized sources:

python evaluate_dirichlet_generalization.py \
    --mesh data/dc.obj \
    --source-spec data/dc_constraints_v0.json \
    --checkpoint results/generalization/dc/model.pt \
    --value-generalize-source 0 --value-generalize-source 1 \
    --value-generalize-source 2 --value-generalize-source 3 \
    --value-generalize-source 4 \
    --g-min -3 --g-max 2 --sdf-res 300 \
    --codes-file results/generalization/dc/eval/codes_interp.txt \
    --output results/generalization/dc/eval/metrics_interp.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from Mesh_geometry import MeshGeometry
from Surfacescreendiffusion import MLP_Scalar
from fem_screen import dijkstra_from, solve_fem_screened


def parse_args() -> argparse.Namespace:
    """Build the command-line parser and parse the process arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare a Dirichlet-value generalization network with FEM at the "
            "normalized mesh vertices. Every code vector assigns one Dirichlet "
            "value to each source selected with --value-generalize-source; "
            "codes are sampled at random (--num-samples, --seed) or read from "
            "--codes-file. One FEM solve is run per code vector, and the "
            "per-code MSE and relative L2 error are written to a TSV text file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mesh", required=True, type=Path,
                        help="input triangle mesh (.obj)")
    parser.add_argument(
        "--source-spec", required=True, type=Path,
        help=(
            "source JSON used for training; it is compared with the sources "
            "stored in the checkpoint"
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help=(
            "trained value-generalization checkpoint; it is unpickled, so "
            "only use files you trust"
        ),
    )
    parser.add_argument(
        "--value-generalize-source", action="append", required=True, type=int,
        metavar="INDEX",
        help=(
            "zero-based source index assigned its own conditioning value; "
            "repeat once per generalized source in ascending source index, "
            "which is the checkpoint code order"
        ),
    )
    parser.add_argument(
        "--g-min", type=float, default=-2.0,
        help=(
            "minimum Dirichlet value: lower sampling bound for random codes; "
            "with --codes-file it is only used for the range check and the "
            "output header"
        ),
    )
    parser.add_argument(
        "--g-max", type=float, default=2.0,
        help=(
            "maximum Dirichlet value: upper sampling bound for random codes; "
            "with --codes-file it is only used for the range check and the "
            "output header"
        ),
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help=(
            "number of independently sampled code vectors (ignored with "
            "--codes-file)"
        ),
    )
    parser.add_argument(
        "--codes-file", type=Path, default=None,
        help=(
            "if set, evaluate these Dirichlet code rows instead of random "
            "sampling (whitespace-separated text; '#' and header lines are "
            "skipped, an optional leading epoch column is ignored)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="NumPy random seed (not used with --codes-file)")
    parser.add_argument("--output", type=Path,
                        default=Path("generalization_mse.txt"),
                        help="output TSV text file")

    parser.add_argument(
        "--lambda-dir", type=float, default=1.0,
        help="FEM surface-gradient weight; must equal the training value",
    )
    parser.add_argument(
        "--lambda-mass", type=float, default=1.0,
        help="FEM screened mass weight; must equal the training value",
    )

    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--normal-source", choices=("mesh", "sdf"), default="sdf",
        help=(
            "normal supplied when geo_size=code_size+3; sdf matches training "
            "and needs CUDA, while mesh is an explicit lower-cost approximation"
        ),
    )
    parser.add_argument(
        "--sdf-res", type=int, default=400,
        help=(
            "SparseGridFDM_SDF resolution when using SDF normals; must equal "
            "the training value"
        ),
    )
    parser.add_argument(
        "--sdf-padding", type=float, default=0.15,
        help=(
            "SparseGridFDM_SDF bounding-box padding; must equal the training "
            "value"
        ),
    )
    parser.add_argument(
        "--fd-halo", type=int, default=2,
        help=(
            "SparseGridFDM_SDF finite-difference halo; must equal the "
            "training value"
        ),
    )

    parser.add_argument(
        "--batch-size", type=int, default=65536,
        help="maximum neural-network rows per forward pass",
    )
    parser.add_argument(
        "--code-batch-size", type=int, default=16,
        help="number of sampled code vectors evaluated together",
    )
    parser.add_argument(
        "--progress-every", type=int, default=100,
        help="print FEM progress every N samples (0 disables)",
    )
    parser.add_argument(
        "--allow-code-extrapolation", action="store_true",
        help=(
            "warn instead of failing when [g_min,g_max] exceeds the "
            "checkpoint code_range or a source value_range"
        ),
    )
    parser.add_argument(
        "--allow-legacy-checkpoint", action="store_true",
        help=(
            "downgrade the error for a checkpoint without generalization_mode "
            "metadata to a warning; this does not make legacy checkpoints "
            "usable: a checkpoint without explicit code_size is always "
            "rejected, and checkpoints written by train_screen.py store both "
            "keys, so the flag has no effect on them"
        ),
    )
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    """Return the resolved path, or raise ValueError if it is not a file."""
    path = path.expanduser()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")
    return path.resolve()


def _finite_float(value: Any, label: str) -> float:
    """Convert ``value`` to a finite float, or raise ValueError naming ``label``."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result!r}")
    return result


def _as_bool(value: Any) -> bool:
    """Interpret a JSON / checkpoint flag (string, scalar tensor or other)."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("boolean checkpoint metadata tensor must be scalar")
        return bool(value.item())
    return bool(value)


def _load_sources(path: Path, num_vertices: int) -> List[Dict[str, Any]]:
    """Load and validate the source list, resolving points_npy by JSON location."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("--source-spec must contain a non-empty JSON list")

    sources: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"source[{index}] must be a JSON object")
        source = dict(item)
        source_type = source.get("type")
        if source_type not in {"vertex", "disk_curve", "curve"}:
            raise ValueError(
                f"source[{index}].type must be vertex, disk_curve or curve; "
                f"got {source_type!r}"
            )
        source["value"] = _finite_float(
            source.get("value", 1.0), f"source[{index}].value"
        )
        source["weight"] = _finite_float(
            source.get("weight", 1.0), f"source[{index}].weight"
        )
        if source["weight"] < 0.0:
            raise ValueError(f"source[{index}].weight must be non-negative")

        if source_type == "vertex":
            ids = source.get("ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"source[{index}].ids must be a non-empty list")
            try:
                ids = [int(vertex_id) for vertex_id in ids]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"source[{index}].ids must contain integers") from exc
            bad = [vertex_id for vertex_id in ids
                   if vertex_id < 0 or vertex_id >= num_vertices]
            if bad:
                raise ValueError(
                    f"source[{index}] has vertex IDs outside [0,{num_vertices - 1}]: "
                    f"{bad[:10]}"
                )
            source["ids"] = ids
            patch_radius = _finite_float(
                source.get("patch_radius", 0.0),
                f"source[{index}].patch_radius",
            )
            if patch_radius < 0.0:
                raise ValueError(f"source[{index}].patch_radius must be non-negative")
            source["patch_radius"] = patch_radius

        elif source_type == "disk_curve":
            if "seed" not in source:
                raise ValueError(f"source[{index}] disk_curve is missing seed")
            source["seed"] = int(source["seed"])
            if not 0 <= source["seed"] < num_vertices:
                raise ValueError(
                    f"source[{index}].seed={source['seed']} is outside "
                    f"[0,{num_vertices - 1}]"
                )
            source["radius"] = _finite_float(
                source.get("radius", 0.1), f"source[{index}].radius"
            )
            if source["radius"] <= 0.0:
                raise ValueError(f"source[{index}].radius must be positive")

        else:  # curve
            if "points_npy" in source:
                npy_path = Path(source.pop("points_npy")).expanduser()
                if not npy_path.is_absolute():
                    npy_path = path.parent / npy_path
                if not npy_path.is_file():
                    raise ValueError(
                        f"source[{index}].points_npy does not exist: {npy_path}"
                    )
                source["points"] = np.load(npy_path, allow_pickle=False)
            points = np.asarray(source.get("points"), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
                raise ValueError(
                    f"source[{index}].points must have finite shape (K,3), K>=2"
                )
            if not np.isfinite(points).all():
                raise ValueError(f"source[{index}].points contains NaN or infinity")
            source["points"] = points
            source["closed"] = _as_bool(source.get("closed", False))

        if "value_range" in source:
            source["value_range"] = _parse_range(
                source["value_range"], f"source[{index}].value_range"
            )
        sources.append(source)
    return sources


def _parse_range(value: Any, label: str) -> Tuple[float, float]:
    """Parse a two-element range ``(low, high)`` and require ``low < high``."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two numbers")
    low = _finite_float(value[0], f"{label}[0]")
    high = _finite_float(value[1], f"{label}[1]")
    if not low < high:
        raise ValueError(f"{label} must be increasing, got [{low}, {high}]")
    return low, high


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    """Load the checkpoint dictionary (full unpickling: trusted files only)."""
    # weights_only=False: the checkpoints also store optimizer state and
    # source dictionaries, and the whole file is unpickled.  Unpickling can
    # execute arbitrary code, so checkpoints must be trusted.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping/dictionary")
    return checkpoint


def _strip_uniform_prefix(
    state: Mapping[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    """Remove ``prefix`` from the keys if every key starts with it."""
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items()}
    return dict(state)


def _extract_model_state(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    """Return the network state dict stored in the checkpoint."""
    if isinstance(checkpoint.get("model_u"), Mapping):
        state = checkpoint["model_u"]
    elif isinstance(checkpoint.get("state_dict"), Mapping):
        state = checkpoint["state_dict"]
    elif checkpoint and all(isinstance(value, torch.Tensor)
                            for value in checkpoint.values()):
        state = checkpoint
    else:
        raise ValueError(
            "checkpoint contains neither a 'model_u' nor a 'state_dict' mapping"
        )
    result: Dict[str, torch.Tensor] = dict(state)
    for prefix in ("module.", "_orig_mod.", "model_u."):
        result = _strip_uniform_prefix(result, prefix)
    if "layer1.weight" not in result:
        raise ValueError("checkpoint model state is missing layer1.weight")
    return result


def _build_model_safely(
    checkpoint: Mapping[str, Any], device: torch.device, expected_code_size: int
) -> Tuple[MLP_Scalar, int, int, int, int, int]:
    """Construct MLP from authoritative state shapes and cross-check metadata.

    Returns ``(model, input_size, geo_size, code_size, normal_size, hidden)``
    with the model in eval mode on ``device``.
    """
    state = _extract_model_state(checkpoint)
    first_weight = state["layer1.weight"]
    if not isinstance(first_weight, torch.Tensor) or first_weight.ndim != 2:
        raise ValueError("checkpoint layer1.weight must be a rank-2 tensor")

    input_size = int(checkpoint.get("input_size", 3))
    if input_size != 3:
        raise ValueError(
            f"checkpoint input_size={input_size}; a 3D mesh evaluator requires 3"
        )
    hidden = int(first_weight.shape[0])

    code_size_raw = _metadata_value(checkpoint, ("code_size",))
    if code_size_raw is None:
        raise ValueError(
            "checkpoint is missing explicit code_size metadata; it is a legacy "
            "shared-scalar checkpoint and cannot be used for independent "
            "per-source Dirichlet generalization"
        )
    if isinstance(code_size_raw, torch.Tensor):
        if code_size_raw.numel() != 1:
            raise ValueError(
                "checkpoint code_size tensor must contain exactly one value"
            )
        code_size_raw = code_size_raw.item()
    try:
        code_size_number = float(code_size_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint code_size must be a positive integer, got {code_size_raw!r}"
        ) from exc
    if (
        isinstance(code_size_raw, (bool, np.bool_))
        or not math.isfinite(code_size_number)
        or not code_size_number.is_integer()
        or code_size_number <= 0
    ):
        raise ValueError(
            f"checkpoint code_size must be a positive integer, got {code_size_raw!r}"
        )
    code_size = int(code_size_number)
    if code_size != expected_code_size:
        raise ValueError(
            "checkpoint code_size does not match the selected independent "
            f"sources: checkpoint={code_size}, selected={expected_code_size}"
        )

    # MLP_Scalar currently has five frequencies and therefore contributes
    # input_size * 2 * len(freqs) positional features before geometry/code.
    probe = MLP_Scalar(input_size=input_size, geo_size=1, hidden=1)
    positional_features = input_size * 2 * len(probe.freqs)
    inferred_geo_size = int(first_weight.shape[1]) - positional_features
    normal_size = inferred_geo_size - code_size
    if normal_size not in (0, 3):
        raise ValueError(
            "cannot interpret checkpoint input layout: layer1 has "
            f"{first_weight.shape[1]} inputs, implying geo_size="
            f"{inferred_geo_size} with code_size={code_size}; expected "
            "geo_size=code_size ([codes]) or code_size+3 ([normal, codes])"
        )

    if "geo_size" in checkpoint:
        declared_geo_size = int(checkpoint["geo_size"])
        if declared_geo_size != inferred_geo_size:
            raise ValueError(
                f"checkpoint geo_size metadata ({declared_geo_size}) disagrees "
                f"with layer1.weight ({inferred_geo_size})"
            )
    if "hidden" in checkpoint and int(checkpoint["hidden"]) != hidden:
        raise ValueError(
            f"checkpoint hidden metadata ({checkpoint['hidden']}) disagrees "
            f"with layer1.weight ({hidden})"
        )

    model = MLP_Scalar(
        input_size=input_size, geo_size=inferred_geo_size, hidden=hidden
    ).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint is incompatible with MLP_Scalar: {exc}") from exc
    model.eval()
    return (
        model,
        input_size,
        inferred_geo_size,
        code_size,
        normal_size,
        hidden,
    )


def _metadata_value(checkpoint: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Return the first of ``keys`` found in the checkpoint, or None.

    The top level is searched first, then the optional ``metadata``,
    ``config`` and ``args`` sub-dictionaries.
    """
    containers: List[Mapping[str, Any]] = [checkpoint]
    for container_key in ("metadata", "config", "args"):
        candidate = checkpoint.get(container_key)
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    for container in containers:
        for key in keys:
            if key in container:
                return container[key]
    return None


def _checkpoint_no_normal_lift(checkpoint: Mapping[str, Any]) -> bool:
    """True if the checkpoint says the network was fed zero normals."""
    value = _metadata_value(checkpoint, ("no_normal_lift", "_no_normal_lift"))
    return False if value is None else _as_bool(value)


def _marked_value_sources(sources: Sequence[Mapping[str, Any]]) -> List[int]:
    """Indices, in ascending order, of the sources marked ``value_generalize``."""
    return [index for index, source in enumerate(sources)
            if _as_bool(source.get("value_generalize", False))]


def _same_float(left: Any, right: Any) -> bool:
    """Float comparison with rel_tol=1e-7, abs_tol=1e-9; False if not numeric."""
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _expanded_vertex_ids(
    source: Mapping[str, Any], vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Sorted vertex ids of a vertex source after ``patch_radius`` expansion.

    With ``patch_radius > 0`` every vertex whose graph distance
    (``fem_screen.dijkstra_from``) to one of the listed ids is at most
    ``patch_radius`` belongs to the source.
    """
    ids = np.asarray(source.get("ids", []), dtype=np.int64).reshape(-1)
    patch_radius = float(source.get("patch_radius", 0.0))
    if patch_radius <= 0.0:
        return np.unique(ids)
    expanded = set()
    for vertex_id in ids:
        distance = dijkstra_from(vertices, faces, int(vertex_id))
        expanded.update(np.where(distance <= patch_radius)[0].tolist())
    return np.asarray(sorted(expanded), dtype=np.int64)


def _validate_source_identity(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    index: int,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Reject a source JSON that describes a different FEM problem."""
    source_type = current["type"]
    for field in ("weight", "value"):
        if not _same_float(saved.get(field, 1.0), current.get(field, 1.0)):
            raise ValueError(
                f"source[{index}].{field} differs between checkpoint "
                f"({saved.get(field)!r}) and JSON ({current.get(field)!r})"
            )

    if source_type == "vertex":
        saved_radius = saved.get("patch_radius", 0.0)
        current_radius = current.get("patch_radius", 0.0)
        if not _same_float(saved_radius, current_radius):
            raise ValueError(
                f"source[{index}].patch_radius differs between checkpoint "
                f"({saved_radius!r}) and JSON ({current_radius!r})"
            )
        expected_ids = _expanded_vertex_ids(current, vertices, faces)
        saved_ids = np.unique(
            np.asarray(saved.get("ids", []), dtype=np.int64).reshape(-1)
        )
        if not np.array_equal(saved_ids, expected_ids):
            raise ValueError(
                f"source[{index}].ids differs between checkpoint and JSON "
                "after applying patch_radius"
            )
    elif source_type == "disk_curve":
        if int(saved.get("seed", -1)) != int(current["seed"]):
            raise ValueError(f"source[{index}].seed differs between checkpoint and JSON")
        if not _same_float(saved.get("radius", 0.1), current["radius"]):
            raise ValueError(
                f"source[{index}].radius differs between checkpoint and JSON"
            )
        if _as_bool(saved.get("generalize", False)) != _as_bool(
            current.get("generalize", False)
        ):
            raise ValueError(
                f"source[{index}].generalize differs between checkpoint and JSON"
            )
        if _as_bool(current.get("generalize", False)):
            saved_range = _parse_range(saved.get("r_range"),
                                       f"checkpoint source[{index}].r_range")
            current_range = _parse_range(current.get("r_range"),
                                         f"source[{index}].r_range")
            if not np.allclose(saved_range, current_range, rtol=1e-7, atol=1e-9):
                raise ValueError(
                    f"source[{index}].r_range differs between checkpoint and JSON"
                )
    else:
        saved_points = np.asarray(saved.get("points"), dtype=np.float64)
        current_points = np.asarray(current["points"], dtype=np.float64)
        if saved_points.shape != current_points.shape or not np.allclose(
            saved_points, current_points, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(
                f"source[{index}].points differs between checkpoint and JSON"
            )
        if _as_bool(saved.get("closed", False)) != _as_bool(
            current.get("closed", False)
        ):
            raise ValueError(
                f"source[{index}].closed differs between checkpoint and JSON"
            )


def _validate_selected_sources(
    sources: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    selected: Sequence[int],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Check the CLI source indices and the source JSON against the checkpoint.

    The indices must be unique and in range, and must equal the
    ``value_generalize`` markers of the JSON and of the checkpoint wherever
    such markers exist.  Every source saved in the checkpoint must describe
    the same constraint as the JSON source with the same index.
    """
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("--value-generalize-source contains duplicate indices")
    bad = [index for index in selected if index < 0 or index >= len(sources)]
    if bad:
        raise ValueError(
            f"source indices outside [0,{len(sources) - 1}]: {bad}"
        )

    json_marked = _marked_value_sources(sources)
    if json_marked and list(selected) != json_marked:
        raise ValueError(
            "CLI generalized source indices/order do not match source JSON "
            f"value_generalize markers: CLI={list(selected)}, "
            f"JSON={json_marked}"
        )

    checkpoint_sources = checkpoint.get("sources")
    if isinstance(checkpoint_sources, Sequence) and not isinstance(
        checkpoint_sources, (str, bytes)
    ):
        if len(checkpoint_sources) != len(sources):
            raise ValueError(
                "checkpoint/source JSON source counts differ: "
                f"{len(checkpoint_sources)} versus {len(sources)}"
            )
        for index, (saved, current) in enumerate(zip(checkpoint_sources, sources)):
            if not isinstance(saved, Mapping):
                raise ValueError(f"checkpoint source[{index}] is not a mapping")
            if saved.get("type") != current.get("type"):
                raise ValueError(
                    f"source[{index}] type differs between checkpoint "
                    f"({saved.get('type')!r}) and JSON ({current.get('type')!r})"
                )
            _validate_source_identity(
                saved, current, index, vertices=vertices, faces=faces
            )
        if all(isinstance(item, Mapping) for item in checkpoint_sources):
            checkpoint_marked = _marked_value_sources(checkpoint_sources)
            if checkpoint_marked and list(selected) != checkpoint_marked:
                raise ValueError(
                    "CLI generalized source indices/order do not match checkpoint "
                    f"value_generalize markers: CLI={list(selected)}, "
                    f"checkpoint={checkpoint_marked}"
                )


def _range_contains(
    trained_range: Tuple[float, float], requested: Tuple[float, float]
) -> bool:
    """True if ``requested`` lies inside ``trained_range`` (relative slack 1e-12)."""
    tolerance = 1e-12 * max(1.0, *(abs(value) for value in trained_range))
    return (requested[0] >= trained_range[0] - tolerance
            and requested[1] <= trained_range[1] + tolerance)


def _validate_generalization_metadata(
    checkpoint: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    requested_range: Tuple[float, float],
    allow_extrapolation: bool,
    allow_legacy: bool,
) -> None:
    """Check the generalization metadata of the checkpoint and the value range.

    The checkpoint must be a Dirichlet-value one whose generalized sources and
    code order equal ``selected``.  ``requested_range`` must lie inside the
    checkpoint ``code_range`` and every available ``value_range``; with
    ``allow_extrapolation`` a violation only issues a warning.
    ``allow_legacy`` only turns the error for a checkpoint without
    ``generalization_mode`` metadata into a warning.
    """
    mode_raw = _metadata_value(checkpoint, ("generalization_mode",))
    mode = None if mode_raw is None else str(mode_raw).strip().lower()
    accepted_modes = {"value", "dirichlet_value", "dirichlet-value"}
    if mode is not None and mode not in accepted_modes:
        raise ValueError(
            f"checkpoint generalization_mode={mode_raw!r}, not Dirichlet-value "
            "conditioning; refusing an invalid FEM comparison"
        )

    checkpoint_sources = checkpoint.get("sources")
    saved_sources = (checkpoint_sources if isinstance(checkpoint_sources, Sequence)
                     and not isinstance(checkpoint_sources, (str, bytes)) else [])
    radius_evidence = _as_bool(checkpoint.get("has_generalization", False)) or any(
        isinstance(source, Mapping) and _as_bool(source.get("generalize", False))
        for source in saved_sources
    )
    if mode in accepted_modes:
        if not saved_sources:
            raise ValueError(
                "Dirichlet-value checkpoint is missing its saved source list"
            )
        saved_marked = _marked_value_sources(saved_sources)
        if saved_marked != list(selected):
            raise ValueError(
                "Dirichlet-value checkpoint source markers/order disagree with CLI: "
                f"checkpoint={saved_marked}, CLI={list(selected)}"
            )
        for code_index, source_index in enumerate(selected):
            saved_code_index = saved_sources[source_index].get("value_code_index")
            if saved_code_index is None:
                raise ValueError(
                    f"checkpoint source[{source_index}] is missing "
                    "value_code_index; this is not an independent-code checkpoint"
                )
            if int(saved_code_index) != code_index:
                raise ValueError(
                    f"checkpoint source[{source_index}].value_code_index="
                    f"{saved_code_index!r}, but CLI position requires {code_index}"
                )
        radius_sources = [
            index for index, source in enumerate(saved_sources)
            if isinstance(source, Mapping)
            and _as_bool(source.get("generalize", False))
        ]
        if radius_sources:
            raise ValueError(
                "checkpoint declares value generalization but also contains "
                f"radius-generalized source(s) {radius_sources}"
            )
        if "has_generalization" in checkpoint and not _as_bool(
            checkpoint["has_generalization"]
        ):
            raise ValueError(
                "checkpoint generalization_mode='value' contradicts "
                "has_generalization=False"
            )
    if mode is None:
        message = (
            "legacy checkpoint has no generalization_mode metadata; its scalar "
            "code semantics cannot be proven"
        )
        if not allow_legacy:
            raise ValueError(
                message
                + ("; radius-generalization metadata was also detected"
                   if radius_evidence else "")
                + ". Use a value-generalization checkpoint, or pass "
                "--allow-legacy-checkpoint only after manually verifying it."
            )
        warnings.warn(
            message + (" (radius metadata also detected)" if radius_evidence else ""),
            RuntimeWarning,
        )

    candidate_ranges: List[Tuple[str, Tuple[float, float]]] = []
    code_range = _metadata_value(checkpoint, ("code_range",))
    if mode in accepted_modes and code_range is None:
        raise ValueError("Dirichlet-value checkpoint is missing code_range metadata")
    if code_range is not None:
        candidate_ranges.append(
            ("checkpoint code_range", _parse_range(code_range, "checkpoint code_range"))
        )
    if mode in accepted_modes:
        checkpoint_range = candidate_ranges[0][1]
        for index in selected:
            if "value_range" not in saved_sources[index]:
                raise ValueError(
                    f"checkpoint source[{index}] is missing value_range"
                )
            saved_range = _parse_range(
                saved_sources[index]["value_range"],
                f"checkpoint source[{index}].value_range",
            )
            if not np.allclose(
                saved_range, checkpoint_range, rtol=1e-7, atol=1e-9
            ):
                raise ValueError(
                    f"checkpoint source[{index}].value_range {saved_range} "
                    f"disagrees with code_range {checkpoint_range}"
                )
            candidate_ranges.append(
                (f"checkpoint source[{index}].value_range", saved_range)
            )
    for index in selected:
        if "value_range" in sources[index]:
            candidate_ranges.append(
                (f"source[{index}].value_range", tuple(sources[index]["value_range"]))
            )
    for label, trained_range in candidate_ranges:
        if not _range_contains(trained_range, requested_range):
            message = (
                f"requested g range {requested_range} exceeds {label} "
                f"{trained_range}"
            )
            if allow_extrapolation:
                warnings.warn(message + "; evaluating extrapolation", RuntimeWarning)
            else:
                raise ValueError(message + "; use --allow-code-extrapolation to override")


def _resolve_device(spec: str) -> torch.device:
    """Turn the --device string into a torch.device ('auto' prefers CUDA)."""
    if spec.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(spec)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid --device {spec!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"--device {spec!r} requested CUDA, but CUDA is unavailable")
    return device


def _compute_vertex_normals(
    mesh: MeshGeometry,
    requested_source: str,
    normal_size: int,
    no_normal_lift: bool,
    device: torch.device,
    sdf_resolution: int,
    sdf_padding: float,
    fd_halo: int,
    batch_size: int,
) -> Tuple[np.ndarray | None, str]:
    """Return the normals fed to the network and the effective normal source.

    The effective source is ``"none"`` (the network has no normal input; the
    array is None), ``"zero"`` (checkpoint trained with no_normal_lift),
    ``"mesh"`` (mesh vertex normals) or ``"sdf"`` (SparseGridFDM_SDF normals
    queried at the mesh vertices; requires a CUDA device).
    """
    if normal_size == 0:
        if requested_source == "sdf":
            warnings.warn(
                "--normal-source=sdf is ignored because this checkpoint has no "
                "normal input (geo_size=code_size)",
                RuntimeWarning,
            )
        return None, "none"
    if normal_size != 3:
        raise ValueError(
            f"unsupported checkpoint normal_size={normal_size}; expected 0 or 3"
        )
    if no_normal_lift:
        print("Checkpoint requests no_normal_lift: supplying zero normals.")
        return np.zeros_like(mesh.vertices_np, dtype=np.float32), "zero"
    if requested_source == "mesh":
        normals = np.asarray(mesh.vertex_normals_np, dtype=np.float32)
        return normals, "mesh"

    if device.type != "cuda":
        raise ValueError(
            "--normal-source=sdf requires --device cuda (or auto resolving to "
            "CUDA) because SparseGridFDM_SDF uses fvdb"
        )
    # Import lazily so mesh-normal evaluation does not require the SDF stack.
    from SparseGridFDM_SDF import SparseGridFDM_SDF

    print(
        "Building SparseGridFDM_SDF once "
        f"(resolution={sdf_resolution}, padding={sdf_padding}, halo={fd_halo}) ..."
    )
    sdf = SparseGridFDM_SDF.from_mesh(
        mesh,
        resolution=sdf_resolution,
        padding=sdf_padding,
        fd_halo=fd_halo,
        device=device,
    )
    vertices = np.asarray(mesh.vertices_np, dtype=np.float32)
    normals = np.empty_like(vertices)
    with torch.no_grad():
        for start in range(0, len(vertices), batch_size):
            end = min(start + batch_size, len(vertices))
            points = torch.from_numpy(vertices[start:end]).to(device)
            normals[start:end] = sdf.normal(points).detach().cpu().numpy()
    if not np.isfinite(normals).all():
        raise RuntimeError("SDF produced non-finite normals at mesh vertices")
    return normals, "sdf"


def _sources_with_selected_values(
    sources: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    values: Sequence[float],
) -> List[Dict[str, Any]]:
    """Deep copy of ``sources`` with the ``value`` of the selected ones replaced."""
    if len(selected) != len(values):
        raise ValueError(
            f"selected/value size mismatch: {len(selected)} versus {len(values)}"
        )
    result = copy.deepcopy(list(sources))
    for index, value in zip(selected, values):
        result[index]["value"] = float(value)
    return result


def _evaluate_mse(
    model: MLP_Scalar,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None,
    codes: np.ndarray,
    sources: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    lambda_dir: float,
    lambda_mass: float,
    code_size: int,
    normal_size: int,
    device: torch.device,
    max_rows: int,
    code_batch_size: int,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-code ``(mse, rel_l2)`` arrays of the network against FEM.

    Despite its name the function computes both metrics.  ``codes`` has shape
    (num_codes, code_size); every row triggers one ``solve_fem_screened`` call
    in which the selected sources take the values of that row.  The network is
    evaluated at all mesh vertices for ``code_batch_size`` codes at a time,
    with at most ``max_rows`` input rows per forward pass.
    """
    num_vertices = len(vertices)
    if codes.ndim != 2 or codes.shape[1] != code_size:
        raise ValueError(
            f"codes must have shape (num_samples, {code_size}), got {codes.shape}"
        )
    if len(selected) != code_size:
        raise ValueError(
            f"selected sources must have length {code_size}, got {len(selected)}"
        )
    mse_result = np.empty(len(codes), dtype=np.float64)
    rel_result = np.empty(len(codes), dtype=np.float64)
    # Keep FEM geometry in float64; the neural network receives its usual
    # float32 copy in the identical vertex order.
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    network_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    if normals is not None:
        normals = np.ascontiguousarray(normals, dtype=np.float32)

    started = time.perf_counter()
    next_progress = 1
    with torch.inference_mode():
        for code_start in range(0, len(codes), code_batch_size):
            code_end = min(code_start + code_batch_size, len(codes))
            code_values = codes[code_start:code_end]
            count = len(code_values)
            sum_squared_error = np.zeros(count, dtype=np.float64)
            sum_squared_fem = np.zeros(count, dtype=np.float64)
            vertex_batch_size = max(1, max_rows // count)

            # Each sampled independent code vector gets its own real FEM solve.
            # Keep only this small code batch in memory; no basis interpolation
            # or response linear combination is used.
            fem_batch = np.empty((count, num_vertices), dtype=np.float64)
            for local_index, code_row in enumerate(code_values):
                sample_index = code_start + local_index
                done = sample_index + 1
                if progress_every > 0 and (
                    done % progress_every == 0 or done == len(codes)
                ):
                    print(
                        f"FEM solve {done}/{len(codes)}: "
                        + ", ".join(
                            f"source[{source_index}]={code_row[code_index]:+.6g}"
                            for code_index, source_index in enumerate(selected)
                        )
                    )
                fem_solution, _ = solve_fem_screened(
                    vertices,
                    faces,
                    _sources_with_selected_values(
                        sources, selected, code_row
                    ),
                    lambda_dir=lambda_dir,
                    lambda_mass=lambda_mass,
                    verbose=False,
                )
                fem_solution = np.asarray(
                    fem_solution, dtype=np.float64
                ).reshape(-1)
                if fem_solution.shape != (num_vertices,):
                    raise RuntimeError(
                        f"FEM sample {sample_index} returned shape "
                        f"{fem_solution.shape}; expected ({num_vertices},)"
                    )
                if not np.isfinite(fem_solution).all():
                    raise RuntimeError(
                        f"FEM sample {sample_index} returned NaN or infinity"
                    )
                fem_batch[local_index] = fem_solution
                sum_squared_fem[local_index] = float(
                    np.dot(fem_solution, fem_solution)
                )

            code_tensor = torch.as_tensor(
                code_values, dtype=torch.float32, device=device
            )
            for vertex_start in range(0, num_vertices, vertex_batch_size):
                vertex_end = min(vertex_start + vertex_batch_size, num_vertices)
                chunk_size = vertex_end - vertex_start
                points_one = torch.from_numpy(
                    network_vertices[vertex_start:vertex_end]
                ).to(device)
                points = points_one.unsqueeze(0).expand(
                    count, chunk_size, 3
                ).reshape(-1, 3)
                code_columns = code_tensor[:, None, :].expand(
                    count, chunk_size, code_size
                ).reshape(-1, code_size)

                if normal_size == 3:
                    if normals is None:
                        raise RuntimeError("normal_size=3 requires a normal array")
                    normals_one = torch.from_numpy(
                        normals[vertex_start:vertex_end]
                    ).to(device)
                    normals_batch = normals_one.unsqueeze(0).expand(
                        count, chunk_size, 3
                    ).reshape(-1, 3)
                    network_input = torch.cat(
                        (points, normals_batch, code_columns), dim=1
                    )
                else:
                    network_input = torch.cat((points, code_columns), dim=1)

                prediction = model(network_input).reshape(count, chunk_size)
                prediction_np = prediction.detach().cpu().numpy().astype(
                    np.float64, copy=False
                )
                difference = (
                    prediction_np - fem_batch[:, vertex_start:vertex_end]
                )
                sum_squared_error += np.einsum(
                    "ij,ij->i", difference, difference, optimize=True
                )

            mse_result[code_start:code_end] = sum_squared_error / num_vertices
            # rel_l2 = sqrt(sum((nn-fem)^2) / sum(fem^2))
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.sqrt(sum_squared_error / sum_squared_fem)
            rel[~np.isfinite(rel)] = np.nan
            rel_result[code_start:code_end] = rel
            completed = code_end
            should_report = (
                progress_every > 0
                and (completed >= next_progress or completed == len(codes))
            )
            if should_report:
                elapsed = time.perf_counter() - started
                rate = completed / max(elapsed, 1e-12)
                remaining = (len(codes) - completed) / max(rate, 1e-12)
                print(
                    f"  evaluated {completed:>{len(str(len(codes)))}}/{len(codes)} "
                    f"({100.0 * completed / len(codes):5.1f}%), "
                    f"elapsed={elapsed:.1f}s, ETA={remaining:.1f}s"
                )
                while next_progress <= completed:
                    next_progress += progress_every
    if not np.isfinite(mse_result).all():
        bad = np.where(~np.isfinite(mse_result))[0][:10].tolist()
        raise RuntimeError(f"non-finite MSE values at samples {bad}")
    return mse_result, rel_result


def _write_tsv(
    path: Path,
    codes: np.ndarray,
    mse: np.ndarray,
    rel_l2: np.ndarray,
    selected: Sequence[int],
    metadata: Mapping[str, Any],
) -> None:
    """Write the result file: format line, ``# key: json`` metadata, TSV table.

    The table columns are ``sample_index``, one ``source_<index>_value`` per
    selected source, ``mse`` and ``rel_l2``.  Other scripts read the table by
    these column names, so they must not change.
    """
    if len(codes) != len(mse) or len(codes) != len(rel_l2):
        raise ValueError(
            f"cannot write TSV: {len(codes)} codes, {len(mse)} MSE, "
            f"{len(rel_l2)} rel_l2 values"
        )
    if codes.ndim != 2 or codes.shape[1] != len(selected):
        raise ValueError(
            "cannot write TSV: codes shape does not match selected sources: "
            f"codes={codes.shape}, selected={list(selected)}"
        )
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# format: dirichlet_generalization_mse_v3\n")
        for key, value in metadata.items():
            handle.write(
                f"# {key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}\n"
            )
        value_columns = [f"source_{index}_value" for index in selected]
        handle.write(
            "\t".join(["sample_index", *value_columns, "mse", "rel_l2"]) + "\n"
        )
        for sample_index, (code_row, error, rel) in enumerate(
            zip(codes, mse, rel_l2)
        ):
            fields = [
                str(sample_index),
                *(f"{float(value):.17g}" for value in code_row),
                f"{float(error):.17g}",
                f"{float(rel):.17g}",
            ]
            handle.write("\t".join(fields) + "\n")


def _load_codes_file(path: Path, code_size: int) -> np.ndarray:
    """Load code rows from a text file as a float64 array (N, code_size).

    Each row holds one whitespace-separated code vector.  Blank lines and
    lines starting with ``#`` are skipped, and so are header lines (first
    token ``epoch``, ``sample_index`` or ``source_...``).  A row with
    ``code_size + 1`` columns whose first column is an integer is taken to
    carry a leading epoch index, which is dropped; any other column count is
    an error.  This function is also imported by ``export_split_samples.py``.
    """
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            lower = parts[0].lower()
            if lower in {"epoch", "sample_index"} or lower.startswith("source_"):
                continue
            # drop leading epoch index when present
            if len(parts) == code_size + 1:
                try:
                    int(parts[0])
                    parts = parts[1:]
                except ValueError:
                    pass
            if len(parts) != code_size:
                raise ValueError(
                    f"{path}: expected {code_size} code columns, got {len(parts)} "
                    f"in line: {line}"
                )
            rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"no code rows in {path}")
    return np.asarray(rows, dtype=np.float64)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate the numeric command-line arguments (floats are normalized in place)."""
    args.g_min = _finite_float(args.g_min, "--g-min")
    args.g_max = _finite_float(args.g_max, "--g-max")
    if not args.g_min < args.g_max:
        raise ValueError("--g-min must be smaller than --g-max")
    if args.num_samples <= 0 and args.codes_file is None:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.code_batch_size <= 0:
        raise ValueError("--code-batch-size must be positive")
    if args.code_batch_size > args.batch_size:
        raise ValueError("--code-batch-size cannot exceed --batch-size")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    args.lambda_dir = _finite_float(args.lambda_dir, "--lambda-dir")
    args.lambda_mass = _finite_float(args.lambda_mass, "--lambda-mass")
    if args.lambda_dir < 0.0 or args.lambda_mass < 0.0:
        raise ValueError("--lambda-dir and --lambda-mass must be non-negative")
    if args.lambda_dir == 0.0 and args.lambda_mass == 0.0:
        raise ValueError("--lambda-dir and --lambda-mass cannot both be zero")
    if args.sdf_res < 2:
        raise ValueError("--sdf-res must be at least 2")
    args.sdf_padding = _finite_float(args.sdf_padding, "--sdf-padding")
    if args.sdf_padding < 0.0:
        raise ValueError("--sdf-padding must be non-negative")
    if args.fd_halo < 2:
        raise ValueError("--fd-halo must be at least 2")


def main() -> int:
    """Run the evaluation; return 0 on success, 2 after printing ``ERROR: ...``."""
    args = parse_args()
    try:
        _validate_args(args)
        mesh_path = _require_file(args.mesh, "--mesh")
        source_path = _require_file(args.source_spec, "--source-spec")
        checkpoint_path = _require_file(args.checkpoint, "--checkpoint")
        output_path = args.output.expanduser().resolve()
        if output_path in {mesh_path, source_path, checkpoint_path}:
            raise ValueError("--output must not overwrite an input file")
        args.output = output_path
        device = _resolve_device(args.device)
        selected = list(args.value_generalize_source)

        print(f"Device: {device}")
        print("Loading and normalizing mesh with MeshGeometry ...")
        mesh = MeshGeometry(str(mesh_path), normalize=True, device=device)
        vertices = np.asarray(mesh.vertices_np, dtype=np.float64)
        faces = np.asarray(mesh.faces_np, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("mesh has no vertices or faces")
        print(f"Evaluation mesh: {len(vertices)} vertices, {len(faces)} faces")

        sources = _load_sources(source_path, len(vertices))
        print(f"Loaded {len(sources)} sources from {source_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = _torch_load(checkpoint_path, device)
        _validate_selected_sources(
            sources,
            checkpoint,
            selected,
            vertices=np.asarray(mesh.vertices_np),
            faces=faces,
        )
        _validate_generalization_metadata(
            checkpoint=checkpoint,
            sources=sources,
            selected=selected,
            requested_range=(args.g_min, args.g_max),
            allow_extrapolation=args.allow_code_extrapolation,
            allow_legacy=args.allow_legacy_checkpoint,
        )
        model, input_size, geo_size, code_size, normal_size, hidden = (
            _build_model_safely(
                checkpoint, device, expected_code_size=len(selected)
            )
        )
        no_normal_lift = _checkpoint_no_normal_lift(checkpoint)
        print(
            f"Model: input_size={input_size}, geo_size={geo_size}, "
            f"code_size={code_size}, normal_size={normal_size}, hidden={hidden}, "
            f"no_normal_lift={no_normal_lift}"
        )
        print(f"Independent code/source order: {selected}")

        normals, effective_normal_source = _compute_vertex_normals(
            mesh=mesh,
            requested_source=args.normal_source,
            normal_size=normal_size,
            no_normal_lift=no_normal_lift,
            device=device,
            sdf_resolution=args.sdf_res,
            sdf_padding=args.sdf_padding,
            fd_halo=args.fd_halo,
            batch_size=args.batch_size,
        )
        print(f"Effective normal source: {effective_normal_source}")

        rng = np.random.default_rng(args.seed)
        if args.codes_file is not None:
            codes_path = _require_file(args.codes_file, "--codes-file")
            codes = _load_codes_file(codes_path, code_size)
            print(
                f"Evaluating {len(codes)} code rows from {codes_path} "
                f"({code_size}-value) with {len(codes)} direct FEM solves ..."
            )
        else:
            codes = rng.uniform(
                args.g_min,
                args.g_max,
                size=(args.num_samples, code_size),
            ).astype(np.float64)
            print(
                f"Evaluating {args.num_samples} independent {code_size}-value "
                f"Dirichlet samples in [{args.g_min}, {args.g_max}] with "
                f"{args.num_samples} direct FEM solves ..."
            )
        mse, rel_l2 = _evaluate_mse(
            model=model,
            vertices=vertices,
            faces=faces,
            normals=normals,
            codes=codes,
            sources=sources,
            selected=selected,
            lambda_dir=args.lambda_dir,
            lambda_mass=args.lambda_mass,
            code_size=code_size,
            normal_size=normal_size,
            device=device,
            max_rows=args.batch_size,
            code_batch_size=args.code_batch_size,
            progress_every=args.progress_every,
        )

        metadata = {
            "mesh": str(mesh_path),
            "source_spec": str(source_path),
            "checkpoint": str(checkpoint_path),
            "mesh_normalized": True,
            "num_vertices": int(len(vertices)),
            "num_faces": int(len(faces)),
            "source_indices": selected,
            "independent_generalization_values": True,
            "code_size": code_size,
            "code_columns": [f"source_{index}_value" for index in selected],
            "g_min": args.g_min,
            "g_max": args.g_max,
            "num_samples": int(len(codes)),
            "codes_file": (None if args.codes_file is None else str(args.codes_file)),
            "seed": args.seed,
            "lambda_dir": args.lambda_dir,
            "lambda_mass": args.lambda_mass,
            "device": str(device),
            "normal_source": effective_normal_source,
            "requested_normal_source": args.normal_source,
            "no_normal_lift": no_normal_lift,
            "geo_size": geo_size,
            "normal_size": normal_size,
            "hidden": hidden,
            "fem_solves": int(len(codes)),
            "fem_mode": "one direct solve_fem_screened call per sampled code vector",
            "mse_definition": "mean over all mesh vertices of (u_nn-u_fem)^2",
            "rel_l2_definition": (
                "sqrt(sum((u_nn-u_fem)^2) / sum(u_fem^2)) over all mesh vertices"
            ),
            "mse_min": float(mse.min()),
            "mse_mean": float(mse.mean()),
            "mse_max": float(mse.max()),
            "rel_l2_min": float(np.nanmin(rel_l2)),
            "rel_l2_mean": float(np.nanmean(rel_l2)),
            "rel_l2_max": float(np.nanmax(rel_l2)),
        }
        _write_tsv(args.output, codes, mse, rel_l2, selected, metadata)
        print(f"Wrote TSV: {args.output.resolve()}")
        print(
            f"MSE summary: min={mse.min():.8g}, mean={mse.mean():.8g}, "
            f"max={mse.max():.8g}"
        )
        print(
            f"rel_l2 summary: min={np.nanmin(rel_l2):.8g}, "
            f"mean={np.nanmean(rel_l2):.8g}, max={np.nanmax(rel_l2):.8g}"
        )
        return 0
    except (
        ValueError,
        RuntimeError,
        OSError,
        ImportError,
        AssertionError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
