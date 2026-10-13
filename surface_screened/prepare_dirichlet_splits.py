#!/usr/bin/env python3
"""Build the train / interp / extrap Dirichlet code lists for the generalization evaluation.

Step 2 of the Dirichlet-value generalization pipeline: train_screen.py (with
--train-codes-out) logs the Dirichlet code vector drawn at every epoch, this
script turns that log into three lists of code vectors, and
evaluate_dirichlet_generalization.py (--codes-file) compares the network with
the FEM reference on every code of a list.  Only numpy is needed.

Input (--train-codes)
---------------------
The log written by train_screen.py --train-codes-out: a tab-separated text
file with the header line

    epoch  source_0  ...  source_{d-1}

followed by one line per epoch,

    <epoch>  <code_0>  ...  <code_{d-1}>

d is the number of code components (one per value-generalized source).  Blank
lines, lines starting with '#' and the header line are skipped, and a first
column that parses as an integer (the epoch) is dropped.  The input therefore
has to be the trainer's log including its epoch column; the code lists written
by this script are not valid input.

Outputs (in --out-dir)
----------------------
Let train_all be all rows of the log.  ONE generator,
rng = numpy.random.default_rng(--seed), is consumed in the order train,
interp, extrap:

1. codes_train.txt -- a random subset of the logged training codes.
   n = min(--num-train, len(train_all)) row indices are drawn with
   rng.choice(len(train_all), size=n, replace=False) and sorted, so the
   selected codes stay in epoch order.  With --num-train <= 0 all rows are
   written and no random number is consumed.
2. codes_interp.txt -- --num-interp vectors, drawn one at a time as
   x = rng.uniform(g_min, g_max, size=d), i.e. uniform in [g_min, g_max]^d,
   excluding any vector that equals a training code (any row of train_all,
   not only the subset above; compared after rounding to 12 decimals).
3. codes_extrap.txt -- --num-extrap vectors, drawn one at a time as
   x = rng.uniform(extrap_min, extrap_max, size=d), i.e. uniform in
   [extrap_min, extrap_max]^d, rejecting every vector that lies fully inside
   the closed box [inner_min, inner_max]^d.  A kept vector has at least one
   component outside the inner box; its other components may be inside.
   --extrap-min / --extrap-max default to --g-min / --g-max.

Because the generator is shared, the interp and extrap lists depend on the
seed, the number of rows in the log, --num-train, d, the boxes and
--num-interp.  Sampling of a list stops with an error after more than
1000 * n draws for n requested vectors, and an extrapolation box that lies
entirely inside the inner box (no vector could be accepted) is refused before
anything is read or written.

Format of the three output files: UTF-8, tab-separated, the header line
"source_0 ... source_{d-1}", then one code vector per line with 17
significant digits.  source_i is the i-th code component (its position in the
code vector), as in the training log.

Usage (from the repository root; the settings of the dc experiment)
-------------------------------------------------------------------
    python prepare_dirichlet_splits.py --train-codes results/generalization/dc/train_codes.txt --out-dir results/generalization/dc/eval --g-min -3 --g-max 2 --extrap-min -3.5 --extrap-max 2.5 --inner-min -3 --inner-max 2 --num-train 2000 --num-interp 2000 --num-extrap 2000 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_train_codes(path: Path) -> np.ndarray:
    """Read the per-epoch training-code log; returns the (rows, d) float64 codes.

    Blank lines, '#' lines and the 'epoch ...' header are skipped; the first
    column is dropped when it parses as an integer (the epoch index).
    """
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if parts[0].lower() == "epoch":
                continue
            # A log row is '<epoch> <code_0> ... <code_{d-1}>'.  This first
            # assignment is always overwritten by the try/except below, which
            # drops the first token only if it parses as an integer.
            vals = parts[1:] if len(parts) > 1 else parts
            try:
                int(parts[0])
                vals = parts[1:]
            except ValueError:
                vals = parts
            rows.append([float(x) for x in vals])
    if not rows:
        raise SystemExit(f"no codes in {path}")
    return np.asarray(rows, dtype=np.float64)


def key(row: np.ndarray) -> tuple:
    """Hashable key of a code vector (rounded to 12 decimals) for the training-code lookup."""
    return tuple(np.round(row, 12).tolist())


def write_codes(path: Path, codes: np.ndarray) -> None:
    """Write codes (n, d) as a TSV: header 'source_0 ... source_{d-1}', one row per code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = codes.shape[1]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(f"source_{i}" for i in range(d)) + "\n")
        for row in codes:
            f.write("\t".join(f"{float(x):.17g}" for x in row) + "\n")


def sample_interp(n, d, lo, hi, forbid, rng):
    """Draw n vectors uniformly in [lo, hi]^d, skipping any whose key is in forbid.

    One rng.uniform(lo, hi, size=d) call per candidate; exits after more than
    n * 1000 candidates.
    """
    out = []
    tries = 0
    while len(out) < n:
        x = rng.uniform(lo, hi, size=d)
        tries += 1
        # Checked on every draw, so a configuration that rejects every
        # candidate stops with an error instead of looping forever.
        if tries > n * 1000:
            raise SystemExit("interp sampling failed: too many collisions")
        if key(x) in forbid:
            continue
        out.append(x)
    return np.asarray(out, dtype=np.float64)


def sample_extrap(n, d, lo, hi, inner_lo, inner_hi, rng):
    """Draw n vectors uniformly in [lo, hi]^d, rejecting those fully inside the inner box.

    A candidate is rejected when all of its components lie in the closed
    interval [inner_lo, inner_hi].  One rng.uniform(lo, hi, size=d) call per
    candidate; exits after more than n * 1000 candidates.
    """
    out = []
    tries = 0
    while len(out) < n:
        x = rng.uniform(lo, hi, size=d)
        tries += 1
        # Checked on every draw, so a configuration that rejects every
        # candidate stops with an error instead of looping forever.
        if tries > n * 1000:
            raise SystemExit("extrap sampling failed: rejection too high")
        if np.all((x >= inner_lo) & (x <= inner_hi)):
            continue
        out.append(x)
    return np.asarray(out, dtype=np.float64)


def main():
    """Parse the arguments and write codes_train.txt, codes_interp.txt, codes_extrap.txt."""
    p = argparse.ArgumentParser()
    p.add_argument("--train-codes", type=Path, required=True,
                   help="per-epoch training-code log written by "
                        "train_screen.py --train-codes-out")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="folder that receives codes_train.txt, "
                        "codes_interp.txt and codes_extrap.txt")
    p.add_argument("--g-min", type=float, default=-3.0,
                   help="lower bound of the interp sampling box; use the "
                        "value range of the training run (default: -3.0)")
    p.add_argument("--g-max", type=float, default=2.0,
                   help="upper bound of the interp sampling box; use the "
                        "value range of the training run (default: 2.0)")
    p.add_argument("--extrap-min", type=float, default=None,
                   help="outer box min for extrap (default: --g-min)")
    p.add_argument("--extrap-max", type=float, default=None,
                   help="outer box max for extrap (default: --g-max)")
    p.add_argument("--inner-min", type=float, default=-3.0,
                   help="drop extrap samples fully inside [inner_min,inner_max]")
    p.add_argument("--inner-max", type=float, default=2.0,
                   help="upper bound of the inner box, see --inner-min "
                        "(default: 2.0)")
    p.add_argument("--num-train", type=int, default=1000,
                   help="random subset of training codes used for FEM eval "
                        "(<=0 means use all)")
    p.add_argument("--num-interp", type=int, default=1000,
                   help="number of interp codes to draw (default: 1000)")
    p.add_argument("--num-extrap", type=int, default=1000,
                   help="number of extrap codes to draw (default: 1000)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the single numpy default_rng shared by the "
                        "train, interp and extrap lists (default: 0)")
    args = p.parse_args()

    extrap_lo = args.g_min if args.extrap_min is None else args.extrap_min
    extrap_hi = args.g_max if args.extrap_max is None else args.extrap_max
    if not extrap_lo < extrap_hi:
        raise SystemExit("--extrap-min must be < --extrap-max")
    if not args.inner_min < args.inner_max:
        raise SystemExit("--inner-min must be < --inner-max")
    # Every extrap candidate lies in [extrap_lo, extrap_hi]; if that box is
    # inside the inner box all candidates would be rejected.
    if extrap_lo >= args.inner_min and extrap_hi <= args.inner_max:
        raise SystemExit(
            f"extrap box [{extrap_lo},{extrap_hi}] lies entirely inside the "
            f"inner box [{args.inner_min},{args.inner_max}], so every extrap "
            "sample would be rejected; pass --extrap-min/--extrap-max (they "
            "default to --g-min/--g-max) so that the extrap box extends "
            "beyond [--inner-min,--inner-max]"
        )

    train_all = load_train_codes(args.train_codes)
    d = train_all.shape[1]
    # exclude ANY training code from interp (full set), not just the eval subset
    forbid = {key(r) for r in train_all}
    # One generator for all three lists, consumed in the order train, interp,
    # extrap; do not reorder the calls below.
    rng = np.random.default_rng(args.seed)

    if args.num_train is not None and args.num_train > 0:
        n = min(int(args.num_train), len(train_all))
        pick = rng.choice(len(train_all), size=n, replace=False)
        pick.sort()
        train_eval = train_all[pick]
    else:
        train_eval = train_all

    interp = sample_interp(
        args.num_interp, d, args.g_min, args.g_max, forbid, rng
    )
    extrap = sample_extrap(
        args.num_extrap, d, extrap_lo, extrap_hi,
        args.inner_min, args.inner_max, rng,
    )

    out = args.out_dir
    write_codes(out / "codes_train.txt", train_eval)
    write_codes(out / "codes_interp.txt", interp)
    write_codes(out / "codes_extrap.txt", extrap)
    print(
        f"train_all={len(train_all)} train_eval={len(train_eval)} "
        f"interp={len(interp)} extrap={len(extrap)} dim={d}"
    )
    print(
        f"interp box=[{args.g_min},{args.g_max}]  "
        f"extrap box=[{extrap_lo},{extrap_hi}]  "
        f"extrap drop-if-inside=[{args.inner_min},{args.inner_max}]"
    )
    print(f"wrote under {out.resolve()}")


if __name__ == "__main__":
    main()
