#!/usr/bin/env python3
# correlation_cooc_percent.py
import argparse
import os
import numpy as np
import pandas as pd  # <-- keep import at module level only

# 12 clusters = 11 official + 'ukjent' fallback
ROOMS_12 = [
    "andre områder", "kjøkken", "stue", "gang", "soverom", "bad",
    "bod", "vaskerom", "wc", "kjeller", "garasje", "ukjent",
]
ROOM_TO_IDX = {r: i for i, r in enumerate(ROOMS_12)}

NUM_ROOMS = 12
NUM_JOBS = 388
N_PAIRS = NUM_ROOMS * NUM_JOBS  # 4656


def room_to_idx(name: str) -> int:
    if not isinstance(name, str):
        return ROOM_TO_IDX["ukjent"]
    lname = name.strip().lower()
    if lname in ROOM_TO_IDX:
        return ROOM_TO_IDX[lname]
    for base in ROOMS_12[:-1]:  # try substring match (exclude 'ukjent')
        if base in lname:
            return ROOM_TO_IDX[base]
    return ROOM_TO_IDX["ukjent"]


def pair_index(room_idx: int, job_idx: int) -> int:
    return room_idx * NUM_JOBS + job_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="TRAIN CSV with columns: id,project_id,room,work_operation_cluster_code")
    ap.add_argument("--project-col", default="project_id")
    ap.add_argument("--room-col", default="room")
    ap.add_argument("--job-code-col", default="work_operation_cluster_code")
    ap.add_argument("--outdir", default="corr_out_simple")
    ap.add_argument("--save-head", action="store_true",
                    help="Also save a small 12x12 preview CSV")
    args = ap.parse_args()

    print(f"Reading {args.csv} ...")
    df = pd.read_csv(args.csv)
    for col in (args.project_col, args.room_col, args.job_code_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV")

    cooc = np.zeros((N_PAIRS, N_PAIRS), dtype=np.int32)
    freq = np.zeros(N_PAIRS, dtype=np.int32)

    print("Grouping by project and collecting active room-job pairs...")
    n_projects = 0
    for pid, g in df.groupby(args.project_col, sort=False):
        n_projects += 1
        active_pairs = set()

        # Each row is (project_id, room, job_code)
        for _, row in g.iterrows():
            r_idx = room_to_idx(row[args.room_col])
            j = int(row[args.job_code_col])
            if 0 <= j < NUM_JOBS:
                active_pairs.add(pair_index(r_idx, j))

        if not active_pairs:
            continue

        alist = list(active_pairs)
        # per-item project frequency
        for i in alist:
            freq[i] += 1
        # co-occurrence (including self)
        for i in alist:
            cooc[i, alist] += 1

    print(f"Processed projects: {n_projects:,}")
    print("Co-occurrence matrix shape:", cooc.shape)

    # Column-wise conditional percentages:
    # P(i|j) = 100 * cooc[i,j] / freq[j]; if freq[j]==0 -> entire column j is 0%
    print("Converting to conditional percentage matrix with 0% for zero-frequency columns...")
    denom = freq.astype(np.float64)
    P = np.zeros_like(cooc, dtype=np.float32)
    nz = denom > 0
    P[:, nz] = (cooc[:, nz].astype(np.float64) / denom[nz]) * 100.0
    P = P.astype(np.float32)

    # Sanity checks
    print("Sanity checks:")
    print("  Nonzero cooc entries:", int(np.count_nonzero(cooc)))
    print("  Sum(cooc):", int(cooc.sum()))
    print("  Mean diag(cooc):", float(np.mean(np.diag(cooc))))
    nz_cols = np.where(nz)[0]
    if nz_cols.size:
        j0 = int(nz_cols[0])
        print(f"  Example column j={j0}: freq={int(freq[j0])}, max P(:,j)={float(P[:, j0].max()):.2f}%")
    print("  P shape:", P.shape)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "cooc_counts.npy"), cooc)
    np.save(os.path.join(args.outdir, "freq.npy"), freq)
    np.save(os.path.join(args.outdir, "P_cond_percent.npy"), P)

    if args.save_head:
        pd.DataFrame(P[:12, :12]).to_csv(os.path.join(args.outdir, "P_head12x12.csv"), index=False)

    print("Saved:")
    print(f"  {args.outdir}/cooc_counts.npy        (int32, {cooc.shape[0]}x{cooc.shape[1]})")
    print(f"  {args.outdir}/freq.npy               (int32, {freq.shape[0]})")
    print(f"  {args.outdir}/P_cond_percent.npy     (float32 %, {P.shape[0]}x{P.shape[1]})")
    if args.save_head:
        print(f"  {args.outdir}/P_head12x12.csv        (preview block)")
    print("Done.")


if __name__ == "__main__":
    main()