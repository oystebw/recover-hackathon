#!/usr/bin/env python3
import argparse
import csv
import numpy as np
from dataset.hackathon import HackathonDataset  # repo layout only

NUM_ROOMS = 12
NUM_JOBS = 388
N_PAIRS = NUM_ROOMS * NUM_JOBS


# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------
def pair_index(room_idx: int, job_idx: np.ndarray | int) -> np.ndarray:
    return room_idx * NUM_JOBS + job_idx


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


# ---------------------------------------------------------------------
# dataset helpers
# ---------------------------------------------------------------------
def build_project_visible(item):
    """Return (visible_4656, x_tgt_388, tgt_room_idx)."""
    visible = np.zeros(N_PAIRS, dtype=np.int8)

    room_oh = to_numpy(item["room_cluster_one_hot"]).astype(np.int8)
    tgt = int(np.argmax(room_oh))

    x_tgt = to_numpy(item["X"]).astype(np.int8)
    if x_tgt.size != NUM_JOBS:
        raise ValueError(f"Unexpected X size {x_tgt.size}, expected {NUM_JOBS}")
    if np.any(x_tgt == 1):
        js = np.where(x_tgt == 1)[0]
        visible[pair_index(tgt, js)] = 1

    calculus = item.get("calculus", []) or []
    for entry in calculus:
        rc_oh = entry.get("room_cluster_one_hot", None)
        r_idx = int(np.argmax(to_numpy(rc_oh))) if rc_oh is not None else 11
        xenc = to_numpy(entry.get("work_operations_index_encoded", []))
        if xenc.size == NUM_JOBS:
            js = np.where(xenc == 1)[0]
            if js.size:
                visible[pair_index(r_idx, js)] = 1

    return visible, x_tgt, tgt


# ---------------------------------------------------------------------
# iterative correlation inference
# ---------------------------------------------------------------------
def infer_binary_iterative(
    P: np.ndarray,
    visible_vec: np.ndarray,
    x_tgt: np.ndarray,
    tgt_room: int,
    threshold: float,
    min_votes: int = 1,
    iters: int = 10,
    same_room_only: bool = True,
) -> np.ndarray:
    """
    Iterative correlation inference.
    - Start with project-visible set (visible_vec) and target room's visible X (x_tgt).
    - On each pass, flip zeros to 1 if at least `min_votes` visible pairs have P(i|j) >= threshold.
    - Newly activated pairs are added to the visible set for subsequent passes.
    - Stops early if a pass makes no new activations.

    Returns: (388,) binary vector of new activations (only for positions where x_tgt==0).
    """
    NUM_JOBS = x_tgt.shape[0]
    pred = np.zeros(NUM_JOBS, dtype=np.int8)

    # current visible indices (global 4656 space)
    idx_visible = np.where(visible_vec == 1)[0]

    # restrict visible evidence to same room if requested
    if same_room_only:
        start, end = tgt_room * NUM_JOBS, (tgt_room + 1) * NUM_JOBS
        idx_visible = idx_visible[(idx_visible >= start) & (idx_visible < end)]

    if idx_visible.size == 0:
        return pred

    # candidates are zeros in target room
    candidates = np.where(x_tgt == 0)[0]

    for _ in range(max(1, iters)):
        newly_on = []
        for j in candidates:
            row = tgt_room * NUM_JOBS + j
            votes = np.sum(P[row, idx_visible] >= threshold)
            if votes >= min_votes:
                pred[j] = 1
                newly_on.append(j)

        if not newly_on:
            break  # converged

        # add newly activated pairs to visible set for next pass
        new_global = tgt_room * NUM_JOBS + np.array(newly_on, dtype=np.int64)
        idx_visible = np.unique(np.concatenate([idx_visible, new_global]))

        # remove activated ones
        candidates = np.setdiff1d(candidates, np.array(newly_on, dtype=np.int64), assume_unique=False)
        if candidates.size == 0:
            break

    return pred


# ---------------------------------------------------------------------
# main: build submission
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--matrix", default="corr_out_simple/P_cond_percent.npy")
    ap.add_argument("--threshold", type=float, default=85.0, help="Percent threshold for activation.")
    ap.add_argument("--min-votes", type=int, default=1, help="Require at least this many links >= threshold.")
    ap.add_argument("--iters", type=int, default=3, help="Number of inference passes.")
    ap.add_argument("--same-room-only", action="store_true", help="Restrict correlations to same room.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-submit-rows", type=int, default=18299)
    ap.add_argument("--out", default="submission_corr_iter.csv")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Load P(i|j) %, zero diagonal
    P = np.load(args.matrix).astype(np.float32)
    if P.shape != (N_PAIRS, N_PAIRS):
        raise ValueError(f"P matrix shape {P.shape}, expected {(N_PAIRS, N_PAIRS)}")
    np.fill_diagonal(P, 0.0)

    test_ds = HackathonDataset(split="test", download=False, seed=args.seed, root=args.root)
    print(f"[info] test samples: {len(test_ds)}")

    N_submit = int(args.n_submit_rows)
    pred_dict = {k: [] for k in range(N_submit)}

    for i in range(len(test_ds)):
        item = test_ds[i]
        submit_id = int(item["id"])
        visible_vec, x_tgt, tgt_room = build_project_visible(item)
        pred = infer_binary_iterative(
            P=P,
            visible_vec=visible_vec,
            x_tgt=x_tgt,
            tgt_room=tgt_room,
            threshold=float(args.threshold),
            min_votes=int(args.min_votes),
            iters=int(args.iters),
            same_room_only=bool(args.same_room_only),
        )
        pred_dict[submit_id] = np.where(pred == 1)[0].tolist()

    # Prefer dataset helper to guarantee format
    if hasattr(test_ds, "create_submission"):
        print("[submit] Using dataset.create_submission(...)")
        try:
            out_path = test_ds.create_submission(pred_dict)
            print(f"[submit] Submission created at: {out_path if out_path else '(dataset-defined path)'}")
            return
        except Exception as e:
            print("[submit] dataset.create_submission failed; falling back to CSV. Error:", e)

    # Fallback wide CSV
    print("[submit] Fallback CSV writer (wide 388-col)")
    cols = ["id"] + [f"{j}" for j in range(NUM_JOBS)]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for submit_id in range(N_submit):
            row_bin = np.zeros(NUM_JOBS, dtype=np.int8)
            if pred_dict[submit_id]:
                row_bin[pred_dict[submit_id]] = 1
            w.writerow([submit_id] + row_bin.tolist())
    print(f"[submit] Wrote {args.out}")


if __name__ == "__main__":
    main()