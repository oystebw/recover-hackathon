#!/usr/bin/env python3
import argparse
import numpy as np

try:
    from dataset.hackathon import HackathonDataset
except Exception:
    from hackathon import HackathonDataset  # type: ignore

NUM_ROOMS = 12
NUM_JOBS = 388
N_PAIRS = NUM_ROOMS * NUM_JOBS


def pair_index(room_idx, job_idx):
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


# ---------------- hackathon score ----------------
def room_score(tp, fp, fn, is_empty_room, predicted_empty):
    s = tp - 0.25 * fp - 0.5 * fn
    if is_empty_room and predicted_empty:
        s += 1.0
    return s


def normalized_rooms_score(y_pred_bin, y_true_bin, verbose=False):
    N = y_true_bin.shape[0]
    total, empty_base, perfect = 0.0, 0.0, 0.0

    # aggregated counts
    tp_sum, fp_sum, fn_sum, tn_sum = 0, 0, 0, 0

    for i in range(N):
        yt = y_true_bin[i]
        yp = y_pred_bin[i]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        # Entirely empty row handling
        is_empty = int(np.sum(yt) == 0)
        pred_empty = int(np.sum(yp) == 0)
        tn = 1 if (is_empty and pred_empty) else 0

        # accumulate
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        tn_sum += tn

        total += room_score(tp, fp, fn, is_empty, pred_empty)
        empty_base += room_score(0, 0, int(np.sum(yt)), is_empty, True)
        perfect += room_score(int(np.sum(yt)), 0, 0, is_empty, is_empty)

    denom = max(perfect - empty_base, 1e-9)
    norm_score = max(0.0, (total - empty_base) / denom * 100.0)

    # Optional printout for debugging
    if verbose:
        precision = tp_sum / max(tp_sum + fp_sum, 1e-9)
        recall = tp_sum / max(tp_sum + fn_sum, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print("\n--- Detailed Metrics ---")
        print(f"TP: {tp_sum:,}")
        print(f"FP: {fp_sum:,}")
        print(f"FN: {fn_sum:,}")
        print(f"TN (empty rows): {tn_sum:,}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 score:  {f1:.3f}")
        print(f"Normalized Score: {norm_score:.2f}%")
        print("------------------------\n")

    return norm_score, (tp_sum, fp_sum, fn_sum, tn_sum)


# ---------------- build project-visible vector + target X ----------------
def build_visible_and_targetX(item):
    """
    visible_vec: 4656-dim binary (project union of visible pairs)
    x_tgt:       388-dim binary (visible jobs in TARGET room only)
    tgt_room:    int room index 0..11
    """
    v = np.zeros(N_PAIRS, dtype=np.int8)

    room_one_hot = to_numpy(item["room_cluster_one_hot"]).astype(np.int8)
    tgt_room = int(np.argmax(room_one_hot))

    # target room visible X
    x_tgt = to_numpy(item["X"]).astype(np.int8)  # (388,)
    if x_tgt.size != NUM_JOBS:
        raise ValueError("Unexpected X shape")
    if np.any(x_tgt == 1):
        v[pair_index(tgt_room, np.where(x_tgt == 1)[0])] = 1

    # calculus: other rooms' visible
    calculus = item.get("calculus", []) or []
    for entry in calculus:
        rc_oh = entry.get("room_cluster_one_hot", None)
        r_idx = int(np.argmax(to_numpy(rc_oh))) if rc_oh is not None else 11
        xenc = to_numpy(entry.get("work_operations_index_encoded", []))
        if xenc.size == NUM_JOBS:
            js = np.where(xenc == 1)[0]
            if js.size:
                v[pair_index(r_idx, js)] = 1

    return v, x_tgt, tgt_room


# ---------------- correlation-based inference ----------------
def infer_binary_iterative(
    P: np.ndarray,
    visible_vec: np.ndarray,
    x_tgt: np.ndarray,
    tgt_room: int,
    threshold: float,
    iters: int = 3,
    same_room_only: bool = True,
) -> np.ndarray:
    """
    Iterative correlation inference.
    - Start with project-visible set (visible_vec) and target room's visible X (x_tgt).
    - On each pass, flip zeros to 1 if at least `min_votes` visible pairs have P(i|j) >= threshold.
    - Newly activated pairs are added to the visible set for subsequent passes.
    - Stops early if a pass makes no new activations.

    Returns:
        pred: (388,) binary vector of new activations (only for positions where x_tgt==0).
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

        # evaluate all remaining candidates using current evidence
        for j in candidates:
            row = tgt_room * NUM_JOBS + j
            if np.any((P[row, idx_visible]) >= threshold):
                pred[j] = 1
                newly_on.append(j)

        if not newly_on:
            break  # no changes → converged

        # add newly activated pairs to visible set for next pass
        new_global = tgt_room * NUM_JOBS + np.array(newly_on, dtype=np.int64)
        if same_room_only:
            # already same-room; just append
            idx_visible = np.unique(np.concatenate([idx_visible, new_global]))
        else:
            # even if cross-room evidence is allowed, new signals are within target room (that’s fine)
            idx_visible = np.unique(np.concatenate([idx_visible, new_global]))

        # remove already-activated candidates
        candidates = np.setdiff1d(candidates, np.array(newly_on, dtype=np.int64), assume_unique=False)

        if candidates.size == 0:
            break

    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--matrix", default="corr_out_simple/P_cond_percent.npy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--thresholds", default="0.25,0.5,1,2,3,4,5,7.5,10", help="Comma-separated thresholds in percentage points."
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    P = np.load(args.matrix).astype(np.float32)
    if P.shape != (N_PAIRS, N_PAIRS):
        raise ValueError(f"Expected {(N_PAIRS, N_PAIRS)}, got {P.shape}")
    # prevent self-triggering
    np.fill_diagonal(P, 0.0)

    val_ds = HackathonDataset(split=args.val_split, root=args.root, seed=args.seed)

    # Precompute visible sets and ground-truth
    Scores_pred_by_thr = {}
    Y_true = []

    # Prepare containers for each threshold
    thrs = [float(t) for t in args.thresholds.split(",")]
    preds_by_thr = {t: [] for t in thrs}

    for i in range(len(val_ds)):
        item = val_ds[i]
        visible_vec, x_tgt, tgt_room = build_visible_and_targetX(item)
        y_true = to_numpy(item["Y"]).astype(np.int8)
        Y_true.append(y_true)

        for t in thrs:
            pred = infer_binary_iterative(P, visible_vec, x_tgt, tgt_room, t)
            preds_by_thr[t].append(pred)

    Y_true = np.vstack(Y_true)

    # Evaluate each threshold and report the best
    best_t, best_score = None, -1.0
    for t in thrs:
        Y_pred = np.vstack(preds_by_thr[t])
        sc, (tp, fp, fn, tn) = normalized_rooms_score(Y_pred, Y_true, verbose=True)
        print(f"threshold={t:>5g}%  -> val normalized score = {sc:.2f}%")
        print(f"→ TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}\n")
        if sc > best_score:
            best_score, best_t = sc, t

    print(f"\nBest threshold = {best_t}%   Best val score = {best_score:.2f}%")


if __name__ == "__main__":
    main()
