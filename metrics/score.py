def get_room_scores(room_preds: list[int], room_targets: list[int]) -> dict[str, float]:
    EMPTY_ROOM_REWARD = 1
    TRUE_POSITIVE_REWARD = 1
    FALSE_POSITIVE_PENALTY = 0.25
    FALSE_NEGATIVE_PENALTY = 0.5

    best_possible_score = 0
    dummy_score = 0
    score = 0

    room_targets_set = set(room_targets)
    room_preds_set = set(room_preds)

    if len(room_targets_set) == 0:
        best_possible_score += EMPTY_ROOM_REWARD
        dummy_score += EMPTY_ROOM_REWARD

        if len(room_preds_set) == 0:
            score += EMPTY_ROOM_REWARD
        else:
            score -= FALSE_POSITIVE_PENALTY * len(room_preds_set)

    else:
        best_possible_score += len(room_targets_set) * TRUE_POSITIVE_REWARD
        dummy_score -= len(room_targets_set) * FALSE_NEGATIVE_PENALTY

        score += TRUE_POSITIVE_REWARD * len(room_targets_set & room_preds_set)
        score -= FALSE_POSITIVE_PENALTY * len(room_preds_set - room_targets_set)
        score -= FALSE_NEGATIVE_PENALTY * len(room_targets_set - room_preds_set)

    # Add a small coefficient to prevent division by zero
    normalized_score = (score - dummy_score) / (best_possible_score - dummy_score + 1e-20)

    return {
        "score": score,
        "dummy_score": dummy_score,
        "best_possible_score": best_possible_score,
        "normalized_score": normalized_score,
    }


def normalized_rooms_score(preds: list[list[int]], targets: list[list[int]]) -> float:
    best_possible_score = 0
    dummy_score = 0
    score = 0

    for room_targets, room_preds in zip(targets, preds, strict=True):
        room_scores = get_room_scores(room_preds, room_targets)
        score += room_scores["score"]
        dummy_score += room_scores["dummy_score"]
        best_possible_score += room_scores["best_possible_score"]

    # Add a small coefficient to prevent division by zero
    return (score - dummy_score) / (best_possible_score - dummy_score + 1e-20)
