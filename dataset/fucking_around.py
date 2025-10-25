from collections import Counter, defaultdict
from pprint import pprint

from work_operations import WorkOperationsDataset


def sample_room_operation_frequencies(
    dataset: WorkOperationsDataset,
    samples_per_room: int = 1000,
    max_passes: int = 5,
    top_k: int = 10,
):
    """Sample rooms and tally how often operations occur per room cluster."""

    room_operation_counts: dict[str, Counter[int]] = defaultdict(Counter)
    room_sample_counts: Counter[str] = Counter()
    target_rooms = list(dataset.room_to_index.keys())

    # Collect frequencies until we either hit the requested quota or exhaust the dataset.
    for _ in range(max_passes):
        length = len(dataset)
        for idx in range(length):
            try:
                item = dataset[idx]
            except Exception:
                # Stop early if the underlying dataframe reshuffled mid-iteration.
                break
            room = item["room_cluster"]
            if room_sample_counts[room] >= samples_per_room:
                continue

            visible_codes = set(int(code) for code in item["X_codes"].tolist())
            hidden_codes = set(int(code) for code in item["Y_codes"].tolist())
            all_codes = visible_codes | hidden_codes

            for code in all_codes:
                room_operation_counts[room][code] += 1

            room_sample_counts[room] += 1

        if all(room_sample_counts.get(room, 0) >= samples_per_room for room in target_rooms):
            break

        dataset.shuffle()

    code_to_name = dataset.code_to_wo

    summaries: list[dict[str, object]] = []
    for room, counts in sorted(room_operation_counts.items()):
        common_ops = counts.most_common(top_k)
        summaries.append(
            {
                "room": room,
                "samples_collected": room_sample_counts[room],
                "top_operations": [
                    {
                        "code": code,
                        "name": code_to_name.get(code, f"Unknown {code}"),
                        "count": occ,
                    }
                    for code, occ in common_ops
                ],
            }
        )

    return summaries


if __name__ == "__main__":
    dataset = WorkOperationsDataset(root="data", split="train", download=False)
    print(len(dataset))
    dataset.shuffle()

    summaries = sample_room_operation_frequencies(dataset)

    pprint(summaries)
