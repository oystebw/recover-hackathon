from collections import Counter
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dataset.work_operations import WorkOperationsDataset

dataset = WorkOperationsDataset(root="data", split="train", download=False)
room_counts = Counter()

for idx in range(len(dataset)):
    sample = dataset[idx]
    room_counts[sample["room_cluster"]] += 1

print("Total samples:", len(dataset))
for room, count in room_counts.most_common():
    print(f"{room}: {count}")