import torch
import pandas as pd
from pathlib import Path

payload = torch.load(Path("precomputed/catboost_features/catboost_train.pt"), map_location="cpu")
labels = payload["labels"]  # shape [156052, 388]

positive_counts = labels.sum(dim=0).to(torch.int64)
df = pd.DataFrame(
    {"job_index": range(labels.shape[1]), "positive_labels": positive_counts.tolist()}
).sort_values("positive_labels", ascending=False)

print(df.head(100))       # most frequent jobs
print(df.tail(10))       # rarest jobs
print("zero-sum jobs:", df.query("positive_labels == 0")["job_index"].tolist())

job_id = 234
job_labels = labels[:, job_id]

counts = torch.bincount(job_labels.to(torch.int64))
num_zeros = counts[0].item()
num_ones = counts[1].item() if len(counts) > 1 else 0

print(f"Job {job_id}: {num_ones} positives, {num_zeros} negatives")