import torch


def collate_fn(batch):
    batch_size = len(batch)
    feature_dim = batch[0]["X"].shape[0] + batch[0]["room_cluster_one_hot"].shape[0]
    max_set_size = max(len(item["calculus"]) if item["calculus"] else 1 for item in batch)
    target_features = torch.empty((batch_size, feature_dim), dtype=batch[0]["X"].dtype)
    context_set = torch.zeros((batch_size, max_set_size, feature_dim), dtype=batch[0]["X"].dtype)
    context_mask = torch.zeros((batch_size, max_set_size), dtype=torch.bool)
    labels = torch.empty((batch_size, batch[0]["Y"].shape[0]), dtype=batch[0]["Y"].dtype)

    for i, item in enumerate(batch):
        target_features[i] = torch.cat([item["X"], item["room_cluster_one_hot"]])
        context = []
        for entry in item["calculus"]:
            work_operations_index_encoded = entry["work_operations_index_encoded"].detach().clone()
            room_cluster_one_hot = entry["room_cluster_one_hot"].detach().clone()
            context.append(torch.cat([work_operations_index_encoded, room_cluster_one_hot]))
        if context:
            context_tensor = torch.stack(context)
            context_set[i, : len(context), :] = context_tensor
            context_mask[i, : len(context)] = True
        else:
            context_set[i, 0, :] = torch.zeros(feature_dim, dtype=batch[0]["X"].dtype)
            context_mask[i, 0] = False
        labels[i] = item["Y"]

    return {
        "X": target_features,
        "Y": labels,
        "context": context_set,
        "context_mask": context_mask,
    }
