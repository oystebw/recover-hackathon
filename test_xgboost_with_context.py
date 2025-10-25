import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
import os
from typing import Tuple
from collections import defaultdict # --- NEW: For grouping ---

# --- Define constants ---
NUM_LABELS = 388
NUM_SUBMISSION_ROWS = 18299
# --- CHANGED: Updated filenames to match new model ---
SUBMISSION_FILENAME = 'submission_xgb_v12_context.csv'
MODEL_DIR = 'xgboost_model_v2_context' # Must match new training dir

# --- Import your custom dataset/loaders ---
try:
    from dataset.hackathon import HackathonDataset
    # We no longer need the collate_fn
except ImportError:
    print("Error: Could not import HackathonDataset.")
    print("Please make sure they are in your Python path.")
    exit()

# --- NEW: Helper function to build features for TEST data ---
def create_feature_matrix_for_test(dataset: HackathonDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes the entire TEST dataset to build advanced features.
    Returns:
    - X_matrix (numpy): The feature matrix (N_samples, 787)
    - ids (numpy): The corresponding submission row IDs (N_samples,)
    """
    print(f"Grouping {len(dataset)} test samples by project_id...")
    projects = defaultdict(list)
    original_items_in_order = [] # To store items in original order
    for i in range(len(dataset)):
        item = dataset[i] 
        projects[item['project_id']].append(item)
        original_items_in_order.append(item)
    print("Grouping complete.")

    X_list = []
    id_list = []
    
    print("Building feature matrix with 'other room' context...")
    for item in original_items_in_order: # Iterate in original order
        project_id = item['project_id']
        current_id = item['id'] # This is the submission row ID
        
        # 1. Base features for this room
        current_X = item['X'].float()
        current_room_features = item['room_cluster_one_hot'].float()
        
        # 2. New Context Feature: 'Other Room' Faults
        all_project_items = projects[project_id]
        other_X_tensors = [
            i['X'].float() for i in all_project_items if i['id'] != current_id
        ]
        
        if not other_X_tensors:
            other_room_faults = torch.zeros(NUM_LABELS, dtype=torch.float)
        else:
            other_X_matrix = torch.stack(other_X_tensors, dim=0)
            other_room_faults = (other_X_matrix.sum(dim=0) > 0).float()
        
        # 3. Combine features
        original_features = torch.cat((current_X, current_room_features), dim=0) # 399
        combined_features = torch.cat((original_features, other_room_faults), dim=0) # 787
        
        X_list.append(combined_features)
        id_list.append(current_id) # Store the submission row ID

    print("Matrix build complete. Stacking tensors...")
    
    X_matrix_tensor = torch.stack(X_list, dim=0)
    
    # Convert IDs to numpy array
    ids_numpy = np.array(id_list, dtype=np.int64)
    
    return X_matrix_tensor.numpy(), ids_numpy
# --- END NEW FUNCTION ---


# === 1. LOAD YOUR UNSEEN (TEST) DATA ===
# --- (This section is REPLACED) ---
print("Loading unseen (test) data...")
test_data = HackathonDataset(split="test", download=True, seed=42)
print(f"Total samples found by HackathonDataset: {len(test_data)}")

print("Processing test data (this may take a moment)...")
X_unseen, test_ids = create_feature_matrix_for_test(test_data)
# --- END NEW DATA PREP ---

num_samples = X_unseen.shape[0]
print(f"Loaded {num_samples} unseen test samples.")
print(f"Feature matrix shape: {X_unseen.shape}") # Should be (N, 787)


# === 2. LOAD ALL TRAINED MODELS ===
# --- (This section is unchanged, but points to new MODEL_DIR) ---
print(f"Loading all {NUM_LABELS} trained XGBoost models from {MODEL_DIR}...")
all_models = {}
for i in range(NUM_LABELS):
    model_filename = os.path.join(MODEL_DIR, f"xgboost_model_label_{i}.json")
    try:
        bst = xgb.Booster()
        bst.load_model(model_filename)
        all_models[i] = bst
    except Exception as e:
        print(f"Error: Could not load model '{model_filename}'.")
        print(f"Details: {e}")
        exit()
print("All models loaded successfully.")


# === 3. MAKE PREDICTIONS (ONE-VS-REST) ===
# --- (This section is unchanged, it will use X_unseen with 787 features) ---
print("Generating predictions from all models...")

print("Converting test data to DMatrix...")
dX_unseen = xgb.DMatrix(X_unseen)

y_pred_matrix = np.zeros((num_samples, NUM_LABELS))

for i in range(NUM_LABELS):
    model_i = all_models[i]
    
    # Predict logits
    y_pred_logits_i = model_i.predict(dX_unseen)
    
    # Convert logits to probabilities
    y_pred_prob_i = 1.0 / (1.0 + np.exp(-y_pred_logits_i))
    
    y_pred_matrix[:, i] = y_pred_prob_i

print("Prediction probability matrix generated.")


# === 4. SAVE SUBMISSION FILE ===
# --- (This section is unchanged) ---
print(f"Saving predictions to {SUBMISSION_FILENAME}...")

# Use the 0.25 threshold from your XGBoost training
y_pred_binary_matrix = (y_pred_matrix > 0.145).astype(int)

print(f"Mapping {num_samples} predictions to {NUM_SUBMISSION_ROWS} submission rows...")
final_ids = np.arange(NUM_SUBMISSION_ROWS)
final_data = np.zeros((NUM_SUBMISSION_ROWS, NUM_LABELS), dtype=int)

mapped_count = 0
for i in range(num_samples): 
    submission_id = test_ids[i] # Get the correct row ID
    
    if 0 <= submission_id < NUM_SUBMISSION_ROWS:
        final_data[submission_id] = y_pred_binary_matrix[i]
        mapped_count += 1
    else:
        print(f"Warning: Found out-of-bounds submission ID {submission_id}. Ignoring.")

print(f"Successfully mapped {mapped_count} predictions.")
final_ids_col = final_ids.reshape(-1, 1)
submission_data = np.hstack((final_ids_col, final_data))
header = ['id'] + [str(i) for i in range(NUM_LABELS)]
df_submission = pd.DataFrame(submission_data, columns=header)
df_submission = df_submission.astype(int)
df_submission.to_csv(SUBMISSION_FILENAME, index=False)

print("\n--- SUBMISSION FILE SAVED ---")
print(f"Successfully saved {NUM_SUBMISSION_ROWS} predictions to {SUBMISSION_FILENAME}")