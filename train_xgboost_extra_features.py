import numpy as np
import pandas as pd
import torch  # --- RE-ENABLED: Needed for feature engineering ---
from torch.utils.data import DataLoader
import xgboost as xgb
import os
from typing import Tuple
from collections import defaultdict # --- NEW: For grouping ---

# --- Import your custom dataset/loaders ---
try:
    from dataset.hackathon import HackathonDataset
    # We no longer need the training collate_fn
    # from dataset.collate import collate_fn 
except ImportError:
    print("Error: Could not import HackathonDataset.")
    print("Please make sure the 'dataset' directory is in your Python path.")
    exit()

# === 1. DEFINE CUSTOM METRIC FUNCTIONS ===
# --- (This section is unchanged) ---
try:
    from metrics.score import normalized_rooms_score
    print("Successfully imported 'normalized_rooms_score' from metrics.py.")
except ImportError:
    print("Warning: Could not import 'normalized_rooms_score' from metrics.py.")
    def normalized_rooms_score(preds, targets):
        print("Using placeholder score!")
        return 0.5

def custom_room_objective(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    y_true = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))
    W_POS = 1.5
    W_NEG = 1.25
    grad = -y_true * W_POS * (1 - p) + (1 - y_true) * W_NEG * p
    hess_weights = (y_true * W_POS + (1 - y_true) * W_NEG)
    hess = hess_weights * p * (1 - p)
    return grad, hess

def eval_per_label_room_score(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y_true = dtrain.get_label().astype(int)
    p = 1.0 / (1.0 + np.exp(-predt))
    THRESHOLD = 0.15
    y_pred = (p > THRESHOLD).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    score = (tp * 1.0) - (fp * 0.25) - (fn * 0.5) + (tn * 1.0)
    return 'RoomScore', score

# --- END CUSTOM FUNCTIONS ---

# --- NEW: Define NUM_LABELS at the top ---
NUM_LABELS = 388

# --- NEW: Helper function to build features ---
def create_feature_matrix_with_context(dataset: HackathonDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes the entire dataset to build advanced features.
    For each sample (room), it adds:
    - item['X'] (388 features for the current room)
    - item['room_cluster_one_hot'] (11 features for room type)
    - other_room_faults (388 features, an 'OR' of faults in *other* rooms
                         in the same project_id)
    """
    print(f"Grouping {len(dataset)} samples by project_id...")
    projects = defaultdict(list)
    original_items_in_order = [] # To store items in original order
    for i in range(len(dataset)):
        item = dataset[i] 
        projects[item['project_id']].append(item)
        original_items_in_order.append(item)
    print("Grouping complete.")

    X_list = []
    Y_list = []
    
    print("Building feature matrix with 'other room' context...")
    # Iterate through the original dataset items
    for item in original_items_in_order:
        project_id = item['project_id']
        current_id = item['id']
        
        # 1. Base features for this room
        current_X = item['X'].float() # (388,)
        current_room_features = item['room_cluster_one_hot'].float() # (11,)
        
        # 2. Target vector
        current_Y = item['Y'] # (388,)
        
        # 3. New Context Feature: 'Other Room' Faults
        all_project_items = projects[project_id]
        
        # Find all 'X' tensors from OTHER rooms in the same project
        other_X_tensors = [
            i['X'].float() for i in all_project_items if i['id'] != current_id
        ]
        
        if not other_X_tensors:
            # This is the only room in the project
            other_room_faults = torch.zeros(NUM_LABELS, dtype=torch.float)
        else:
            # Stack the other tensors and 'OR' them
            other_X_matrix = torch.stack(other_X_tensors, dim=0)
            # Sum vertically, check if > 0, cast to float
            other_room_faults = (other_X_matrix.sum(dim=0) > 0).float()
        
        # Combine all features:
        # [X (388), Room (11)]
        original_features = torch.cat((current_X, current_room_features), dim=0) # (399,)
        
        # --- FINAL FEATURE VECTOR ---
        # [Original (399), Other_X (388)]
        combined_features = torch.cat((original_features, other_room_faults), dim=0) # (787,)
        
        X_list.append(combined_features)
        Y_list.append(current_Y)

    print("Matrix build complete. Stacking tensors...")
    
    # Stack all samples into a single batch
    X_matrix_tensor = torch.stack(X_list, dim=0)
    Y_matrix_tensor = torch.stack(Y_list, dim=0)
    
    return X_matrix_tensor.numpy(), Y_matrix_tensor.numpy()
# --- END NEW FUNCTION ---


# === 2. LOAD AND PREPARE DATA ===
# --- (This section is REPLACED) ---
print("Loading dataset...")
train_data = HackathonDataset(split="train", download=True, seed=42)
val_data = HackathonDataset(split="val", download=True, seed=42) 
print("Datasets loaded.")

# --- NEW: Process data using our custom function ---
print("Processing training data (this may take a moment)...")
X_train, y_train = create_feature_matrix_with_context(train_data)

print("Processing validation data...")
X_val, y_val = create_feature_matrix_with_context(val_data)
# --- END NEW DATA PREP ---

print(f"Data aggregated. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Data aggregated. X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")


# === 3. DEFINE XGBOOST PARAMETERS ===
# --- (This section is unchanged) ---
params = {
    'disable_default_eval_metric': 1,
    'booster': 'gbtree',
    'max_leaves': 128,
    'learning_rate': 0.05,
    'colsample_bytree': 0.9,
    'verbosity': 0
}
MODEL_DIR = 'xgboost_model_v3_context' # <-- Renamed model dir
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Models will be saved to '{MODEL_DIR}' directory.")


# === 4. TRAIN MULTI-LABEL MODEL (ONE-VS-REST) ===
# --- (This section is unchanged, it will just use X_train with 787 features) ---
print("\nTraining XGBoost model (One-vs-Rest strategy)...")

num_labels = y_train.shape[1]
all_models = {}
y_val_pred_matrix = np.zeros(y_val.shape)

# Create DMatrix for validation prediction (once)
dval_pred = xgb.DMatrix(X_val)

for i in range(num_labels):
    print(f"--- Training model for label {i+1}/{num_labels} ---")
    
    y_train_i = y_train[:, i]
    y_val_i = y_val[:, i]
    
    dtrain = xgb.DMatrix(X_train, label=y_train_i)
    dval = xgb.DMatrix(X_val, label=y_val_i)
    
    bst_i = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, 'validation')],
        obj=custom_room_objective,
        custom_metric=eval_per_label_room_score,
        maximize=True,
        early_stopping_rounds=100,
        verbose_eval=50
    )
    
    print(f"Best iteration: {bst_i.best_iteration}, Best RoomScore: {bst_i.best_score:.4f}")
    
    all_models[i] = bst_i
    
    y_val_pred_i = bst_i.predict(
        dval_pred, 
        iteration_range=(0, bst_i.best_iteration)
    )
    y_val_pred_matrix[:, i] = y_val_pred_i
    
    model_filename = f"xgboost_model_label_{i}.json"
    bst_i.save_model(os.path.join(MODEL_DIR, model_filename))

print(f"\nAll {num_labels} models trained and saved individually.")


# === 5. EVALUATE FULL MULTI-LABEL MODEL ===
# --- (This section is unchanged) ---
print("Evaluating full model on validation data...")

y_val_pred_binary_matrix = (y_val_pred_matrix > 0.15).astype(int)
final_score = normalized_rooms_score(y_val_pred_binary_matrix, y_val)
print(f"\n--- FINAL VALIDATION SCORE: {final_score:.6f} ---")