import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import xgboost as xgb  # --- CHANGED: Import xgboost ---
import os

# --- Define constants ---
NUM_LABELS = 388
NUM_SUBMISSION_ROWS = 18299
# --- CHANGED: Updated filenames ---
SUBMISSION_FILENAME = 'submission_xgb_custom_obj.csv'
MODEL_DIR = 'xgboost_model' # Directory where XGBoost models are saved

# === 1. LOAD YOUR UNSEEN (TEST) DATA ===
# --- (This entire section is identical to your lightgbm script) ---
# --- (It correctly loads the 399 features needed for prediction) ---
try:
    from dataset.hackathon import HackathonDataset
    from dataset.collate import collate_fn 
except ImportError:
    print("Error: Could not import HackathonDataset or collate_fn.")
    print("Please make sure they are in your Python path.")
    exit()

def predict_collate_fn(batch):
    """
    Custom collate function for the test loader.
    It extracts 'id' and stacks 'X' and 'room_cluster_one_hot' to
    create the 399-feature vector your models were trained on.
    """
    ids = []
    features = []
    
    for item in batch:
        ids.append(item['id'])
        
        try:
            x_features = item['X']
            room_features = item['room_cluster_one_hot']
            combined_features = torch.cat((x_features, room_features), dim=0)
            features.append(combined_features)
        except KeyError as e:
            print(f"Error: Item {item['id']} is missing expected key {e}.")
            print("Please check your HackathonDataset 'test' split structure.")
            return None 
            
    try:
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        features_tensor = torch.stack(features, dim=0)
    except Exception as e:
        print(f"Error stacking tensors in predict_collate_fn: {e}")
        return None 

    return {
        'id': ids_tensor,
        'X': features_tensor
    }

print("Loading unseen (test) data...")
test_data = HackathonDataset(split="test", download=True, seed=42)
print(f"Total samples found by HackathonDataset: {len(test_data)}")

full_test_loader = DataLoader(
    test_data,
    batch_size=len(test_data), 
    collate_fn=predict_collate_fn 
)

print("Unpacking data...")
test_batch_data = next(iter(full_test_loader)) 

if test_batch_data is None:
    print("Error during data loading. Exiting.")
    exit()

try:
    X_unseen_tensor = test_batch_data['X']
    test_ids_tensor = test_batch_data['id'] 
except KeyError as e:
    print(f"Error: Failed to find key {e} in test data batch.")
    print("The predict_collate_fn did not work as expected.")
    print("Available keys:", test_batch_data.keys())
    exit()

X_unseen = X_unseen_tensor.numpy()
test_ids = test_ids_tensor.numpy() 

num_samples = X_unseen.shape[0]

print(f"Loaded {num_samples} unseen test samples.")


# === 2. LOAD ALL TRAINED MODELS ===
# --- (This section is CHANGED to load XGBoost models) ---
print(f"Loading all {NUM_LABELS} trained XGBoost models...")
all_models = {}
for i in range(NUM_LABELS):
    # --- CHANGED: Model filename format ---
    model_filename = os.path.join(MODEL_DIR, f"xgboost_model_label_{i}.json")
    try:
        # --- CHANGED: Loading logic for XGBoost ---
        bst = xgb.Booster() # Create a new booster
        bst.load_model(model_filename) # Load the saved model
        all_models[i] = bst
    except Exception as e: # Use a general exception for xgb
        print(f"Error: Could not load model '{model_filename}'.")
        print(f"Details: {e}")
        print("Please make sure you have run the training script first.")
        exit()

print("All models loaded successfully.")


# === 3. MAKE PREDICTIONS (ONE-VS-REST) ===
# --- (This section is CHANGED to handle XGBoost DMatrix and logit output) ---
print("Generating predictions from all models...")

# --- NEW: Convert numpy test data to DMatrix for XGBoost ---
print("Converting test data to DMatrix...")
dX_unseen = xgb.DMatrix(X_unseen)

# Create an empty matrix to hold the *probability* predictions
y_pred_matrix = np.zeros((num_samples, NUM_LABELS))

for i in range(NUM_LABELS):
    # Get the specific model for this label
    model_i = all_models[i]
    
    # --- CHANGED: Predict using DMatrix ---
    # Because you used a custom objective, model_i.predict()
    # will output raw *logits*, not probabilities.
    y_pred_logits_i = model_i.predict(dX_unseen)
    
    # --- NEW: Convert logits to probabilities using sigmoid ---
    # This is necessary to match the logic in your custom eval metric
    y_pred_prob_i = 1.0 / (1.0 + np.exp(-y_pred_logits_i))
    
    # Store these *probabilities* in the i-th column
    y_pred_matrix[:, i] = y_pred_prob_i

print("Prediction probability matrix generated.")


# === 4. SAVE SUBMISSION FILE ===
# --- (This section is identical, except for the threshold) ---
print(f"Saving predictions to {SUBMISSION_FILENAME}...")

# --- CHANGED: Use the 0.25 threshold from your XGBoost training ---
y_pred_binary_matrix = (y_pred_matrix > 0.25).astype(int)

print(f"Mapping {num_samples} predictions to {NUM_SUBMISSION_ROWS} submission rows...")

# Create the full submission data containers
final_ids = np.arange(NUM_SUBMISSION_ROWS)
final_data = np.zeros((NUM_SUBMISSION_ROWS, NUM_LABELS), dtype=int)

# Use the test_ids to place predictions in the correct row
mapped_count = 0
for i in range(num_samples): 
    submission_id = test_ids[i]
    
    if submission_id >= 0 and submission_id < NUM_SUBMISSION_ROWS:
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