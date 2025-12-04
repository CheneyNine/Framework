# Model Adaptation Guide: STGCN on Sacra_Jan Dataset

This document outlines the modifications made to adapt the STGCN model and the data pipeline to work with the `Sacra_Jan` dataset (derived from Sacramento data), which has a different dimensional structure compared to the original Taxi datasets.

## 1. Data Generation (`data/generate_data_for_training.py`)

The data generation script was updated to produce data compatible with the framework's expected format, but with an additional feature dimension.

*   **Output Format**: Generates `.npz` files (`_train.npz`, `_val.npz`, `_test.npz`) containing:
    *   `data`: Shape `(time_steps, nodes, feature_dim)`. Note the extra `feature_dim` compared to Taxi datasets which are typically `(time_steps, nodes)`.
    *   `index`: Node indices.
    *   `start_time`: Timestamps.
*   **Time Selection**: Added `--start_date` and `--end_date` arguments to filter specific time ranges (e.g., January 2023).
*   **Adjacency Matrix**: Automatically copies the adjacency matrix to the new dataset folder.

## 2. Data Loading (`data/factory.py` & `data/datamodule.py`)

### `data/factory.py`
*   Added a new configuration block for `Sacra_Jan` in `get_dataset_paths`.
*   Maps the dataset name to the specific folder `dataset/Sacra_Jan2023` and file prefixes.

### `data/datamodule.py`
*   **Dimension Handling**: Modified `setup()` to handle 3D input data `(time, nodes, features)`.
    *   The `StandardScaler` expects 2D input `(samples, features)`.
    *   The code now reshapes 3D data to 2D `(time * nodes, features)` before fitting/transforming, and then reshapes it back to 3D.

## 3. Model Architecture (`models/STGCN.py`)

The STGCN model expects a specific 4D input tensor `(batch, nodes, 1, time)`. The `Sacra_Jan` data pipeline produces a 5D tensor `(batch, features, 1, nodes, time)` because of the extra feature dimension.

*   **Forward Pass**: Added a check at the beginning of `forward(self, x, adj)`:
    ```python
    if x.dim() == 5:
        x = x[:, 0, :, :, :] # Select the first feature -> (batch, 1, nodes, time)
        x = x.permute(0, 2, 1, 3) # Permute to -> (batch, nodes, 1, time)
    ```
    This effectively selects the first feature (usually the target traffic flow/speed) for the model to process.

## 4. Training & Evaluation (`models/base_model.py`)

### `_get_reals_and_preds`
*   Similar to the model input, the ground truth `y` also comes in as 5D.
*   Added logic to slice and permute `y` to match the model output shape `(batch, nodes, 1, time)` for loss calculation.

### `_unscale_data`
*   **Scaler Compatibility**: The `StandardScaler` for `Sacra` data is fitted on `features` (shape `(F,)`), whereas for Taxi data it might be fitted on `nodes` or be a single value.
*   Added logic to detect this mismatch:
    ```python
    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != original_shape[1]:
         # Sacra style: scaler is (features,), data is (batch*time, nodes)
         # We assume we are predicting the first feature
         mean = scaler.mean_[0]
         std = scaler.scale_[0]
         unscaled_data_reshaped = data_reshaped * std + mean
    ```
    This ensures that the inverse transformation uses the correct mean and standard deviation corresponding to the target feature (index 0).

## 5. Configuration (`configs/global_sacra.yaml`)

*   Created a new global configuration file `configs/global_sacra.yaml`.
*   Sets `data.datasets` to `['Sacra_Jan']`.
*   Adjusts GPU settings and other hyperparameters as needed.

## How to Run

1.  **Generate Data** (if not already done):
    ```bash
    python data/generate_data_for_training.py --start_date 2023-01-01 --end_date 2023-01-31 --output_dir Sacra_Jan2023 --dataset_name Sacra_Jan
    ```

2.  **Train Model**:
    ```bash
    python main.py --config configs/global_sacra.yaml --model_config configs/STGCN.yaml
    ```
