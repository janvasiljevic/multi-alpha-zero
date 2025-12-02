import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset


class SalvageSettings:
    # --- REQUIRED: CONFIGURE THESE PATHS ---
    MODEL_TO_SALVAGE = Path("../../testing/149.pt")
    SALVAGED_MODEL_SAVE_PATH = Path("../../testing/149_salvaged.pt")
    # Use a small, high-quality subset of your data
    GOLDEN_DATASET_FILES = [Path("../../testing/samples_148.parquet")]

    # --- SALVAGE HYPERPARAMETERS ---
    SALVAGE_LEARNING_RATE = 1e-5  # Small LR for gentle fine-tuning
    SALVAGE_WEIGHT_DECAY = 5e-3  # HIGH weight decay to shrink weights
    SALVAGE_EPOCHS = 3  # 1-5 epochs is usually enough
    BATCH_SIZE = 1024  # Adjust based on your GPU memory
    NUM_WORKERS = 4

    # --- LOSS WEIGHTS (should match original training) ---
    PI_WEIGHT = 1.0
    VALUES_Z_WEIGHT = 1.0
    VALUES_Q_WEIGHT = 0.5


# Dummy StaticDataset class from your script
class StaticDataset(Dataset):
    def __init__(self, capacity: int, files_to_load: List[Path], state_shape: Tuple, policy_shape: Tuple,
                 value_shape: Tuple):
        self.capacity = capacity;
        self.files_to_load = files_to_load;
        self.state_shape = state_shape;
        self.policy_shape = policy_shape;
        self.value_shape = value_shape
        self.states_np = np.zeros((capacity, *state_shape), dtype=np.bool_);
        self.pis_np = np.zeros((capacity, *policy_shape), dtype=np.float16);
        self.moves_mask = np.zeros((capacity, *policy_shape), dtype=np.bool_);
        self.z_value_np = np.zeros((capacity, *value_shape), dtype=np.float32);
        self.q_value_np = np.zeros((capacity, *value_shape), dtype=np.float32)
        self.pos = 0;
        self.load_files()

    def __len__(self) -> int:
        return self.pos

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (self.states_np[idx], self.pis_np[idx], self.z_value_np[idx], self.q_value_np[idx], self.moves_mask[idx])

    def _add_data_to_buffer(self, states, pis, z_values, moves_mask, q_values):
        num_samples = states.shape[0]
        if self.pos + num_samples > self.capacity: raise ValueError("Capacity exceeded")
        end_idx = self.pos + num_samples;
        self.states_np[self.pos:end_idx] = states;
        self.pis_np[self.pos:end_idx] = pis;
        self.moves_mask[self.pos:end_idx] = moves_mask;
        self.z_value_np[self.pos:end_idx] = z_values;
        self.q_value_np[self.pos:end_idx] = q_values;
        self.pos = end_idx

    def load_files(self):
        for file_path in self.files_to_load:
            table = pq.read_table(file_path);
            num_rows = table.num_rows
            s_flat = np.stack(table['state'].to_numpy(zero_copy_only=False));
            pi_flat = np.stack(table['policy'].to_numpy(zero_copy_only=False));
            moves_mask_flat = np.stack(table['masked_policy'].to_numpy(zero_copy_only=False));
            z_values_flat = np.stack(table['value'].to_numpy(zero_copy_only=False));
            q_values_flat = np.stack(table['q_value'].to_numpy(zero_copy_only=False))
            states = s_flat.reshape(num_rows, *self.state_shape);
            pis = pi_flat.reshape(num_rows, *self.policy_shape);
            moves_mask = moves_mask_flat.reshape(num_rows, *self.policy_shape);
            z_values = z_values_flat.reshape(num_rows, *self.value_shape);
            q_values = q_values_flat.reshape(num_rows, *self.value_shape)
            self._add_data_to_buffer(states, pis, z_values, moves_mask, q_values)


def monitor_weights(model: ThreePlayerChessformerBertV2, prefix: str):
    """Prints the min, max, and std of key weight tensors."""
    logging.info(f"--- {prefix} Weight Statistics ---")
    with torch.no_grad():
        rb_weights = model.relative_bias_table.weight.detach().cpu()
        pe_weights = model.position_embeddings.weight.detach().cpu()

        logging.info(
            f"model.relative_bias_table.weight: min: {rb_weights.min():.6f} max: {rb_weights.max():.6f} std: {rb_weights.std():.6f}")
        logging.info(
            f"model.position_embeddings.weight: min: {pe_weights.min():.6f} max: {pe_weights.max():.6f} std: {pe_weights.std():.6f}")
    logging.info("-" * (len(prefix) + 24))


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    settings = SalvageSettings()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: '{device}'")

    # --- 1. Load the problematic model ---
    # Dummy config for model instantiation, will be overwritten by checkpoint
    config = {"d_model": 240, "n_layers": 8, "n_heads": 5, "d_ff": 960, "dropout_rate": 0.1}
    model = ThreePlayerChessformerBertV2(**config)

    logging.info(f"Loading model to salvage from: {settings.MODEL_TO_SALVAGE}")
    checkpoint = torch.load(settings.MODEL_TO_SALVAGE, map_location=device)

    # Handle potential prefixes if the model was compiled
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

    # Monitor weights BEFORE the fix
    monitor_weights(model, prefix="Before Salvage")

    # --- 2. Prepare the "Golden" Dataset ---
    logging.info("Scanning dataset files for salvage operation...")
    if not settings.GOLDEN_DATASET_FILES:
        raise ValueError("Please specify at least one file in GOLDEN_DATASET_FILES.")

    golden_samples_count = sum(pq.read_metadata(f).num_rows for f in settings.GOLDEN_DATASET_FILES)
    logging.info(f"Found {golden_samples_count:,} samples in the golden dataset.")

    golden_dataset = StaticDataset(
        capacity=golden_samples_count, files_to_load=settings.GOLDEN_DATASET_FILES,
        state_shape=model.state_shape(), policy_shape=model.policy_shape(), value_shape=model.value_shape()
    )
    golden_loader = DataLoader(
        dataset=golden_dataset, batch_size=settings.BATCH_SIZE, shuffle=True,
        num_workers=settings.NUM_WORKERS, persistent_workers=settings.NUM_WORKERS > 0,
        pin_memory=device.type == 'cuda', drop_last=True
    )

    # --- 3. Set up the Salvage Optimizer ---
    salvage_optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.SALVAGE_LEARNING_RATE,
        weight_decay=settings.SALVAGE_WEIGHT_DECAY
    )

    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)

    # --- 4. Run the Salvage Fine-Tuning Loop ---
    logging.info(f"Starting salvage fine-tuning for {settings.SALVAGE_EPOCHS} epochs...")
    model.train()
    for epoch in range(1, settings.SALVAGE_EPOCHS + 1):
        total_loss = 0
        epoch_start_time = time.time()
        for batch_data in golden_loader:
            batch_states, batch_pis, batch_z_values, batch_q_values, moves_mask = batch_data

            batch_states = batch_states.to(device, non_blocking=True).float()
            batch_pis = batch_pis.to(device, non_blocking=True).float()
            batch_z_values = batch_z_values.to(device, non_blocking=True)
            batch_q_values = batch_q_values.to(device, non_blocking=True)
            moves_mask = moves_mask.to(device, non_blocking=True)

            salvage_optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_pi_logits, pred_z_values, pred_q_values = model(batch_states)
                masked_logits = pred_pi_logits.float().masked_fill(~moves_mask, -1e9)
                loss_pi = F.cross_entropy(masked_logits, batch_pis)
                loss_z_values = F.mse_loss(pred_z_values, batch_z_values)
                loss_q_values = F.mse_loss(pred_q_values, batch_q_values)
                loss = (settings.PI_WEIGHT * loss_pi +
                        settings.VALUES_Z_WEIGHT * loss_z_values +
                        settings.VALUES_Q_WEIGHT * loss_q_values)

            scaler.scale(loss).backward()
            scaler.unscale_(salvage_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            scaler.step(salvage_optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(golden_loader)
        duration = time.time() - epoch_start_time
        logging.info(
            f"Salvage Epoch {epoch}/{settings.SALVAGE_EPOCHS} | Avg Loss: {avg_loss:.4f} | Duration: {duration:.2f}s")

    # --- 5. Final Steps ---
    logging.info("Salvage operation completed.")

    # Monitor weights AFTER the fix
    monitor_weights(model, prefix="After Salvage")

    # Save the final, fixed model
    logging.info(f"Saving salvaged model to: {settings.SALVAGED_MODEL_SAVE_PATH}")
    # We save the state_dict directly for maximum compatibility
    torch.save(model.state_dict(), settings.SALVAGED_MODEL_SAVE_PATH)
    logging.info("Done.")


if __name__ == "__main__":
    main()
