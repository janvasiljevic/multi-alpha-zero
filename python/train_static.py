import logging
import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import time
import torch
import torch.nn.functional as F
from torch import optim
# --- CHANGE 1: Import amp components ---
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from neural.model_factory import model_factory
from neural.model_wrapper import ModelWrapper

# PYTHONPATH=/shared/home/jan.vasiljevic/hex-testing/python python train_static.py

class TrainSettings:
    SAMPLES_PATH = Path("../runs/chess_big_bert_fixed_gradients_data_gen/samples_new")
    PYTORCH_DIR = Path("../runs/chess_pre_training_bert_v2_10m_amp/pytorch")
    OPTIMIZER_DIR = Path("../runs/chess_pre_training_bert_v2_10m_amp/optimizer")
    TENSORBOARD_DIR = Path("../runs/chess_pre_training_bert_v2_10m_amp/tensorboard")

    MODEL_KEY = 'ChessBigBertV2'
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 2048
    EPOCHS = 200  # A high maximum limit; early stopping will likely trigger first
    NUM_WORKERS = 4

    WARMUP_STEPS = 2000
    WARMUP_START_FACTOR = 0.01

    # Same as server.py
    PI_WEIGHT = 1.0
    VALUES_Z_WEIGHT = 1.0
    VALUES_Q_WEIGHT = 0.5

    VALIDATION_SPLIT_RATIO = 0.06
    EARLY_STOPPING_PATIENCE = 10  # Stop if validation loss doesn't improve for 10 epochs


class StaticDataset(Dataset):
    """
    Needs a lot of ram
    """

    def __init__(self, capacity: int, files_to_load: List[Path], state_shape: Tuple, policy_shape: Tuple,
                 value_shape: Tuple):
        self.capacity = capacity
        self.files_to_load = files_to_load
        self.state_shape = state_shape
        self.policy_shape = policy_shape
        self.value_shape = value_shape

        logging.info(f"Pre-allocating memory for {capacity:,} samples for this dataset...")
        self.states_np = np.zeros((capacity, *state_shape), dtype=np.bool_)
        self.pis_np = np.zeros((capacity, *policy_shape), dtype=np.float16)
        self.moves_mask = np.zeros((capacity, *policy_shape), dtype=np.bool_)
        self.z_value_np = np.zeros((capacity, *value_shape), dtype=np.float32)
        self.q_value_np = np.zeros((capacity, *value_shape), dtype=np.float32)

        self.pos = 0
        self._log_ram_usage()
        self.load_files()

    def __len__(self) -> int:
        return self.pos

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.states_np[idx], self.pis_np[idx], self.z_value_np[idx],
            self.q_value_np[idx], self.moves_mask[idx]
        )

    def _add_data_to_buffer(self, states, pis, z_values, moves_mask, q_values):
        num_samples = states.shape[0]
        if self.pos + num_samples > self.capacity:
            raise ValueError(
                f"Capacity exceeded: Trying to add {num_samples} samples to a buffer with {self.capacity - self.pos} space left.")

        end_idx = self.pos + num_samples
        self.states_np[self.pos:end_idx] = states
        self.pis_np[self.pos:end_idx] = pis
        self.moves_mask[self.pos:end_idx] = moves_mask
        self.z_value_np[self.pos:end_idx] = z_values
        self.q_value_np[self.pos:end_idx] = q_values
        self.pos = end_idx

    def load_specific_file(self, file_path: Path):
        try:
            table = pq.read_table(file_path)
            num_rows = table.num_rows

            s_flat = np.stack(table['state'].to_numpy(zero_copy_only=False))
            pi_flat = np.stack(table['policy'].to_numpy(zero_copy_only=False))
            moves_mask_flat = np.stack(table['masked_policy'].to_numpy(zero_copy_only=False))
            z_values_flat = np.stack(table['value'].to_numpy(zero_copy_only=False))
            q_values_flat = np.stack(table['q_value'].to_numpy(zero_copy_only=False))

            states = s_flat.reshape(num_rows, *self.state_shape)
            pis = pi_flat.reshape(num_rows, *self.policy_shape)
            moves_mask = moves_mask_flat.reshape(num_rows, *self.policy_shape)
            z_values = z_values_flat.reshape(num_rows, *self.value_shape)
            q_values = q_values_flat.reshape(num_rows, *self.value_shape)

            self._add_data_to_buffer(states, pis, z_values, moves_mask, q_values)
        except Exception as e:
            logging.error(f"Error loading file '{file_path}': {e}")

    def load_files(self):
        start_time = time.time()
        if not self.files_to_load:
            logging.warning("No files provided to load for this dataset.")
            return

        logging.info(f"Loading data from {len(self.files_to_load)} specified files...")
        i = 0
        for file_path in self.files_to_load:
            i += 1
            self.load_specific_file(file_path)
            if i % 10 == 0 or i == len(self.files_to_load):
                logging.info(f"Loaded {i}/{len(self.files_to_load)} files... Current total samples: {self.pos:,}")
        logging.info(
            f"Loading completed in {time.time() - start_time:.2f} seconds. Total samples in this dataset: {self.pos:,}")

    def _log_ram_usage(self):
        total_bytes = sum(
            arr.nbytes for arr in [self.states_np, self.pis_np, self.z_value_np, self.q_value_np, self.moves_mask])
        logging.info(
            f"{self.__class__.__name__}: Allocated RAM for data: {total_bytes / 1024 ** 3:.2f} GB for {self.capacity:,} samples.")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    settings = TrainSettings()

    for path in [settings.PYTORCH_DIR, settings.OPTIMIZER_DIR, settings.TENSORBOARD_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: '{device}'")
    net = model_factory(settings.MODEL_KEY)

    logging.info("Scanning dataset files and preparing split...")
    all_files = list(settings.SAMPLES_PATH.glob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No .parquet files found in {settings.SAMPLES_PATH}")

    random.shuffle(all_files)
    split_index = int(len(all_files) * (1 - settings.VALIDATION_SPLIT_RATIO))
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # train_files = train_files[:30]  # --- IGNORE ---
    # val_files = val_files[:10]      # --- IGNORE ---

    logging.info(f"Splitting by file: {len(train_files)} files for training, {len(val_files)} for validation.")
    train_samples_count = sum(pq.read_metadata(f).num_rows for f in train_files)
    val_samples_count = sum(pq.read_metadata(f).num_rows for f in val_files)

    train_dataset = StaticDataset(
        capacity=train_samples_count, files_to_load=train_files,
        state_shape=net.state_shape(), policy_shape=net.policy_shape(), value_shape=net.value_shape()
    )
    val_dataset = StaticDataset(
        capacity=val_samples_count, files_to_load=val_files,
        state_shape=net.state_shape(), policy_shape=net.policy_shape(), value_shape=net.value_shape()
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True,
        num_workers=settings.NUM_WORKERS, persistent_workers=settings.NUM_WORKERS > 0,
        pin_memory=device.type == 'cuda', drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False,
        num_workers=settings.NUM_WORKERS, persistent_workers=settings.NUM_WORKERS > 0,
        pin_memory=device.type == 'cuda', drop_last=False
    )
    net.to(device)
    net.pre_onnx_export()

    using_torch_compile = False
    if device.type == "cuda":
        logging.info("Compiling the model for CUDA with torch.compile...")
        net = torch.compile(net, mode="default", fullgraph=True)
        using_torch_compile = True
        logging.info("Enabling Automatic Mixed Precision (AMP).")

    optimizer = optim.AdamW(net.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=settings.WARMUP_START_FACTOR, end_factor=1.0,
                                                  total_iters=settings.WARMUP_STEPS)

    model_wrapper: ModelWrapper
    latest_model_version = -1
    for f in settings.PYTORCH_DIR.glob("*.pt"):
        if match := re.match(r"(\d+)\.pt", f.name):
            version = int(match.group(1))
            if version > latest_model_version:
                latest_model_version = version

    if latest_model_version > -1:
        model_path = settings.PYTORCH_DIR / f"{latest_model_version}.pt"
        optimizer_path = settings.OPTIMIZER_DIR / f"{latest_model_version}.pt"
        logging.info(f"Loading existing model from {model_path} and optimizer from {optimizer_path}")

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        if not using_torch_compile and any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = OrderedDict((k.replace("_orig_mod.", "", 1), v) for k, v in state_dict.items())

        net.load_state_dict(state_dict)
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
        model_wrapper = ModelWrapper(model=net, optimizer=optimizer, version=latest_model_version)
    else:
        logging.info(f"Initializing new model with #parameters: {sum(p.numel() for p in net.parameters()):,}")
        model_wrapper = ModelWrapper(model=net, optimizer=optimizer, version=0)
        model_wrapper.save_checkpoint(pytorch_dir=settings.PYTORCH_DIR, optimizer_dir=settings.OPTIMIZER_DIR)

    summary_writer = SummaryWriter(str(settings.TENSORBOARD_DIR))
    total_training_steps = 0
    np.set_printoptions(precision=2)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_version = model_wrapper.version
    
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(init_scale=2.**16, enabled=use_amp) 

    logging.info(f"Starting training for up to {settings.EPOCHS} epochs...")
    for epoch in range(1, settings.EPOCHS + 1):
        # --- Training Phase ---
        model_wrapper.model.train()
        total_train_loss = 0
        epoch_start_time = time.time()
        for batch_states, batch_pis, batch_z_values, batch_q_values, moves_mask in train_loader:
            batch_states = batch_states.to(device, non_blocking=True).float()
            batch_pis = batch_pis.to(device, non_blocking=True).float()
            batch_z_values = batch_z_values.to(device, non_blocking=True)
            batch_q_values = batch_q_values.to(device, non_blocking=True)
            moves_mask = moves_mask.to(device, non_blocking=True)

            # Use set_to_none=True for a minor performance improvement
            optimizer.zero_grad(set_to_none=True)

            # --- CHANGE 3: Use autocast for forward pass and loss calculation ---
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_pi_logits, pred_z_values, pred_q_values = model_wrapper.model(batch_states)
                masked_logits = pred_pi_logits.float().masked_fill(~moves_mask, -1e9)

                loss_pi = F.cross_entropy(masked_logits, batch_pis)
                loss_z_values = F.mse_loss(pred_z_values, batch_z_values)
                loss_q_values = F.mse_loss(pred_q_values, batch_q_values)
                loss = (
                        settings.PI_WEIGHT * loss_pi + settings.VALUES_Z_WEIGHT * loss_z_values + settings.VALUES_Q_WEIGHT * loss_q_values)

            # --- CHANGE 4: Use scaler for backward pass and optimizer step ---
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            if total_training_steps < settings.WARMUP_STEPS: scheduler.step()
            total_train_loss += loss.item()
            total_training_steps += 1
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model_wrapper.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_states, batch_pis, batch_z_values, batch_q_values, moves_mask in val_loader:
                batch_states = batch_states.to(device, non_blocking=True).float()
                batch_pis = batch_pis.to(device, non_blocking=True).float()
                batch_z_values = batch_z_values.to(device, non_blocking=True)
                batch_q_values = batch_q_values.to(device, non_blocking=True)
                moves_mask = moves_mask.to(device, non_blocking=True)
                
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred_pi_logits, pred_z_values, pred_q_values = model_wrapper.model(batch_states)
                    masked_logits = pred_pi_logits.float().masked_fill(~moves_mask, -1e9)

                    loss_pi = F.cross_entropy(masked_logits, batch_pis)
                    loss_z_values = F.mse_loss(pred_z_values, batch_z_values)
                    loss_q_values = F.mse_loss(pred_q_values, batch_q_values)
                    val_loss = (
                            settings.PI_WEIGHT * loss_pi + settings.VALUES_Z_WEIGHT * loss_z_values + settings.VALUES_Q_WEIGHT * loss_q_values)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        # ... (Logging and checkpointing logic is unchanged) ...
        model_wrapper.debug() # This might be removed if it causes issues with AMP/compile
        
        duration_seconds = time.time() - epoch_start_time
        logging.info(
            f"Epoch {epoch}/{settings.EPOCHS} ({duration_seconds:.2f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        summary_writer.add_scalars('Loss', {'train': avg_train_loss, 'validation': avg_val_loss}, epoch)
        summary_writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model_wrapper.version += 1
            best_model_version = model_wrapper.version

            model_wrapper.save_checkpoint(pytorch_dir=settings.PYTORCH_DIR, optimizer_dir=settings.OPTIMIZER_DIR)
            logging.info(
                f"New best model found! Val loss: {best_val_loss:.4f}. Saved checkpoint version {best_model_version}")
        else:
            epochs_no_improve += 1
            logging.info(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= settings.EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logging.info("=" * 50)
    logging.info("Training finished.")
    logging.info(f"The best model was version {best_model_version} with a validation loss of {best_val_loss:.4f}.")
    logging.info(f"Checkpoints are saved in: {settings.PYTORCH_DIR}")
    logging.info("=" * 50)

    summary_writer.close()


if __name__ == "__main__":
    main()