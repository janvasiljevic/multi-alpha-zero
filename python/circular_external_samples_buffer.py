import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq
import time
from torch.utils.data import Dataset


class CircularExternalSamplesBuffer(Dataset):
    """
    A circular replay buffer with a fixed capacity that loads data explicitly from disk.

    This buffer pre-allocates memory and allows for loading data from Parquet files.
    When new data is loaded beyond the buffer's capacity, it overwrites the oldest
    existing data in a circular fashion.
    """

    def __init__(self, capacity: int,
                 persistence_path: Path,
                 state_shape: Tuple,
                 policy_shape: Tuple,
                 value_shape: Tuple,
                 aux_features_shape: Tuple,
                 is_resuming: bool = False,
                 ):
        self.capacity = capacity
        self.persistence_path = persistence_path

        self.state_shape = state_shape
        self.policy_shape = policy_shape
        self.value_shape = value_shape
        self.aux_features_shape = aux_features_shape

        self.states_np = np.zeros((capacity, *state_shape), dtype=np.bool_)
        self.pis_np = np.zeros((capacity, *policy_shape), dtype=np.float16)
        self.moves_mask = np.zeros((capacity, *policy_shape), dtype=np.bool_)
        self.z_value_np = np.zeros((capacity, *value_shape), dtype=np.float32)
        self.q_value_np = np.zeros((capacity, *aux_features_shape), dtype=np.float32)
        self.moves_left_np = np.zeros((capacity, 1), dtype=np.int32)

        # Pointers for managing the circular buffer
        self.pos = 0  # Next position to write to
        self._current_size = 0

        self.persistence_path.mkdir(parents=True, exist_ok=True)

        if is_resuming:
            logging.info("Resuming from existing data in folder: %s", self.persistence_path)
            logging.info("This may take a while depending on the amount of data...")
            self.load_all_files_from_folder()

        self._log_ram_usage()

    def __len__(self) -> int:
        """Returns the current number of valid samples in the buffer."""
        return self._current_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Used for data loading in PyTorch.
        """
        return (
            self.states_np[idx],
            self.pis_np[idx],
            self.z_value_np[idx],
            self.q_value_np[idx],
            self.moves_mask[idx]
        )

    def safe_get_item(self, raw_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves an item by its logical index (0 = oldest, len-1 = newest).
        This correctly maps the logical index to the physical index in the circular buffer.
        """
        if not 0 <= raw_idx < self._current_size:
            raise IndexError(f"Index {raw_idx} is out of range for buffer of size {self._current_size}")

        # If the buffer is not yet full, data is stored contiguously from 0.
        if self._current_size < self.capacity:
            idx = raw_idx
        else:
            # If the buffer is full, the logical index 0 corresponds to the oldest data,
            # which is located at the current `pos` pointer.
            idx = (self.pos + raw_idx) % self.capacity

        return self.__getitem__(idx)

    def _add_data_to_buffer(self, states: np.ndarray, pis: np.ndarray, z_values: np.ndarray,
                            moves_left: np.ndarray, moves_mask: np.ndarray, q_values: np.ndarray):
        """A private helper to add a batch of data to the circular buffer."""
        num_samples = states.shape[0]
        if num_samples == 0:
            return

        start_idx = self.pos
        end_idx = start_idx + num_samples

        # Write data, handling wrap-around if necessary
        if end_idx <= self.capacity:
            self.states_np[start_idx:end_idx] = states
            self.pis_np[start_idx:end_idx] = pis
            self.moves_mask[start_idx:end_idx] = moves_mask
            self.z_value_np[start_idx:end_idx] = z_values
            self.q_value_np[start_idx:end_idx] = q_values
            self.moves_left_np[start_idx:end_idx] = moves_left
        else:  # Data needs to wrap around
            fit_len = self.capacity - start_idx
            self.states_np[start_idx:] = states[:fit_len]
            self.pis_np[start_idx:] = pis[:fit_len]
            self.moves_mask[start_idx:] = moves_mask[:fit_len]
            self.z_value_np[start_idx:] = z_values[:fit_len]
            self.q_value_np[start_idx:] = q_values[:fit_len]
            self.moves_left_np[start_idx:] = moves_left[:fit_len]

            rem_len = num_samples - fit_len
            self.states_np[:rem_len] = states[fit_len:]
            self.pis_np[:rem_len] = pis[fit_len:]
            self.moves_mask[:rem_len] = moves_mask[fit_len:]
            self.z_value_np[:rem_len] = z_values[fit_len:]
            self.q_value_np[:rem_len] = q_values[fit_len:]
            self.moves_left_np[:rem_len] = moves_left[fit_len:]

        # Update pointers
        self.pos = (self.pos + num_samples) % self.capacity
        self._current_size = min(self._current_size + num_samples, self.capacity)

    def load_specific_file(self, file_path: Path) -> int:
        """
        Loads data from a single Parquet file into the buffer.
        If the buffer becomes full, older data will be overwritten.
        Returns 'samples_loaded'.
        """
        if not file_path.exists():
            logging.info("Skipping non-existent file (might have been deleted): %s", file_path)
            return 0
        try:
            table = pq.read_table(file_path)
            num_rows: int = table.num_rows
            logging.info(f"Loading {num_rows} samples from {file_path.name}...")

            # Assumes parquet stores flattened arrays which need to be stacked and reshaped.
            s_flat = np.stack(table['state'].to_numpy(zero_copy_only=False))
            pi_flat = np.stack(table['policy'].to_numpy(zero_copy_only=False))
            moves_mask_flat = np.stack(table['masked_policy'].to_numpy(zero_copy_only=False))
            z_values_flat = np.stack(table['value'].to_numpy(zero_copy_only=False))
            q_values_flat = np.stack(table['q_value'].to_numpy(zero_copy_only=False))
            moves_left_flat = np.stack(table['moves_left'].to_numpy(zero_copy_only=False))

            states = s_flat.reshape(num_rows, *self.state_shape)
            pis = pi_flat.reshape(num_rows, *self.policy_shape)
            moves_mask = moves_mask_flat.reshape(num_rows, *self.policy_shape)
            z_values = z_values_flat.reshape(num_rows, *self.value_shape)
            q_values = q_values_flat.reshape(num_rows, *self.aux_features_shape)
            moves_left = moves_left_flat.reshape(num_rows, 1)

            self._add_data_to_buffer(states, pis, z_values, moves_left, moves_mask, q_values)

            return num_rows

        except Exception as e:
            raise RuntimeError(f"Error loading file '{file_path}': {e}") from e

    def expand_capacity(self, new_capacity: int):
        """
        Re-allocates memory for the internal numpy arrays and
        copies the existing data. If the buffer was full and had
        wrapped around, the data is "unrolled" into its correct logical
        order (oldest to newest) in the new, larger arrays.
        """
        if new_capacity <= self.capacity:
            raise ValueError(
                f"New capacity ({new_capacity}) must be strictly "
                f"greater than the current capacity ({self.capacity})."
            )

        logging.info(f"Expanding buffer capacity from {self.capacity} to {new_capacity}.")

        # Larger arrays
        new_states = np.zeros((new_capacity, *self.state_shape), dtype=self.states_np.dtype)
        new_pis = np.zeros((new_capacity, *self.policy_shape), dtype=self.pis_np.dtype)
        new_z_values = np.zeros((new_capacity, *self.value_shape), dtype=self.z_value_np.dtype)
        new_q_values = np.zeros((new_capacity, *self.aux_features_shape), dtype=self.q_value_np.dtype)
        new_moves_left = np.zeros((new_capacity, 1), dtype=self.moves_left_np.dtype)
        new_moves_mask = np.zeros((new_capacity, *self.policy_shape), dtype=self.moves_mask.dtype)

        # Copy
        if self._current_size > 0:
            if self._current_size < self.capacity:
                # Case 1: Buffer is not full, data is stored linearly from index 0
                new_states[:self._current_size] = self.states_np[:self._current_size]
                new_pis[:self._current_size] = self.pis_np[:self._current_size]
                new_moves_mask[:self._current_size] = self.moves_mask[:self._current_size]
                new_z_values[:self._current_size] = self.z_value_np[:self._current_size]
                new_q_values[:self._current_size] = self.q_value_np[:self._current_size]
                new_moves_left[:self._current_size] = self.moves_left_np[:self._current_size]
            else:
                # Case 2: Buffer is full and wrapped. Unroll it using np.roll (shifting to the left)
                ordered_states = np.roll(self.states_np, shift=-self.pos, axis=0)
                ordered_pis = np.roll(self.pis_np, shift=-self.pos, axis=0)
                ordered_moves_mask = np.roll(self.moves_mask, shift=-self.pos, axis=0)
                ordered_z_values = np.roll(self.z_value_np, shift=-self.pos, axis=0)
                ordered_q_values = np.roll(self.q_value_np, shift=-self.pos, axis=0)
                ordered_moves_left = np.roll(self.moves_left_np, shift=-self.pos, axis=0)

                # Copy the now-ordered data into the new arrays
                new_states[:self._current_size] = ordered_states
                new_pis[:self._current_size] = ordered_pis
                new_moves_mask[:self._current_size] = ordered_moves_mask
                new_z_values[:self._current_size] = ordered_z_values
                new_q_values[:self._current_size] = ordered_q_values
                new_moves_left[:self._current_size] = ordered_moves_left

                # Now the pos pointer should point to the end of the new data
                self.pos = self._current_size

        self.states_np = new_states
        self.pis_np = new_pis
        self.moves_mask = new_moves_mask
        self.z_value_np = new_z_values
        self.q_value_np = new_q_values
        self.moves_left_np = new_moves_left

        self.capacity = new_capacity

        self._log_ram_usage()

    def _log_ram_usage(self):
        """
        Logs the current RAM usage of the buffer.
        """
        total_bytes = sum(
            arr.nbytes for arr in
            [self.states_np, self.pis_np, self.z_value_np, self.q_value_np, self.moves_left_np, self.moves_mask])
        logging.info(
            f"{self.__class__.__name__}: Current RAM usage: {total_bytes / 1024 ** 2:.2f} MB. Cap.: {self.capacity}/ Pos.: {self.pos}/ Size: {self._current_size}")

    def load_all_files_from_folder(self):
        """
        Scans the persistence path for Parquet files, sorts them by modification
        time (newest first), and loads them until the buffer is full.
        """
        existing_files = list(self.persistence_path.glob("*.parquet"))
        if not existing_files:
            logging.info("No existing files found in the folder.")
            return

        start_time = time.time()

        # Sort files by modification time, oldest first, so the new ones push out the old ones
        # if the buffer fills up.
        # use the same function as rust
        existing_files.sort(key=lambda f: (int(f.stem.split('_')[0]), int(f.stem.split('_')[1])), reverse=False)

        logging.info(f"Found {len(existing_files)} files. Loading newest ones first to fill buffer.")

        for file_path in existing_files:
            # Load the file. The internal logic will handle capacity.
            self.load_specific_file(file_path)

        logging.info(f"Loading completed in {time.time() - start_time:.2f} seconds.")
