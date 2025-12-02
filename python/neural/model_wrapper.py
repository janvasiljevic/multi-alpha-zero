import logging
from pathlib import Path

import torch
from torch import optim

from neural.base_interface_az import AlphaZeroNet
from neural.onnx_wrapper import OnnxExportWrapper


class ModelWrapper:
    version: int
    model: AlphaZeroNet
    optimizer: optim.Optimizer

    def __init__(self, model: AlphaZeroNet, optimizer: optim.Optimizer, version: int = 0):
        self.model = model
        self.optimizer = optimizer
        self.version = version

        self.original_model = getattr(model, '_orig_mod', model)

    def save_checkpoint(self, pytorch_dir: Path, optimizer_dir: Path) -> Path:
        model_path = pytorch_dir / f"{self.version}.pt"
        optimizer_path = optimizer_dir / f"{self.version}.pt"

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

        logging.info(f"Model saved at {model_path} and optimizer state at {optimizer_path}")

        return model_path

    def save_to_onnx(self, onnx_dir: Path, batch_size) -> Path:
        onnx_wrapper_model = OnnxExportWrapper(self.original_model)

        onnx_path = onnx_dir / f"{self.version}.onnx"

        torch.onnx.export(
            onnx_wrapper_model,
            (onnx_wrapper_model.get_trace_input(batch_size).to(self.model.device),),
            onnx_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["policy_logits", "value"],
        )

        return onnx_path

    def debug(self):
        self.model.debug()
