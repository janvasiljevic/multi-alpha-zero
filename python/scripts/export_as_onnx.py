import logging
import os
import pathlib

import torch
from onnxruntime.tools.check_onnx_model_mobile_usability import usability_checker

from logging_setup import setup_logging
from neural.chess_model_basic import ThreePlayerChessformerBasic
from neural.chess_model_bert_cls import ThreePlayerChessformerBert
from neural.chess_model_bert_cls_v2 import ThreePlayerChessformerBertV2
from neural.chess_model_relative import ThreePlayerChessformerRelative
from neural.chess_model_shaw import ThreePlayerChessformerShaw
from neural.hex_model import HexAlphaZeroNet
from neural.hex_model_axiom_bias_bert_cls import Hex5AxiomBiasWithBertCls
from neural.hex_model_relative import Hex5AxiomBias
from neural.onnx_wrapper import OnnxExportWrapper

# PYTHONPATH=/shared/home/jan.vasiljevic/hex-testing/python python scripts/export_as_onnx.py

# setup_logging()
#
# m = ThreePlayerChessformerBert(
#     d_model=192,
#     d_ff=512,
#     n_heads=4,
#     n_layers=8,
# )
#

config = {
    "d_model": 5 * 64,
    "n_layers": 9,
    "n_heads": 5,
    "d_ff": 5 * 64 * 4,      # 4 * d_model
    "dropout_rate": 0.1
}

setup_logging()

m = ThreePlayerChessformerShaw(**config)

print(f"Model parameter count: {sum(p.numel() for p in m.parameters() if p.requires_grad)}")
m.eval()
m.pre_onnx_export()

name = "chess_bert_v2"

# Acts differently on Frida...
folder = f"./onnx/{name}"

if not os.path.exists(folder):
    os.makedirs(folder)

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            os.rmdir(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

batch_sizes = [32, 48] #256]
opset_version = 19

for batch_size in batch_sizes:

    onnx_path = f"{folder}/{name}_model_batched_{batch_size}.onnx"

    to_export = OnnxExportWrapper(m)

    torch.onnx.export(
        to_export,
        (to_export.get_trace_input(batch_size),),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy_logits", "value"],
    )

    to_export.model.debug()

    try:
        logger = logging.getLogger("check_usability")
        path = pathlib.Path(os.path.abspath(onnx_path))
        usability_checker.analyze_model(path, skip_optimize=False, logger=None)
    except Exception as e:
        print(f"Error checking ONNX model for batch size {batch_size}: {e}")

# get all model sizes in the folder
onnx_files = [f for f in os.listdir(folder) if f.endswith('.onnx')]
total_file_size = 0
for onnx_file in onnx_files:
    file_path = os.path.join(folder, onnx_file)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # size in MB
    total_file_size += file_size

print(f"Total size of ONNX models in '{folder}': {total_file_size:.2f} MB")
