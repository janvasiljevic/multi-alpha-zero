import pathlib
from collections import OrderedDict

import torch
from onnxruntime.tools.check_onnx_model_mobile_usability import usability_checker

from logging_setup import setup_logging
from neural.chess_model_bert_cls import ThreePlayerChessformerBert
from neural.model_factory import model_factory
from neural.onnx_wrapper import OnnxExportWrapper
from train_static import StaticDataset


# get all models from ../../testing/torch
models = []
for file in pathlib.Path("../../testing/torch").rglob("*.pt"):
    models.append(str(file))

# sort by filename
models = sorted(models, key=lambda x: int(pathlib.Path(x).stem))

setup_logging()

for m in models:
    print(f"Loading model {m}")

    model = model_factory("ChessBigBertV2")

    model.pre_onnx_export()

    checkpoint = torch.load(m, map_location="cpu")
    state_dict = checkpoint

    # remove the "_orig_mod." prefix from all keys
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_state_dict[k.replace("_orig_mod.", "")] = v

    model.load_state_dict(new_state_dict)
    model.debug()


# static_dataset = StaticDataset(
#     6000,
#     files_to_load=["../../testing/samples_148.parquet"],
#     state_shape=model.state_shape(), policy_shape=model.policy_shape(), value_shape=model.value_shape()
# )
#
# model.eval()
# # # perform 6000 forward passes
# for i in range(6000):
#     state, _, _, _, _ = static_dataset[i]
#     state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#     with torch.no_grad():
#         _ = model(state)
#
#     if i % 100 == 0:
#         print(f"Completed {i} forward passes")
#
#     if i > 200:
#         break
#
#
