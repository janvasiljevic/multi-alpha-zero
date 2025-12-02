
from collections import OrderedDict

import torch
from onnxruntime.tools.check_onnx_model_mobile_usability import usability_checker

from logging_setup import setup_logging
from neural.chess_model_bert_cls import ThreePlayerChessformerBert
from neural.model_factory import model_factory
from neural.onnx_wrapper import OnnxExportWrapper

model_to_load = "../../testing/115.pt"

model = model_factory("ChessBigBertV2")


model.pre_onnx_export()

checkpoint = torch.load(model_to_load, map_location="cpu")
state_dict = checkpoint

# remove the "_orig_mod." prefix from all keys
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    new_state_dict[k.replace("_orig_mod.", "")] = v


setup_logging()
model.load_state_dict(new_state_dict)
model.all_debug()

wrapper_model = OnnxExportWrapper(model)



