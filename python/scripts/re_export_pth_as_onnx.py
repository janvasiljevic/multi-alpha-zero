import logging
import os
import pathlib
from collections import OrderedDict

import torch
from onnxruntime.tools.check_onnx_model_mobile_usability import usability_checker
from torch import nn

from neural.chess_model_bert_cls import ThreePlayerChessformerBert
from neural.model_factory import model_factory
from neural.onnx_wrapper import OnnxExportWrapper

models = [
    # {
    #     "dir": "./../../testing/torch_bert/",
    #     "label": "Hex5Bert",
    #     "items": [
    #         "20.pt",
    #         "35.pt",
    #         "50.pt",
    #         "65.pt",
    #         "80.pt",
    #         "95.pt",
    #         "110.pt",
    #         "125.pt",
    #         "140.pt",
    #     ],
    # },
    # {
    #     "dir": "./../../testing/torch_chess_final/",
    #     "label": "Chess24m",
    #     "items": [
    #         "50.pt",
    #         "80.pt",
    #         "110.pt",
    #         "140.pt",
    #         "170.pt",
    #         "200.pt",
    #         "230.pt",
    #         "260.pt",
    #         "290.pt",
    #         "320.pt",
    #         "350.pt",
    #         "380.pt",
    #         "410.pt",
    #         "440.pt",
    #         "470.pt",
    #         "500.pt",
    #     ],
    # }
    # {
    #     "dir": "./../../testing/chess2p/",
    #     "label": "ChessDomain",
    #     "items": [
    #         # "50.pt",
    #         # "100.pt",
    #         # "150.pt",
    #         # "200.pt",
    #         # "250.pt",
    #         # "300.pt",
    #         # "350.pt",
    #         # "400.pt",
    #         "538.pt",
    #     ]
    # },
    {
        "dir": "./../../testing/",
        "label": "ChessDomain",
        "items": [
            "249.pt",
        ]
    }
]

class OnnxDebugWrapper(nn.Module):
    def __init__(self, model_to_wrap: nn.Module):
        super().__init__()
        self.model = model_to_wrap.float()
        self.model.eval()
        self.model.pre_onnx_export()

    def forward(self, boolean_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        float_input = boolean_input.float()

        policy_logits, z_value, material, from_logits, to_logits = self.model(float_input)

        fp16 = policy_logits.to(torch.float16)

        return fp16, z_value, material, from_logits, to_logits

    def get_trace_input(self, batch_size: int) -> torch.Tensor:
        state_shapes = self.model.state_shape()
        return torch.randint(
            low=0,
            high=2,
            size=(batch_size, state_shapes[0], state_shapes[1]),
            dtype=torch.bool,
        )

# out_dir = "../../testing/arena/Hex5Bert"
# model = model_factory("Hex5CanonicalAxiomBiasBertCls")
# out_dir = "../../testing/play/v2"

out_dir = "../../testing/play/ChessDomain32"
model = model_factory("ChessDomain")
set_to_debug = False
batch_size = 16

actually_set_to_debug = False

if set_to_debug:
    if hasattr(model, "set_debug_mode"):
        model.set_debug_mode(True)
        actually_set_to_debug = True


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model.pre_onnx_export() # This calls _compute_and_register_bias()

for model_info in models:
    label = model_info["label"]
    dir_path = model_info["dir"]

    for item in model_info["items"]:
        checkpoint = torch.load(f"{dir_path}{item}", map_location="cpu")
        state_dict = checkpoint

        # remove the "_orig_mod." prefix from all keys
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            new_state_dict[k.replace("_orig_mod.", "")] = v

        model.load_state_dict(new_state_dict)

        if actually_set_to_debug:
            wrapper_model = OnnxDebugWrapper(model)
        else:
            wrapper_model = OnnxExportWrapper(model)

        onnx_path = f"{out_dir}/{label}_{item.replace('.pt', '.onnx')}"

        print(f"Exporting to {onnx_path}")

        torch.onnx.export(
            wrapper_model,
            (wrapper_model.get_trace_input(batch_size),),
            onnx_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["policy_logits", "value", "material", "from_logits", "to_logits"] if actually_set_to_debug else ["policy_logits", "value"],
        )

        try:
            logger = logging.getLogger("check_usability")
            path = pathlib.Path(os.path.abspath(onnx_path))
            usability_checker.analyze_model(path, skip_optimize=False, logger=None)
        except Exception as e:
            print(f"Error checking ONNX model for batch size {batch_size}: {e}")
