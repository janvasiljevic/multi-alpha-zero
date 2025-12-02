from typing import Tuple

import torch
from torch import nn

from neural.base_interface_az import AlphaZeroNet


class OnnxExportWrapper(nn.Module):
    def __init__(self, model_to_wrap: AlphaZeroNet):
        super().__init__()
        self.model = model_to_wrap.float()
        self.model.eval()
        self.model.pre_onnx_export()

    def forward(self, boolean_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        float_input = boolean_input.float()

        policy_logits, z_value, _ = self.model(float_input)

        fp16 = policy_logits.to(torch.float16)

        return fp16, z_value

    def get_trace_input(self, batch_size: int) -> torch.Tensor:
        state_shapes = self.model.state_shape()
        return torch.randint(
            low=0,
            high=2,
            size=(batch_size, state_shapes[0], state_shapes[1]),
            dtype=torch.bool,
        )


