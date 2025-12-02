import logging

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logging_setup import setup_logging
from neural.base_interface_az import AlphaZeroNet
from neural.hex_5_coords import all_hex_5_coords


class FeedForward(nn.Module):
    """Standard FeedForward layer with GELU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.gelu(self.linear_1(x)))


class RelativeBiasAttention(nn.Module):
    """
    T5-style self-attention with a learnable relative position bias.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 1. Standard content scores
        scores = torch.matmul(q, k.transpose(-1, -2))

        # 2. Add the pre-computed relative position bias
        scores = (scores / math.sqrt(self.d_head)) + relative_bias

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.w_o(context)


class EncoderLayer(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Layer with T5-style relative attention.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = RelativeBiasAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        # Pre-LN implementation
        x = x + self.attn(self.norm1(x), relative_bias=relative_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class Hex5AxiomBias(AlphaZeroNet):
    """
    UPGRADED: This network now uses T5-style relative positional embeddings and
    the corresponding EncoderLayer architecture from the reference model.
    """

    def pre_onnx_export(self):
        pass

    def debug(self):
        pass

    def __init__(
            self,
            d_model: int = 256,
            nhead: int = 4,
            num_encoder_layers: int = 4,
            d_ff: int = 1024,
    ) -> None:
        super().__init__()
        self.board_side_length = 5
        self.num_cells = 3 * self.board_side_length * (self.board_side_length - 1) + 1
        self.input_dim = 4
        self.nhead = nhead

        # Input embedding layer
        self.input_embedding = nn.Linear(self.input_dim, d_model)

        # T5-style Relative Positional Bias Setup
        all_hex_coords = all_hex_5_coords
        if all_hex_coords.shape[0] != self.num_cells:
            raise ValueError(
                f"Coordinate LUT size ({all_hex_coords.shape[0]}) does not match "
                f"expected number of cells ({self.num_cells})."
            )

        # TODO: A bi to blo kul da bi blo dejank 3 * daljse?
        # ka tko bi lahko sestel q + r + s in noter bi imel max delta q + max delta r + max delta s
        # in potem naredil nek lookup na podlgi tega? Idk ceb delalo
        max_coord_val = self.board_side_length - 1
        max_rel_dist = 2 * max_coord_val
        num_rel_embeddings = 2 * max_rel_dist + 1

        self.relative_bias_table = nn.Embedding(num_rel_embeddings, self.nhead)
        self.relative_bias_projection = nn.Linear(self.nhead * 3, self.nhead)

        with torch.no_grad():
            rel_indices = self._create_relative_indices(all_hex_coords, max_rel_dist)
            self.register_buffer('rel_indices', rel_indices)

        # --- MODIFIED: Use the new EncoderLayer ---
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, d_ff)
                for _ in range(num_encoder_layers)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        self.z_value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Tanh(),
        )

        self.q_value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    def _create_relative_indices(self, coords: torch.Tensor, max_rel_dist: int) -> torch.Tensor:
        q_coords, r_coords, s_coords = coords.chunk(3, dim=-1)
        q_coords, r_coords, s_coords = q_coords.squeeze(-1), r_coords.squeeze(-1), s_coords.squeeze(-1)
        rel_q = q_coords.unsqueeze(1) - q_coords.unsqueeze(0)
        rel_r = r_coords.unsqueeze(1) - r_coords.unsqueeze(0)
        rel_s = s_coords.unsqueeze(1) - s_coords.unsqueeze(0)
        offset = max_rel_dist
        rel_q_indices = rel_q + offset
        rel_r_indices = rel_r + offset
        rel_s_indices = rel_s + offset
        return torch.stack([rel_q_indices, rel_r_indices, rel_s_indices], dim=-1).long()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if hasattr(self, 'policy_head') and module == self.policy_head[2]:
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if hasattr(self, 'q_value_head') and module == self.q_value_head[2]:
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if hasattr(self, 'z_value_head') and module == self.z_value_head[2]:
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def device(self):
        return self.input_embedding.weight.device

    def validated_forward(self, x: torch.Tensor):
        if x.shape[1] != self.num_cells or x.ndim != 3:
            raise ValueError(
                f"Input tensor shape must be [batch_size, {self.num_cells}, input_dim]. "
                f"Got {x.shape} instead."
            )
        return self.forward(x)

    def forward(self, x: torch.Tensor):
        # Dynamically compute the relative bias for the attention layers
        rel_q_indices, rel_r_indices, rel_s_indices = self.rel_indices.chunk(3, dim=-1)
        bias_q = self.relative_bias_table(rel_q_indices.squeeze(-1))
        bias_r = self.relative_bias_table(rel_r_indices.squeeze(-1))
        bias_s = self.relative_bias_table(rel_s_indices.squeeze(-1))
        stacked_biases = torch.cat([bias_q, bias_r, bias_s], dim=-1)
        projected_biases = self.relative_bias_projection(stacked_biases)
        relative_bias = projected_biases.permute(2, 0, 1).unsqueeze(0)

        x = self.input_embedding(x)

        for layer in self.encoder_layers:
            x = layer(x, relative_bias)

        pooled_output = x.mean(dim=1)

        z_value = self.z_value_head(pooled_output)
        q_value = self.q_value_head(pooled_output)

        # This is new and corrected...
        policy_logits = self.policy_head(x)

        policy_logits = policy_logits.squeeze(-1)

        return policy_logits, z_value, q_value

    def state_shape(self):
        return self.num_cells, self.input_dim

    def policy_shape(self):
        return (self.num_cells,)

    def value_shape(self):
        return (3,)

    def log_gradients(self, epoch: int):
        pi_head_grad_norm = torch.linalg.norm(self.policy_head[2].weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head[2].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head[2].weight.grad).item()
        # MODIFIED: Path to attention weights is different in the new encoder
        encoder_grad_norm = torch.linalg.norm(self.encoder_layers[0].attn.w_q.weight.grad).item()

        logging.info(
            f"Grad norms for epoch {epoch + 1} are PI: {pi_head_grad_norm:.6f}, Z value: {z_val_head_grad_norm:.6f}, Q value: {q_val_head_grad_norm:.6f}, Encoder: {encoder_grad_norm:.6f}")

    def debug(self):
        # print the relative bias table
        # set print options to show all, no sci notation and max 2 decimals (numpy)
        torch.set_printoptions(profile="full", precision=2, sci_mode=False)
        np.set_printoptions(precision=2, suppress=True, linewidth=200)

        builder = ""

        for i in range(self.relative_bias_table.weight.size(0)):
            builder += f"Relative bias for distance {i - 14}: {self.relative_bias_table.weight[i].cpu().detach().numpy()}\n"

        logging.info("Debug info:\n" + builder)


if __name__ == '__main__':
    setup_logging()
    model_config = {
        "d_model": 128,
        "nhead": 4,
        "num_encoder_layers": 3,
        "d_ff": 512,
    }

    model = Hex5AxiomBias(**model_config)
    print(f"Model created successfully on device: {model.device}")
    print(f"Number of cells for Hex-5: {model.num_cells}")

    batch_size = 8
    dummy_input = torch.randn(batch_size, model.num_cells, model.input_dim).to(model.device)

    with torch.no_grad():
        policy_logits, z_value, q_value = model(dummy_input)

    print("\n--- Output Shapes ---")
    print(f"Policy logits shape: {policy_logits.shape}")  # Expected: (batch_size, num_cells)
    print(f"Z-Value shape: {z_value.shape}")  # Expected: (batch_size, 3)
    print(f"Q-Value shape: {q_value.shape}")  # Expected: (batch_size, 3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    model.debug()
