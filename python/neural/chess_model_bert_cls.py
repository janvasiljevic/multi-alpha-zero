import logging
from typing import Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural.base_interface_az import AlphaZeroNet
from neural.chess_model_relative_coords import all_hex_coords


class FeedForward(nn.Module):
    """Standard FeedForward layer with GELU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.gelu(self.linear_1(x)))


class AttentionPolicyHead(nn.Module):
    def __init__(self, d_model: int, d_policy_head: int = 32):
        super().__init__()
        self.d_policy_head = d_policy_head
        self.q_proj = nn.Linear(d_model, d_policy_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_policy_head, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_policy_head)
        return logits.view(logits.size(0), -1)


class ValueHead(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RelativeBiasMQA(nn.Module):
    """
    T5-style self-attention with Multi-Query Attention (MQA) for faster inference.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q projection is still per-head
        self.w_q = nn.Linear(d_model, d_model, bias=False)

        # K and V projections are shared across all heads
        self.w_k = nn.Linear(d_model, self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, self.d_head, bias=False)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # K and V are now [batch_size, seq_len, d_head]. We need to add a head dim for broadcasting.
        k = self.w_k(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)

        # Matmul will broadcast k and v across the head dimension
        scores = torch.matmul(q, k.transpose(-1, -2))

        scores = (scores / math.sqrt(self.d_head)) + relative_bias
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(context)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = RelativeBiasMQA(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        # Pre-LN implementation
        x = x + self.attn(self.norm1(x), relative_bias=relative_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class ThreePlayerChessformerBert(AlphaZeroNet):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()
        self.seq_len = 96
        self.input_features = 28  # Updated to 27 input features - 18 piece planes + 2 * 3 castling + 3 en passant + 1 grace period
        self.n_heads = n_heads  # Store n_heads
        self.n_layers = n_layers  # Store n_layers

        self.token_embed = nn.Linear(self.input_features, d_model)

        # Prepend a CLS token to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        nn.init.normal_(self.cls_token, std=0.02)

        max_coord_val = 7
        max_rel_dist = 2 * max_coord_val
        num_rel_embeddings = 2 * max_rel_dist + 1

        self.relative_bias_table = nn.Embedding(num_rel_embeddings, self.n_heads)
        self.relative_bias_projection = nn.Linear(self.n_heads * 3, self.n_heads)

        # Pre-compute the indices, as they are constant. Store them in a buffer.
        with torch.no_grad():
            rel_indices = self._create_relative_indices(all_hex_coords, max_rel_dist)
            # We will use these indices in the forward pass
            self.register_buffer('rel_indices', rel_indices)

        # --- Standard Model Layers ---
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.policy_head = AttentionPolicyHead(d_model)
        self.z_value_head = ValueHead(d_model)
        self.q_value_head = ValueHead(d_model)

        self.apply(self._init_weights)

        self.register_buffer('relative_bias', None, persistent=True)

    def _init_weights(self, module):
        """
        Initializes weights of the model for better stability.
        This is crucial for preventing large initial values that can cause
        overflow in FP16.
        """
        if isinstance(module, nn.Linear):
            # Apply Kaiming for most layers
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if isinstance(module, nn.Embedding):
            # For 'relative_bias_table' / 'cls_token', we want small initial weights centered around zero.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Special initialization for residual projections
        if isinstance(module, RelativeBiasMQA):
            nn.init.normal_(module.w_o.weight, std=0.02 / math.sqrt(2 * self.n_layers))

        if isinstance(module, FeedForward):
            nn.init.normal_(module.linear_2.weight, std=0.02 / math.sqrt(2 * self.n_layers))

        if isinstance(module, ValueHead):
            nn.init.constant_(module.network[2].weight, 0)
            if module.network[2].bias is not None:
                nn.init.constant_(module.network[2].bias, 0)

        if isinstance(module, AttentionPolicyHead):
            nn.init.constant_(module.q_proj.weight, 0)
            nn.init.constant_(module.k_proj.weight, 0)

        if isinstance(module, nn.LayerNorm):
            # Standard practice for LayerNorm
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)


    def _compute_and_register_bias(self):
        with torch.no_grad():
            rel_q_indices, rel_r_indices, rel_s_indices = self.rel_indices.chunk(3, dim=-1)
            bias_q = self.relative_bias_table(rel_q_indices.squeeze(-1))
            bias_r = self.relative_bias_table(rel_r_indices.squeeze(-1))
            bias_s = self.relative_bias_table(rel_s_indices.squeeze(-1))
            stacked_biases = torch.cat([bias_q, bias_r, bias_s], dim=-1)
            projected_biases = self.relative_bias_projection(stacked_biases)
            bias = projected_biases.permute(2, 0, 1).unsqueeze(0)

            # --- Modified for CLS token ---
            # NOTE: Determine if you are using a CLS token. If so, pad it.
            # If not (e.g., using Global Average Pooling), you don't need to pad.
            # Let's assume you still use the CLS token for this example.
            bias = F.pad(bias, (1, 0, 1, 0))

            # Register it to the buffer
            self.register_buffer('relative_bias', bias)
        print("Relative position bias has been pre-computed and registered.")


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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # During training, we compute the bias on the fly.
        # For a pre-processed inference model, we use the buffer.
        if self.training:
            # Dynamically compute for training to allow bias weights to be learned
            rel_q_indices, rel_r_indices, rel_s_indices = self.rel_indices.chunk(3, dim=-1)

            # Look up bias from the learnable table using the pre-computed indices
            bias_q = self.relative_bias_table(rel_q_indices.squeeze(-1))
            bias_r = self.relative_bias_table(rel_r_indices.squeeze(-1))
            bias_s = self.relative_bias_table(rel_s_indices.squeeze(-1))

            # Concatenate the biases along the feature dimension (n_heads).
            # Shape goes from (seq_len, seq_len, n_heads) * 3 -> (seq_len, seq_len, n_heads * 3)
            stacked_biases = torch.cat([bias_q, bias_r, bias_s], dim=-1)

            # Project them back down to the desired size using the linear layer.
            # Shape goes from (seq_len, seq_len, n_heads * 3) -> (seq_len, seq_len, n_heads)
            projected_biases = self.relative_bias_projection(stacked_biases)

            # Permute and unsqueeze for broadcasting, as before -> Shape becomes (1, n_heads, seq_len, seq_len)
            relative_bias = projected_biases.permute(2, 0, 1).unsqueeze(0)

            # Permute and unsqueeze for broadcasting, as before -> Shape becomes (1, n_heads, seq_len, seq_len)
            relative_bias = F.pad(relative_bias, (1, 0, 1, 0))
        else:
            # Use the pre-computed bias during evaluation/inference
            if self.relative_bias is None:
                raise RuntimeError(
                    "Relative bias is not computed. Call `model.eval()` and then "
                    "`model._compute_and_register_bias()` before running inference."
                )
            relative_bias = self.relative_bias

        x = self.token_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [batch, num_cells + 1, d_model]

        for layer in self.encoder_layers:
            x = layer(x, relative_bias=relative_bias)

        # Separate the [CLS] token output from the board cell outputs
        # torch.split is ONNX friendly
        # x has shape [batch, 1 + num_cells, d_model] -> split it along dim 1 into a chunk of size 1 and a chunk of size num_cells.
        cls_output_with_seq_dim, board_output = torch.split(x, [1, 96], dim=1)

        # Squeeze the sequence dimension from the CLS token output for the value heads
        # Shape changes from [batch, 1, d_model] to [batch, d_model]
        cls_output = cls_output_with_seq_dim.squeeze(1)

        # Basically do Pre-Norm for the Policy head
        normalized_board_output = self.final_norm(board_output)

        # Pass the normalized board cell outputs to the policy head
        policy_logits = self.policy_head(normalized_board_output)

        z_value = self.z_value_head(cls_output)
        q_value = self.q_value_head(cls_output)

        return policy_logits, z_value, q_value

    def state_shape(self) -> Tuple[int, ...]:
        return self.seq_len, self.input_features

    def policy_shape(self) -> Tuple[int, ...]:
        return (self.seq_len * self.seq_len,)

    def value_shape(self) -> Tuple[int, ...]:
        return (3,)

    @property
    def device(self):
        return self.token_embed.weight.device

    def pre_onnx_export(self):
        self._compute_and_register_bias()

    def log_gradients(self, epoch: int):
        pi_head_grad_norm = torch.linalg.norm(self.policy_head.q_proj.weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head.network[2].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head.network[2].weight.grad).item()
        encoder_grad_norm = torch.linalg.norm(self.encoder_layers[0].attn.w_q.weight.grad).item()

        logging.info(
            f"Grad norms for epoch {epoch + 1} are PI: {pi_head_grad_norm:.6f}, Z value: {z_val_head_grad_norm:.6f}, Q value: {q_val_head_grad_norm:.6f}, Encoder: {encoder_grad_norm:.6f}")

    def debug(self):

        torch.set_printoptions(profile="full", precision=2, sci_mode=False)
        np.set_printoptions(precision=2, suppress=True, linewidth=200)

        builder = ""

        for i in range(self.relative_bias_table.weight.size(0)):
            builder += f"Relative bias for distance {i - 14}: {self.relative_bias_table.weight[i].cpu().detach().numpy()}\n"

        logging.info("Debug info:\n" + builder)


if __name__ == '__main__':
    model_config = {
        "d_model": 180,
        "n_layers": 6,
        "n_heads": 4,
        "d_ff": 512,
    }

    model = ThreePlayerChessformerBert(**model_config)
    print(f"Model created on device: {model.device}")

    batch_size = 4
    dummy_input = torch.randn(batch_size, 96, 18)

    with torch.no_grad():
        policy_logits, z_value, q_value = model(dummy_input)

    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Z-Value shape: {z_value.shape}")
    print(f"Q-Value shape: {q_value.shape}")

    print("Number of model parameters:", sum(p.numel() for p in model.parameters()))
