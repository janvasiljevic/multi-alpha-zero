import logging
from typing import Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural.base_interface_az import AlphaZeroNet
from neural.chess_model_relative_coords import all_hex_coords


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-4): # NOTE: Increased default epsilon
        """
        A robust implementation of RMSNorm, designed for NATIVE STABILITY in FP16.
        It avoids upcasting to float32 by using a larger, FP16-safe epsilon.
        - d_model: The feature dimension of the input tensor.
        - eps: A value to prevent division by zero. Must be large enough to be
               meaningful in FP16. 1e-4 is a safe choice.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This forward pass contains NO data type casting. It will run in whatever
        # dtype it receives (e.g., float16).

        # The key is that self.eps (1e-4) is large enough to prevent
        # (variance + self.eps) from becoming zero, even if variance underflows.
        variance = x.pow(2).mean(-1, keepdim=True)
        rsqrt_val = torch.rsqrt(variance + self.eps)

        return self.weight * x * rsqrt_val

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
        self.q_proj = nn.Linear(d_model, d_policy_head)
        self.k_proj = nn.Linear(d_model, d_policy_head)

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
        pooled_output = x.mean(dim=1)
        return self.network(pooled_output)

# --- New T5-Style Relative Attention Module ---
class RelativeBiasAttention(nn.Module):
    """
    T5-style self-attention with a learnable relative position bias.
    This is extremely fast and hardware-accelerator friendly.
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

        # Po testih zgleda da ne rabim q_norm in k_norm, ce imam preln
        # + velik hitrje se uci (240k vs 400k inference na GH200)
        # Puscam tuki za odcommentat ker zej testiram za stare test runne
        # self.q_norm = RMSNorm(self.d_head)
        # self.k_norm = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # q = self.q_norm(q)
        # k = self.k_norm(k)
        #
        # 1. Standard content scores
        scores = torch.matmul(q, k.transpose(-1, -2))

        # 2. Add the pre-computed relative position bias
        # The bias tensor has shape (1, n_heads, seq_len, seq_len)
        # and will be broadcasted across the batch dimension.
        scores = (scores / math.sqrt(self.d_head)) + relative_bias

        # Idk if this helps, but why not
        scores = torch.clamp(scores, -10, 10)

        # content_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        #
        # max_content_val = torch.max(torch.abs(content_scores)).item()
        # max_bias_val = torch.max(torch.abs(relative_bias)).item()
        #
        # scores = content_scores + relative_bias
        # max_total_val = torch.max(torch.abs(scores)).item()
        #
        # if max_total_val > 10.0:
        #     print(f"[DEBUG] High score detected! Max Content: {max_content_val:.2f}, Max Bias: {max_bias_val:.2f}, Max Total: {max_total_val:.2f}")

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.w_o(context)


# --- EncoderLayer now passes the bias tensor ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = RelativeBiasAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    # def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
    #     # Pass the pre-computed bias tensor down to the attention layer
    #     x = self.norm1(x + self.attn(x, relative_bias=relative_bias))
    #     x = self.norm2(x + self.ffn(x))
    #     return x

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        # Pre-LN implementation
        x = x + self.attn(self.norm1(x), relative_bias=relative_bias)
        x = x + self.ffn(self.norm2(x))
        return x

class ThreePlayerChessformerRelative(AlphaZeroNet):
    def pre_onnx_export(self):
        pass

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()
        self.seq_len = 96
        self.input_features = 18
        self.n_heads = n_heads  # Store n_heads

        self.token_embed = nn.Linear(self.input_features, d_model)

        # --- CORRECTED T5-style Bias Logic ---
        max_coord_val = 7
        max_rel_dist = 2 * max_coord_val
        num_rel_embeddings = 2 * max_rel_dist + 1

        # This table remains a learnable module
        self.relative_bias_table = nn.Embedding(num_rel_embeddings, self.n_heads)

        # --- NEW: Projection layer for combining biases ---
        # This replaces the simple summation. It learns the best way to combine the
        # three relative coordinate biases, providing more control and stability.
        # Input: 3 * n_heads (from concatenating q, r, s biases)
        # Output: n_heads (the final bias for each head)
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

    def _init_weights(self, module):
        """
        Initializes weights of the model for better stability.
        This is crucial for preventing large initial values that can cause
        overflow in FP16.
        """
        if isinstance(module, nn.Linear):
            # Use a normal distribution with a small standard deviation for linear layers.
            # This prevents activations from exploding early in training.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # This is especially important for your `relative_bias_table`.
            # Initializing its weights to be small and centered at zero ensures
            # that the initial relative biases don't dominate the attention scores.
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard practice for LayerNorm
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    # _create_relative_indices remains the same...
    def _create_relative_indices(self, coords: torch.Tensor, max_rel_dist: int) -> torch.Tensor:
        # ... (implementation is correct)
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
        # --- Dynamically compute the relative bias ---
        rel_q_indices, rel_r_indices, rel_s_indices = self.rel_indices.chunk(3, dim=-1)

        # Look up bias from the learnable table using the pre-computed indices
        bias_q = self.relative_bias_table(rel_q_indices.squeeze(-1))
        bias_r = self.relative_bias_table(rel_r_indices.squeeze(-1))
        bias_s = self.relative_bias_table(rel_s_indices.squeeze(-1))


        # --- MODIFIED: Use concatenation and projection instead of summation ---
        # 1. Concatenate the biases along the feature dimension (n_heads).
        # Shape goes from (seq_len, seq_len, n_heads) * 3
        # to (seq_len, seq_len, n_heads * 3)
        stacked_biases = torch.cat([bias_q, bias_r, bias_s], dim=-1)

        # 2. Project them back down to the desired size using the linear layer.
        # Shape goes from (seq_len, seq_len, n_heads * 3)
        # to (seq_len, seq_len, n_heads)
        projected_biases = self.relative_bias_projection(stacked_biases)

        # 3. Permute and unsqueeze for broadcasting, as before.
        # Shape becomes (1, n_heads, seq_len, seq_len)
        relative_bias = projected_biases.permute(2, 0, 1).unsqueeze(0)

        # --- Model forward pass ---
        x = self.token_embed(x)

        for layer in self.encoder_layers:
            # Pass the newly computed bias to each layer
            x = layer(x, relative_bias=relative_bias)

        encoder_output = self.final_norm(x)
        policy_logits = self.policy_head(encoder_output)
        z_value = self.z_value_head(encoder_output)
        q_value = self.q_value_head(encoder_output)
        return policy_logits, z_value, q_value

#
# # --- Main Model: Pre-computes the BIAS tensor ---
# class ThreePlayerChessformerRelative(AlphaZeroNet):
#
#
#     def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int):
#         super().__init__()
#         self.seq_len = 96
#         self.input_features = 18
#
#         self.token_embed = nn.Linear(self.input_features, d_model)
#
#         # --- T5-style Bias Logic ---
#         # We only need one embedding table for each axis, which learns a scalar bias.
#         max_coord_val = 7
#         max_rel_dist = 2 * max_coord_val
#         num_rel_embeddings = 2 * max_rel_dist + 1
#
#         # This single table learns a different bias for each head.
#         self.relative_bias_table = nn.Embedding(num_rel_embeddings, n_heads)
#         # print(self.relative_bias_table.weight.shape)  # Should be (29, n_heads)
#
#         # Pre-compute the final bias tensor
#         with torch.no_grad():
#             rel_indices = self._create_relative_indices(all_hex_coords, max_rel_dist)
#             # print the whole vector - set torch print options
#             # torch.set_printoptions(profile="full")
#             # print(rel_indices[:5, :5, :])  # Print a small slice for brevity
#
#             rel_q_indices, rel_r_indices, rel_s_indices = rel_indices.chunk(3, dim=-1)
#
#             # print(rel_q_indices[:5, :5])  # Print a small slice for brevity
#             # print(rel_r_indices[:5, :5])  # Print a small slice for brev
#             # print(rel_s_indices[:5, :5])  # Print a small slice for brevity
#
#             # Look up bias from each table and add them up
#             bias_q = self.relative_bias_table(rel_q_indices.squeeze(-1))
#             bias_r = self.relative_bias_table(rel_r_indices.squeeze(-1))
#             bias_s = self.relative_bias_table(rel_s_indices.squeeze(-1))
#
#
#             # Final bias is the sum, permuted to (n_heads, seq_len, seq_len)
#             relative_bias = (bias_q + bias_r + bias_s).permute(2, 0, 1)
#
#         # print(relative_bias.shape)  # Should be (n_heads, 96, 96)
#
#         # Register as a buffer and add a batch dimension for broadcasting
#         self.register_buffer('relative_bias', relative_bias.unsqueeze(0))
#
#         # --- Standard Model Layers ---
#         self.encoder_layers = nn.ModuleList(
#             [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
#         )
#         self.final_norm = nn.LayerNorm(d_model)
#         # ... (Policy and Value Heads are unchanged) ...
#         self.policy_head = AttentionPolicyHead(d_model)
#         self.z_value_head = ValueHead(d_model)
#         self.q_value_head = ValueHead(d_model)
#
#     def _create_relative_indices(self, coords: torch.Tensor, max_rel_dist: int) -> torch.Tensor:
#         # This function is unchanged
#         q_coords, r_coords, s_coords = coords.chunk(3, dim=-1)
#         q_coords, r_coords, s_coords = q_coords.squeeze(-1), r_coords.squeeze(-1), s_coords.squeeze(-1)
#         rel_q = q_coords.unsqueeze(1) - q_coords.unsqueeze(0)
#         rel_r = r_coords.unsqueeze(1) - r_coords.unsqueeze(0)
#         rel_s = s_coords.unsqueeze(1) - s_coords.unsqueeze(0)
#
#         offset = max_rel_dist
#         rel_q_indices = rel_q + offset
#         rel_r_indices = rel_r + offset
#         rel_s_indices = rel_s + offset
#
#         return torch.stack([rel_q_indices, rel_r_indices, rel_s_indices], dim=-1).long()
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         x = self.token_embed(x)
#
#         for layer in self.encoder_layers:
#             x = layer(x, relative_bias=self.relative_bias)
#
#         encoder_output = self.final_norm(x)
#         policy_logits = self.policy_head(encoder_output)
#         z_value = self.z_value_head(encoder_output)
#         q_value = self.q_value_head(encoder_output)
#         return policy_logits, z_value, q_value

    def state_shape(self) -> Tuple[int, ...]:
        return self.seq_len, self.input_features

    def policy_shape(self) -> Tuple[int, ...]:
        return (self.seq_len * self.seq_len,)

    def value_shape(self) -> Tuple[int, ...]:
        return (3,)

    @property
    def device(self):
        return self.token_embed.weight.device

    def log_gradients(self, epoch: int):
        # TODO
        pi_head_grad_norm = torch.linalg.norm(self.policy_head.q_proj.weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head.network[2].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head.network[2].weight.grad).item()
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
    model_config = {
        "d_model": 180,
        "n_layers": 6,
        "n_heads": 6,
        "d_ff": 512,
    }

    model = ThreePlayerChessformerRelative(**model_config)
    print(f"Model created on device: {model.device}")

    batch_size = 4
    dummy_input = torch.randn(batch_size, 96, 18)  # (B, seq_len, input_features)

    with torch.no_grad():
        policy_logits, z_value, q_value = model(dummy_input)

    print(f"Policy logits shape: {policy_logits.shape}")  # Expected: (4, 9216)
    print(f"Z-Value shape: {z_value.shape}")  # Expected: (4, 3)
    print(f"Q-Value shape: {q_value.shape}")  # Expected: (4, 3)

    print("Number of model parameters:", sum(p.numel() for p in model.parameters()))

    # print(f"\nShape of the pre-computed relative index buffer: {model.rel_indices.shape}")  # Expected: (96, 96, 3)
    # print(f"Data type of the buffer: {model.rel_indices.dtype}")  # Expected: torch.int64
