import logging
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural.base_interface_az import AlphaZeroNet


class SelfAttention(nn.Module):
    """Standard multi-head self-attention with absolute learned positional embeddings."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # QKV projections - no bias as per Appendix A.2 of the paper
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection - has bias
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. Reshape for multi-head attention
        # (B, S, D) -> (B, H, S, D_h)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Scaled dot-product attention
        # Using F.scaled_dot_product_attention for efficiency and clarity
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # 4. Reshape and final projection
        # (B, H, S, D_h) -> (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)

        return output


class FeedForward(nn.Module):
    """Standard FeedForward layer with Mish activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.gelu(self.linear_1(x)))


class EncoderLayer(nn.Module):
    """
    A single Transformer Encoder layer with Pre-LN.
    This version uses standard self-attention (no relative embeddings).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        # Pre-LN: LayerNorm is applied before the sub-layer.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        x = x + self.attn(self.norm1(x))
        # Feed-forward block
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionPolicyHead(nn.Module):
    """
    Policy head based on scaled dot-product attention.
    It computes a logit for every possible (from_square, to_square) move pair.
    """

    def __init__(self, d_model: int, d_policy_head: int = 32):
        super().__init__()
        self.d_policy_head = d_policy_head

        # Projections to get query (from_square) and key (to_square) vectors
        self.q_proj = nn.Linear(d_model, d_policy_head)
        self.k_proj = nn.Linear(d_model, d_policy_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to queries and keys
        q = self.q_proj(x)  # (B, 96, D_policy_head)
        k = self.k_proj(x)  # (B, 96, D_policy_head)

        # Calculate logits for all (from, to) pairs
        # (B, 96, D_head) @ (B, D_head, 96) -> (B, 96, 96)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_policy_head)

        # Flatten logits to match the expected output shape
        # (B, 96, 96) -> (B, 96 * 96)
        logits_flat = logits.view(logits.size(0), -1)

        return logits_flat


class ValueHead(nn.Module):
    """
    Value head for a 3-player game.
    It pools the transformer output and predicts a value for each player.
    """

    def __init__(self, d_model: int, d_hidden: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),  # 3 players
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pool the features from all 96 squares. Mean pooling is a simple choice.
        # (B, 96, D_model) -> (B, D_model)
        pooled_output = x.mean(dim=1)
        return self.network(pooled_output)


class ThreePlayerChessformerBasic(AlphaZeroNet):
    """
    A Transformer-based model for a 3-player, 96-square chess variant,
    adhering to the AlphaZeroNet interface.
    """

    def pre_onnx_export(self):
        pass

    def debug(self):
        pass

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()

        # --- Game and Model Config ---
        self.seq_len = 96
        self.input_features = 18
        self.d_model = d_model

        # --- Model Architecture ---
        # Input embedding layer
        self.token_embed = nn.Linear(self.input_features, d_model)

        # Absolute positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(self.seq_len, d_model))

        # Transformer Encoder stack
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        # Final LayerNorm for the encoder output
        self.final_norm = nn.LayerNorm(d_model)

        # Policy Head
        self.policy_head = AttentionPolicyHead(d_model)

        # Primary value head: learns the final game outcome `z`
        self.z_value_head = ValueHead(d_model)
        # Auxiliary value head: learns the MCTS search value `q`
        self.q_value_head = ValueHead(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 96, 18).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - policy_logits: (batch, 96, 96) logits for (from, to) moves.
            - value: (batch, 3) estimated scores for each of the 3 players.
        """
        # 1. Generate token embeddings and add positional info
        x = self.token_embed(x)
        x = x + self.positional_embeddings

        # 2. Pass through Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # 3. Apply final normalization
        encoder_output = self.final_norm(x)

        # 4. Compute head outputs
        policy_logits = self.policy_head(encoder_output)

        z_value = self.z_value_head(encoder_output)

        q_value = self.q_value_head(encoder_output)

        return policy_logits, z_value, q_value

    def state_shape(self) -> Tuple[int, ...]:
        return self.seq_len, self.input_features

    def policy_shape(self) -> Tuple[int, ...]:
        return (self.seq_len * self.seq_len,)

    def value_shape(self) -> Tuple[int, ...]:
        return (3,)

    @property
    def device(self):
        """Return the device of the model's parameters."""
        return self.positional_embeddings.device

    def log_gradients(self, epoch: int):
        pi_head_grad_norm = torch.linalg.norm(self.policy_head.q_proj.weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head.network[2].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head.network[2].weight.grad).item()
        encoder_grad_norm = torch.linalg.norm(self.encoder_layers[0].attn.w_q.weight.grad).item()

        logging.info(
            f"Grad norms for epoch {epoch + 1} are PI: {pi_head_grad_norm:.6f}, Z value: {z_val_head_grad_norm:.6f}, Q value: {q_val_head_grad_norm:.6f}, Encoder: {encoder_grad_norm:.6f}")


if __name__ == '__main__':
    config = {
        "d_model": 180,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
    }

    model = ThreePlayerChessformerBasic(**config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration: {config}")
    print(f"Estimated Parameters: {num_params / 1e6:.2f}M\n")

    batch_size = 4
    dummy_input = torch.randn(batch_size, model.seq_len, model.input_features)

    with torch.no_grad():
        policy_logits, value = model(dummy_input)

    print(f"Input Shape:          {dummy_input.shape}")
    print(f"Policy Logits Shape:  {policy_logits.shape}  (Expected: [{batch_size}, 96, 96])")
    print(f"Value Shape:          {value.shape}          (Expected: [{batch_size}, 3])")

    print(f"state_shape():  {model.state_shape()} (Expected: (96, 18))")
    print(f"policy_shape(): {model.policy_shape()} (Expected: (96, 96))")
    print(f"value_shape():  {model.value_shape()}  (Expected: (3,))")
