import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from logging_setup import setup_logging
from neural.base_interface_az import AlphaZeroNet
from neural.chess_model_relative_coords import all_hex_coords


class FeedForward(nn.Module):
    """Standard FeedForward layer with GELU activation and optional dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(self.gelu(self.linear_1(x))))

class AttentionPolicyHead(nn.Module):
    """Your original attention-based policy head."""
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
    """Standard value head with a Tanh activation for the [-1, 1] range."""
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
    """T5-style self-attention with Multi-Query Attention (MQA)."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, self.d_head, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = (scores / math.sqrt(self.d_head)) + relative_bias
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(context)

class EncoderLayer(nn.Module):
    """Pre-LN Transformer Encoder Layer with dropout."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.attn = RelativeBiasMQA(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, relative_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), relative_bias=relative_bias))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# =============================================================================
# 2. THE COMPLETE V2 MODEL
# =============================================================================

class ThreePlayerChessformerBertV2(AlphaZeroNet):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.seq_len = 96
        self.input_features = 28
        self.n_heads = n_heads
        self.n_layers = n_layers

        # --- TOKEN AND ABSOLUTE POSITION EMBEDDINGS ---
        self.token_embed = nn.Linear(self.input_features, d_model)
        self.position_embeddings = nn.Embedding(self.seq_len + 1, d_model) # +1 for CLS
        self.embedding_layernorm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # --- JOINT RELATIVE POSITION EMBEDDINGS ---
        max_coord_val = 7
        self.max_rel_dist = 2 * max_coord_val
        self.range_size = 2 * self.max_rel_dist + 1
        num_rel_embeddings = self.range_size * self.range_size
        self.relative_bias_table = nn.Embedding(num_rel_embeddings, self.n_heads)

        # --- STANDARD MODEL LAYERS ---
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.policy_head = AttentionPolicyHead(d_model)
        self.z_value_head = ValueHead(d_model)
        self.q_value_head = ValueHead(d_model)

        self.apply(self._init_weights)
        self.register_buffer('relative_bias', None, persistent=True)
        # Note: rel_indices buffer is created by calling register_hex_coords()

        with torch.no_grad():
            if not hasattr(self, 'rel_indices'):
                with torch.no_grad():
                    rel_indices = self._create_joint_relative_indices(all_hex_coords)
                    self.register_buffer('rel_indices', rel_indices)
            print("Hex coordinates and relative indices have been registered.")

    def _init_weights(self, module):
        """Initializes weights of the model for better stability."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

        # Special initialization for residual projections
        if isinstance(module, RelativeBiasMQA):
            nn.init.normal_(module.w_o.weight, std=0.02 / math.sqrt(2 * self.n_layers))
        if isinstance(module, FeedForward):
            nn.init.normal_(module.linear_2.weight, std=0.02 / math.sqrt(2 * self.n_layers))

        # Zero-out the final layers of heads
        if isinstance(module, ValueHead):
            nn.init.constant_(module.network[2].weight, 0)
            if module.network[2].bias is not None:
                nn.init.constant_(module.network[2].bias, 0)
        if isinstance(module, AttentionPolicyHead):
            nn.init.constant_(module.q_proj.weight, 0)
            nn.init.constant_(module.k_proj.weight, 0)

    def _compute_and_register_bias(self):
        """Pre-computes the relative attention bias for inference."""
        if not hasattr(self, 'rel_indices'):
            raise RuntimeError("Hex coordinates not registered. Call `model.register_hex_coords()` first.")
        with torch.no_grad():
            biases = self.relative_bias_table(self.rel_indices)
            bias = biases.permute(2, 0, 1).unsqueeze(0)
            bias = F.pad(bias, (1, 0, 1, 0))
            self.register_buffer('relative_bias', bias)
        print("Joint relative position bias has been pre-computed for inference.")

    def _create_joint_relative_indices(self, coords: torch.Tensor) -> torch.Tensor:
        """Creates a 2D matrix of unique IDs for each relative position vector (dq, dr)."""
        q_coords, r_coords = coords[:, 0], coords[:, 1]
        rel_q = q_coords.unsqueeze(1) - q_coords.unsqueeze(0)
        rel_r = r_coords.unsqueeze(1) - r_coords.unsqueeze(0)
        offset = self.max_rel_dist
        rel_q_indices = rel_q + offset
        rel_r_indices = rel_r + offset
        joint_indices = rel_q_indices * self.range_size + rel_r_indices
        return joint_indices.long()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        if self.training:
            if not hasattr(self, 'rel_indices'):
                raise RuntimeError("Hex coordinates not registered. Call `model.register_hex_coords()` first.")
            biases = self.relative_bias_table(self.rel_indices)
            relative_bias = biases.permute(2, 0, 1).unsqueeze(0)
            relative_bias = F.pad(relative_bias, (1, 0, 1, 0))
        else:
            if self.relative_bias is None:
                raise RuntimeError("Relative bias not computed for inference. Call `model.pre_onnx_export()` first.")
            relative_bias = self.relative_bias

        x = self.token_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        positions = torch.arange(0, self.seq_len + 1, device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(positions)
        x = self.embedding_layernorm(x)
        x = self.embedding_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, relative_bias=relative_bias)

        cls_output_with_seq_dim, board_output = torch.split(x, [1, 96], dim=1)
        cls_output = cls_output_with_seq_dim.squeeze(1)

        normalized_board_output = self.final_norm(board_output)
        policy_logits = self.policy_head(normalized_board_output)

        z_value = self.z_value_head(cls_output)
        q_value = self.q_value_head(cls_output)

        return policy_logits, z_value, q_value

    def pre_onnx_export(self):
        self._compute_and_register_bias()

    def state_shape(self) -> Tuple[int, ...]:
        return self.seq_len, self.input_features

    def policy_shape(self) -> Tuple[int, ...]:
        return (self.seq_len * self.seq_len,)

    def value_shape(self) -> Tuple[int, ...]:
        return (3,)

    def log_gradients(self, epoch: int):
        pi_head_grad_norm = torch.linalg.norm(self.policy_head.q_proj.weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head.network[2].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head.network[2].weight.grad).item()
        encoder_grad_norm = torch.linalg.norm(self.encoder_layers[0].attn.w_q.weight.grad).item()

        logging.info(
            f"Grad norms for epoch {epoch + 1} are PI: {pi_head_grad_norm:.6f}, Z value: {z_val_head_grad_norm:.6f}, Q value: {q_val_head_grad_norm:.6f}, Encoder: {encoder_grad_norm:.6f}")

    def _get_index_for_rel_pos(self, dq: int, dr: int) -> int:
        """Helper function to convert a relative (dq, dr) vector to its joint index."""
        q_idx = dq + self.max_rel_dist
        r_idx = dr + self.max_rel_dist

        if not (0 <= q_idx < self.range_size and 0 <= r_idx < self.range_size):
            raise ValueError(f"Relative position (dq={dq}, dr={dr}) is out of bounds.")

        return q_idx * self.range_size + r_idx

    def _get_rel_pos_from_index(self, index: int) -> tuple[int, int]:
        """
        Helper function to convert a joint index back to a relative (dq, dr) vector.
        This is the reverse of _get_index_for_rel_pos.
        """
        if not (0 <= index < self.range_size * self.range_size):
            raise ValueError(f"Index {index} is out of bounds.")

        q_idx = index // self.range_size
        r_idx = index % self.range_size

        dq = q_idx - self.max_rel_dist
        dr = r_idx - self.max_rel_dist

        return dq, dr

    def debug(self):
        """
        Prints the learned relative position embeddings for a selection of
        meaningful geometric relationships on the hex board.
        """
        torch.set_printoptions(profile="full", precision=3, sci_mode=False)
        np.set_printoptions(precision=3, suppress=True, linewidth=120)

        builder = "--- Debugging Joint Relative Position Embeddings ---\n"
        builder += f"Embedding table size: {self.relative_bias_table.weight.size()}\n\n"

        pos_to_check = [
            # To itself
            (0, 0),
            # Adjacent hexes
            (1, 0),
            (0, 1),
            (-1, 1),
            # Straight line moves
            (2, 0),
            (0, 2),
            (-2, 2),
            # Knight
            (2, -1),
            (1, 1),
            (-1, 2),
        ]

        for dq, dr in pos_to_check:
            try:
                # The sum of RELATIVE differences (dq+dr+ds) is always 0,
                # even if the sum of ABSOLUTE coordinates (q+r+s) is -1.
                ds = -dq - dr
                joint_index = self._get_index_for_rel_pos(dq, dr)

                embedding = self.relative_bias_table.weight[joint_index]
                embedding_np = embedding.cpu().detach().numpy()

                coord_str = f"(q={dq:>2}, r={dr:>2}, s={ds:>2})"
                builder += f"Pos: {coord_str} | Index: {joint_index:<4} | Embedding: {embedding_np}\n"

            # Should never happen TBH, but just in case :)
            except ValueError as e:
                builder += f"Could not check (dq={dq}, dr={dr}): {e}\n"


        key_layers = {
            "relative_bias_table": self.relative_bias_table.weight,
            "position_embeddings": self.position_embeddings.weight,
            "token_embed": self.token_embed.weight,
            "cls_token": self.cls_token,
            "encoder_0_wq": self.encoder_layers[0].attn.w_q.weight,
            "encoder_0_ffn_lin2": self.encoder_layers[0].ffn.linear_2.weight,
            "policy_head_q": self.policy_head.q_proj.weight,
            "z_value_head_out": self.z_value_head.network[2].weight,
        }

        stats = {
            "weights": {},
            "gradients": {}
        }


        with torch.no_grad():
            for name, p in key_layers.items():
                if p is not None:
                    # Log weight statistics
                    stats["weights"][name] = {
                        "mean": p.mean().item(),
                        "std": p.std().item(),
                        "max": p.max().item(),
                        "min": p.min().item(),
                        "norm": torch.linalg.norm(p).item()
                    }

        builder += f"\n"
        builder += "{:<25} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}\n".format(
            "Layer", "Wgt Mean", "Wgt Std", "Wgt Min", "Wgt Max", "Wgt L2 Norm"
        )
        builder += "-"*94 + "\n"
        for name, data in stats["weights"].items():
            builder += "{:<25} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>12.4f}\n".format(
                name, data['mean'], data['std'], data['min'], data['max'], data['norm']
            )

        logging.info(builder)

    def all_debug(self):
        # Show all positional embeddings and then even absolute
        torch.set_printoptions(profile="full", precision=3, sci_mode=False)
        np.set_printoptions(precision=3, suppress=True, linewidth=120)
        builder = "--- Full Joint Relative Position Embeddings ---\n"
        builder += f"Embedding table size: {self.relative_bias_table.weight.size()}\n"
        for idx in range(self.relative_bias_table.weight.size(0)):
            qrs = self._get_rel_pos_from_index(idx)
            embedding = self.relative_bias_table.weight[idx]
            embedding_np = embedding.cpu().detach().numpy()
            builder += f"QRS: {qrs} | Index: {idx:<4} | Embedding: {embedding_np}\n"
        logging.info("Full debug info:\n" + builder)

        # absolute positions
        builder = "--- Absolute Position Embeddings ---\n"
        builder += f"Embedding table size: {self.position_embeddings.weight.size()}\n"
        for idx in range(self.position_embeddings.weight.size(0)):
            embedding = self.position_embeddings.weight[idx]
            embedding_np = embedding.cpu().detach().numpy()
            builder += f"Index: {idx:<4} | Embedding: {embedding_np}\n"
        logging.info("Full absolute position embeddings:\n" + builder)


    @property
    def device(self):
        return self.token_embed.weight.device


if __name__ == '__main__':
    config = {
        "d_model": 5 * 48,
        "n_layers": 8,
        "n_heads": 5,
        "d_ff": 240 * 4,      # 4 * d_model
        "dropout_rate": 0.1
    }

    setup_logging()

    model = ThreePlayerChessformerBertV2(**config)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Configuration: {config}")
    print(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")

    model.train()

    batch_size = 4
    dummy_input = torch.randn(batch_size, model.seq_len, model.input_features)

    policy_logits, z_value, q_value = model(dummy_input)

    loss = policy_logits.sum() + z_value.sum() + q_value.sum()
    loss.backward()

    print(f"\n--- Forward Pass Output Shapes ---")
    print(f"Policy Logits Shape: {policy_logits.shape}") # Expected: [4, 9216]
    print(f"Z-Value Shape:       {z_value.shape}")       # Expected: [4, 3]
    print(f"Q-Value Shape:       {q_value.shape}")       # Expected: [4, 3]


    model.log_gradients(epoch=0)

    model.debug()

    model.eval()
    model.pre_onnx_export()

    with torch.no_grad():
        policy_logits_inf, _, _ = model(dummy_input)

    print("\nInference check successful.")
