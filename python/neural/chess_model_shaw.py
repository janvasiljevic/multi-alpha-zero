# chess_model_shaw.py

import logging
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ShawAttentionMQA(nn.Module):
    """
    Attention mechanism based on Shaw et al. ("Self-Attention with Relative Position Representations").
    The attention score is the sum of content-to-content and content-to-position terms.
    This is faster than full DeBERTa-style disentanglement.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_head, bias=False)
        self.w_v = nn.Linear(d_model, self.d_head, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.scale = 1 / math.sqrt(self.d_head)

    # def forward(self, x: torch.Tensor, relative_pos_embeddings: torch.Tensor) -> torch.Tensor:
    #     batch_size, seq_len, _ = x.shape
    #
    #     q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    #     k = self.w_k(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
    #     v = self.w_v(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
    #
    #     # Term 1: content-to-content (c2c) - Standard attention
    #     c2c_score = torch.matmul(q, k.transpose(-1, -2))
    #
    #     # ==============================================================================
    #     # Term 2: content-to-position (c2p) - EINSUM REPLACEMENT
    #     #
    #     # Original (incompatible) einsum:
    #     # c2p_score = torch.einsum('bhid,ijd->bhij', q, relative_pos_embeddings)
    #     #
    #     # New implementation using matmul:
    #     # 1. Reshape query: [B, H, L, d_h] -> [B*H, L, d_h]
    #     q_reshaped = q.contiguous().view(batch_size * self.n_heads, seq_len, self.d_head)
    #
    #     # 2. Reshape relative embeddings: [L, L, d_h] -> [L, d_h, L] (transpose)
    #     rel_pos_reshaped = relative_pos_embeddings.permute(0, 2, 1)
    #
    #     # 3. Perform batched matrix multiplication:
    #     #    [B*H, L, d_h] @ [L, d_h, L] is not valid.
    #     #    We need to multiply each query vector q_i with all rel_pos_{i,j}.
    #     #    So, we do [B*H, L, d_h] @ [d_h, L] for each of the L positions.
    #     #    The most direct matmul is (q @ rel_pos.T)
    #     #    q: [B, H, L, d_h]
    #     #    rel_pos: [L, L, d_h] -> transpose -> [L, d_h, L]
    #     #    Let's permute `rel_pos` to [d_h, L, L] and q to [B, H, L, d_h]
    #     #    The most efficient way is a direct matmul with broadcasting.
    #
    #     # [B, H, L, d_h] @ [d_h, L] for each of the L query positions.
    #     # Let's reshape q to [B, H, L, 1, d_h] and rel_pos to [1, 1, L, d_h, L]
    #     # This is getting complicated. Let's use the simplest, most robust method:
    #     # broadcasting + element-wise multiplication + sum, which is a dot product.
    #
    #     q_expanded = q.unsqueeze(3)  # Shape: [B, H, L, 1, d_h]
    #     rel_pos_expanded = relative_pos_embeddings.unsqueeze(0).unsqueeze(0) # Shape: [1, 1, L, L, d_h]
    #
    #     # Broadcasting creates a temporary tensor of shape [B, H, L, L, d_h]
    #     # We multiply element-wise and then sum over the last dimension (d_h)
    #     # This is mathematically identical to the dot product performed by einsum.
    #     c2p_score = (q_expanded * rel_pos_expanded).sum(dim=-1) # Shape: [B, H, L, L]
    #     # ==============================================================================
    #
    #     scores = (c2c_score + c2p_score) * self.scale
    #
    #     attn_weights = F.softmax(scores, dim=-1)
    #     context = torch.matmul(attn_weights, v)
    #     context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    #
    #     return self.w_o(context)
    #
    # def forward(self, x: torch.Tensor, relative_pos_embeddings: torch.Tensor) -> torch.Tensor:
    #     batch_size, seq_len, _ = x.shape
    #     q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
    #     k = self.w_k(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
    #     v = self.w_v(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
    #     c2c_score = torch.matmul(q, k.transpose(-1, -2))
    #     q_expanded = q.unsqueeze(3)
    #     rel_pos_expanded = relative_pos_embeddings.unsqueeze(0).unsqueeze(0)
    #     c2p_score = (q_expanded * rel_pos_expanded).sum(dim=-1)
    #     scores = (c2c_score + c2p_score) * self.scale
    #     attn_weights = F.softmax(scores, dim=-1)
    #     context = torch.matmul(attn_weights, v)
    #     context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    #     return self.w_o(context)

    def forward(self, x: torch.Tensor, relative_pos_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)

        # Term 1: content-to-content (c2c) - Standard attention
        c2c_score = torch.matmul(q, k.transpose(-1, -2))

        # === CORRECTED & OPTIMIZED c2p_score CALCULATION ===
        # Original einsum: torch.einsum('bhid,ijd->bhij', q, relative_pos_embeddings)
        # q:               [B, H, L, d_h]
        # rel_pos_embeds:  [L, L, d_h]
        # Target score:    [B, H, L, L]

        # Reshape query for matmul: [B, H, L, d_h] -> [B*H, L, d_h]
        q_reshaped = q.contiguous().view(batch_size * self.n_heads, seq_len, self.d_head)
        
        # We want to compute: score[i] = q[i] @ rel_pos[i].T
        # This can be done by treating the sequence length 'L' as a batch dimension.
        # Transpose q to [L, B*H, d_h] and add a dimension for matmul.
        q_for_c2p = q_reshaped.transpose(0, 1).unsqueeze(2) # Shape: [L, B*H, 1, d_h]
        
        # Transpose relative embeddings for matmul: [L, L, d_h] -> [L, d_h, L]
        # Add a dimension to broadcast against the B*H dimension of q.
        rel_pos_T = relative_pos_embeddings.transpose(1, 2).unsqueeze(1) # Shape: [L, 1, d_h, L]
        
        # Batched matmul. Broadcasting rules handle the rest:
        # [L, B*H, 1, d_h] @ [L, 1, d_h, L] -> [L, B*H, 1, L]
        c2p_score = torch.matmul(q_for_c2p, rel_pos_T)
        
        # Reshape score back to the original layout:
        # [L, B*H, 1, L] -> [L, B, H, L] -> [B, H, L, L]
        c2p_score = c2p_score.squeeze(2).view(seq_len, batch_size, self.n_heads, seq_len).permute(1, 2, 0, 3)
        # =========================================================

        scores = (c2c_score + c2p_score) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.w_o(context)

class EncoderLayerShaw(nn.Module):
    """Pre-LN Transformer Encoder Layer using Shaw-style attention."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.attn = ShawAttentionMQA(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, relative_pos_embeddings: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), relative_pos_embeddings=relative_pos_embeddings))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# =============================================================================
# 2. THE COMPLETE SHAW-STYLE MODEL
# =============================================================================

class ThreePlayerChessformerShaw(AlphaZeroNet):
    def pre_onnx_export(self):
        self.precompute_for_inference()

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        # ... (Config is the same) ...
        self.seq_len = 96
        self.input_features = 28
        self.n_heads = n_heads
        self.n_layers = n_layers
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads

        # --- Learnable Embeddings ---
        self.token_embed = nn.Linear(self.input_features, d_model)
        self.position_embeddings = nn.Embedding(self.seq_len + 1, d_model) # ABSOLUTE
        max_coord_val = 7
        self.max_rel_dist = 2 * max_coord_val
        self.range_size = 2 * self.max_rel_dist + 1
        num_rel_embeddings = self.range_size * self.range_size
        self.relative_pos_embedding_table = nn.Embedding(num_rel_embeddings, self.d_head) # RELATIVE

        # --- Standard Layers ---
        self.embedding_layernorm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.encoder_layers = nn.ModuleList([EncoderLayerShaw(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.policy_head = AttentionPolicyHead(d_model)
        self.z_value_head = ValueHead(d_model)
        self.q_value_head = ValueHead(d_model)

        self.apply(self._init_weights)

        # --- Buffers ---
        with torch.no_grad():
            rel_indices = self._create_joint_relative_indices(all_hex_coords)
            self.register_buffer('rel_indices', rel_indices, persistent=False)

        # Buffers for pre-computed embeddings for inference
        self.register_buffer('rel_pos_embeds_inf_buffer', None, persistent=False)
        self.register_buffer('abs_pos_embeds_inf_buffer', None, persistent=False) # NEW BUFFER
        print("Model initialized. Call `precompute_for_inference()` before exporting.")


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

        if isinstance(module, (ShawAttentionMQA)):
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

    def _create_joint_relative_indices(self, coords: torch.Tensor) -> torch.Tensor:
        q_coords, r_coords = coords[:, 0], coords[:, 1]
        rel_q = q_coords.unsqueeze(1) - q_coords.unsqueeze(0)
        rel_r = r_coords.unsqueeze(1) - r_coords.unsqueeze(0)
        offset = self.max_rel_dist
        rel_q_indices = rel_q + offset
        rel_r_indices = rel_r + offset
        joint_indices = rel_q_indices * self.range_size + rel_r_indices
        return joint_indices.long()

    def precompute_for_inference(self):
        """
        Pre-computes ALL embedding lookups and DETACHES them from the computation graph
        to avoid `Gather` ops and `requires_grad` errors during ONNX export.
        """
        print("Pre-computing ALL embeddings for inference...")
        with torch.no_grad():
            # 1. Pre-compute relative position embeddings and detach
            relative_pos_embeddings = self.relative_pos_embedding_table(self.rel_indices).detach()
            rel_pos_embed_padded = F.pad(relative_pos_embeddings, (0, 0, 1, 0, 1, 0))
            self.rel_pos_embeds_inf_buffer = rel_pos_embed_padded

            # 2. Pre-compute absolute position embeddings and detach
            abs_embeds = self.position_embeddings.weight.unsqueeze(0).detach()
            self.abs_pos_embeds_inf_buffer = abs_embeds

        print("Inference buffers created successfully for both relative and absolute positions.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # --- Token Embedding ---
        x = self.token_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # --- Positional Embeddings (Training vs Inference Path) ---
        if self.training:
            # Training: Dynamic lookups for backpropagation
            relative_pos_embeddings = self.relative_pos_embedding_table(self.rel_indices)
            rel_pos_embed_padded = F.pad(relative_pos_embeddings, (0, 0, 1, 0, 1, 0))

            positions = torch.arange(0, self.seq_len + 1, device=x.device).unsqueeze(0)
            abs_pos_embeds = self.position_embeddings(positions)

        else:
            # Inference/Export: Use pre-computed buffers
            if self.rel_pos_embeds_inf_buffer is None or self.abs_pos_embeds_inf_buffer is None:
                raise RuntimeError("Inference buffers not created. Call `model.precompute_for_inference()` before running in eval mode or exporting.")
            rel_pos_embed_padded = self.rel_pos_embeds_inf_buffer
            abs_pos_embeds = self.abs_pos_embeds_inf_buffer

        # Apply absolute position embeddings
        x = x + abs_pos_embeds
        x = self.embedding_layernorm(x)
        x = self.embedding_dropout(x)

        # --- Encoder Layers ---
        for layer in self.encoder_layers:
            x = layer(x, relative_pos_embeddings=rel_pos_embed_padded)

        cls_output_with_seq_dim, board_output = torch.split(x, [1, 96], dim=1)
        cls_output = cls_output_with_seq_dim.squeeze(1)

        normalized_board_output = self.final_norm(board_output)
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

    def log_gradients(self, epoch: int):
        pi_head_grad_norm = torch.linalg.norm(self.policy_head.q_proj.weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head.network[2].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head.network[2].weight.grad).item()
        encoder_grad_norm = torch.linalg.norm(self.encoder_layers[0].attn.w_q.weight.grad).item()

        logging.info(
            f"Grad norms for epoch {epoch + 1} are PI: {pi_head_grad_norm:.6f}, Z value: {z_val_head_grad_norm:.6f}, Q value: {q_val_head_grad_norm:.6f}, Encoder: {encoder_grad_norm:.6f}")

    def _get_index_for_rel_pos(self, dq: int, dr: int) -> int:
        q_idx = dq + self.max_rel_dist
        r_idx = dr + self.max_rel_dist
        if not (0 <= q_idx < self.range_size and 0 <= r_idx < self.range_size):
            raise ValueError(f"Relative position (dq={dq}, dr={dr}) is out of bounds.")
        return q_idx * self.range_size + r_idx

    def _get_rel_pos_from_index(self, index: int) -> tuple[int, int]:
        if not (0 <= index < self.range_size * self.range_size):
            raise ValueError(f"Index {index} is out of bounds.")
        q_idx = index // self.range_size
        r_idx = index % self.range_size
        dq = q_idx - self.max_rel_dist
        dr = r_idx - self.max_rel_dist
        return dq, dr

    def debug(self):
        torch.set_printoptions(profile="full", precision=3, sci_mode=False)
        np.set_printoptions(precision=3, suppress=True, linewidth=120)

        builder = "--- Debugging Joint Relative Position Embeddings ---\n"
        builder += f"Embedding table size: {self.relative_pos_embedding_table.weight.size()}\n\n"

        pos_to_check = [(0, 0), (1, 0), (0, 1), (-1, 1), (2, 0), (0, 2), (-2, 2), (2, -1), (1, 1), (-1, 2)]
        for dq, dr in pos_to_check:
            try:
                ds = -dq - dr
                joint_index = self._get_index_for_rel_pos(dq, dr)
                embedding = self.relative_pos_embedding_table.weight[joint_index]
                embedding_np = embedding.cpu().detach().numpy()
                coord_str = f"(q={dq:>2}, r={dr:>2}, s={ds:>2})"
                builder += f"Pos: {coord_str} | Index: {joint_index:<4} | Embedding (first 5 vals): {embedding_np[:5]}\n"
            except ValueError as e:
                builder += f"Could not check (dq={dq}, dr={dr}): {e}\n"

        key_layers = {
            "rel_pos_embed_table": self.relative_pos_embedding_table.weight,
            "position_embeddings": self.position_embeddings.weight,
            "token_embed": self.token_embed.weight,
            "cls_token": self.cls_token,
            "encoder_0_wq": self.encoder_layers[0].attn.w_q.weight,
            "encoder_0_ffn_lin2": self.encoder_layers[0].ffn.linear_2.weight,
            "policy_head_q": self.policy_head.q_proj.weight,
            "z_value_head_out": self.z_value_head.network[2].weight,
        }

        stats = {"weights": {}}
        with torch.no_grad():
            for name, p in key_layers.items():
                if p is not None:
                    stats["weights"][name] = {
                        "mean": p.mean().item(), "std": p.std().item(),
                        "max": p.max().item(), "min": p.min().item(),
                        "norm": torch.linalg.norm(p).item()}

        builder += f"\n"
        builder += "{:<25} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}\n".format(
            "Layer", "Wgt Mean", "Wgt Std", "Wgt Min", "Wgt Max", "Wgt L2 Norm")
        builder += "-"*94 + "\n"
        for name, data in stats["weights"].items():
            builder += "{:<25} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>12.4f}\n".format(
                name, data['mean'], data['std'], data['min'], data['max'], data['norm'])

        logging.info(builder)

    def all_debug(self):
        torch.set_printoptions(profile="full", precision=3, sci_mode=False)
        np.set_printoptions(precision=3, suppress=True, linewidth=120)
        builder = "--- Full Joint Relative Position Embeddings ---\n"
        builder += f"Embedding table size: {self.relative_pos_embedding_table.weight.size()}\n"
        for idx in range(self.relative_pos_embedding_table.weight.size(0)):
            qrs = self._get_rel_pos_from_index(idx)
            embedding = self.relative_pos_embedding_table.weight[idx]
            embedding_np = embedding.cpu().detach().numpy()
            builder += f"QRS: {qrs} | Index: {idx:<4} | Embedding (first 5 vals): {embedding_np[:5]}\n"
        logging.info("Full debug info:\n" + builder)

        builder = "--- Absolute Position Embeddings ---\n"
        builder += f"Embedding table size: {self.position_embeddings.weight.size()}\n"
        for idx in range(self.position_embeddings.weight.size(0)):
            embedding = self.position_embeddings.weight[idx]
            embedding_np = embedding.cpu().detach().numpy()
            builder += f"Index: {idx:<4} | Embedding (first 5 vals): {embedding_np[:5]}\n"
        logging.info("Full absolute position embeddings:\n" + builder)


    @property
    def device(self):
        return self.token_embed.weight.device


if __name__ == '__main__':
    config = {
        "d_model": 5 * 48,
        "n_layers": 8,
        "n_heads": 5,
        "d_ff": 240 * 4,
        "dropout_rate": 0.1
    }

    setup_logging()

    model = ThreePlayerChessformerShaw(**config)

    model.pre_onnx_export()

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
    with torch.no_grad():
        policy_logits_inf, _, _ = model(dummy_input)

    # Note: The `pre_onnx_export` method is no longer needed because the model
    # is now fully dynamic and traceable by default. You can now directly
    # call `torch.onnx.export(model, dummy_input, ...)`
    print("\nInference check successful. Model is ready for ONNX export.")