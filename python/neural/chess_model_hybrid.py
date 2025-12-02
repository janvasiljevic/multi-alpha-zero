import logging
from typing import Tuple

import math
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


class FactoredPolicyHead(nn.Module):
    """
    A policy head that factors the action space into 'from' and 'to' logits.
    It learns two separate scores for each square: one for being a source of a move,
    and one for being a destination. The final move logit is the sum of these scores.
    This provides a strong inductive bias and stabilizes training on large action spaces.
    """

    def __init__(self, d_model: int, debug_mode: bool = False):
        super().__init__()
        self.from_head = nn.Linear(d_model, 1)
        self.to_head = nn.Linear(d_model, 1)
        self.debug_mode = debug_mode

    # Return either combined logits or detailed components for debugging
    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x has shape [B, seq_len, d_model], where seq_len is 96
        from_logits = self.from_head(x)  # Shape: [B, 96, 1]
        to_logits = self.to_head(x)  # Shape: [B, 96, 1]

        # Use broadcasting to create the outer-product sum:
        # [B, 96, 1] + [B, 1, 96] -> [B, 96, 96]
        to_logits_transposed = to_logits.transpose(1, 2)
        combined_logits = from_logits + to_logits_transposed

        if self.debug_mode:
            return combined_logits.view(combined_logits.size(0), -1), from_logits, to_logits
        else:
            # Flatten the result to match the expected single policy vector output
            return combined_logits.view(combined_logits.size(0), -1)


class PaperValueHead(nn.Module):
    """
    A stable value head using Global Average Pooling.
    1. Project all board tokens to a smaller dimension (`d_value`).
    2. Aggregate token representations by taking the mean across the sequence.
    3. Use a small MLP on the aggregated vector to predict the final value.
    This is much more stable than flattening the entire sequence.
    """

    def __init__(self, d_model: int, seq_len: int, d_value: int = 32, d_hidden: int = 128):
        super().__init__()
        # Note: seq_len is unused but kept for API consistency if needed later
        self.project_down = nn.Linear(d_model, d_value)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_value, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 3),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, seq_len, d_model]
        projected = self.project_down(x)  # -> [B, seq_len, d_value]
        # Aggregate across the sequence dimension (dim=1)
        pooled = torch.mean(projected, dim=1)  # -> [B, d_value]
        return self.mlp(pooled)


class MaterialHead(nn.Module):
    """
    A stable material head using Global Average Pooling.
    Architecturally identical to PaperValueHead but with a Sigmoid activation.
    """

    def __init__(self, d_model: int, seq_len: int, d_value: int = 32, d_hidden: int = 128, aux_features: int = 3):
        super().__init__()
        self.project_down = nn.Linear(d_model, d_value)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_value, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, aux_features),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, seq_len, d_model]
        projected = self.project_down(x)  # -> [B, seq_len, d_value]
        # Aggregate across the sequence dimension (dim=1)
        pooled = torch.mean(projected, dim=1)  # -> [B, d_value]
        return self.mlp(pooled)


class ShawAttentionMQA(nn.Module):
    """Attention mechanism based on Shaw et al. (Unchanged)."""

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

    def forward(self, x: torch.Tensor, relative_pos_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, 1, self.d_head).transpose(1, 2)
        c2c_score = torch.matmul(q, k.transpose(-1, -2))

        q_reshaped = q.contiguous().view(batch_size * self.n_heads, seq_len, self.d_head)
        q_for_c2p = q_reshaped.transpose(0, 1).unsqueeze(2)
        rel_pos_T = relative_pos_embeddings.transpose(1, 2).unsqueeze(1)
        c2p_score = torch.matmul(q_for_c2p, rel_pos_T)
        c2p_score = c2p_score.squeeze(2).view(seq_len, batch_size, self.n_heads, seq_len).permute(1, 2, 0, 3)

        scores = (c2c_score + c2p_score) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.w_o(context)


class EncoderLayerShaw(nn.Module):
    """Pre-LN Transformer Encoder Layer using Shaw-style attention. (Unchanged)"""

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


class ThreePlayerHybrid(AlphaZeroNet):
    def pre_onnx_export(self):
        self.precompute_for_inference()

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1,
                 input_features: int = 28, aux_features: int = 3, debug_mode: bool = False):
        super().__init__()
        self.seq_len = 96
        self.input_features = input_features
        self.n_heads = n_heads
        self.n_layers = n_layers
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads

        # --- Learnable Embeddings ---
        self.token_embed = nn.Linear(self.input_features, d_model)
        self.position_embeddings = nn.Embedding(self.seq_len, d_model)  # No CLS token
        max_coord_val = 7
        self.max_rel_dist = 2 * max_coord_val
        self.debug_mode = debug_mode
        self.range_size = 2 * self.max_rel_dist + 1
        num_rel_embeddings = self.range_size * self.range_size
        self.relative_pos_embedding_table = nn.Embedding(num_rel_embeddings, self.d_head)

        self.embedding_layernorm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayerShaw(d_model, n_heads, d_ff, dropout_rate) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.policy_head = FactoredPolicyHead(d_model, debug_mode=debug_mode)
        self.z_value_head = PaperValueHead(d_model, self.seq_len)
        self.q_value_head = MaterialHead(d_model, self.seq_len, aux_features=aux_features)

        self.apply(self._init_weights)

        # --- Buffers ---
        with torch.no_grad():
            rel_indices = self._create_joint_relative_indices(all_hex_coords)
            self.register_buffer('rel_indices', rel_indices, persistent=False)
        self.register_buffer('rel_pos_embeds_inf_buffer', None, persistent=False)
        self.register_buffer('abs_pos_embeds_inf_buffer', None, persistent=False)
        print("Model initialized with FactoredPolicyHead and PaperValueHead.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

        if isinstance(module, ShawAttentionMQA):
            nn.init.normal_(module.w_o.weight, std=0.02 / math.sqrt(2 * self.n_layers))
        if isinstance(module, FeedForward):
            nn.init.normal_(module.linear_2.weight, std=0.02 / math.sqrt(2 * self.n_layers))

        # Initializations for the new heads.
        # The final linear layer is now at index 3 of the mlp Sequential block.
        if isinstance(module, PaperValueHead):
            final_layer = module.mlp[3]
            nn.init.constant_(final_layer.weight, 0)
            if final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, 0)

        # Add the corresponding initialization for the MaterialHead.
        if isinstance(module, MaterialHead):
            final_layer = module.mlp[3]
            nn.init.constant_(final_layer.weight, 0)
            if final_layer.bias is not None:
                nn.init.constant_(final_layer.bias, 0)

        if isinstance(module, FactoredPolicyHead):
            nn.init.constant_(module.from_head.weight, 0)
            nn.init.constant_(module.to_head.weight, 0)
            if module.from_head.bias is not None:
                nn.init.constant_(module.from_head.bias, 0)
            if module.to_head.bias is not None:
                nn.init.constant_(module.to_head.bias, 0)

    def _create_joint_relative_indices(self, coords: torch.Tensor) -> torch.Tensor:
        q_coords, r_coords = coords[:, 0], coords[:, 1]
        rel_q = q_coords.unsqueeze(1) - q_coords.unsqueeze(0)
        rel_r = r_coords.unsqueeze(1) - r_coords.unsqueeze(0)
        offset = self.max_rel_dist
        joint_indices = (rel_q + offset) * self.range_size + (rel_r + offset)
        return joint_indices.long()

    def set_debug_mode(self, debug: bool):
        self.debug_mode = debug
        self.policy_head.debug_mode = debug

    def precompute_for_inference(self):
        with torch.no_grad():
            relative_pos_embeddings = self.relative_pos_embedding_table(self.rel_indices).detach()
            self.rel_pos_embeds_inf_buffer = relative_pos_embeddings
            abs_embeds = self.position_embeddings.weight.unsqueeze(0).detach()
            self.abs_pos_embeds_inf_buffer = abs_embeds

        logging.info("Inference buffers created successfully.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Token Embedding
        x = self.token_embed(x)

        # Positional Embeddings
        if self.training:
            relative_pos_embeddings = self.relative_pos_embedding_table(self.rel_indices)
            positions = torch.arange(0, self.seq_len, device=x.device).unsqueeze(0)
            abs_pos_embeds = self.position_embeddings(positions)
        else:
            if self.rel_pos_embeds_inf_buffer is None or self.abs_pos_embeds_inf_buffer is None:
                raise RuntimeError("Call `model.precompute_for_inference()` before eval/export.")
            relative_pos_embeddings = self.rel_pos_embeds_inf_buffer
            abs_pos_embeds = self.abs_pos_embeds_inf_buffer

        x = x + abs_pos_embeds
        x = self.embedding_layernorm(x)
        x = self.embedding_dropout(x)

        # Encoder Layers
        for layer in self.encoder_layers:
            x = layer(x, relative_pos_embeddings=relative_pos_embeddings)

        # Final processing for heads
        normalized_board_output = self.final_norm(x)
        z_value = self.z_value_head(normalized_board_output)
        q_value = self.q_value_head(normalized_board_output)

        if self.debug_mode:
            policy_logits, from_logits, to_logits = self.policy_head(normalized_board_output)
            return policy_logits, z_value, q_value, from_logits, to_logits # type: ignore
        else:
            policy_logits = self.policy_head(normalized_board_output)

            return policy_logits, z_value, q_value

    def state_shape(self) -> Tuple[int, ...]:
        return (self.seq_len, self.input_features)

    def policy_shape(self) -> Tuple[int, ...]:
        return (self.seq_len * self.seq_len,)

    def value_shape(self) -> Tuple[int, ...]:
        return (3,)

    def log_gradients(self, epoch: int):
        pi_head_grad_norm = torch.linalg.norm(self.policy_head.from_head.weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head.mlp[3].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head.mlp[3].weight.grad).item()
        encoder_grad_norm = torch.linalg.norm(self.encoder_layers[0].attn.w_q.weight.grad).item()
        logging.info(
            f"Grad norms for epoch {epoch + 1} are PI: {pi_head_grad_norm:.6f}, Z value: {z_val_head_grad_norm:.6f}, Q value: {q_val_head_grad_norm:.6f}, Encoder: {encoder_grad_norm:.6f}")

    def debug(self):
        key_layers = {
            "rel_pos_embed_table": self.relative_pos_embedding_table.weight,
            "position_embeddings": self.position_embeddings.weight,
            "token_embed": self.token_embed.weight,
            "encoder_0_wq": self.encoder_layers[0].attn.w_q.weight,
            "policy_head_from": self.policy_head.from_head.weight,
            "policy_head_to": self.policy_head.to_head.weight,
            "z_value_head_out": self.z_value_head.mlp[3].weight,
            "material_head_out": self.q_value_head.mlp[3].weight,
        }

        stats = {"weights": {}}
        with torch.no_grad():
            for name, p in key_layers.items():
                if p is not None:
                    stats["weights"][name] = {
                        "mean": p.mean().item(), "std": p.std().item(),
                        "max": p.max().item(), "min": p.min().item(),
                        "norm": torch.linalg.norm(p).item()}
        builder = "\n{:<25} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12}\n".format(
            "Layer", "Wgt Mean", "Wgt Std", "Wgt Min", "Wgt Max", "Wgt L2 Norm")
        builder += "-" * 94 + "\n"
        for name, data in stats["weights"].items():
            builder += "{:<25} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>12.4f}\n".format(
                name, data['mean'], data['std'], data['min'], data['max'], data['norm'])
        logging.info("--- Model Weight Statistics ---" + builder)

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
    model = ThreePlayerHybrid(**config)
    model.pre_onnx_export()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model Configuration: {config}")
    logging.info(f"Total Trainable Parameters: {total_params / 1e6:.2f}M")

    model.train()
    batch_size = 4
    dummy_input = torch.randn(batch_size, model.seq_len, model.input_features)

    policy_logits, z_value, q_value = model(dummy_input)

    loss = policy_logits.sum() + z_value.sum() + q_value.sum()
    loss.backward()

    print(f"\n--- Forward Pass Output Shapes ---")
    print(f"Policy Logits Shape: {policy_logits.shape}")  # Expected: [4, 9216]
    print(f"Z-Value Shape:       {z_value.shape}")  # Expected: [4, 3]
    print(f"Q-Value Shape:       {q_value.shape}")  # Expected: [4, 3]

    model.log_gradients(epoch=0)
    model.debug()

    model.eval()
    with torch.no_grad():
        policy_logits_inf, _, _ = model(dummy_input)

    print("\nInference check successful. Model is ready for ONNX export.")
