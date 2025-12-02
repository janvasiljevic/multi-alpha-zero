import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple

# As per Appendix A.2, the paper uses the Mish activation function.
class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class RelativeAttention(nn.Module):
    """
    Custom Self-Attention with Relative Position Encodings.
    This implementation follows the complex formulation described in formulas (6) and (7)
    of the "Mastering Chess with a Transformer Model" paper, which is a variant of the
    encoding scheme from Shaw et al. (2018). It introduces learnable relative position
    vectors for queries, keys, and values.

    Note: This is computationally intensive due to the creation of large intermediate tensors,
    but the paper states this is negligible for a 64-token context length (8x8 board).
    """
    def __init__(self, d_model: int, n_heads: int, board_size: int = 8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.board_size = board_size
        self.seq_len = board_size * board_size

        # QKV projections - no bias as per Appendix A.2
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection - has bias
        self.w_o = nn.Linear(d_model, d_model)

        # Learnable relative position embeddings for queries, keys, and values.
        # The maximum relative distance is board_size - 1. We need embeddings for
        # displacements from -(board_size-1) to +(board_size-1).
        self.max_rel_pos = self.board_size - 1
        num_rel_embeddings = 2 * self.max_rel_pos + 1

        # We create separate embeddings for horizontal (width) and vertical (height) displacements.
        self.rel_emb_q_w = nn.Embedding(num_rel_embeddings, self.d_head)
        self.rel_emb_q_h = nn.Embedding(num_rel_embeddings, self.d_head)
        self.rel_emb_k_w = nn.Embedding(num_rel_embeddings, self.d_head)
        self.rel_emb_k_h = nn.Embedding(num_rel_embeddings, self.d_head)
        self.rel_emb_v_w = nn.Embedding(num_rel_embeddings, self.d_head)
        self.rel_emb_v_h = nn.Embedding(num_rel_embeddings, self.d_head)

        # Register a buffer for relative indices to avoid re-computation
        self.register_buffer('rel_indices', self._create_relative_indices())

    def _create_relative_indices(self) -> torch.Tensor:
        """Creates a tensor of relative indices for width and height."""
        torch.set_printoptions(profile="full")
        torch.set_printoptions(linewidth=400)

        coords = torch.arange(self.seq_len)
        row_coords = coords // self.board_size
        col_coords = coords % self.board_size

        print(f"Row coords: {row_coords}")
        print(f"Col coords: {col_coords}")

        rel_row = row_coords.unsqueeze(1) - row_coords.unsqueeze(0)
        rel_col = col_coords.unsqueeze(1) - col_coords.unsqueeze(0)

        print(f"Relative row: {rel_row}")
        print(f"Relative col: {rel_col}")

        # Shift indices to be non-negative for embedding lookup
        rel_row += self.max_rel_pos
        rel_col += self.max_rel_pos

        print(f"Relative abs. row: {rel_row}")
        print(f"Relative abs. col: {rel_col}")

        exit()

        return torch.stack([rel_row, rel_col], dim=-1)

    def _get_relative_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Looks up and combines relative position embeddings."""
        rel_row_indices = self.rel_indices[..., 0]
        rel_col_indices = self.rel_indices[..., 1]

        # Shape of each is (seq_len, seq_len, d_head)
        rel_q = self.rel_emb_q_h(rel_row_indices) + self.rel_emb_q_w(rel_col_indices)
        rel_k = self.rel_emb_k_h(rel_row_indices) + self.rel_emb_k_w(rel_col_indices)
        rel_v = self.rel_emb_v_h(rel_row_indices) + self.rel_emb_v_w(rel_col_indices)

        return rel_q, rel_k, rel_v

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

        # 3. Get relative position embeddings
        # Shape of each is (S, S, D_h)
        rel_q, rel_k, rel_v = self._get_relative_embeddings()

        # 4. Compute attention scores according to formula (6)
        # (q_i + a^Q_ij) * (k_j + a^K_ij)^T
        # This requires broadcasting and is memory intensive.
        q_exp = q.unsqueeze(3) # (B, H, S, 1, D_h)
        rel_q_exp = rel_q.view(1, 1, seq_len, seq_len, self.d_head) # (1, 1, S, S, D_h)
        q_with_rel = q_exp + rel_q_exp

        k_exp = k.unsqueeze(2) # (B, H, 1, S, D_h)
        rel_k_exp = rel_k.view(1, 1, seq_len, seq_len, self.d_head) # (1, 1, S, S, D_h)
        k_with_rel = k_exp + rel_k_exp

        # Element-wise product and sum over the head dimension
        scores = (q_with_rel * k_with_rel).sum(dim=-1) # (B, H, S, S)
        scores = scores / math.sqrt(self.d_head)

        attn_weights = F.softmax(scores, dim=-1) # (B, H, S, S)

        # 5. Compute output values according to formula (7)
        # sum_j alpha_ij * (v_j + a^V_ij)
        v_exp = v.unsqueeze(2) # (B, H, 1, S, D_h)
        rel_v_exp = rel_v.view(1, 1, seq_len, seq_len, self.d_head) # (1, 1, S, S, D_h)
        v_with_rel = v_exp + rel_v_exp # (B, H, S, S, D_h)

        attn_weights_exp = attn_weights.unsqueeze(-1) # (B, H, S, S, 1)

        # Weighted sum
        context = (attn_weights_exp * v_with_rel).sum(dim=3) # (B, H, S, D_h)

        # 6. Reshape and final projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)

        return output

class FeedForward(nn.Module):
    """Standard FeedForward layer with Mish activation."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.mish = Mish()
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.mish(self.linear_1(x)))

class EncoderLayer(nn.Module):
    """A single Transformer Encoder layer with Post-LN and custom attention."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = RelativeAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        # Post-LN: LayerNorm is applied after the residual connection.
        # As per paper, biases are omitted in encoder normalization layers.
        # Setting elementwise_affine=False removes learnable gain and bias.
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        attn_output = self.attn(x)
        x = self.norm1(x + attn_output)

        # Feed-forward block
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class AttentionPolicyHead(nn.Module):
    """
    Policy head based on scaled dot-product attention, as described in Appendix A.3.
    It computes a logit for every possible (from_square, to_square) move pair.
    """
    def __init__(self, d_model: int, d_policy_head: int = 32):
        super().__init__()
        # Dense layer to generate policy embeddings
        self.embed_proj = nn.Linear(d_model, d_model)

        # Projections to get query (from_square) and key (to_square) vectors
        self.q_proj = nn.Linear(d_model, d_policy_head)
        self.k_proj = nn.Linear(d_model, d_policy_head)
        self.d_policy_head = d_policy_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate policy embeddings from the final encoder output
        # Using ReLU as a non-linearity, which is a common choice.
        policy_embed = F.relu(self.embed_proj(x))

        # Project to queries and keys
        q = self.q_proj(policy_embed) # (B, 64, D_policy_head)
        k = self.k_proj(policy_embed) # (B, 64, D_policy_head)

        # Calculate logits for all (from, to) pairs
        # (B, 64, D_head) @ (B, D_head, 64) -> (B, 64, 64)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_policy_head)

        return logits

class ValueHead(nn.Module):
    """
    A single value head pathway, as described in Appendix A.3.
    It generates a 128-dim "value embedding" from the encoder output.
    This module will be instantiated multiple times for different value targets.
    """
    def __init__(self, d_model: int, d_value_embed: int = 128, d_value_hidden: int = 32):
        super().__init__()
        seq_len = 64
        # Projections to create the value embedding
        self.proj1 = nn.Linear(d_model, d_value_hidden)
        self.flatten = nn.Flatten()
        self.proj2 = nn.Linear(seq_len * d_value_hidden, d_value_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj1(x))
        x = self.flatten(x)
        value_embedding = F.relu(self.proj2(x))
        return value_embedding

class ChessFormer(nn.Module):
    """
    The main ChessFormer model, combining the transformer encoder with policy and value heads.
    
    Example instantiation for CF-6M (6M parameters):
        model = ChessFormer(
            d_model=256,
            n_layers=8,
            n_heads=8,
            d_ff=256, # As specified in the paper
            input_features=112,
            board_size=8
        )
        
    Example instantiation for CF-240M (243M parameters):
        model = ChessFormer(
            d_model=1024,
            n_layers=15,
            n_heads=32,
            d_ff=4096,
            input_features=112,
            board_size=8
        )
    """
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 input_features: int = 112, board_size: int = 8):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Input embedding layer
        self.token_embed = nn.Linear(input_features, d_model)

        # Transformer Encoder stack
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )

        # Final LayerNorm for the encoder output
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # --- Output Heads ---

        # Policy Head (for both vanilla and soft policy targets)
        self.policy_head = AttentionPolicyHead(d_model)

        # Value Heads
        # The paper states that each value head generates its own embedding separately.
        # We create two separate pathways as requested.

        # Pathway 1: For the Game Result head (Win/Draw/Loss)
        self.game_result_value_path = ValueHead(d_model)
        self.game_result_predictor = nn.Linear(128, 3) # 3 classes: win, draw, loss

        # Pathway 2: For the L2 Value head (scalar reward prediction)
        self.l2_value_path = ValueHead(d_model)
        self.l2_value_predictor = nn.Linear(128, 1) # scalar value

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 64, input_features)
                              representing the 64 squares of the chessboard.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the model's predictions.
                - 'policy_logits': (B, 64, 64) logits for (from, to) moves.
                - 'game_result_logits': (B, 3) logits for win/draw/loss.
                - 'l2_value': (B, 1) scalar value prediction.
        """
        # 1. Generate token embeddings
        x = self.token_embed(x)

        # 2. Pass through Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Apply final normalization
        encoder_output = self.final_norm(x)

        # 3. Compute head outputs
        policy_logits = self.policy_head(encoder_output)

        # Game result head
        game_result_embedding = self.game_result_value_path(encoder_output)
        game_result_logits = self.game_result_predictor(game_result_embedding)

        # L2 value head
        l2_value_embedding = self.l2_value_path(encoder_output)
        l2_value = self.l2_value_predictor(l2_value_embedding)

        return {
            "policy_logits": policy_logits,
            "game_result_logits": game_result_logits,
            "l2_value": l2_value,
        }

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Configure and create the CF-6M model
    print("--- Testing CF-6M Configuration ---")
    cf_6m_config = {
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 8,
        "d_ff": 256, # Unusual, but as specified for CF-6M
        "input_features": 112,
        "board_size": 8
    }
    model_6m = ChessFormer(**cf_6m_config)

    # Count parameters
    num_params_6m = sum(p.numel() for p in model_6m.parameters() if p.requires_grad)
    print(f"CF-6M Estimated Parameters: {num_params_6m / 1e6:.2f}M")

    # Create a dummy input tensor
    # Batch size = 4, 64 squares, 112 features per square
    dummy_input = torch.randn(4, 64, 112)

    # Forward pass
    with torch.no_grad():
        output_6m = model_6m(dummy_input)

    # Print output shapes
    print("\nOutput shapes for CF-6M:")
    for name, tensor in output_6m.items():
        print(f"{name}: {tensor.shape}")

    exit()

    # 2. Configure and create the CF-240M model
    print("\n--- Testing CF-240M Configuration ---")
    cf_240m_config = {
        "d_model": 1024,
        "n_layers": 15,
        "n_heads": 32,
        "d_ff": 4096,
        "input_features": 112,
        "board_size": 8
    }
    model_240m = ChessFormer(**cf_240m_config)

    num_params_240m = sum(p.numel() for p in model_240m.parameters() if p.requires_grad)
    print(f"CF-240M Estimated Parameters: {num_params_240m / 1e6:.2f}M")

    # Forward pass with the same dummy input
    with torch.no_grad():
        output_240m = model_240m(dummy_input)

    print("\nOutput shapes for CF-240M:")
    for name, tensor in output_240m.items():
        print(f"{name}: {tensor.shape}")