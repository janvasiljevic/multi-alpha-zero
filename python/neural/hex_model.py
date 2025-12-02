import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural.base_interface_az import AlphaZeroNet


class CoreMLSafeEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model

        # Attention projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.alpha_attn = nn.Parameter(torch.ones(1))
        self.alpha_ff = nn.Parameter(torch.ones(1))

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 1. PRE-NORM ATTENTION BLOCK ---
        norm_x = self.norm1(x)

        # Project to Q, K, V
        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)

        # Manually compute attention. This avoids the ONNX export issue.
        # Reshape for multi-head attention: (B, L, E) -> (B, L, H, E/H) -> (B, H, L, E/H)
        # B=batch, L=seq_len, E=embed_dim, H=nhead
        B, L, E = q.shape
        head_dim = E // self.nhead
        q = q.view(B, L, self.nhead, head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, head_dim).transpose(1, 2)

        # Use the scaled dot-product attention function
        # This is the core of the attention mechanism and is ONNX-friendly
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, E)

        # Final output projection
        attn_output = self.out_proj(attn_output)

        # First residual connection
        # x = x + self.dropout(attn_output)
        x = x + self.alpha_attn * attn_output

        # --- 2. PRE-NORM FEED-FORWARD BLOCK ---
        norm_x_ff = self.norm2(x)
        ff_output = self.ff(norm_x_ff)

        # Second residual connection
        x = x + self.alpha_ff * ff_output

        # Second residual connection
        # x = x + self.dropout(ff_output)

        return x


class HexAlphaZeroNet(AlphaZeroNet):
    """
    The main network for 3-player Hex, combining an encoder with policy and value heads.

    Args:
        board_side_length (int): The side length of the hexagonal board.
                                 The total number of cells is calculated from this.
        input_dim (int): The number of feature channels for each cell on the board.
                         Default: 7 (4 for piece state [Empty, P1, P2, P3] + 3 for player turn).
        d_model (int): The dimensionality of the model's hidden layers.
        nhead (int): The number of heads in the multi-head attention mechanism.
        num_encoder_layers (int): The number of stacked encoder layers.
        d_ff (int): The dimensionality of the feed-forward network inside the encoder.

        IMO we don't need dropout since we are doing alpha zero training -
        the data is already noisy.
    """

    def pre_onnx_export(self):
        pass

    def debug(self):
        pass

    def __init__(
            self,
            board_side_length: int,
            input_dim: int = 7,
            d_model: int = 256,
            nhead: int = 4,
            num_encoder_layers: int = 4,
            d_ff: int = 1024,
    ):
        super().__init__()
        self.board_side_length = board_side_length
        # The number of cells in a regular hexagonal grid of side length s
        self.num_cells = 3 * board_side_length * (board_side_length - 1) + 1
        self.input_dim = input_dim

        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        # Learned positional embeddings for each cell on the board
        self.positional_embeddings = nn.Parameter(torch.randn(self.num_cells, d_model))

        self.encoder_layers = nn.ModuleList(
            [
                CoreMLSafeEncoderLayer(d_model, nhead, d_ff)
                for _ in range(num_encoder_layers)
            ]
        )

        # self.policy_head = nn.Sequential(
        #     nn.Linear(d_model, 256),  # Input is the pooled d_model vector
        #     nn.GELU(),
        #     nn.Linear(256, self.num_cells)  # Output is a logit for EACH cell
        # )

        self.policy_head = nn.Linear(d_model, 1)

        # Value Head: Predicts the expected outcome for each of the 3 players.
        # The output is a 3-element vector, e.g., [-0.8, 0.9, -0.2], where each
        # value is in [-1, 1] representing the expected score (e.g., win=+1, lose=-0.5).
        self.z_value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Tanh(),  # Tanh activation to scale the output to [-1, 1]
        )

        self.q_value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.Tanh(),  # Tanh activation to scale the output to [-1, 1]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # General case for most linear layers
            # This will now apply to the final layers as well, which is what we want.
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

        # --- SPECIAL CASES FOR HEADS ---
        # Initialize the final policy layer to output zeros. This creates a uniform
        # policy at the start, which is a stable starting point.
        # self.policy_head[2] is the Linear layer before softmax (which we don't apply here)
        if hasattr(self, 'policy_head') and module == self.policy_head:
            # print("Initializing final policy layer weights to zero.")
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # Initialize the final value layer to output zero. This prevents Tanh saturation
        # at the start of training.
        # self.value_head[2] is the Linear layer before Tanh
        if hasattr(self, 'q_value_head') and module == self.q_value_head[2]:
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if hasattr(self, 'z_value_head') and module == self.z_value_head[2]:
            nn.init.constant_(module.weight, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @property
    def device(self):  # JIT compatibility
        return self.positional_embeddings.device

    def validated_forward(self, x: torch.Tensor):
        """
        Safe forward pass that checks the input shape.
        """
        if x.shape[1] != self.num_cells or x.ndim != 3:
            raise ValueError(
                f"Input tensor shape must be [batch_size, {self.num_cells}, input_dim]. "
                f"Got {x.shape} instead."
            )

        return self.forward(x)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor representing the board state.
                              Shape: [batch_size, num_cells, input_dim].
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - policy_logits (torch.Tensor): Logits for the move probabilities.
                                            Shape: [batch_size, num_cells].
            - value (torch.Tensor): The estimated value for each of the 3 players.
                                    Shape: [batch_size, 3].
        """
        # 1. Embed input and add positional embeddings
        # [batch, num_cells, input_dim] -> [batch, num_cells, d_model]
        x = self.input_embedding(x)
        x = x + self.positional_embeddings.unsqueeze(0)

        # 2. Pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # 3. Policy Head
        # [batch, num_cells, d_model] -> [batch, num_cells, 1] -> [batch, num_cells]
        # policy_logits = self.policy_head(x).squeeze(-1)

        # Tko sm mel ucasih, probimo ce deluje boljs....
        policy_logits = self.policy_head(x)
        # Squeeze the last dimension to get the correct output shape.
        # [batch, num_cells, 1] -> [batch, num_cells]
        policy_logits = policy_logits.squeeze(-1)

        # 4. Value Head
        # First, pool the features from all cells. Mean pooling is a simple and effective choice.
        # [batch, num_cells, d_model] -> [batch, d_model]
        pooled_output = x.mean(dim=1)
        # [batch, d_model] -> [batch, 3]
        # value = self.value_head(pooled_output)

        # policy_logits = self.policy_head(pooled_output)  # Shape: [batch, num_cells]
        z_value = self.z_value_head(pooled_output)  # Shape: [batch, 3]
        q_value = self.q_value_head(pooled_output)

        return policy_logits, z_value, q_value

    def state_shape(self):
        return self.num_cells, self.input_dim

    def policy_shape(self):
        return (self.num_cells,)

    def value_shape(self):
        return (3,)

    def log_gradients(self, epoch: int):
        pi_head_grad_norm = torch.linalg.norm(self.policy_head.weight.grad).item()
        z_val_head_grad_norm = torch.linalg.norm(self.z_value_head[2].weight.grad).item()
        q_val_head_grad_norm = torch.linalg.norm(self.q_value_head[2].weight.grad).item()
        encoder_grad_norm = torch.linalg.norm(self.encoder_layers[0].q_proj.weight.grad).item()

        logging.info(
            f"Grad norms for epoch {epoch + 1} are PI: {pi_head_grad_norm:.6f}, Z value: {z_val_head_grad_norm:.6f}, Q value: {q_val_head_grad_norm:.6f}, Encoder: {encoder_grad_norm:.6f}")
