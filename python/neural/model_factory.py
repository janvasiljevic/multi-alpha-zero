from neural.base_interface_az import AlphaZeroNet
from neural.chess_model_basic import ThreePlayerChessformerBasic
from neural.chess_model_bert_cls import ThreePlayerChessformerBert
from neural.chess_model_bert_cls_v2 import ThreePlayerChessformerBertV2
from neural.chess_model_hybrid import ThreePlayerHybrid
from neural.chess_model_relative import ThreePlayerChessformerRelative
from neural.chess_model_shaw import ThreePlayerChessformerShaw
from neural.hex_model import HexAlphaZeroNet
from neural.hex_model_axiom_bias_bert_cls import Hex5AxiomBiasWithBertCls
from neural.hex_model_relative import Hex5AxiomBias


def model_factory(model_key: str) -> AlphaZeroNet:
    if model_key == "Hex2Absolute":
        return HexAlphaZeroNet(
            board_side_length=3,
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            input_dim=7,  # one-hot[Empty, P1, P2, P3, P1_turn, P2_turn, P3_turn]
        )

    if model_key == "Hex2Canonical":
        return HexAlphaZeroNet(
            board_side_length=3,
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            input_dim=4,  # one-hot[Empty, P1, P2, P3] (no player turn)
        )

    if model_key == "Hex4Absolute":
        return HexAlphaZeroNet(
            board_side_length=4,
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            input_dim=7,  # one-hot[Empty, P1, P2, P3, P1_turn, P2_turn, P3_turn]
        )
    if model_key == "Hex4Canonical":
        return HexAlphaZeroNet(
            board_side_length=4,
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            input_dim=4,  # one-hot[Empty, P1, P2, P3] (no player turn)
        )

    if model_key == "Hex5Absolute":
        return HexAlphaZeroNet(
            board_side_length=5,
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            input_dim=7,  # one-hot[Empty, P1, P2, P3, P1_turn, P2_turn, P3_turn]
        )
    if model_key == "Hex5Canonical":
        return HexAlphaZeroNet(
            board_side_length=5,
            d_model=128,
            nhead=4,
            num_encoder_layers=4,
            input_dim=4,  # one-hot[Empty, P1, P2, P3] (no player turn)
        )

    if model_key == "Hex5CanonicalAxiomBias":
        return Hex5AxiomBias(
            d_model=128,
            d_ff=1024,
            nhead=4,
            num_encoder_layers=4,
        )

    if model_key == "Hex5CanonicalAxiomBiasBertCls":
        return Hex5AxiomBiasWithBertCls(
            d_model=128,
            d_ff=1024,
            nhead=4,
            num_encoder_layers=4,
        )

    if model_key == "ChessAbsPositionalEnc":
        config = {
            "d_model": 180,
            "n_layers": 4,
            "n_heads": 4,
            "d_ff": 512,
        }
        return ThreePlayerChessformerBasic(**config)

    if model_key == "ChessAxiomBias":
        config = {
            "d_model": 192,
            "n_layers": 4,
            "n_heads": 4,
            "d_ff": 512,
        }

        return ThreePlayerChessformerRelative(**config)

    if model_key == "ChessBigBert":
        return ThreePlayerChessformerBert(
            d_model=192,
            d_ff=512,
            n_heads=4,
            n_layers=8
        )

    # 10M parameters
    if model_key == "ChessBigBertV2":
        return ThreePlayerChessformerBertV2(
            d_model=5 * 64,
            d_ff=5 * 64 * 4,      # 4 * d_model
            n_heads=5,
            n_layers=9,
            dropout_rate=0.1
        )

    # 10M parameters
    if model_key == "ChessShaw":
        return ThreePlayerChessformerShaw(
            d_model=5 * 64,
            d_ff=5 * 64 * 4,      # 4 * d_model
            n_heads=5,
            n_layers=9,
            dropout_rate=0.1
        )

    # 10M parameters
    if model_key == "ChessHybrid":
        return ThreePlayerHybrid(
            d_model=5 * 64,
            d_ff=5 * 64 * 4,      # 4 * d_model
            n_heads=5,
            n_layers=9,
            dropout_rate=0.1,
            input_features=24
        )

    if model_key == "ChessDomain":
        return ThreePlayerHybrid(
            d_model=5 * 64,
            d_ff=5 * 64 * 4,      # 4 * d_model
            n_heads=5,
            n_layers=9,
            dropout_rate=0.1,
            input_features=53,
            aux_features=4 # One for third move counter
        )

    raise ValueError(f"Unknown model key: {model_key}.")
