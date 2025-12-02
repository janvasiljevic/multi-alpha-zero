use crate::mapping::chess_canonical_mapper::ChessCanonicalMapper;
use crate::mapping::chess_domain_canonical_mapper::ChessDomainMapper;
use crate::mapping::chess_extended_canonical_mapper::ChessExtendedCanonicalMapper;
use crate::mapping::chess_hybrid_canonical_mapper::ChessHybridCanonicalMapper;
use crate::mapping::{InputMapper, MetaPerformanceMapper, PolicyMapper, ReverseInputMapper};
use game_tri_chess::basics::Color;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, PseudoLegalMove};
use ndarray::{ArrayView2, ArrayViewMut2};

#[derive(Clone, Debug)]
pub enum ChessWrapperMapper {
    Basic(ChessCanonicalMapper),
    Extended(ChessExtendedCanonicalMapper),
    Hybrid(ChessHybridCanonicalMapper),
    Domain(ChessDomainMapper),
}

pub const ALL_CHESS_MAPPERS: [ChessWrapperMapper; 4] = [
    ChessWrapperMapper::Basic(ChessCanonicalMapper),
    ChessWrapperMapper::Extended(ChessExtendedCanonicalMapper),
    ChessWrapperMapper::Hybrid(ChessHybridCanonicalMapper),
    ChessWrapperMapper::Domain(ChessDomainMapper),
];

pub fn auto_detect_chess_mapper(sample_input_flattened: &[bool]) -> ChessWrapperMapper {
    let detected_mapper = ALL_CHESS_MAPPERS
        .iter()
        .find(|mapper| {
            let shape = mapper.input_board_shape();
            shape[0] * shape[1] == sample_input_flattened.len()
        })
        .cloned();

    detected_mapper
        .expect("Could not auto-detect a suitable ChessWrapperMapper for the given input shape.")
}


impl ChessWrapperMapper {
    pub fn get_name(&self) -> &'static str {
        match self {
            ChessWrapperMapper::Basic(_) => "Basic",
            ChessWrapperMapper::Extended(_) => "Extended",
            ChessWrapperMapper::Hybrid(_) => "Hybrid",
            ChessWrapperMapper::Domain(_) => "Domain",
        }
    }
}

impl InputMapper<TriHexChess> for ChessWrapperMapper {
    fn input_board_shape(&self) -> [usize; 2] {
        match self {
            ChessWrapperMapper::Basic(m) => m.input_board_shape(),
            ChessWrapperMapper::Extended(m) => m.input_board_shape(),
            ChessWrapperMapper::Hybrid(m) => m.input_board_shape(),
            ChessWrapperMapper::Domain(m) => m.input_board_shape(),
        }
    }

    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &TriHexChess) {
        match self {
            ChessWrapperMapper::Basic(m) => m.encode_input(input_view, board),
            ChessWrapperMapper::Extended(m) => m.encode_input(input_view, board),
            ChessWrapperMapper::Hybrid(m) => m.encode_input(input_view, board),
            ChessWrapperMapper::Domain(m) => m.encode_input(input_view, board),
        }
    }

    fn is_absolute(&self) -> bool {
        false
    }
}
impl PolicyMapper<TriHexChess> for ChessWrapperMapper {
    fn policy_len(&self) -> usize {
        match self {
            ChessWrapperMapper::Basic(m) => m.policy_len(),
            ChessWrapperMapper::Extended(m) => m.policy_len(),
            ChessWrapperMapper::Hybrid(m) => m.policy_len(),
            ChessWrapperMapper::Domain(m) => m.policy_len(),
        }
    }
    fn move_to_index(&self, player: Color, mv: PseudoLegalMove) -> usize {
        match self {
            ChessWrapperMapper::Basic(m) => m.move_to_index(player, mv),
            ChessWrapperMapper::Extended(m) => m.move_to_index(player, mv),
            ChessWrapperMapper::Hybrid(m) => m.move_to_index(player, mv),
            ChessWrapperMapper::Domain(m) => m.move_to_index(player, mv),
        }
    }
    fn index_to_move(
        &self,
        board: &TriHexChess,
        move_store: &ChessMoveStore,
        index: usize,
    ) -> Option<PseudoLegalMove> {
        match self {
            ChessWrapperMapper::Basic(m) => m.index_to_move(board, move_store, index),
            ChessWrapperMapper::Extended(m) => m.index_to_move(board, move_store, index),
            ChessWrapperMapper::Hybrid(m) => m.index_to_move(board, move_store, index),
            ChessWrapperMapper::Domain(m) => m.index_to_move(board, move_store, index),
        }
    }
}

impl ReverseInputMapper<TriHexChess> for ChessWrapperMapper {
    fn decode_input(&self, input_view: &ArrayView2<'_, bool>, scalars: &Vec<f32>) -> TriHexChess {
        match self {
            ChessWrapperMapper::Basic(m) => m.decode_input(input_view, scalars),
            ChessWrapperMapper::Extended(m) => m.decode_input(input_view, scalars),
            ChessWrapperMapper::Hybrid(m) => m.decode_input(input_view, scalars),
            ChessWrapperMapper::Domain(m) => m.decode_input(input_view, scalars),
        }
    }
}

impl MetaPerformanceMapper<TriHexChess> for ChessWrapperMapper {
    fn average_number_of_moves(&self) -> usize {
        match self {
            ChessWrapperMapper::Basic(m) => m.average_number_of_moves(),
            ChessWrapperMapper::Extended(m) => m.average_number_of_moves(),
            ChessWrapperMapper::Hybrid(m) => m.average_number_of_moves(),
            ChessWrapperMapper::Domain(m) => m.average_number_of_moves(),
        }
    }
}
