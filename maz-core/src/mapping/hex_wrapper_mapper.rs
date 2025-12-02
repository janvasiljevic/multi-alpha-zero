use crate::mapping::hex_absolute_mapper::HexAbsoluteMapper;
use crate::mapping::hex_canonical_mapper::HexCanonicalMapper;
use crate::mapping::{InputMapper, MetaPerformanceMapper, PolicyMapper};
use game_hex::coords::AxialCoord;
use game_hex::game_hex::{HexGame, HexPlayer};
use ndarray::ArrayViewMut2;

#[derive(Clone, Debug)]
pub enum HexWrapperMapper {
    Absolute(HexAbsoluteMapper),
    Canonical(HexCanonicalMapper),
}

impl InputMapper<HexGame> for HexWrapperMapper {
    fn input_board_shape(&self) -> [usize; 2] {
        match self {
            HexWrapperMapper::Absolute(m) => m.input_board_shape(),
            HexWrapperMapper::Canonical(m) => m.input_board_shape(),
        }
    }
    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &HexGame) {
        match self {
            HexWrapperMapper::Absolute(m) => m.encode_input(input_view, board),
            HexWrapperMapper::Canonical(m) => m.encode_input(input_view, board),
        }
    }
    fn is_absolute(&self) -> bool {
        match self {
            HexWrapperMapper::Absolute(m) => m.is_absolute(),
            HexWrapperMapper::Canonical(m) => m.is_absolute(),
        }
    }
}
impl PolicyMapper<HexGame> for HexWrapperMapper {
    fn policy_len(&self) -> usize {
        match self {
            HexWrapperMapper::Absolute(m) => m.policy_len(),
            HexWrapperMapper::Canonical(m) => m.policy_len(),
        }
    }
    fn move_to_index(&self, player: HexPlayer, mv: AxialCoord) -> usize {
        match self {
            HexWrapperMapper::Absolute(m) => m.move_to_index(player, mv),
            HexWrapperMapper::Canonical(m) => m.move_to_index(player, mv),
        }
    }
    fn index_to_move(
        &self,
        board: &HexGame,
        move_store: &Vec<AxialCoord>,
        index: usize,
    ) -> Option<AxialCoord> {
        match self {
            HexWrapperMapper::Absolute(m) => m.index_to_move(board, move_store, index),
            HexWrapperMapper::Canonical(m) => m.index_to_move(board, move_store, index),
        }
    }
}

impl MetaPerformanceMapper<HexGame> for HexWrapperMapper {
    fn average_number_of_moves(&self) -> usize {
        match self {
            HexWrapperMapper::Absolute(m) => m.average_number_of_moves(),
            HexWrapperMapper::Canonical(m) => m.average_number_of_moves(),
        }
    }
}
