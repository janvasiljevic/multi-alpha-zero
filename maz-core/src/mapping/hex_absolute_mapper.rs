use crate::mapping::{InputMapper, MetaPerformanceMapper, PolicyMapper, ReverseInputMapper};
use ndarray::{ArrayView2, ArrayViewMut2};
use game_hex::coords::AxialCoord;
use game_hex::game_hex::CellState::Occupied;
use game_hex::game_hex::{CellState, HexGame, HexPlayer};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HexAbsoluteMapper {
    // For O(1) index -> coord mapping. (Unchanged)
    index_to_coord: Vec<AxialCoord>,
    // The new O(1) coord -> index lookup table.
    coord_to_index_lut: Vec<usize>,
    // We need to store these to perform the coordinate transformation.
    radius: i32,
    width: i32,
    num_of_hexes: u16,
}

const INVALID_CELL_INDEX: usize = usize::MAX;

impl HexAbsoluteMapper {
    pub fn new(game: &HexGame) -> Self {
        let num_of_hexes = game.num_of_hexes as usize;
        let radius = game.radius;
        let width = 2 * radius + 1;
        let lut_size = (width * width) as usize;

        let mut index_to_coord = Vec::with_capacity(num_of_hexes);

        let mut coord_to_index_lut = vec![INVALID_CELL_INDEX; lut_size];

        let mut current_policy_index = 0;
        for r in -radius..=radius {
            for q in -radius..=radius {
                let coord = AxialCoord::new(q, r);
                if game.is_in(&coord) {
                    // This is a valid hex cell.

                    // Calculate its position in our flat LUT.
                    let lut_q = q + radius;
                    let lut_r = r + radius;
                    let lut_index = (lut_r * width + lut_q) as usize;

                    // Store the current policy index at that position.
                    coord_to_index_lut[lut_index] = current_policy_index;

                    // Also populate the reverse-lookup table.
                    index_to_coord.push(coord);

                    current_policy_index += 1;
                }
            }
        }

        debug_assert_eq!(
            current_policy_index, num_of_hexes,
            "Number of cells processed does not match num_of_hexes"
        );
        debug_assert_eq!(
            index_to_coord.len(),
            num_of_hexes,
            "index_to_coord LUT has the wrong size"
        );

        HexAbsoluteMapper {
            index_to_coord,
            coord_to_index_lut,
            radius,
            width,
            num_of_hexes: game.num_of_hexes,
        }
    }
}

impl PolicyMapper<HexGame> for HexAbsoluteMapper {
    fn policy_len(&self) -> usize {
        self.index_to_coord.len()
    }

    fn move_to_index(&self, _: HexPlayer, mv: AxialCoord) -> usize {
        // Transform the game coordinate (q, r) to a 0-based LUT index.
        let lut_q = mv.q + self.radius;
        let lut_r = mv.r + self.radius;
        let lut_index = (lut_r * self.width + lut_q) as usize;

        debug_assert!(
            lut_index < self.coord_to_index_lut.len(),
            "Calculated LUT index is out of bounds."
        );

        let policy_index = self.coord_to_index_lut[lut_index];

        debug_assert!(
            policy_index != INVALID_CELL_INDEX,
            "Attempted to get index for a move outside the board: {:?}",
            mv
        );

        policy_index
    }

    // This implementation remains identical to the previous version.
    fn index_to_move(
        &self,
        _: &HexGame,
        _move_store: &Vec<AxialCoord>,
        index: usize,
    ) -> Option<AxialCoord> {
        self.index_to_coord.get(index).copied()
    }
}
impl MetaPerformanceMapper<HexGame> for HexAbsoluteMapper {
    fn average_number_of_moves(&self) -> usize {
        // The average number of moves in a Hex game is approximately 2/3 of the number of hexes.
        // This is a rough estimate based on the average game length.
        (self.num_of_hexes as f32 * 2.0 / 3.0) as usize
    }
}

const P1_CHANNEL: usize = 0;
const P2_CHANNEL: usize = 1;
const P3_CHANNEL: usize = 2;
const EMPTY_CHANNEL: usize = 3;
const TURN_P1_CHANNEL: usize = 4;
const TURN_P2_CHANNEL: usize = 5;
const TURN_P3_CHANNEL: usize = 6;

impl InputMapper<HexGame> for HexAbsoluteMapper {
    /// w*7 (num of hexes * {P1, P2, P3, empty, P1_turn, P2_turn, P3_turn})
    fn input_board_shape(&self) -> [usize; 2] {
        [self.num_of_hexes as usize, 7]
    }

    fn encode_input(
        &self,
        input_view: &mut ArrayViewMut2<'_, bool>,
        board: &HexGame,
    ) {
        input_view.fill(false);

        for (index, &coord) in self.index_to_coord.iter().enumerate() {
            let tile_state = board.get_state(coord);

            match tile_state {
                Occupied(HexPlayer::P1) => input_view[[index, P1_CHANNEL]] = true,
                Occupied(HexPlayer::P2) => input_view[[index, P2_CHANNEL]] = true,
                Occupied(HexPlayer::P3) => input_view[[index, P3_CHANNEL]] = true,
                CellState::Empty => input_view[[index, EMPTY_CHANNEL]] = true,
            }
        }

        let turn_channel = match board.current_turn {
            HexPlayer::P1 => TURN_P1_CHANNEL,
            HexPlayer::P2 => TURN_P2_CHANNEL,
            HexPlayer::P3 => TURN_P3_CHANNEL,
        };

        let mut turn_column = input_view.slice_mut(ndarray::s![.., turn_channel]);

        turn_column.fill(true);
    }

    fn is_absolute(&self) -> bool {
        true
    }
}

impl ReverseInputMapper<HexGame> for HexAbsoluteMapper {
    fn decode_input(&self, input_view: &ArrayView2<'_, bool>, _: &Vec<f32>) -> HexGame {
        debug_assert_eq!(
            input_view.shape(),
            &[self.num_of_hexes as usize, 7],
            "Input view shape does not match expected shape."
        );

        let mut game = HexGame::new(self.radius + 1).unwrap();

        for (index, &coord) in self.index_to_coord.iter().enumerate() {
            let p1_value = input_view[[index, P1_CHANNEL]];
            let p2_value = input_view[[index, P2_CHANNEL]];
            let p3_value = input_view[[index, P3_CHANNEL]];
            let empty_value = input_view[[index, EMPTY_CHANNEL]];

            if p1_value {
                game.set_state(coord, Occupied(HexPlayer::P1));
            } else if p2_value {
                game.set_state(coord, Occupied(HexPlayer::P2));
            } else if p3_value {
                game.set_state(coord, Occupied(HexPlayer::P3));
            } else if empty_value {
                game.set_state(coord, CellState::Empty);
            }
        }

        // Set the current turn based on the turn channels.
        if input_view[[0, TURN_P1_CHANNEL]] {
            game.current_turn = HexPlayer::P1;
        } else if input_view[[0, TURN_P2_CHANNEL]] {
            game.current_turn = HexPlayer::P2;
        } else if input_view[[0, TURN_P3_CHANNEL]] {
            game.current_turn = HexPlayer::P3;
        }

        game
    }
}

#[cfg(test)]
mod mapper_tests {
    use super::*;
    use crate::mapping::MoveStore;
    use std::collections::HashSet;
    use colored::Colorize;
    use ndarray::Array2;

    fn pretty_print_tensor_format(input: &Array2<bool>) -> String {
        let mut output = String::new();

        for i in 0..input.shape()[1] {
            for j in 0..input.shape()[0] {
                if input[[j, i]] {
                    let char = "□";

                    let colored_string = match i {
                        0 => char.red(),   // White pieces
                        1 => char.blue(),  // Gray pieces
                        2 => char.green(), // Black pieces
                        3 => char.white(), // Empty
                        4 => char.red().bold(),   // White turn
                        5 => char.blue().bold(),  // Gray turn
                        6 => char.green().bold(), // Black turn
                        _ => char.normal(),
                    };

                    output.push_str(&format!("{colored_string}"));
                } else {
                    output.push_str("_");
                }
            }
            output.push('\n');
        }

        output
    }

    fn test_mapper_for_size(size: i32) {
        let game = HexGame::new(size).unwrap();
        let mapper = HexAbsoluteMapper::new(&game);
        let all_valid_coords = game.get_valid_empty_cells();

        assert_eq!(mapper.policy_len(), all_valid_coords.len());
        assert_eq!(mapper.policy_len(), game.num_of_hexes as usize);

        let mut seen_indices = HashSet::new();

        for coord in all_valid_coords.iter() {
            let index = mapper.move_to_index(HexPlayer::P1, coord);
            let mapped_back_coord = mapper.index_to_move(&game, &all_valid_coords, index);

            // Check that the index is unique and within bounds
            assert!(
                index < mapper.policy_len(),
                "Index {} out of bounds for size {}",
                index,
                size
            );
            assert!(
                seen_indices.insert(index),
                "Index {} was generated twice for size {}",
                index,
                size
            );

            // Check that the mapping is symmetric
            assert_eq!(
                Some(coord),
                mapped_back_coord,
                "Coord-Index-Coord mapping failed for {:?} -> {} -> {:?}",
                coord,
                index,
                mapped_back_coord
            );
        }

        println!("{:?}", mapper.index_to_coord);
        println!("{:?}", mapper.coord_to_index_lut);

        // Ensure all indices from 0 to len-1 were seen
        assert_eq!(seen_indices.len(), mapper.policy_len());
    }

    #[test]
    fn test_mapper_symmetry_size_1() {
        test_mapper_for_size(1); // radius 0, 1 hex
    }

    #[test]
    fn test_mapper_symmetry_size_2() {
        test_mapper_for_size(2); // radius 1, 7 hexes
    }

    #[test]
    fn test_mapper_symmetry_size_4() {
        test_mapper_for_size(4); // radius 3, 37 hexes
    }

    #[test]
    fn test_specific_mappings_size_2() {
        // For size=2 (radius=1), the iteration order is:
        // r=-1: (0,-1), (1,-1)
        // r= 0: (-1,0), (0,0), (1,0)
        // r= 1: (-1,1), (0,1)
        let game = HexGame::new(2).unwrap();
        let mapper = HexAbsoluteMapper::new(&game);

        const UK_P: HexPlayer = HexPlayer::P1;

        let coord0 = AxialCoord::new(0, -1);
        assert_eq!(mapper.move_to_index(UK_P, coord0), 0);
        assert_eq!(mapper.index_to_move(&game, &vec![], 0), Some(coord0));

        let coord2 = AxialCoord::new(-1, 0);
        assert_eq!(mapper.move_to_index(UK_P, coord2), 2);
        assert_eq!(mapper.index_to_move(&game, &vec![], 2), Some(coord2));

        let coord_last = AxialCoord::new(0, 1);
        assert_eq!(mapper.move_to_index(UK_P, coord_last), 6);
        assert_eq!(mapper.index_to_move(&game, &vec![], 6), Some(coord_last));
    }

    #[test]
    fn test_out_of_bounds_index() {
        let game = HexGame::new(3).unwrap();
        let mapper = HexAbsoluteMapper::new(&game);

        assert_eq!(mapper.policy_len(), 19);
        assert!(mapper.index_to_move(&game, &vec![], 18).is_some());
        assert!(mapper.index_to_move(&game, &vec![], 19).is_none());
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_move_to_index_invalid_move_debug() {
        let game = HexGame::new(2).unwrap(); // radius 1
        let mapper = HexAbsoluteMapper::new(&game);
        let invalid_coord = AxialCoord::new(1, 1); // Not in a size 2 hex grid
        mapper.move_to_index(HexPlayer::P1, invalid_coord);
    }

    #[test]
    fn test_encode_decode_input() {
        let mut game = HexGame::new(5).unwrap(); // radius 2

        let mapper = HexAbsoluteMapper::new(&game);
        let mut input_view = Array2::from_elem(mapper.input_board_shape(), false);

        mapper.encode_input(&mut input_view.view_mut(), &game);

        println!("Encoded Input with P1 turn:\n{}", pretty_print_tensor_format(&input_view));

        game.current_turn = HexPlayer::P2;

        mapper.encode_input(&mut input_view.view_mut(), &game);

        println!("Encoded Input with P2 turn:\n{}", pretty_print_tensor_format(&input_view));

        game.current_turn = HexPlayer::P3;

        mapper.encode_input(&mut input_view.view_mut(), &game);

        println!("Encoded Input with P3 turn:\n{}", pretty_print_tensor_format(&input_view));
    }
}
