use crate::mapping::{
    BoardPlayer, InputMapper, MetaPerformanceMapper, PolicyMapper, ReverseInputMapper,
};
use game_hex::coords::AxialCoord;
use game_hex::game_hex::CellState::{self, Occupied};
use game_hex::game_hex::{HexGame, HexPlayer};
use ndarray::{ArrayView2, ArrayViewMut2};

// A rotation 60° right (clockwise ↻) shoves each coordinate one slot to the left ←:
//
// [ q,  r,  s]
// to  [-r, -s, -q]
// to     [  s,  q,  r]
// A rotation 60° left (counter-clockwise ↺) shoves each coordinate one slot to the right →:
//
//           [ q,  r,  s]
// to    [-s, -q, -r]
// to [r,  s,  q]

/// Maps a coordinate from a specific player's perspective TO the canonical (P1) perspective.
/// This requires applying the *inverse* rotation (i.e., rotating left/CCW).
fn map_from_canonical_to_p1(abs_coord: AxialCoord, player: HexPlayer) -> AxialCoord {
    match player {
        // P1 is the canonical player, no rotation needed.
        HexPlayer::P1 => abs_coord,

        // To map P2's view to P1's, we apply a 60° rotation left (CCW).
        // The rule is [q, r, s] -> [-s, -q, -r].
        // new_q = -s, new_r = -q
        HexPlayer::P2 => AxialCoord::new(-abs_coord.s(), -abs_coord.q),

        // To map P3's view to P1's, we apply a 120° rotation left (CCW).
        // The rule is [q, r, s] -> [r, s, q].
        // new_q = r, new_r = s
        HexPlayer::P3 => AxialCoord::new(abs_coord.r, abs_coord.s()),
    }
}

/// Maps a coordinate FROM the canonical (P1) perspective to a specific player's perspective.
/// This requires applying the *forward* rotation (i.e., rotating right/CW).
pub fn map_from_p1_to_canonical(canonical_coord: AxialCoord, player: HexPlayer) -> AxialCoord {
    match player {
        // P1 is the canonical player, no rotation needed.
        HexPlayer::P1 => canonical_coord,

        // To map from P1's view to P2's, we apply a 60° rotation right (CW).
        // The rule is [q, r, s] -> [-r, -s, -q].
        // new_q = -r, new_r = -s
        HexPlayer::P2 => AxialCoord::new(-canonical_coord.r, -canonical_coord.s()),

        // To map from P1's view to P3's, we apply a 120° rotation right (CW).
        // The rule is [q, r, s] -> [s, q, r].
        // new_q = s, new_r = q
        HexPlayer::P3 => AxialCoord::new(canonical_coord.s(), canonical_coord.q),
    }
} // --- The New Relative Canonical Mapper ---

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HexCanonicalMapper {
    // For O(1) index -> coord mapping from the canonical perspective.
    pub index_to_coord: Vec<AxialCoord>,
    // O(1) canonical coord -> index lookup table.
    pub coord_to_index_lut: Vec<usize>,

    pub radius: i32,
    pub width: i32,
    pub num_of_hexes: u16,
}

const INVALID_CELL_INDEX: usize = usize::MAX;

impl HexCanonicalMapper {
    /// Creates a new mapper. The internal data structures are built once and represent
    /// the board from the canonical (Player 1) perspective.
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
                    let lut_q = q + radius;
                    let lut_r = r + radius;
                    let lut_index = (lut_r * width + lut_q) as usize;

                    coord_to_index_lut[lut_index] = current_policy_index;
                    index_to_coord.push(coord);

                    current_policy_index += 1;
                }
            }
        }

        debug_assert_eq!(current_policy_index, num_of_hexes);
        debug_assert_eq!(index_to_coord.len(), num_of_hexes);

        HexCanonicalMapper {
            index_to_coord,
            coord_to_index_lut,
            radius,
            width,
            num_of_hexes: game.num_of_hexes,
        }
    }
}

impl PolicyMapper<HexGame> for HexCanonicalMapper {
    fn policy_len(&self) -> usize {
        self.index_to_coord.len()
    }

    /// Converts a move from the perspective of a given player to its canonical policy index.
    /// NOTE: This function's signature is changed from the original trait to include `player`.
    fn move_to_index(&self, player: HexPlayer, mv: AxialCoord) -> usize {
        let canonical_move = map_from_p1_to_canonical(mv, player);

        let lut_q = canonical_move.q + self.radius;
        let lut_r = canonical_move.r + self.radius;
        let lut_index = (lut_r * self.width + lut_q) as usize;

        debug_assert!(lut_index < self.coord_to_index_lut.len());
        let policy_index = self.coord_to_index_lut[lut_index];

        debug_assert!(
            policy_index != INVALID_CELL_INDEX,
            "Attempted to get index for a move outside the board: {:?}, canonical: {:?}",
            mv,
            canonical_move
        );

        policy_index
    }

    /// Converts a canonical policy index to a move, from the perspective of the board's current player.
    fn index_to_move(
        &self,
        board: &HexGame,
        _move_store: &Vec<AxialCoord>,
        index: usize,
    ) -> Option<AxialCoord> {
        let player = board.current_turn;

        if index >= self.policy_len() {
            return None;
        }

        let canonical_coord = self.index_to_coord[index];
        let absolute_coord = map_from_canonical_to_p1(canonical_coord, player);
        Some(absolute_coord)
    }
}
impl MetaPerformanceMapper<HexGame> for HexCanonicalMapper {
    // This implementation is unchanged.
    fn average_number_of_moves(&self) -> usize {
        (self.num_of_hexes as f32 * 2.0 / 3.0) as usize
    }
}

const ME_CH: usize = 0;
const OPP_NEXT_CH: usize = 1;
const OPP_NEXT_NEXT_CH: usize = 2;
const EMPTY_CH: usize = 3;

impl InputMapper<HexGame> for HexCanonicalMapper {
    fn input_board_shape(&self) -> [usize; 2] {
        [self.num_of_hexes as usize, 4]
    }

    /// Encodes the board state into a canonical input tensor for the neural network.
    /// It iterates through each canonical position, finds the corresponding absolute cell on the board,
    /// and encodes its state relative to the current player.
    fn encode_input(&self, input_view: &mut ArrayViewMut2<'_, bool>, board: &HexGame) {
        input_view.fill(false);

        let me = board.current_turn;
        let opponent1 = me.next();
        let opponent2 = opponent1.next();

        // `index_to_coord` holds all valid coordinates from the canonical (P1) perspective.
        for (_, &abs_coord) in self.index_to_coord.iter().enumerate() {
            let canonical_coord = map_from_p1_to_canonical(abs_coord, me);

            // now we need to find the index of this canonical_coord in the index_to_coord array
            let lut_q = canonical_coord.q + self.radius;
            let lut_r = canonical_coord.r + self.radius;
            let lut_index = (lut_r * self.width + lut_q) as usize;
            let index = self.coord_to_index_lut[lut_index];

            match board.get_state(abs_coord) {
                Occupied(p) if p == me => input_view[[index, ME_CH]] = true,
                Occupied(p) if p == opponent1 => input_view[[index, OPP_NEXT_CH]] = true,
                Occupied(p) if p == opponent2 => input_view[[index, OPP_NEXT_NEXT_CH]] = true,
                CellState::Empty => input_view[[index, EMPTY_CH]] = true,
                _ => {} // Should not be reached in a valid game state.
            }
        }
    }

    fn is_absolute(&self) -> bool {
        false
    }
}

impl ReverseInputMapper<HexGame> for HexCanonicalMapper {
    /// Decodes a canonical input tensor back into an absolute game state.
    /// It assumes the player perspective is provided as the first scalar.
    fn decode_input(&self, input_view: &ArrayView2<'_, bool>, scalars: &Vec<f32>) -> HexGame {
        debug_assert_eq!(input_view.shape(), &[self.num_of_hexes as usize, 4]);
        debug_assert_eq!(
            scalars.len(),
            1,
            "Player turn must be provided as a scalar."
        );

        let original_player = HexPlayer::from_usize(scalars[0] as usize);

        let (me_player, opp1_player, opp2_player) = match original_player {
            HexPlayer::P1 => (HexPlayer::P1, HexPlayer::P2, HexPlayer::P3),
            HexPlayer::P2 => (HexPlayer::P2, HexPlayer::P3, HexPlayer::P1),
            HexPlayer::P3 => (HexPlayer::P3, HexPlayer::P1, HexPlayer::P2),
        };

        let mut game = HexGame::new(self.radius + 1).unwrap();

        // Iterate through each canonical index.
        for (_, &canonical_coord) in self.index_to_coord.iter().enumerate() {
            let absolute_coord = map_from_canonical_to_p1(canonical_coord, original_player);

            let lut_q = canonical_coord.q + self.radius;
            let lut_r = canonical_coord.r + self.radius;
            let lut_index = (lut_r * self.width + lut_q) as usize;
            let index = self.coord_to_index_lut[lut_index];

            // Determine which player occupies this cell from the input view.
            if input_view[[index, ME_CH]] {
                game.set_state(absolute_coord, Occupied(me_player));
            } else if input_view[[index, OPP_NEXT_CH]] {
                game.set_state(absolute_coord, Occupied(opp1_player));
            } else if input_view[[index, OPP_NEXT_NEXT_CH]] {
                game.set_state(absolute_coord, Occupied(opp2_player));
            } else {
                game.set_state(absolute_coord, CellState::Empty);
            }
        }

        game.rebuild_internal_state(original_player);
        game
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapping::Board;
    use colored::Colorize;
    use ndarray::Array2;
    use rand::prelude::IndexedRandom;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;

    fn index_to_coord_without_de_canon(
        mapper: &HexCanonicalMapper,
        index: usize,
        player: HexPlayer,
    ) -> Option<AxialCoord> {
        if index >= mapper.policy_len() {
            return None;
        }

        let canonical_coord = mapper.index_to_coord[index];
        let absolute_coord = canonical_coord; // No de-canonization
        Some(absolute_coord)
    }

    fn reverse_decode_without_de_canon(
        mapper: &HexCanonicalMapper,
        input_view: &ArrayView2<'_, bool>,
        scalars: &Vec<f32>,
    ) -> HexGame {
        debug_assert_eq!(input_view.shape(), &[mapper.num_of_hexes as usize, 4]);
        debug_assert_eq!(
            scalars.len(),
            1,
            "Player turn must be provided as a scalar."
        );

        let original_player = HexPlayer::from_usize(scalars[0] as usize);

        let (me_player, opp1_player, opp2_player) = match original_player {
            HexPlayer::P1 => (HexPlayer::P1, HexPlayer::P2, HexPlayer::P3),
            HexPlayer::P2 => (HexPlayer::P1, HexPlayer::P2, HexPlayer::P3),
            HexPlayer::P3 => (HexPlayer::P1, HexPlayer::P2, HexPlayer::P3),
            // HexPlayer::P2 => (HexPlayer::P2, HexPlayer::P3, HexPlayer::P1),
            // HexPlayer::P3 => (HexPlayer::P3, HexPlayer::P1, HexPlayer::P2),
        };

        let mut game = HexGame::new(mapper.radius + 1).unwrap();

        // Iterate through each canonical index.
        for (_, &canonical_coord) in mapper.index_to_coord.iter().enumerate() {
            let absolute_coord = canonical_coord; // No de-canonization

            let lut_q = canonical_coord.q + mapper.radius;
            let lut_r = canonical_coord.r + mapper.radius;
            let lut_index = (lut_r * mapper.width + lut_q) as usize;
            let index = mapper.coord_to_index_lut[lut_index];

            // Determine which player occupies this cell from the input view.
            if input_view[[index, ME_CH]] {
                game.set_state(absolute_coord, Occupied(me_player));
            } else if input_view[[index, OPP_NEXT_CH]] {
                game.set_state(absolute_coord, Occupied(opp1_player));
            } else if input_view[[index, OPP_NEXT_NEXT_CH]] {
                game.set_state(absolute_coord, Occupied(opp2_player));
            } else {
                game.set_state(absolute_coord, CellState::Empty);
            }
        }

        game.rebuild_internal_state(original_player);
        game
    }

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

    fn pretty_print_output_mask(game: &mut HexGame, mapper: &HexCanonicalMapper) -> String {
        let mut move_store = <HexGame as Board>::MoveStore::default();
        game.fill_move_store(&mut move_store);

        let mut output = String::new();

        let mut vector = vec![false; mapper.policy_len()];

        for mv in move_store {
            let index = mapper.move_to_index(game.current_turn, mv);
            vector[index] = true;
        }

        for i in 0..mapper.policy_len() {
            if vector[i] {
                output.push('□');
            } else {
                output.push('_');
            }
        }

        output
    }

    /// Helper function to run symmetry tests on the base (P1) mapping.
    fn test_p1_mapper_for_size(size: i32) {
        let mut game = HexGame::new(size).unwrap();
        // Ensure we are testing from P1's perspective for index_to_move
        game.current_turn = HexPlayer::P1;

        let mapper = HexCanonicalMapper::new(&game);
        let all_valid_coords = game.get_valid_empty_cells();

        assert_eq!(mapper.policy_len(), all_valid_coords.len());
        assert_eq!(mapper.policy_len(), game.num_of_hexes as usize);

        let mut seen_indices = HashSet::new();

        for coord in all_valid_coords {
            // Test move_to_index from P1's perspective
            let index = mapper.move_to_index(HexPlayer::P1, coord);
            // Test index_to_move from P1's perspective (set in game.current_turn)
            let mapped_back_coord = mapper.index_to_move(&game, &vec![], index);

            assert!(index < mapper.policy_len(), "Index out of bounds");
            assert!(seen_indices.insert(index), "Index generated twice");
            assert_eq!(Some(coord), mapped_back_coord, "Coord-Index-Coord failed");
        }

        assert_eq!(seen_indices.len(), mapper.policy_len());
    }

    #[test]
    fn test_conversions() {
        let coord = AxialCoord::new(5, -2);

        debug_assert_eq!(coord.s(), -3);

        let coord_p2 = map_from_p1_to_canonical(coord, HexPlayer::P2);

        debug_assert_eq!(coord_p2.q, 2);
        debug_assert_eq!(coord_p2.r, 3);
        debug_assert_eq!(coord_p2.s(), -5);
    }

    #[test]
    fn test_mapper_p1_symmetry_size_1() {
        test_p1_mapper_for_size(1);
    }

    #[test]
    fn test_mapper_p1_symmetry_size_2() {
        test_p1_mapper_for_size(2);
    }

    #[test]
    fn test_mapper_p1_symmetry_size_4() {
        test_p1_mapper_for_size(4);
    }

    #[test]
    fn test_specific_p1_mappings_size_2() {
        let mut game = HexGame::new(2).unwrap();
        game.current_turn = HexPlayer::P1;
        let mapper = HexCanonicalMapper::new(&game);

        let coord0 = AxialCoord::new(0, -1);
        assert_eq!(mapper.move_to_index(HexPlayer::P1, coord0), 0);
        assert_eq!(mapper.index_to_move(&game, &vec![], 0), Some(coord0));

        let coord2 = AxialCoord::new(-1, 0);
        assert_eq!(mapper.move_to_index(HexPlayer::P1, coord2), 2);
        assert_eq!(mapper.index_to_move(&game, &vec![], 2), Some(coord2));
    }

    #[test]
    fn test_out_of_bounds_index() {
        let game = HexGame::new(3).unwrap();
        let mapper = HexCanonicalMapper::new(&game);
        assert_eq!(mapper.policy_len(), 19);
        assert!(mapper.index_to_move(&game, &vec![], 18).is_some());
        assert!(mapper.index_to_move(&game, &vec![], 19).is_none());
    }

    #[test]
    fn print_qrs_to_index_mapping_for_different_players() {
        let game = HexGame::new(2).unwrap();

        let mapper = HexCanonicalMapper::new(&game);

        println!("Index to (q,r,s) mapping from canonical (P1) perspective:");

        for player in &[HexPlayer::P1, HexPlayer::P2, HexPlayer::P3] {
            println!("\nMapping from {:?} perspective:", player);
            for index in 0..mapper.policy_len() {
                let abs_coord = mapper.index_to_coord[index];
                let canonical_coord = map_from_p1_to_canonical(abs_coord, *player);

                let lut_q = canonical_coord.q + mapper.radius;
                let lut_r = canonical_coord.r + mapper.radius;
                let lut_index = (lut_r * mapper.width + lut_q) as usize;
                let mapped_index = mapper.coord_to_index_lut[lut_index];

                println!(
                    "Index {:2} -> Abs Coord {:?} -> Canonical Coord {:?} -> Mapped Index {:2}",
                    index, abs_coord, canonical_coord, mapped_index
                );
            }

            for index in 0..mapper.policy_len() {
                let abs_coord = mapper.index_to_coord[index];
                let canonical_coord = map_from_p1_to_canonical(abs_coord, *player);

                // assert_eq!(index, mapped_index, "Index mismatch for player {:?} at index {}", player, index);
            }
        }
    }

    #[test]
    fn exhaustive_mapper_test() {
        let mut game = HexGame::new(5).unwrap();

        let mapper = HexCanonicalMapper::new(&game);

        let rng = StdRng::seed_from_u64(42);

        let mut move_store = <HexGame as Board>::MoveStore::default();

        for _ in 0..30 {
            let mut game_clone = game.clone();
            game_clone.current_turn = HexPlayer::P1;

            let mut rng = rng.clone();

            game.fill_move_store(&mut move_store);

            while !game_clone.is_terminal() {
                let mut input_view = Array2::from_elem(mapper.input_board_shape(), false);

                mapper.encode_input(&mut input_view.view_mut(), &game_clone);

                for coord in &move_store {
                    let index = mapper.move_to_index(game_clone.current_turn, *coord);
                    let mapped_back_coord =
                        mapper.index_to_move(&game_clone, &vec![], index).unwrap();

                    assert_eq!(
                        *coord, mapped_back_coord,
                        "Mapping failed for coord {:?} at index {}",
                        coord, index
                    );

                    // Must be 0 on the empty cells
                    assert!(
                        input_view[[index, EMPTY_CH]],
                        "Expected empty cell at index {}, got {}",
                        index,
                        input_view[[index, EMPTY_CH]]
                    );
                }

                if move_store.is_empty() {
                    break;
                }
                let random_move = move_store.choose(&mut rng).unwrap().clone();

                let prev_player = game_clone.current_turn;

                game_clone.play_move_mut_with_store(&random_move, &mut move_store, None);

                let next_player = game_clone.current_turn;

                game_clone.current_turn = prev_player; // Reset to previous player for encoding

                mapper.encode_input(&mut input_view.view_mut(), &game_clone);

                let index = mapper.move_to_index(game_clone.current_turn, random_move);

                assert!(
                    !input_view[[index, EMPTY_CH]],
                    "Expected non-empty cell at index {}, got {}",
                    index,
                    input_view[[index, EMPTY_CH]]
                );

                assert!(
                    input_view[[index, ME_CH]],
                    "Expected player cell at index {}, got {}",
                    index,
                    input_view[[index, ME_CH]]
                );

                game_clone.current_turn = next_player; // Restore the current turn for the next iteration
            }
        }
    }

    #[test]
    fn test_rotational_symmetry_index_to_move() {
        let mut game = HexGame::new(4).unwrap();
        let mapper = HexCanonicalMapper::new(&game);

        for index in 0..mapper.policy_len() {
            game.current_turn = HexPlayer::P1;
            let p1_move = mapper.index_to_move(&game, &vec![], index).unwrap();

            game.current_turn = HexPlayer::P2;
            let p2_move = mapper.index_to_move(&game, &vec![], index).unwrap();

            game.current_turn = HexPlayer::P3;
            let p3_move = mapper.index_to_move(&game, &vec![], index).unwrap();

            // The point is that p2_move and p3_move are generated from completely
            // indexes - index 0 in policy map for P1, P2 and P3 SHOULD NOT be the same
            let p1_move_from_p2 = map_from_p1_to_canonical(p2_move, HexPlayer::P2);
            let p1_move_from_p3 = map_from_p1_to_canonical(p3_move, HexPlayer::P3);

            assert_eq!(
                p1_move, p1_move_from_p2,
                "P2 move does not map back to P1 move"
            );
            assert_eq!(
                p1_move, p1_move_from_p3,
                "P3 move does not map back to P1 move"
            );

            println!(
                "Index {:2}: P1 Move {:?}, P2 Move {:?}, P3 Move {:?}",
                index, p1_move, p2_move, p3_move
            );
        }
    }

    #[test]
    fn test_rotational_symmetry_encode_input() {
        let size = 2;
        let mut game1 = HexGame::new(size).unwrap();
        let mapper = HexCanonicalMapper::new(&game1);

        game1.set_state(AxialCoord::new(0, 1), Occupied(HexPlayer::P1));
        game1.set_state(AxialCoord::new(1, 0), Occupied(HexPlayer::P2));
        game1.set_state(AxialCoord::new(1, -1), Occupied(HexPlayer::P3));
        game1.current_turn = HexPlayer::P1;

        println!("Base game state for P1:\n{}", game1.fancy_debug());

        // --- Create a rotated version for Player 2 ---
        let mut game2 = game1.clone();

        game2.current_turn = HexPlayer::P2;

        let mut game3 = game1.clone();

        game3.current_turn = HexPlayer::P3;

        // --- Encode all three boards ---
        let mut input1 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input2 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input3 = Array2::from_elem(mapper.input_board_shape(), false);

        mapper.encode_input(&mut input1.view_mut(), &game1);
        mapper.encode_input(&mut input2.view_mut(), &game2);
        mapper.encode_input(&mut input3.view_mut(), &game3);

        println!("Encoded input from P1 perspective:");
        println!("{}", pretty_print_tensor_format(&input1));
        println!("Encoded input from P2 perspective:");
        println!("{}", pretty_print_tensor_format(&input2));
        println!("Encoded input from P3 perspective:");
        println!("{}", pretty_print_tensor_format(&input3));

        let game_1_decoded =
            mapper.decode_input(&input1.view(), &vec![HexPlayer::P1 as usize as f32]);
        let game_2_decoded =
            mapper.decode_input(&input2.view(), &vec![HexPlayer::P2 as usize as f32]);
        let game_3_decoded =
            mapper.decode_input(&input3.view(), &vec![HexPlayer::P3 as usize as f32]);

        println!(
            "Decoded game state from P1 input:\n{}",
            game_1_decoded.fancy_debug()
        );
        println!(
            "Decoded game state from P2 input:\n{}",
            game_2_decoded.fancy_debug()
        );
        println!(
            "Decoded game state from P3 input:\n{}",
            game_3_decoded.fancy_debug()
        );
    }

    #[test]
    fn test_2() {
        let mut game_1 = HexGame::new(2).unwrap();

        game_1.set_state(AxialCoord::new(0, 1), Occupied(HexPlayer::P2));
        game_1.set_state(AxialCoord::new(1, 0), Occupied(HexPlayer::P3));
        game_1.current_turn = HexPlayer::P1;

        println!("Base game state for P1:\n{}", game_1.fancy_debug());

        let mut game_2 = HexGame::new(2).unwrap();

        game_2.set_state(AxialCoord::new(1, 0), Occupied(HexPlayer::P3));
        game_2.set_state(AxialCoord::new(1, -1), Occupied(HexPlayer::P1));

        game_2.current_turn = HexPlayer::P2;

        println!("Rotated game state for P2:\n{}", game_2.fancy_debug());

        let mut game_3 = HexGame::new(2).unwrap();

        game_3.set_state(AxialCoord::new(1, -1), Occupied(HexPlayer::P1));
        game_3.set_state(AxialCoord::new(0, -1), Occupied(HexPlayer::P2));

        game_3.current_turn = HexPlayer::P3;

        println!("Rotated game state for P3:\n{}", game_3.fancy_debug());

        let mapper = HexCanonicalMapper::new(&game_1);

        let mut input1 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input2 = Array2::from_elem(mapper.input_board_shape(), false);
        let mut input3 = Array2::from_elem(mapper.input_board_shape(), false);

        mapper.encode_input(&mut input1.view_mut(), &game_1);
        mapper.encode_input(&mut input2.view_mut(), &game_2);
        mapper.encode_input(&mut input3.view_mut(), &game_3);

        println!("Encoded input from P1 perspective:");
        println!("{}", pretty_print_tensor_format(&input1));

        println!("Encoded input from P2 perspective:");
        println!("{}", pretty_print_tensor_format(&input2));

        println!("Encoded input from P3 perspective:");
        println!("{}", pretty_print_tensor_format(&input3));

        assert_eq!(input1, input2);
        assert_eq!(input1, input3);

        // print all the available moves
        let mut move_store = <HexGame as Board>::MoveStore::default();
        game_1.fill_move_store(&mut move_store);

        println!("Available moves:");
        for mv in &move_store {
            let index = mapper.move_to_index(HexPlayer::P1, *mv);
            let move_from_index = mapper.index_to_move(&game_1, &move_store, index).unwrap();
            println!(
                "From P1 perspective, Move {:?} at index {} which maps back to move {:?}",
                mv, index, move_from_index
            );

            let index_2 = mapper.move_to_index(HexPlayer::P2, *mv);
            let move_from_index_2 = mapper.index_to_move(&game_2, &move_store, index_2).unwrap();

            println!(
                "From P2 perspective, Move {:?} at index {} which maps back to move {:?}",
                mv, index_2, move_from_index_2
            );

            let index_3 = mapper.move_to_index(HexPlayer::P3, *mv);
            let move_from_index_3 = mapper.index_to_move(&game_3, &move_store, index_3).unwrap();

            println!(
                "From P3 perspective, Move {:?} at index {} which maps back to move {:?}",
                mv, index_3, move_from_index_3
            );

            assert_eq!(mv, &move_from_index);
            assert_eq!(mv, &move_from_index_2);
            assert_eq!(mv, &move_from_index_3);
        }
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn test_move_to_index_invalid_move_debug() {
        let game = HexGame::new(2).unwrap();
        let mapper = HexCanonicalMapper::new(&game);
        let invalid_coord = AxialCoord::new(1, 1);
        // Any player perspective is fine here
        mapper.move_to_index(HexPlayer::P1, invalid_coord);
    }

    #[test]
    fn testing_moves() {
        let mut game = HexGame::new(2).unwrap();
        let mapper = HexCanonicalMapper::new(&game);
        let mut move_store = <HexGame as Board>::MoveStore::default();

        game.set_state(AxialCoord::new(0, 1), Occupied(HexPlayer::P1));
        game.fill_move_store(&mut move_store);

        let mut input_array = Array2::from_elem(mapper.input_board_shape(), false);

        mapper.encode_input(&mut input_array.view_mut(), &game);

        for mv in &move_store {
            let index = mapper.move_to_index(game.current_turn, *mv);
            let mapped_back_move = mapper.index_to_move(&game, &move_store, index).unwrap();
            assert_eq!(*mv, mapped_back_move);
            println!(
                "Move {:?} maps to index {} and back to move {:?}",
                mv, index, mapped_back_move
            );
        }

        println!(
            "Tensor from P1 perspective:\n{}",
            pretty_print_tensor_format(&input_array)
        );
        println!(
            "Policy mask from P1 perspective:\n{}",
            pretty_print_output_mask(&mut game, &mapper)
        );

        // set turn to P2 and test again
        game.current_turn = HexPlayer::P2;

        move_store.clear();

        game.fill_move_store(&mut move_store);

        println!("\nTesting from P2 perspective:");

        for mv in &move_store {
            let index = mapper.move_to_index(game.current_turn, *mv);
            let mapped_back_move = mapper.index_to_move(&game, &move_store, index).unwrap();
            assert_eq!(*mv, mapped_back_move);
            println!(
                "Move {:?} maps to index {} and back to move {:?}",
                mv, index, mapped_back_move
            );
        }

        mapper.encode_input(&mut input_array.view_mut(), &game);

        println!(
            "Tensor from P2 perspective:\n{}",
            pretty_print_tensor_format(&input_array)
        );
        println!(
            "Policy mask from P2 perspective:\n{}",
            pretty_print_output_mask(&mut game, &mapper)
        );
    }

    #[test]
    fn test_input_mapper_from_and_to() {
        let mut game = HexGame::new(5).unwrap();
        let mapper = HexCanonicalMapper::new(&game);

        let mut move_store = <HexGame as Board>::MoveStore::default();

        let mut rng = StdRng::seed_from_u64(42);
        let mut input_view = Array2::from_elem(mapper.input_board_shape(), false);

        game.fill_move_store(&mut move_store);

        for _ in 0..16 {
            // Run a few more times to cycle through players
            if move_store.is_empty() {
                break;
            }

            let random_move = move_store.choose(&mut rng).unwrap().clone();

            game.play_move_mut_with_store(&random_move, &mut move_store, None);

            mapper.encode_input(&mut input_view.view_mut(), &game);

            // Check one-hot encoding property
            for row in input_view.rows() {
                let sum: f32 = row.iter().map(|&v| if v { 1.0 } else { 0.0 }).sum();
                assert!((sum - 1.0).abs() < 1e-6, "Row must be one-hot");
            }

            let player_scalar = vec![game.current_turn as usize as f32];
            let mut decoded_game = mapper.decode_input(&input_view.view(), &player_scalar);
            let decoded_game_no_decanon =
                reverse_decode_without_de_canon(&mapper, &input_view.view(), &player_scalar);

            let mut move_store = <HexGame as Board>::MoveStore::default();
            decoded_game.fill_move_store(&mut move_store);

            let mut policy_vector = vec![0.0; mapper.policy_len()];

            for mv in &move_store {
                let index = mapper.move_to_index(decoded_game.current_turn, *mv);
                policy_vector[index] = 1.0;
            }

            let hashmap_of_moves_to_values: std::collections::HashMap<AxialCoord, f32> =
                policy_vector
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let mv = mapper
                            .index_to_move(&decoded_game, &move_store, i)
                            .expect("Index should map to a valid move");
                        (mv, v)
                    })
                    .collect();

            assert_eq!(
                decoded_game, game,
                "Decoded game state does not match original"
            );

            println!(
                "Current game state:\n{}",
                game.fancy_debug_visualize_board_indented(Some((
                    hashmap_of_moves_to_values,
                    vec![0.0, 0.5, 1.0]
                )))
            );

            let hashmap_of_moves_to_values_no_decanon: std::collections::HashMap<AxialCoord, f32> =
                policy_vector
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let mv = index_to_coord_without_de_canon(&mapper, i, game.current_turn)
                            .expect("Index should map to a valid move");
                        (mv, v)
                    })
                    .collect();

            println!(
                "From perspective of current player: {}",
                decoded_game_no_decanon.fancy_debug_visualize_board_indented(Some((
                    hashmap_of_moves_to_values_no_decanon,
                    vec![0.0, 0.5, 1.0]
                )))
            );

            println!(
                "Encoded input tensor:\n{}",
                pretty_print_tensor_format(&input_view)
            );

            println!(
                "Policy mask from current player's perspective:\n{}",
                pretty_print_output_mask(&mut game, &mapper)
            )
        }
    }
}
