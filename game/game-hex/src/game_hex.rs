use crate::coords::AxialCoord;
use colored::{ColoredString, Colorize};
use once_cell::sync::Lazy;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum HexPlayer {
    P1,
    P2,
    P3,
}

pub const ALL_HEX_PLAYERS: [HexPlayer; 3] = [HexPlayer::P1, HexPlayer::P2, HexPlayer::P3];

impl From<usize> for HexPlayer {
    fn from(value: usize) -> Self {
        match value {
            0 => HexPlayer::P1,
            1 => HexPlayer::P2,
            2 => HexPlayer::P3,
            _ => panic!("Invalid value for HexPlayer: {}", value),
        }
    }
}

#[derive(Clone)]
pub struct ZobristTable {
    pieces: Vec<u64>,
    turn: [u8; 3],
}

impl ZobristTable {
    pub fn new(num_cells: usize) -> Self {
        // Using a fixed seed ensures hashes are deterministic.
        let mut rng = StdRng::seed_from_u64(0x_ABAD_CAFE_DEAD_BEEF);

        let pieces = (0..(num_cells * 3)).map(|_| rng.random()).collect();

        let mut turn = [0; 3];
        rng.fill(&mut turn);

        Self { pieces, turn }
    }

    /// Gets the hash value for a piece at a specific cell index.
    #[inline(always)]
    fn piece(&self, index: usize, player: HexPlayer) -> u64 {
        self.pieces[index * 3 + (player as usize)]
    }

    /// Gets the hash value for the current player's turn.
    #[inline(always)]
    fn turn(&self, player: HexPlayer) -> u64 {
        self.turn[player as usize] as u64
    }
}

struct ZobristHexTables {
    tables: HashMap<i32, Arc<ZobristTable>>,
}

static PRECOMPUTED_HEX_ZOBRIST: Lazy<ZobristHexTables> = Lazy::new(|| {
    let supported_sizes = [1, 2, 3, 4, 5, 6];

    let tables = supported_sizes
        .into_iter()
        .map(|size| {
            let radius = size - 1;
            let num_cells = ((2 * radius + 1) * (2 * radius + 1)) as usize;
            let table = Arc::new(ZobristTable::new(num_cells));
            (size, table)
        })
        .collect();

    ZobristHexTables { tables }
});

/// Gets a shared, pre-computed ZobristTable for a given board size.
/// Panics if the size is not supported. Lock-free after the first call.
fn get_hex_zobrist_table(size: i32) -> Arc<ZobristTable> {
    PRECOMPUTED_HEX_ZOBRIST
        .tables
        .get(&size)
        .cloned()
        .expect(&format!(
            "Zobrist table for size {} is not pre-computed.",
            size
        ))
}

impl HexPlayer {
    pub fn from_usize(val: usize) -> Self {
        debug_assert!(val < 3, "HexPlayer value must be in range [0, 2].");
        match val {
            0 => HexPlayer::P1,
            1 => HexPlayer::P2,
            2 => HexPlayer::P3,
            _ => unreachable!(),
        }
    }
}

impl From<HexPlayer> for usize {
    fn from(val: HexPlayer) -> Self {
        match val {
            HexPlayer::P1 => 0,
            HexPlayer::P2 => 1,
            HexPlayer::P3 => 2,
        }
    }
}

#[derive(Clone, Copy)]
struct Rgb(u8, u8, u8);

// A true, perceptually-uniform Viridis colormap
const VIRIDIS_MAP: [Rgb; 11] = [
    Rgb(68, 1, 84),    // 0.0
    Rgb(72, 40, 120),  // 0.1
    Rgb(62, 74, 137),  // 0.2
    Rgb(49, 104, 142), // 0.3
    Rgb(38, 130, 142), // 0.4
    Rgb(31, 158, 137), // 0.5
    Rgb(53, 183, 121), // 0.6
    Rgb(109, 205, 89), // 0.7
    Rgb(180, 222, 44), // 0.8
    Rgb(251, 231, 37), // 0.9
    Rgb(253, 231, 37), // 1.0 (Same as 0.9 for a bright end)
];

fn lerp_color(val: f32, map: &[Rgb]) -> Rgb {
    let val = val.clamp(0.0, 1.0);
    let scaled = val * (map.len() - 1) as f32;
    let i = scaled.floor() as usize;
    let t = scaled - i as f32;

    let Rgb(r1, g1, b1) = map[i];
    let Rgb(r2, g2, b2) = if i + 1 < map.len() {
        map[i + 1]
    } else {
        map[i]
    };

    let r = ((1.0 - t) * r1 as f32 + t * r2 as f32) as u8;
    let g = ((1.0 - t) * g1 as f32 + t * g2 as f32) as u8;
    let b = ((1.0 - t) * b1 as f32 + t * b2 as f32) as u8;

    Rgb(r, g, b)
}
fn policy_color_viridis(value: f32, cs: ColoredString) -> ColoredString {
    let color = lerp_color(value, &VIRIDIS_MAP);
    cs.on_truecolor(color.0, color.1, color.2)
}

pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    /// Creates a new Union-Find structure with `n` disjoint sets.
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// Finds the representative (root) of the set containing `i`, with path compression.
    pub fn find(&mut self, mut i: usize) -> usize {
        let mut root = i;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression
        while self.parent[i] != root {
            let next_i = self.parent[i];
            self.parent[i] = root;
            i = next_i;
        }
        root
    }

    /// Merges the sets containing `i` and `j`, by size.
    /// Returns true if they were different sets, false otherwise.
    pub fn union(&mut self, i: usize, j: usize) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j {
            // Union by size
            if self.size[root_i] < self.size[root_j] {
                self.parent[root_i] = root_j;
                self.size[root_j] += self.size[root_i];
            } else {
                self.parent[root_j] = root_i;
                self.size[root_i] += self.size[root_j];
            }
            return true;
        }
        false
    }
}

impl HexPlayer {
    pub fn next_hex_player(&self) -> HexPlayer {
        match self {
            HexPlayer::P1 => HexPlayer::P2,
            HexPlayer::P2 => HexPlayer::P3,
            HexPlayer::P3 => HexPlayer::P1,
        }
    }

    pub fn all() -> [HexPlayer; 3] {
        [HexPlayer::P1, HexPlayer::P2, HexPlayer::P3]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    Empty,
    Occupied(HexPlayer),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexGameOutcome {
    Win { winner: HexPlayer },
}

#[derive(Clone)]
pub struct HexGame {
    pub radius: i32, // Max coordinate value (size - 1)
    pub board: Vec<CellState>,
    pub current_turn: HexPlayer,
    pub eliminated_players: HashSet<HexPlayer>,
    pub outcome: Option<HexGameOutcome>,
    pub num_of_hexes: u16,
    zobrist_table: Arc<ZobristTable>,
    pub zobrist_hash: u64,
}

impl HexGame {}

impl Hash for HexGame {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.zobrist_hash.hash(state);
    }
}

// When two hashes are equal, a Hasher will use PartialEq to resolve the
// collision. For correctness, this must compare the actual board state.
impl PartialEq for HexGame {
    fn eq(&self, other: &Self) -> bool {
        // The hash is a quick check. If they are different, the boards can't be equal.
        // If they are the same, we must do a full check to handle rare hash collisions.
        self.zobrist_hash == other.zobrist_hash
            && self.current_turn == other.current_turn
            && self.board == other.board
    }
}

impl Eq for HexGame {}

impl Debug for HexGame {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.debug_visualize_board_indented())
    }
}

impl HexGame {
    pub fn new(size: i32) -> Result<Self, String> {
        assert!(size >= 1, "Size must be non-negative.");

        let radius = size - 1;
        let bounding_box_width = 2 * radius + 1;
        let vec_capacity = (bounding_box_width * bounding_box_width) as usize;
        let mut board = vec![CellState::Empty; vec_capacity];

        let zobrist_table = get_hex_zobrist_table(size);

        board.shrink_to_fit();

        let num_of_hexes = 3 * size * (size - 1) + 1;

        let current_turn = HexPlayer::P1;

        let initial_hash = zobrist_table.turn(current_turn);

        Ok(Self {
            radius,
            board,
            current_turn,
            eliminated_players: HashSet::new(),
            outcome: None,
            num_of_hexes: num_of_hexes as u16,
            zobrist_table,
            zobrist_hash: initial_hash,
        })
    }

    pub fn set_state(&mut self, p0: AxialCoord, p1: CellState) {
        let i = self.index_from_qr(p0.q, p0.r);
        if i < self.board.len() {
            self.board[i] = p1;
        } else {
            panic!("Index out of bounds: {i}");
        }
    }

    pub fn set_to_player(&mut self, q: i32, r: i32, p: HexPlayer) {
        let i = self.index_from_qr(q, r);
        if i < self.board.len() {
            self.board[i] = CellState::Occupied(p);
        } else {
            panic!("Index out of bounds: {i}");
        }
    }

    fn calculate_full_hash(&self) -> u64 {
        let mut hash = 0u64;

        // 1. Start with the hash for the current turn
        hash ^= self.zobrist_table.turn(self.current_turn);

        // 2. XOR the hash for each piece on the board
        for i in 0..self.board.len() {
            if let CellState::Occupied(player) = self.board[i] {
                // If the coordinate is valid (not padding in the square vec)
                let coord = self.get_axial_from_index(i as i32);
                if self.is_in(&coord) {
                    hash ^= self.zobrist_table.piece(i, player);
                }
            }
        }
        hash
    }

    pub fn rebuild_internal_state(&mut self, current_turn: HexPlayer) {
        self.current_turn = current_turn;

        self.eliminated_players.clear();
        for &p in HexPlayer::all().iter() {
            if p != current_turn && !self.is_connected_optimized(p, true) {
                self.eliminated_players.insert(p);
            }
        }

        if self.eliminated_players.len() == 2 {
            self.outcome = Some(HexGameOutcome::Win {
                winner: current_turn,
            });
        } else {
            self.outcome = None;
        }

        // Recalculate the hash based on the current state
        self.zobrist_hash = self.calculate_full_hash();
    }

    fn index_from_qr(&self, q: i32, r: i32) -> usize {
        debug_assert!(
            self.is_in(&AxialCoord::new(q, r)),
            "Coordinates out of bounds: ({q}, {r})"
        );

        let size = 2 * self.radius + 1;
        let center = self.radius;
        let row = r + center;
        let col = q + center;
        (row * size + col) as usize
    }

    pub fn get_current_player(&self) -> HexPlayer {
        self.current_turn
    }

    pub fn get_outcome(&self) -> Option<HexGameOutcome> {
        self.outcome
    }

    pub fn get_state(&self, coord: AxialCoord) -> CellState {
        let i = self.index_from_qr(coord.q, coord.r);
        self.board.get(i).copied().expect("Index out of bounds")
    }

    pub fn get_board_state(&self) -> HashMap<AxialCoord, CellState> {
        let mut board_state = HashMap::new();
        for i in 0..self.board.len() {
            let coord = self.get_axial_from_index(i as i32);
            if self.is_in(&coord) {
                board_state.insert(coord, self.board[i]);
            }
        }
        board_state
    }

    fn is_on_player_start_side(&self, player: HexPlayer, coord: AxialCoord) -> bool {
        match player {
            HexPlayer::P1 => coord.r == self.radius,
            HexPlayer::P2 => coord.s() == -self.radius,
            HexPlayer::P3 => coord.q == self.radius,
        }
    }

    fn is_on_player_end_side(&self, player: HexPlayer, coord: AxialCoord) -> bool {
        match player {
            HexPlayer::P1 => coord.r == -self.radius,
            HexPlayer::P2 => coord.s() == self.radius,
            HexPlayer::P3 => coord.q == -self.radius,
        }
    }

    pub fn get_axial_from_index(&self, index: i32) -> AxialCoord {
        let size = 2 * self.radius + 1;
        let center = self.radius;
        let row = index / size;
        let col = index % size;
        AxialCoord::new(col - center, row - center)
    }

    fn reconstruct_path(
        &self,
        end_coord: AxialCoord,
        parents: &HashMap<AxialCoord, Option<AxialCoord>>,
    ) -> Vec<AxialCoord> {
        let mut path = Vec::new();
        let mut current_opt = Some(end_coord);

        // Trace back from the end coordinate using the parents map
        while let Some(current) = current_opt {
            path.push(current);
            // Find the parent of the current node.
            // The loop terminates when we reach a start node, whose parent is `None`.
            current_opt = *parents.get(&current).unwrap_or(&None);
        }

        // The path is constructed from end to start, so we reverse it.
        path.reverse();
        path
    }

    /// Checks if a player has a connection between their sides.
    /// If `allow_empty_cells` is true, the path can go through empty cells (used for elimination check).
    /// If `allow_empty_cells` is false, the path must consist only of the player's stones (used for win check).
    pub fn check_connectivity(
        &self,
        player_to_check: HexPlayer,
        allow_empty_cells: bool,
    ) -> (bool, Option<Vec<AxialCoord>>) {
        let mut queue = VecDeque::new();

        let mut parents: HashMap<AxialCoord, Option<AxialCoord>> = HashMap::new();

        for i in 0..self.board.len() {
            let coord = self.get_axial_from_index(i as i32);

            if !self.is_in(&coord) {
                continue; // Skip coordinates outside the hexagonal grid
            }

            // Check if this coordinate is on the starting side for the player
            if self.is_on_player_start_side(player_to_check, coord) {
                let cell_part_of_path = match self.board[i] {
                    CellState::Occupied(p) => p == player_to_check,
                    CellState::Empty => allow_empty_cells,
                };

                if cell_part_of_path {
                    // This is a valid starting node for our search.
                    queue.push_back(coord);
                    // Mark as visited by adding to parents map. Start nodes have no parent.
                    parents.insert(coord, None);

                    // Edge case: A single cell can be a winning path if it's on both sides.
                    if self.is_on_player_end_side(player_to_check, coord) {
                        return (true, Some(vec![coord]));
                    }
                }
            }
        }

        while let Some(current_coord) = queue.pop_front() {
            // This was already checked for the initial cells.
            if self.is_on_player_end_side(player_to_check, current_coord) {
                return (true, Some(self.reconstruct_path(current_coord, &parents)));
            }

            for neighbor_coord in current_coord.neighbors() {
                // Skip neighbors that are out of bounds or already visited.
                // A cell is "visited" if it's in our `parents` map.
                if !self.is_in(&neighbor_coord) || parents.contains_key(&neighbor_coord) {
                    continue;
                }

                let i = self.index_from_qr(neighbor_coord.q, neighbor_coord.r);

                let cell_part_of_path = match self.board.get(i) {
                    Some(&CellState::Occupied(p)) => p == player_to_check,
                    Some(&CellState::Empty) => allow_empty_cells,
                    None => false, // Should not happen due to `is_in` check, but good to be safe.
                };

                if cell_part_of_path {
                    // This neighbor is part of a potential path.
                    // Record its parent and add it to the queue to visit later.
                    parents.insert(neighbor_coord, Some(current_coord));
                    queue.push_back(neighbor_coord);

                    // Optimization: Check if this neighbor immediately completes the path.
                    if self.is_on_player_end_side(player_to_check, neighbor_coord) {
                        return (true, Some(self.reconstruct_path(neighbor_coord, &parents))); // Path found!
                    }
                }
            }
        }

        (false, None) // No path found
    }

    fn is_connected_optimized(&self, player_to_check: HexPlayer, allow_empty_cells: bool) -> bool {
        let num_cells = self.board.len();
        let mut queue = VecDeque::with_capacity(num_cells / 2);
        let mut visited = vec![false; num_cells];

        for r in -self.radius..=self.radius {
            let q_start = (-self.radius).max(-r - self.radius);
            let q_end = self.radius.min(-r + self.radius);

            for q in q_start..=q_end {
                let coord = AxialCoord::new(q, r);
                if !self.is_on_player_start_side(player_to_check, coord) {
                    continue;
                }

                let i = self.index_from_qr(coord.q, coord.r);

                let cell_part_of_path = match self.board[i] {
                    CellState::Occupied(p) => p == player_to_check,
                    CellState::Empty => allow_empty_cells,
                };

                if cell_part_of_path {
                    if self.is_on_player_end_side(player_to_check, coord) {
                        return true; // Edge case: stone connects both sides.
                    }

                    if !visited[i] {
                        visited[i] = true;
                        queue.push_back(coord);
                    }
                }
            }
        }

        // BFS
        while let Some(current_coord) = queue.pop_front() {
            for neighbor_coord in current_coord.neighbors() {
                if !self.is_in(&neighbor_coord) {
                    continue;
                }

                let neighbor_idx = self.index_from_qr(neighbor_coord.q, neighbor_coord.r);

                if visited[neighbor_idx] {
                    continue;
                }

                let cell_part_of_path = match self.board[neighbor_idx] {
                    CellState::Occupied(p) => p == player_to_check,
                    CellState::Empty => allow_empty_cells,
                };

                if cell_part_of_path {
                    if self.is_on_player_end_side(player_to_check, neighbor_coord) {
                        return true; // Path found so terminate early.
                    }

                    visited[neighbor_idx] = true;
                    queue.push_back(neighbor_coord);
                }
            }
        }

        false
    }

    pub fn make_move_mut(&mut self, coord: AxialCoord) -> Result<(), String> {
        if self.outcome.is_some() {
            return Err("Game is already over.".to_string());
        }

        debug_assert!(
            !self.eliminated_players.contains(&self.current_turn),
            "Current player {:?} should not be eliminated at this point.",
            self.current_turn
        );

        let i = self.index_from_qr(coord.q, coord.r);

        assert!(i < self.board.len(), "Index out of bounds: {i}");

        match self.board.get(i) {
            Some(&CellState::Empty) => {}
            Some(_) => return Err(format!("Cell {coord:?} is already occupied.")),
            None => unreachable!(), // already asserted
        }

        self.board[i] = CellState::Occupied(self.current_turn);

        let player_who_moved = self.current_turn;

        // XOR out the previous turn's hash.
        self.zobrist_hash ^= self.zobrist_table.turn(player_who_moved);

        // XOR in the new piece's hash.
        self.zobrist_hash ^= self.zobrist_table.piece(i, player_who_moved);

        if self.is_connected_optimized(player_who_moved, false) {
            // If the player has a winning path, we set the outcome.
            self.outcome = Some(HexGameOutcome::Win {
                winner: player_who_moved,
            });
            // Also update the eliminated player, since if one player actually connects,
            // the other two implicitly cannot.
            for &p in HexPlayer::all().iter() {
                if p != player_who_moved {
                    self.eliminated_players.insert(p);
                }
            }

            return Ok(());
        }

        for &p_to_check in HexPlayer::all().iter() {
            if !self.eliminated_players.contains(&p_to_check) && p_to_check != player_who_moved {
                // If player p_to_check can NO LONGER connect (even with empty cells allowed in path)
                if !self.is_connected_optimized(p_to_check, true) {
                    self.eliminated_players.insert(p_to_check);
                }
            }
        }

        if self.eliminated_players.len() == 2 {
            self.outcome = Some(HexGameOutcome::Win {
                winner: player_who_moved,
            });
            return Ok(());
        }

        let mut next_player_candidate = player_who_moved.next_hex_player();

        loop {
            if !self.eliminated_players.contains(&next_player_candidate) {
                self.current_turn = next_player_candidate;
                break;
            }
            next_player_candidate = next_player_candidate.next_hex_player();
        }

        // XOR in the new turn's hash.
        self.zobrist_hash ^= self.zobrist_table.turn(self.current_turn);

        Ok(())
    }

    // Helper to get valid empty cells for a player to choose from
    pub fn get_valid_empty_cells(&self) -> Vec<AxialCoord> {
        debug_assert!(
            self.outcome.is_none(),
            "Cannot get valid empty cells on a terminal state."
        );

        self.board
            .iter()
            .enumerate()
            .filter_map(|(i, &state)| {
                let coord = self.get_axial_from_index(i as i32);
                if !self.is_in(&coord) {
                    return None;
                }
                if state == CellState::Empty {
                    Some(coord)
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn fill_vector_with_moves(&self, store: &mut Vec<AxialCoord>) {
        debug_assert!(
            self.outcome.is_none(),
            "Cannot get valid empty cells on a terminal state."
        );

        store.clear(); // Ensure the store is empty before populating

        // Same logic as `get_valid_empty_cells`, but fills the provided vector
        for (i, &state) in self.board.iter().enumerate() {
            let coord = self.get_axial_from_index(i as i32);
            if !self.is_in(&coord) {
                continue;
            }
            if state == CellState::Empty {
                store.push(coord);
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (AxialCoord, CellState)> + '_ {
        self.board
            .iter()
            .enumerate()
            .filter_map(move |(i, &state)| {
                let coord = self.get_axial_from_index(i as i32);
                if self.is_in(&coord) {
                    Some((coord, state))
                } else {
                    None
                }
            })
    }

    pub fn is_in(&self, AxialCoord { q, r }: &AxialCoord) -> bool {
        q.abs() <= self.radius
            && r.abs() <= self.radius
            && q + r >= -self.radius
            && q + r <= self.radius
    }

    pub fn fancy_debug_visualize_board_indented(
        &self,
        net: Option<(HashMap<AxialCoord, f32>, Vec<f32>)>,
    ) -> String {
        if let Some((moves_with_policy, value)) = &net {
            assert_eq!(
                moves_with_policy.len(),
                self.num_of_hexes as usize,
                "Policy length must match board size."
            );
            assert_eq!(value.len(), 3, "Value length must be 3 for three players.");
            for v in value {
                assert!(
                    -1.0 <= *v && *v <= 1.0,
                    "Value must be in range [-1, 1]. Was: {value:?}"
                );
            }
            for (_, p) in moves_with_policy.iter() {
                assert!(0.0 <= *p && *p <= 1.0, "Policy must be in range [0, 1].");
            }
        }

        let mut output = String::new();

        colored::control::set_override(true);

        let (connected, connections) = if let Some(outcome) = self.outcome {
            match outcome {
                HexGameOutcome::Win { winner } => self.check_connectivity(winner, false),
            }
        } else {
            (false, None)
        };

        if let Some(outcome) = self.outcome {
            match outcome {
                HexGameOutcome::Win { winner } => {
                    // Color the winner announcement for extra flair
                    let winner_text = match winner {
                        HexPlayer::P1 => "P1".red().bold(),
                        HexPlayer::P2 => "P2".blue().bold(),
                        HexPlayer::P3 => "P3".yellow().bold(),
                    };

                    output.push_str(&format!("[!] Winner of hex: {winner_text}\n"));

                    if connected && let Some(path) = &connections {
                        output.push_str("Winning path: ");
                        for coord in path {
                            output.push_str(&format!("(q={}, r={}) ", coord.q, coord.r));
                        }
                        output.push('\n');
                    } else {
                        output.push_str("No winning path found, but the game is over.\n");
                    }
                }
            }
        } else {
            output.push_str(&format!(
                "It's {} turn.\n",
                match self.current_turn {
                    HexPlayer::P1 => "P1".red().bold(),
                    HexPlayer::P2 => "P2".blue().bold(),
                    HexPlayer::P3 => "P3".yellow().bold(),
                },
            ));
        }

        // Iterate through each row (r)
        for r in -self.radius - 1..=self.radius + 1 {
            // Indent each row to form the hexagonal shape
            let indent_amount = r.abs();
            output.push_str(&" ".repeat(indent_amount as usize));

            // For a given r, q has a specific valid range in a hexagon.
            let q_start = (-self.radius).max(-r - self.radius);
            let q_end = self.radius.min(-r + self.radius);

            for q in q_start - 1..=q_end + 1 {
                let s = -q - r;

                let distance = (q.abs() + r.abs() + s.abs()) / 2;

                if distance == self.radius + 1 {
                    let vis_radius = self.radius + 1;

                    if r.abs() == vis_radius {
                        output.push_str(&"XX".on_red().to_string());
                    }
                    // P2's sides are top-right/bottom-left (s = +/- radius)
                    else if s.abs() == vis_radius {
                        output.push_str(&"XX".on_blue().to_string());
                    }
                    // P3's sides are top-left/bottom-right (q = +/- radius)
                    else if q.abs() == vis_radius {
                        output.push_str(&"XX".on_yellow().to_string());
                    } else {
                        // This case should not be reached with the current loop bounds,
                        // but is here for safety.
                        output.push_str("  ");
                    }

                    continue; // Skip coordinates outside the hexagonal grid
                } else if distance > self.radius {
                    continue;
                }

                let i = self.index_from_qr(q, r);

                match self.board[i] {
                    CellState::Empty => {
                        let mut dot = ColoredString::from(". ");

                        if let Some((moves_with_policy, _)) = &net {
                            // Highlight the cell based on the policy value
                            dot = policy_color_viridis(
                                *moves_with_policy
                                    .get(&AxialCoord::new(q, r))
                                    .expect("Policy missing for coord"),
                                dot,
                            )
                        }

                        output.push_str(&dot.to_string());
                    }
                    CellState::Occupied(player) => {
                        let mut player_char = match player {
                            HexPlayer::P1 => "1 ".red(),
                            HexPlayer::P2 => "2 ".blue(),
                            HexPlayer::P3 => "3 ".yellow(),
                        };

                        if let Some((moves_with_policy, _)) = &net {
                            // Highlight the cell based on the policy value
                            player_char = policy_color_viridis(
                                *moves_with_policy
                                    .get(&AxialCoord::new(q, r))
                                    .expect("Policy missing for coord"),
                                player_char,
                            );
                        }

                        if connected && let Some(path) = &connections {
                            // Highlight the winning path
                            if path.contains(&AxialCoord::new(q, r)) {
                                output.push_str(&player_char.underline().to_string());
                            } else {
                                output.push_str(&player_char.to_string());
                            }
                        } else {
                            output.push_str(&player_char.to_string());
                        }
                    }
                }
            }
            output.push('\n');
        }

        if let Some((_, value)) = net {
            for (v, player) in value.iter().zip(HexPlayer::all().iter()) {
                let mut color = match player {
                    HexPlayer::P1 => "P1".red(),
                    HexPlayer::P2 => "P2".blue(),
                    HexPlayer::P3 => "P3".yellow(),
                };

                if self.eliminated_players.contains(player) {
                    color = color.strikethrough()
                }

                if *player == self.current_turn {
                    color = color.underline()
                }

                output.push_str(&format!("{}: {:.2}  ", color.bold(), v));
            }
        }

        output
    }

    pub fn debug_visualize_board_indented(&self) -> String {
        let mut output = String::new();

        if let Some(outcome) = self.outcome {
            match outcome {
                HexGameOutcome::Win { winner } => {
                    output.push_str(&format!("[!] Winner of hex: {winner:?}\n"));
                }
            }
        } else {
            output.push_str("... Hex game is still ongoing...\n");
        }

        // Iterate through each row (r)
        for r in -self.radius..=self.radius {
            let indent_amount = r.abs();
            output.push_str(&" ".repeat(indent_amount as usize));

            // For a given r, q has a specific valid range in a hexagon.
            // This is more efficient than iterating over the full square.
            // The constraint comes from cube coordinates (q+r+s=0) where |s| <= radius.
            let q_start = (-self.radius).max(-r - self.radius);
            let q_end = self.radius.min(-r + self.radius);

            for q in q_start..=q_end {
                // We know the coordinate is valid because of our loop bounds.
                let i = self.index_from_qr(q, r); // .unwrap() is safe here

                match self.board[i] {
                    CellState::Empty => output.push_str(". "),
                    CellState::Occupied(player) => {
                        let player_char = match player {
                            HexPlayer::P1 => "1",
                            HexPlayer::P2 => "2",
                            HexPlayer::P3 => "3",
                        };
                        output.push_str(player_char);
                        output.push(' ');
                    }
                }
            }
            output.push('\n');
        }
        output
    }

    pub fn play_random_move(&mut self, rng: &mut impl Rng) {
        if self.outcome.is_some() {
            return; // Game is already over
        }

        let valid_moves = self.get_valid_empty_cells();
        if valid_moves.is_empty() {
            return; // No valid moves available
        }
        let random_index = rng.random_range(0..valid_moves.len());
        let chosen_move = valid_moves[random_index];
        self.make_move_mut(chosen_move)
            .expect("Random move should be valid");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_game_new() {
        let game = HexGame::new(4).unwrap();

        println!("{:?}", game.get_valid_empty_cells());

        let game_size_1 = HexGame::new(4).unwrap();

        println!("{}", game_size_1.debug_visualize_board_indented());
    }

    #[test]
    fn test_player_next_turn() {
        assert_eq!(HexPlayer::P1.next_hex_player(), HexPlayer::P2);
        assert_eq!(HexPlayer::P2.next_hex_player(), HexPlayer::P3);
        assert_eq!(HexPlayer::P3.next_hex_player(), HexPlayer::P1);
    }

    #[test]
    fn test_size() {
        let game = HexGame::new(2).unwrap();
        assert_eq!(game.radius, 1); // size 2 means radius 1
        assert_eq!(game.board.len(), 9);
        assert_eq!(game.num_of_hexes, 7);
    }

    #[test]
    fn test_simple_win_p1_size2() {
        let mut game = HexGame::new(2).unwrap();

        game.make_move_mut(AxialCoord::new(-1, 0)).unwrap(); // P1
        assert_eq!(game.get_current_player(), HexPlayer::P2);
        game.make_move_mut(AxialCoord::new(0, 1)).unwrap(); // P2
        assert_eq!(game.get_current_player(), HexPlayer::P3);
        game.make_move_mut(AxialCoord::new(1, -1)).unwrap(); // P3
        assert_eq!(game.get_current_player(), HexPlayer::P1);

        println!("{game:?}");

        game.make_move_mut(AxialCoord::new(0, 0)).unwrap(); // P1
        println!("{game:?}");

        assert_eq!(
            game.get_outcome(),
            Some(HexGameOutcome::Win {
                winner: HexPlayer::P1
            })
        );
        assert_eq!(game.eliminated_players.len(), 2);
    }

    #[test]
    fn test_win_on_size_1_board() {
        let mut game = HexGame::new(1).unwrap();
        // assert!(game.board.contains_key(&AxialCoord::new(0, 0)));

        game.make_move_mut(AxialCoord::new(0, 0)).unwrap();
        assert_eq!(
            game.get_outcome(),
            Some(HexGameOutcome::Win {
                winner: HexPlayer::P1
            })
        );
    }

    #[test]
    fn test_elimination_scenario() {
        let mut game = HexGame::new(2).unwrap(); // R = 1

        game.make_move_mut(AxialCoord::new(0, 1)).unwrap();
        game.make_move_mut(AxialCoord::new(0, 0)).unwrap();
        // P3 moves
        game.make_move_mut(AxialCoord::new(-1, 1)).unwrap();

        println!("{:?}", game.eliminated_players);

        println!("{:?}", game);

        assert!(
            !game.eliminated_players.contains(&HexPlayer::P2),
            "P2 should not be eliminated yet."
        );

        game.make_move_mut(AxialCoord::new(0, -1)).unwrap();

        assert!(
            game.eliminated_players.contains(&HexPlayer::P3),
            "P3 Should be eliminated after P1's move."
        );

        println!("{:?}", game.eliminated_players);
        println!("{game:?}");

        // P2's turn
        game.make_move_mut(AxialCoord::new(1, 0)).unwrap();

        println!("{:?}", game.eliminated_players);
        println!("{game:?}");

        assert!(
            game.eliminated_players.contains(&HexPlayer::P1),
            "P1 should be eliminated."
        );
        assert_eq!(
            game.outcome.unwrap(),
            HexGameOutcome::Win {
                winner: HexPlayer::P2
            }
        );
    }

    #[test]
    fn test_winner_by_elimination_of_others() {
        let mut game = HexGame::new(2).unwrap();

        game.eliminated_players.insert(HexPlayer::P2);
        game.eliminated_players.insert(HexPlayer::P3);

        // P1 makes any valid move.
        game.make_move_mut(AxialCoord::new(0, 0)).unwrap();

        assert_eq!(
            game.get_outcome(),
            Some(HexGameOutcome::Win {
                winner: HexPlayer::P1
            })
        );
    }

    #[test]
    fn test_fancy_output() {
        let mut board = HexGame::new(3).unwrap();

        colored::control::set_override(true);

        println!("First: Test {}", "colored output".green().bold());

        board.make_move_mut(AxialCoord::new(0, 0)).unwrap(); // P1
        board.make_move_mut(AxialCoord::new(1, 0)).unwrap(); // P2

        board.make_move_mut(AxialCoord::new(0, 1)).unwrap(); // P3

        println!("{}", board.fancy_debug_visualize_board_indented(None));
    }

    #[test]
    fn test_fancy_output_with_policy() {
        let mut board = HexGame::new(3).unwrap();

        colored::control::set_override(true);

        println!("First: Test {}", "colored output".green().bold());

        let all_available_moves = board.get_valid_empty_cells().clone();

        board.make_move_mut(AxialCoord::new(0, 0)).unwrap(); // P1
        board.make_move_mut(AxialCoord::new(1, 0)).unwrap(); // P2

        board.make_move_mut(AxialCoord::new(0, 1)).unwrap(); // P3

        let mut rng = rand::rng();

        let mut policy = HashMap::new();

        for coord in all_available_moves {
            policy.insert(coord, rng.random_range(0.0..1.0));
        }

        let value = vec![0.1, 0.2, 0.3]; // Dummy value for each player

        // Prints all colors from 0-255, but you need to have COLORTERM=truecolor in env vars
        for col_num in 0..77 {
            let r = 255 - (col_num * 255 / 76);
            let g = col_num * 510 / 76;
            let b = col_num * 255 / 76;
            let g = if g > 255 { 510 - g } else { g };
            let color = ".".on_truecolor(r as u8, g as u8, b as u8);
            print!("{}", color);
        }

        println!();
        println!(
            "{}",
            board.fancy_debug_visualize_board_indented(Some((policy, value)))
        );
    }

    #[test]
    fn test_fancy_output_2() {
        let mut board = HexGame::new(2).unwrap();

        // Place some pieces for demonstration
        board.make_move_mut(AxialCoord::new(0, 0)).unwrap(); // P1
        board.make_move_mut(AxialCoord::new(1, 0)).unwrap(); // P2
        board.make_move_mut(AxialCoord::new(0, 1)).unwrap(); // P3
        board.make_move_mut(AxialCoord::new(-1, 1)).unwrap(); // P1
        board.make_move_mut(AxialCoord::new(0, -1)).unwrap(); // P2

        let mut other_possible_board = board.clone();

        board.make_move_mut(AxialCoord::new(1, -1)).unwrap(); // P1

        println!("{}", board.fancy_debug_visualize_board_indented(None));

        other_possible_board
            .make_move_mut(AxialCoord::new(-1, 0))
            .unwrap(); // P2

        println!("{:?}", other_possible_board.get_valid_empty_cells());

        other_possible_board
            .make_move_mut(AxialCoord::new(1, -1))
            .unwrap(); // P3

        println!(
            "{}",
            other_possible_board.fancy_debug_visualize_board_indented(None)
        );
    }
}
