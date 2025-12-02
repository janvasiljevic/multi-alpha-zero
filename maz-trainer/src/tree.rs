use crate::node::{Node, UctContext};
use itertools::Itertools;
use maz_core::mapping::Board;
use maz_core::values_const::ValuesAbs;
use maz_util::display::display_option;
use std::borrow::Cow;
use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::num::{NonZero, NonZeroUsize};
use std::ops::{Index, IndexMut};

/// The result of a zero search.
#[derive(Debug, Clone)]
pub struct Tree<B: Board, const N: usize> {
    root_board: B,
    pub nodes: Vec<Node<B::Move, N>>,
}

// Not sure if we even need this - I think it's more used in runtime for recycling trees not as
// part of training...
// #[derive(Debug, Copy, Clone)]
// pub enum KeepMoveError {
//     Outcome { depth: u32, outcome: Outcome },
//     NotVisitedYet { depth: u32 },
// }

pub const ROOT_NODE_ID: usize = 0;

#[allow(dead_code)] // A lot of the methods are used outside of this crate or only in tests.
impl<B: Board, const N: usize> Tree<B, N> {
    pub fn new_with_capacity(root_board: B, capacity_hint: usize) -> Self {
        assert!(
            !root_board.is_terminal(),
            "Cannot build tree for terminal board"
        );

        let root = Node::new(
            None,
            None,
            0,
            f32::NAN,
            Some(root_board.player_current().into()),
        );

        let mut nodes = Vec::with_capacity(capacity_hint);

        nodes.push(root);

        Tree { root_board, nodes }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_board(&self) -> &B {
        &self.root_board
    }

    pub fn uct_context(&self, node: usize) -> UctContext<N> {
        let node = &self[node];

        let visited_policy_mass = node.children.map_or(0.0, |children| {
            children
                .iter()
                .map(|c| {
                    let child = &self[c];
                    if child.total_visits() > 0 {
                        child.net_policy
                    } else {
                        0.0
                    }
                })
                .sum()
        });

        node.uct_context(visited_policy_mass)
    }

    pub fn capacity(&self) -> usize {
        self.nodes.capacity()
    }

    /// Retain a subtree with the given `new_root_node_id`.
    /// This will promote the node at `new_root_node_id` to be the new root of the tree,
    /// but only update the new root's children to point to it.
    /// This leaves a lot of old nodes "dead" in the vector, but avoids costly memory operations.
    /// Call [`defragment_and_retain_subtree`] periodically to clean up the tree and reclaim memory,
    /// or else the tree will grow indefinitely with dead nodes.
    /// Explanation: If we have a game with 70 move avg., 400 simulations and 37 avg. branch factor,
    /// this can quickly make the tree grow to 1M nodes, which is around 200MB of memory.
    /// With 2k concurrent games, this is just too much....
    pub fn retain_subtree(&mut self, new_board: B, new_root_node_id: NonZeroUsize) {
        assert!(
            new_root_node_id < NonZeroUsize::new(self.nodes.len()).expect("Tree is empty"),
            "New root node ID is out of bounds."
        );

        let new_root_node_data = self.nodes[new_root_node_id.get()].clone();

        self.nodes[ROOT_NODE_ID] = new_root_node_data;
        self.nodes[ROOT_NODE_ID].set_root_properties();

        // Update the parent pointers of the new root's direct children.
        if let Some(children) = self.nodes[ROOT_NODE_ID].children {
            for child_id in children {
                self.nodes[child_id].parent = Some(ROOT_NODE_ID);
            }
        }

        self.root_board = new_board;
    }

    /// Same as [`retain_subtree`], but also defragments the tree by removing dead nodes.
    /// It's much slower than [`retain_subtree`], so use it only when necessary.
    pub fn retain_subtree_with_defrag(
        &mut self,
        new_board: B,
        old_new_root_id: NonZeroUsize,
        avg_num_of_moves: u64, // Performance hint
    ) {
        assert!(
            old_new_root_id < NonZeroUsize::new(self.nodes.len()).expect("Tree is empty"),
            "Old new root ID is out of bounds."
        );

        let old_nodes = std::mem::take(&mut self.nodes);

        let reasonable_capacity_guess = old_nodes.len(); // Maybe needs fine-tuning, not sure yet.
        let mut new_nodes = Vec::with_capacity(reasonable_capacity_guess);

        // Old (unfragmented) to new (fragmented) ID mapping.
        let mut old_to_new: std::collections::HashMap<NonZeroUsize, usize> =
            std::collections::HashMap::new();

        let mut bfs_queue = std::collections::VecDeque::with_capacity(
            (old_nodes[old_new_root_id.get()].complete_visits * avg_num_of_moves) as usize,
        );

        bfs_queue.push_back(old_new_root_id);
        old_to_new.insert(old_new_root_id, ROOT_NODE_ID);

        let mut new_id_counter = 1;
        let mut q_read_idx = ROOT_NODE_ID; // Chases the length of the queue

        while q_read_idx < bfs_queue.len() {
            let parent_old_id = bfs_queue[q_read_idx];
            q_read_idx += 1;

            if let Some(children) = old_nodes[parent_old_id.get()].children {
                for child_old_id in children {
                    // Use this if we implement DAG (Directed Acyclic Graph) support in the future.
                    // if !old_to_new.contains_key(&child_old_id) {

                    let child_old_id =
                        NonZeroUsize::new(child_old_id).expect("Children IDs must be non-zero");

                    old_to_new.insert(child_old_id, new_id_counter);
                    new_id_counter += 1;
                    bfs_queue.push_back(child_old_id);
                }
            }
        }

        // println!("Total queue length: {}", bfs_queue.len()); // Debugging stats for memcpy

        // Pre allocate the new nodes vector.
        new_nodes.resize(bfs_queue.len(), Node::new(None, None, 0, f32::NAN, None));

        for old_id in bfs_queue {
            let mut node_copy = old_nodes[old_id.get()].clone();
            let new_id = old_to_new[&old_id];

            // Fix self parent
            node_copy.parent = node_copy.parent.and_then(|p_old| {
                if p_old == ROOT_NODE_ID {
                    None // Root node has no parent
                } else {
                    // Map old parent ID to new ID
                    old_to_new
                        .get(&NonZero::new(p_old).expect("Old parent couldn't have been 0!"))
                        .copied()
                }
            });

            // Fix children range!
            if let Some(mut children) = node_copy.children {
                children.start = NonZero::new(old_to_new[&children.start])
                    .expect("Children start index must be non-zero");
                node_copy.children = Some(children);
            }

            new_nodes[new_id] = node_copy;
        }

        self.nodes = new_nodes;
        self.nodes[ROOT_NODE_ID].set_root_properties();
        self.root_board = new_board;
    }

    pub fn best_child(&self, node: usize) -> Option<usize> {
        self[node].children.map(|children| {
            children
                .iter()
                .max_by_key(|&child| {
                    (
                        self[child].complete_visits,
                        decorum::Total::from(self[child].net_policy),
                    )
                })
                .unwrap()
        })
    }

    // Find the principal variation up to `max_len` moves. (so best moves until a leaf or max_len)
    // pub fn principal_variation(&self, max_len: usize) -> Vec<B::Move> {
    //     std::iter::successors(Some(0), |&n| self.best_child(n))
    //         .skip(1)
    //         .take(max_len)
    //         .map(|n| self[n].last_move.unwrap())
    //         .collect()
    // }

    pub fn root_visits(&self) -> u64 {
        self[ROOT_NODE_ID].complete_visits
    }

    pub fn root_virtual_visits(&self) -> u64 {
        self[ROOT_NODE_ID].virtual_visits
    }

    /// Return `(min, max)` where `min` is the depth of the shallowest un-evaluated node
    /// and `max` is the depth of the deepest evaluated node.
    /// Used for sample statistics and debugging.
    pub fn depth_range(&self, node: usize) -> (usize, usize) {
        match self[node].children {
            None => (0, 0),
            Some(children) => {
                let mut total_min = usize::MAX;
                let mut total_max = usize::MIN;

                for child in children {
                    let (c_min, c_max) = self.depth_range(child);
                    total_min = min(total_min, c_min);
                    total_max = max(total_max, c_max);
                }

                (total_min + 1, total_max + 1)
            }
        }
    }

    /// Reconstructs the policy vector for the root node, with a length of `len`.
    /// Used for sample statistics and debugging.
    pub fn net_policy(&self, len: usize) -> Cow<'_, [f32]> {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let mut policy = vec![0.0; len];

        for c in self[0].children.unwrap() {
            let node = &self[c];

            policy[node.last_move_policy_index] = node.net_policy;
        }

        Cow::Owned(policy)
    }

    pub fn root_q_abs_values(&self) -> ValuesAbs<N> {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        self[0].values()
    }

    /// Either returns the Q values of the best child of the root node,
    /// or the root node's values if there are no children.
    pub fn best_child_q_abs_values(&self) -> ValuesAbs<N> {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let best_child_id = self.best_child(ROOT_NODE_ID);

        match best_child_id {
            Some(id) => {
                if self[id].complete_visits > 0 {
                    self[id].values()
                } else {
                    self[ROOT_NODE_ID].values()
                }
            }
            None => self[ROOT_NODE_ID].values(),
        }
    }

    /// Fills the given `policy_buffer` with the improved policy from MCTS and
    /// the `mask_buffer` with valid moves.
    pub fn improved_policy_vector(
        &self,
        policy_buffer: &mut Vec<f32>,
        mask_buffer: &mut Vec<bool>,
    ) {
        debug_assert!(self.len() > 1, "Must have run for at least 1 iteration");

        policy_buffer.fill(0.0);
        mask_buffer.fill(false);

        for c in self[ROOT_NODE_ID].children.unwrap() {
            let policy_index = self[c].last_move_policy_index;

            policy_buffer[policy_index] = self.get_node_improved_policy(c);
            mask_buffer[policy_index] = true;
        }
    }

    pub fn get_node_improved_policy(&self, node_id: usize) -> f32 {
        debug_assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let node = &self[node_id];
        let parent_id = node.parent.expect("Node must have a parent");
        let parent = &self[parent_id];

        let visits = node.complete_visits as f32;

        if visits > 0.0 {
            visits / (parent.complete_visits as f32 - 1.0).max(0.0)
        } else {
            0.0
        }
    }

    pub fn root_nodes(&self) -> impl Iterator<Item = Node<B::Move, N>> + '_ {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        self[ROOT_NODE_ID]
            .children
            .iter()
            .flat_map(|children| children.iter().map(|c| self[c].clone()))
    }

    #[must_use]
    pub fn display(
        &self,
        max_depth: usize,
        sort: bool,
        max_children: usize,
        expand_all: bool,
    ) -> TreeDisplay<'_, B, N> {
        TreeDisplay {
            tree: self,

            node: ROOT_NODE_ID,

            curr_depth: 0,

            max_depth,
            max_children,
            sort,
            expand_all,
        }
    }
}

#[derive(Debug)]
pub struct TreeDisplay<'a, B: Board, const N: usize> {
    tree: &'a Tree<B, N>,

    node: usize,
    curr_depth: usize,

    max_depth: usize,
    max_children: usize,
    sort: bool,
    expand_all: bool,
}

impl<B: Board, const N: usize> Display for TreeDisplay<'_, B, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tree = self.tree;

        for _ in 0..self.curr_depth {
            write!(f, "  ")?
        }

        let node = &self.tree[self.node];
        let parent = node.parent.map(|p| &self.tree[p]);

        let virtual_visits = if node.virtual_visits != 0 {
            format!("+{}", node.virtual_visits)
        } else {
            String::default()
        };

        let node_values = node.values();

        let net_values = node.net_values.unwrap_or_else(|| ValuesAbs::default());

        let parent_complete_visits = parent.map_or(f32::NAN, |p| p.complete_visits as f32);

        let zero_policy = if parent_complete_visits > 0.0 {
            (node.complete_visits as f32) / (parent_complete_visits - 1.0)
        } else {
            f32::NAN
        };

        writeln!(
            f,
            "{} {}: {:?} {}{} zero({}, {:.4}) net({}, {:.4})",
            display_option(node.player_index),
            display_option(node.last_move),
            if node.is_terminal_node() {
                "Term."
            } else {
                "/"
            },
            node.complete_visits,
            virtual_visits,
            node_values,
            zero_policy,
            net_values,
            node.net_policy,
        )?;

        if self.curr_depth == self.max_depth {
            return Ok(());
        }

        if let Some(children) = node.children {
            let mut children = children.iter().collect_vec();

            let best_child = if self.sort {
                // sort by visits first, then by policy
                children.sort_by_key(|&c| {
                    (
                        self.tree[c].complete_visits,
                        decorum::Total::from(self.tree[c].net_policy),
                    )
                });
                children.reverse();
                children[0]
            } else {
                children
                    .iter()
                    .copied()
                    .max_by_key(|&c| self.tree[c].complete_visits)
                    .unwrap()
            };

            for (i, &child) in children.iter().enumerate() {
                assert_eq!(tree[child].parent, Some(self.node));

                if i == self.max_children {
                    for _ in 0..(self.curr_depth + 1) {
                        write!(f, "  ")?
                    }
                    writeln!(f, "...")?;
                    break;
                }

                let next_max_depth = if self.expand_all || child == best_child {
                    self.max_depth
                } else {
                    self.curr_depth + 1
                };

                let child_display = TreeDisplay {
                    tree: self.tree,

                    node: child,
                    // parent_player: curr_player,
                    curr_depth: self.curr_depth + 1,

                    max_depth: next_max_depth,
                    max_children: self.max_children,
                    sort: self.sort,
                    expand_all: self.expand_all,
                };
                write!(f, "{}", child_display)?;
            }
        }

        Ok(())
    }
}

impl<B: Board, const N: usize> Index<usize> for Tree<B, N> {
    type Output = Node<B::Move, N>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl<B: Board, const N: usize> IndexMut<usize> for Tree<B, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}
