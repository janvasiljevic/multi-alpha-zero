use maz_core::mapping::Board;
use schemars::JsonSchema;
use serde::Serialize;
use std::fmt::{Debug, Display};
use maz_trainer::tree::Tree;

// The attributes for each node in the JSON tree.
#[derive(Serialize, JsonSchema)]
pub struct ApiNodeAttributes {
    /// The number of times this node was visited during the MCTS search.
    visit_count: u64,
    /// The raw policy value from the neural network for the move leading to this node.
    raw_policy: f32,

    net_value: Option<Vec<f32>>,

    sum_values: Vec<f32>,

    player_to_move: Option<usize>,

    is_terminal: bool, // Indicates if this node is a terminal state (game over)

    move_indexes: Vec<usize>, // The indexes of the moves in the policy vector that led to this node
}

// The recursive structure for a single node in the JSON tree.
#[derive(Serialize, JsonSchema)]
pub struct ApiNode {
    /// A human-readable name for the node, typically the move that was played to reach it.
    name: String,

    /// MCTS-specific data for this node.
    attributes: ApiNodeAttributes,

    /// The children of this node, representing subsequent possible moves.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<ApiNode>,
}

/// Helper function to build the tree recursively
fn build_recursive<B, const N: usize>(
    tree: &Tree<B, N>,
    node_id: usize,
    depth: usize,
    max_depth: usize,
    max_children: usize,
    moves_indexes: Vec<usize>,
) -> ApiNode
where
    B: Board,
    B::Move: Display,
{
    let node = &tree[node_id];
    let node_name = if node_id == 0 {
        "Root".to_string()
    } else {
        // Format the move that led to this node.
        format!(
            "{}",
            node.last_move.expect("Non-root node must have a move")
        )
    };

    let attributes = ApiNodeAttributes {
        visit_count: node.complete_visits,
        raw_policy: node.net_policy,
        sum_values: node.values().value_abs.to_vec(),
        net_value: node.net_values.as_ref().map(|v| v.value_abs.to_vec()),
        player_to_move: node.player_index,
        is_terminal: node.is_terminal_node(),
        move_indexes: moves_indexes.clone(), // Pass the move indexes down to children
    };

    let mut children = Vec::new();
    if depth < max_depth {
        if let Some(child_indices) = node.children {
            let mut sorted_children: Vec<usize> = child_indices.iter().collect();

            // Sort children by visit count in descending order to show the most promising lines first.
            sorted_children
                .sort_by_key(|&child_id| std::cmp::Reverse(tree[child_id].complete_visits));

            for child_id in sorted_children.iter().take(max_children) {
                // We only include children that were actually visited to keep the tree clean.
                if tree[*child_id].complete_visits > 0 {
                    children.push(build_recursive(
                        tree,
                        *child_id,
                        depth + 1,
                        max_depth,
                        max_children,
                        moves_indexes
                            .iter()
                            .chain(std::iter::once(&tree[*child_id].last_move_policy_index))
                            .cloned()
                            .collect::<Vec<_>>(),
                    ));
                }
            }
        }
    }

    ApiNode {
        name: node_name,
        attributes,
        children,
    }
}

/// Public entry point to build the API-serializable tree from the MCTS result.
///
/// # Arguments
/// * `tree` - The MCTS search tree.
/// * `max_depth` - The maximum depth to serialize. Prevents gigantic JSON responses.
/// * `max_children` - The maximum number of children per node to serialize (sorted by visit count).
pub fn build_api_tree<B, const N: usize>(
    tree: &Tree<B, N>,
    max_depth: usize,
    max_children: usize,
) -> Option<ApiNode>
where
    B: Board,
    B::Move: Debug,
{
    Some(build_recursive(tree, 0, 0, max_depth, max_children, vec![]))
}
