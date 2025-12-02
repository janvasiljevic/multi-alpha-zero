use maz_core::mapping::chess_wrapper_mapper::{ChessWrapperMapper, ALL_CHESS_MAPPERS};
use maz_core::mapping::{InputMapper, Outcome};
use maz_util::network::NetShapesInfo;

pub fn outcome_to_ranks(outcome: Outcome) -> [u32; 3] {
    match outcome {
        Outcome::WonBy(winner_idx) => {
            let mut ranks = [2, 2, 2];
            ranks[winner_idx as usize] = 1;
            ranks
        }
        Outcome::AllDraw => [1, 1, 1],
        Outcome::PartialDraw(draw_mask) => {
            let mut ranks = [2, 2, 2];
            for i in 0..3 {
                if (draw_mask >> i) & 1 == 1 {
                    ranks[i] = 1;
                }
            }
            ranks
        }
    }
}

pub fn chess_mapper_from_network_shapes(
    input_shape: NetShapesInfo,
) -> Result<ChessWrapperMapper, String> {
    let input_shape = input_shape.input_shape[1..3].to_vec();
    let shape_size = input_shape[0] * input_shape[1];
    for mapper in ALL_CHESS_MAPPERS {
        let mapper_shape = mapper.input_board_shape();
        if mapper_shape[0] * mapper_shape[1] == shape_size {
            return Ok(mapper);
        }
    }

    Err(format!(
        "No suitable ChessWrapperMapper found for input shape {:?}",
        input_shape
    ))
}
