use crate::vis_chess_util::{SIZE, get_move_color, get_x, get_y};
use analysis_util::VisArrow;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, MoveType};
use svg::node::element::Group;

pub fn draw_demo_moves(move_store: &ChessMoveStore, board: &TriHexChess) -> Group {
    let mut circle_group = Group::new().set("id", "demo-move-circles");

    let mut moves = move_store.iter().cloned().collect::<Vec<_>>();

    moves.sort_by_key(|m| match m.move_type {
        MoveType::DoublePawnPush => -1,
        MoveType::CastleKingSide => 2,
        MoveType::CastleQueenSide => 2,
        _ => 0,
    });

    for m in moves.iter() {
        let qrs_to = m.to.to_qrs_global();

        let x = get_x(qrs_to.q);
        let y = get_y(qrs_to.q, qrs_to.r);

        let color = get_move_color(*m, board);
        let color = color.as_str();

        let circle = svg::node::element::Circle::new()
            .set("cx", x)
            .set("cy", y)
            .set("r", SIZE * 0.3)
            .set("fill", color)
            .set("fill-opacity", 0.7)
            .set("stroke", color)
            .set("stroke-width", 2)
            .set("stroke-opacity", 1.0);

        circle_group = circle_group.add(circle);
    }

    circle_group
}

pub fn chess_moves_to_arrows(
    moves: &ChessMoveStore,
    board: &TriHexChess,
    add_notation: bool,
) -> Vec<VisArrow> {
    let mut moves = moves.iter().cloned().collect::<Vec<_>>();

    moves.sort_by_key(|m| match m.move_type {
        MoveType::DoublePawnPush => -1,
        _ => 0,
    });

    let mut out = moves
        .into_iter()
        .map(|m| {
            let qrs_from = m.from.to_qrs_global();
            let qrs_to = m.to.to_qrs_global();
            let color = get_move_color(m, board);
            VisArrow {
                from_q: qrs_from.q,
                from_r: qrs_from.r,
                from_i: m.from,
                to_q: qrs_to.q,
                to_r: qrs_to.r,
                to_i: m.to,
                opacity: 1.0,
                text: match add_notation {
                    true => Some(m.notation_lan(&board.state.buffer).to_string()),
                    false => None,
                },
                color,
            }
        })
        .collect::<Vec<_>>();

    out.sort_by(|a, b| {
        let da = ((a.to_q - a.from_q).pow(2) + (a.to_r - a.from_r).pow(2)) as f64;
        let db = ((b.to_q - b.from_q).pow(2) + (b.to_r - b.from_r).pow(2)) as f64;
        db.partial_cmp(&da).unwrap()
    });

    out
}
