use chess_visualize::vis_chess_render::render_board;
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use maz_core::mapping::Board;

fn main() {
    let mut board = TriHexChess::new_with_fen("r1bqkbnr/5p2/2p1p1p1/pp1p3p/X/8/8/3n4/8 X/rnkq1bnr/2p2pp1/3p4/1p2p2p/X X/X/rnbq1knr/2p1p1p1/8/pp3p1p G qk---- --- 0 5".as_ref(), false).unwrap();

    let mut move_store = ChessMoveStore::default();

    board.fill_move_store(&mut move_store);

    let document = render_board(&board, vec![].as_ref(), None, None, None, false);

    let svg_filename = format!("fen.svg");

    svg::save(svg_filename.as_str(), &document).unwrap();
}
