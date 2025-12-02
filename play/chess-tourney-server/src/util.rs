use api_autogen::models;
use game_tri_chess::basics::MemoryPos;

pub fn memory_pos_to_qrs(pos: MemoryPos) -> models::Position {
    let qrs = pos.to_qrs_global();
    models::Position {
        q: qrs.q as i32,
        r: qrs.r as i32,
        s: qrs.s as i32,
        i: pos.0 as i32,
    }
}