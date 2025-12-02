export const todos = [
    "Game logic: 50-move rule and 'third' moves", // TODO
    "Algebraic notation",
    "Logic and UI: Undo last move", // TODO
    "UI: Hover effect on pieces (show attack squares)", // TODO
    "UI: Show the direction in which pawns move", // TODO
    "UI: Prettier promotion menu", // TODO: Low priority
    "UI: Rasterize chess piece SVG to base64 so they can be loaded directly in CSS with one request - also a performance improvement during rendering", // TODO: Low priority
    "UI: Add color 'mixer' for figures and the board", // TODO: Low priority
    "UI: Perhaps add some notes for how to play", // TODO: Low priority
    "-- FEN: Third move",
    "-- Game logic: Stalemate",
    "-- Game logic: En passant", // Implemented. However the code is janky and there is a slight question if it is correct. Wait for the email response
    "-- FEN: En passant", // Saved as triple chars in fen
    "-- UI: Show last move played", // Shows the last 2 moves played. Perhaps not the best, but works
    "-- Fix: Fix FEN generation", // Perhaps this would benefit from a fuzzing test. E.g. generate a million different position and then check if the covnversion <-> works
    "-- Logic: Conversions between internal game state <-> algebraic notation", // Also added to debug tiles
    "-- UI: Add actual (algebraic) notation to the sides", // Perhaps cleaner font / toggle
    "-- UI: Show material count for all three players", // Not the prettiest probably, but it works
    "-- FEN: Castling",
    "-- Game logic: Checkmate", // Current players are stored in an array and the game is over when the length is 1
    "-- Fix pawn bug (king check move generation?) - the pawn can capture, but the king can still move there...",
    "-- UI: Resizable chessboard", // Created, however it is slightly janky: Sizing for the dragging isn't the best
    "-- Game logic: Pawn promotion",
    "-- Game logic: Pawn promotion on the sides",
    "-- FIX: Can castle while in check",
    "-- Game logic: Castling",
    "-- Game logic: Turns",
    "-- FEN: Turns",
    "-- Game logic: Checks",
    "-- Game logic: Pins",
];
