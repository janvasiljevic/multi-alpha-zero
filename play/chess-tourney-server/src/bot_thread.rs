use crate::entity::db_color::ColorDb;
use crate::entity::games::GameStatusDb::InProgress;
use crate::entity::players::GameRelationDb;
use crate::entity::users::UserTypeDb;
use crate::entity::{bot, games, history};
use crate::errors::err_encapsulate;
use crate::game_events::GameEvent;
use crate::model_provider::{ChessModelActorRequest, ThinkRequestChess};
use crate::server::ServerImpl;
use crate::util::memory_pos_to_qrs;
use anyhow::Result;
use api_autogen::models;
use api_autogen::models::{Move, PlayerColor, WsEvent, WsEventMoveMade, WsMoveMade};
use game_tri_chess::basics::Color::{Black, Gray, White};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::ChessMoveStore;
use game_tri_chess::pos::MemoryPos;
use maz_core::mapping::Board;
use sea_orm::prelude::Expr;
use sea_orm::prelude::StringLen::N;
use sea_orm::ColumnTrait;
use sea_orm::{ActiveModelTrait, EntityTrait, QueryFilter, Set, TransactionTrait};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub struct BotGameContext {
    pub game: games::ModelEx,
    pub board: TriHexChess,
}

pub async fn start_bot_worker(
    server: Arc<ServerImpl>,
    mut rx: mpsc::Receiver<BotGameContext>, // Update Type
) {
    info!("Bot worker thread started");

    while let Some(ctx) = rx.recv().await {
        let server = server.clone();
        // Returns immediately, processes next game in queue
        tokio::spawn(async move {
            if let Err(e) = process_bot_move(&server, ctx).await {
                error!("Error processing bot move: {:?}", e);
            }
        });
    }
}

async fn process_bot_move(server: &Arc<ServerImpl>, mut ctx: BotGameContext) -> Result<()> {
    let mut board = ctx.board.clone();

    if board.is_terminal() {
        warn!(
            "Game {} is already terminal, skipping bot move",
            ctx.game.id
        );
        return Ok(());
    }

    if ctx.game.game_status != InProgress {
        warn!(
            "Game {} is not in progress (status: {:?}), skipping bot move",
            ctx.game.id, ctx.game.game_status
        );
        return Ok(());
    }

    let current_turn_color = board.get_turn().expect("Game not terminal");

    let db_turn_color = match current_turn_color {
        White => ColorDb::White,
        Gray => ColorDb::Gray,
        Black => ColorDb::Black,
    };

    let (user_id, is_current_player_bot) = ctx
        .game
        .players
        .iter()
        .find(|p| p.color == db_turn_color)
        .map(|p| {
            p.users
                .is_loaded()
                .then(|| {
                    let user = p.users.as_ref().unwrap();
                    (Some(user.id), user.user_type == UserTypeDb::Bot)
                })
                .unwrap_or((None, false))
        })
        .unwrap_or((None, false));

    if !is_current_player_bot {
        info!("It's not bot's turn for game {}, skipping...", ctx.game.id);
        return Ok(());
    }

    let (tx, rx) = oneshot::channel();

    // filter by bot user_id
    let Some(bot_db) = bot::Entity::find()
        .filter(bot::Column::UserId.eq(user_id.expect("Bot player should have user_id")))
        .one(&server.db)
        .await?
    else {
        warn!("No bot configuration found for user_id {:?}", user_id);
        return Ok(());
    };

    let model_handle = server
        .model_service
        .get_model(bot_db.model_key.as_deref().unwrap_or(""));

    let request = ChessModelActorRequest::Think(ThinkRequestChess {
        board: board.clone(),
        num_of_rollouts: bot_db.playouts as u64,
        exploration_factor: bot_db.exploration_factor,
        contempt: bot_db.contempt,
        virtual_loss_weight: bot_db.virtual_loss_weight,
        responder: tx,
    });

    model_handle.tx.send(request).await?;
    let result = rx.await?;

    let best_move_stats = result
        .moves
        .iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

    let Some(best_move_wrap) = best_move_stats else {
        warn!("Model returned no moves for game {}", ctx.game.id);
        return Ok(());
    };

    let chess_move = best_move_wrap.inner.clone();

    let prev_fen = board.to_fen();
    let prev_turn_counter = board.state.turn_counter;

    let mut move_store = ChessMoveStore::default();
    board.play_move_mut_with_store(&chess_move, &mut move_store, None);

    let new_fen = board.to_fen();
    let move_uci = chess_move.notation_uci();

    server
        .db
        .transaction::<_, _, anyhow::Error>(|txn| {
            let new_fen_db = new_fen.clone();
            let move_uci_db = move_uci.clone();
            let gid = ctx.game.id;

            Box::pin(async move {
                games::Entity::update_many()
                    .col_expr(games::Column::Fen, Expr::Value(new_fen_db.into()))
                    .filter(games::Column::Id.eq(gid))
                    .exec(txn)
                    .await?;

                history::Entity::insert(history::ActiveModel {
                    fen: Set(prev_fen),
                    move_uci: Set(move_uci_db),
                    turn: Set(prev_turn_counter as i64),
                    color: Set(current_turn_color.into()),
                    game_id: Set(gid),
                    ..Default::default()
                })
                .exec(txn)
                .await?;

                Ok(())
            })
        })
        .await?;

    server
        .game_events
        .send(GameEvent {
            game_id: ctx.game.id,
            event: WsEvent::WsEventMoveMade(Box::new(WsEventMoveMade {
                kind: "".to_string(),
                value: WsMoveMade {
                    r#move: Move {
                        from: memory_pos_to_qrs(chess_move.from),
                        to: memory_pos_to_qrs(chess_move.to),
                        move_uci,
                    },
                    new_fen,
                    new_turn: board.get_turn().map_or(PlayerColor::White, |c| match c {
                        White => PlayerColor::White,
                        Gray => PlayerColor::Grey,
                        Black => PlayerColor::Black,
                    }),
                },
            })),
        })
        .await;

    // 8. Handle Termination
    if board.is_terminal() {
        server.finish_terminal_game(ctx.game.id, &board).await?;
    } else if (board.state.turn_counter > 500) {
        info!(
            "Game {} exceeded maximum turn count, finishing as full draw",
            ctx.game.id
        );
        server
            .set_finished_game(vec![White, Gray, Black], GameRelationDb::Draw, ctx.game.id)
            .await
            .map_err(err_encapsulate)?;
    } else if (board.can_game_end_early()) {
        info!(
            "Game {} reached early draw condition, finishing as full draw",
            ctx.game.id
        );

        server
            .set_finished_game(vec![White, Gray, Black], GameRelationDb::Draw, ctx.game.id)
            .await
            .map_err(err_encapsulate)?;
    } else {
        let tx_clone = server.bot_processing_tx.clone();
        let game = server
            .find_game(ctx.game.id)
            .await
            .expect("Error refetching game for bot move")
            .expect("Game should exist");

        let ctx = BotGameContext { game, board };

        tokio::spawn(async move {
            // Just a small delay
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = tx_clone.send(ctx).await;
        });
    }

    Ok(())
}
