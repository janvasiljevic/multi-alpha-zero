use crate::bot_thread::BotGameContext;
use crate::entity::db_color::ColorDb;
use crate::entity::games::GameStatusDb;
use crate::entity::players::GameRelationDb;
use crate::entity::prelude::Games;
use crate::entity::users::UserTypeDb::Admin;
use crate::entity::users::{Entity, UserTypeDb};
use crate::entity::{games, players, users};
use crate::errors::{err_encapsulate, error_403_forbidden, error_404_miss, error_409_conflict};
use crate::game_events::GameEvent;
use crate::server::{Claims, ServerImpl};
use crate::util::memory_pos_to_qrs;
use anyhow::Error;
use api_autogen::apis::bot_management::BotsAssignBotResponse;
use api_autogen::apis::game_management::{
    GamesCreateGameResponse, GamesFinishGameInResponse, GamesGetGameStateResponse,
    GamesJoinGameResponse, GamesLeaveGameResponse, GamesListGamesResponse, GamesMakeMoveResponse,
    GamesStartGameResponse, GamesSubscribeToGameEventsResponse,
};
use api_autogen::models;
use api_autogen::models::WsEvent::{
    WsEventEnded, WsEventJoined, WsEventLeft, WsEventMoveMade, WsEventStarted,
};
use api_autogen::models::{
    CreateGame, FilterGames, FinishGameIn, GameRelation, GameState, GameStatus,
    GamesFinishGameInPathParams, GamesGetGameStatePathParams, GamesJoinGamePathParams,
    GamesLeaveGamePathParams, GamesMakeMovePathParams, GamesStartGamePathParams,
    GamesSubscribeToGameEventsPathParams, JoinGamePayload, MakeMovePayload, PlayerColor,
    PlayerUpdate, Position, WsGameStarted, WsMoveMade, WsPlayerJoined, WsPlayerLeft,
};
use api_autogen::types::Nullable;
use async_trait::async_trait;
use axum::http::Method;
use axum_extra::extract::{CookieJar, Host};
use futures::future::join_all;
use game_tri_chess::basics::Color::{Black, Gray, White};
use game_tri_chess::basics::{Color, Piece};
use game_tri_chess::chess_game::TriHexChess;
use game_tri_chess::moves::{ChessMoveStore, MoveType, PseudoLegalMove};
use game_tri_chess::phase::Phase;
use game_tri_chess::pos::MemoryPos;
use maz_core::mapping::Board;
use sea_orm::compound::HasOne;
use sea_orm::prelude::{Expr, HasMany};
use sea_orm::{
    ColumnTrait, EntityLoaderTrait, EntityTrait, ExprTrait, QueryFilter, Set, TransactionTrait,
};
use tracing::{error, info};

impl ServerImpl {
    async fn player_to_dto(
        &self,
        player: &players::ModelEx,
        players_masked: bool,
    ) -> Nullable<models::Player> {
        match &player.users {
            HasOne::Unloaded => Nullable::Null,
            HasOne::NotFound => Nullable::Null,
            HasOne::Loaded(user) => Nullable::Present(models::Player {
                id: user.id,
                username: if players_masked {
                    "Masked".to_string()
                } else {
                    user.username.clone()
                },
                relation: match player.relation {
                    GameRelationDb::None => Nullable::Null,
                    GameRelationDb::Winner => Nullable::Present(GameRelation::Winner),
                    GameRelationDb::Draw => Nullable::Present(GameRelation::Draw),
                    GameRelationDb::Loser => Nullable::Present(GameRelation::Loser),
                },
                is_owner: false,
                is_connected_to_game: self.game_events.is_in_room(user.id, player.game_id).await,
            }),
        }
    }

    fn find_player_in_vec(
        players: &Vec<players::ModelEx>,
        color: ColorDb,
    ) -> Option<&players::ModelEx> {
        players.iter().find(|p| p.color == color)
    }

    pub async fn players_to_dto(
        &self,
        players: &Vec<players::ModelEx>,
        players_masked: bool,
    ) -> PlayerUpdate {
        PlayerUpdate {
            white: match Self::find_player_in_vec(players, ColorDb::White) {
                Some(player) => self.player_to_dto(player, players_masked).await,
                None => Nullable::Null,
            },
            grey: match Self::find_player_in_vec(players, ColorDb::Gray) {
                Some(player) => self.player_to_dto(player, players_masked).await,
                None => Nullable::Null,
            },
            black: match Self::find_player_in_vec(players, ColorDb::Black) {
                Some(player) => self.player_to_dto(player, players_masked).await,
                None => Nullable::Null,
            },
        }
    }

    pub async fn map_game_to_dto(&self, game: games::ModelEx, include_players: bool) -> GameState {
        let players_masked = game.names_masked.unwrap_or(false);
        let mut out_dto = GameState {
            game_id: game.id,
            name: game.name,
            fen: game.fen,
            owner_id: game.owner_id.unwrap_or(-1),
            owner_username: game
                .owner
                .as_ref()
                .map_or("".to_string(), |u| u.username.clone()),
            players: Nullable::Null,
            status: match game.game_status {
                GameStatusDb::Waiting => GameStatus::Waiting,
                GameStatusDb::InProgress => GameStatus::InProgress,
                GameStatusDb::CompletedWin => GameStatus::FinishedWin,
                GameStatusDb::CompletedSemiDraw => GameStatus::FinishedSemiDraw,
                GameStatusDb::CompletedFullDraw => GameStatus::FinishedDraw,
            },
            players_masked,
            material_masked: game.material_masked.unwrap_or(false),
            training_mode: game.training_mode.unwrap_or(false),
            suggested_move_time_seconds: game
                .suggested_time_for_move_secs
                .map_or(Nullable::Null, |secs| Nullable::Present(secs)),
            tournament_id: game
                .tournament_id
                .map_or(Nullable::Null, |id| Nullable::Present(id)),
        };

        if include_players {
            if let HasMany::Loaded(loaded_players) = &game.players {
                out_dto.players =
                    Nullable::Present(self.players_to_dto(loaded_players, players_masked).await);
            }
        }

        out_dto
    }

    pub async fn find_game(&self, game_id: i64) -> Result<Option<games::ModelEx>, anyhow::Error> {
        let game = match games::Entity::load()
            .filter_by_id(game_id)
            .with(users::Entity)
            .with(players::Entity)
            .with((players::Entity, users::Entity))
            .one(&self.db)
            .await?
        {
            Some(game) => game,
            None => {
                return Ok(None);
            }
        };

        Ok(Some(game))
    }

    pub async fn find_players_for_game(
        &self,
        game_id: i64,
    ) -> Result<Vec<players::ModelEx>, anyhow::Error> {
        let players = players::Entity::load()
            .with(users::Entity)
            .filter(players::Column::GameId.eq(game_id))
            .all(&self.db)
            .await?;

        Ok(players)
    }

    pub async fn set_finished_game(
        &self,
        not_losers: Vec<Color>,
        relation: GameRelationDb,
        game_id: i64,
    ) -> Result<(), anyhow::Error> {
        let players = players::Entity::find()
            .filter(players::Column::GameId.eq(game_id))
            .all(&self.db)
            .await?;

        self.db
            .transaction::<_, _, anyhow::Error>(|txn| {
                let players = players.clone();
                let winners = not_losers.clone();
                let relation = relation.clone();

                Box::pin(async move {
                    for player in players {
                        let new_relation = if winners.contains(&match player.color {
                            ColorDb::White => White,
                            ColorDb::Gray => Gray,
                            ColorDb::Black => Black,
                        }) {
                            relation.clone()
                        } else {
                            GameRelationDb::Loser
                        };

                        players::Entity::update_many()
                            .filter(players::Column::GameId.eq(game_id))
                            .filter(players::Column::UserId.eq(player.user_id))
                            .col_expr(players::Column::Relation, Expr::Value(new_relation.into()))
                            .exec(txn)
                            .await?;
                    }

                    let status = match winners.len() {
                        3 => GameStatusDb::CompletedFullDraw,
                        2 => GameStatusDb::CompletedSemiDraw,
                        1 => GameStatusDb::CompletedWin,
                        _ => {
                            return Err(anyhow::Error::msg(
                                "Invalid number of winners when finishing game",
                            ));
                        }
                    };

                    Games::update_many()
                        .filter(games::Column::Id.eq(game_id))
                        .col_expr(games::Column::GameStatus, Expr::Value(status.into()))
                        .col_expr(games::Column::MaterialMasked, Expr::Value(false.into()))
                        .col_expr(games::Column::NamesMasked, Expr::Value(false.into()))
                        .exec(txn)
                        .await?;

                    Ok(())
                })
            })
            .await?;

        let game = self
            .find_game(game_id)
            .await?
            .expect("Game just verified to exist");

        self.game_events
            .send(GameEvent {
                game_id,
                event: WsEventEnded(Box::new(models::WsEventEnded::new(
                    "".into(),
                    models::WsGameEnded {
                        game: self.map_game_to_dto(game, true).await,
                    },
                ))),
            })
            .await;

        Ok(())
    }

    pub async fn finish_terminal_game(
        &self,
        game_id: i64,
        board: &TriHexChess,
    ) -> Result<(), anyhow::Error> {
        if !board.is_terminal() {
            return Err(anyhow::Error::msg(
                "Attempted to finish a non-terminal game",
            ));
        }

        let phase = board.state.phase;

        let (relation, winners) = match phase {
            Phase::Normal(_) => {
                return Err(anyhow::Error::msg(
                    "Attempted to finish a non-terminal game",
                ));
            }
            Phase::Won(winner) => {
                let winners = vec![winner];
                (players::GameRelationDb::Winner, winners)
            }
            Phase::Draw(draw_state) => {
                let winners = draw_state.get_drawn_players();
                (players::GameRelationDb::Draw, winners)
            }
        };

        self.set_finished_game(winners, relation, game_id).await?;

        Ok(())
    }
}

#[allow(unused_variables)]
#[async_trait]
impl api_autogen::apis::game_management::GameManagement<anyhow::Error> for ServerImpl {
    type Claims = Claims;

    async fn games_create_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &CreateGame,
    ) -> Result<GamesCreateGameResponse, anyhow::Error> {
        let default_fen = TriHexChess::default().to_fen();

        let game = Games::insert(games::ActiveModel {
            id: Default::default(),
            name: Set(body.name.clone()),
            fen: Set(default_fen.clone()),
            owner_id: Set(Some(claims.sub)),
            game_status: Set(GameStatusDb::Waiting),
            names_masked: Set(Some(body.names_masked)),
            material_masked: Set(Some(body.material_masked)),
            training_mode: Set(Some(body.training_mode)),
            suggested_time_for_move_secs: Set(body.suggested_move_time_seconds),
            tournament_id: body.tournament_id.map_or(Set(None), |id| Set(Some(id))),
        })
        .exec_with_returning(&self.db)
        .await?;

        info!(
            "Game (ID: {}) '{}' created by user ID {}",
            game.id, game.name, claims.sub
        );

        let game_created = GamesCreateGameResponse::Status200_TheRequestHasSucceeded(GameState {
            game_id: game.id,
            name: game.name,
            fen: default_fen,
            owner_id: claims.sub,
            owner_username: "".to_string(),
            players: Nullable::Null,
            status: GameStatus::Waiting,
            players_masked: body.names_masked,
            material_masked: body.material_masked,
            training_mode: body.training_mode,
            suggested_move_time_seconds: body
                .suggested_move_time_seconds
                .map_or(Nullable::Null, |secs| Nullable::Present(secs)),
            tournament_id: body
                .tournament_id
                .map_or(Nullable::Null, |id| Nullable::Present(id)),
        });

        Ok(game_created)
    }

    async fn games_finish_game_in(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &GamesFinishGameInPathParams,
        body: &FinishGameIn,
    ) -> Result<GamesFinishGameInResponse, Error> {
        let Some(game) = self.find_game(path_params.game_id).await? else {
            return Ok(GamesFinishGameInResponse::Status404_NotFound(
                error_404_miss(),
            ));
        };

        if claims.userType != Admin {
            info!(
                "User {} is not an admin and cannot finish game {}",
                claims.sub, path_params.game_id
            );
            return Ok(GamesFinishGameInResponse::Status403_Forbidden(
                error_403_forbidden(),
            ));
        }

        if game.game_status != GameStatusDb::InProgress {
            info!(
                "Game {} is not in progress, admin user {} cannot finish it",
                path_params.game_id, claims.sub
            );
            return Ok(GamesFinishGameInResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        };

        if game.players.len() != 3 {
            info!(
                "Game {} does not have all player slots filled, admin user {} cannot finish it",
                path_params.game_id, claims.sub
            );
            return Ok(GamesFinishGameInResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        };

        let relations: [(_, GameRelationDb); 3] = [
            (ColorDb::White, body.white.into()),
            (ColorDb::Gray, body.grey.into()),
            (ColorDb::Black, body.black.into()),
        ];

        info!(
            "Setting game {} relations to: White: {:?}, Grey: {:?}, Black: {:?}",
            path_params.game_id, body.white, body.grey, body.black
        );

        self.db
            .transaction::<_, _, anyhow::Error>(|txn| {
                let game_id = path_params.game_id;

                Box::pin(async move {
                    for (color, relation) in relations.clone() {
                        players::Entity::update_many()
                            .filter(players::Column::GameId.eq(game_id))
                            .filter(players::Column::Color.eq(color))
                            .col_expr(players::Column::Relation, Expr::Value(relation.into()))
                            .exec(txn)
                            .await?;
                    }

                    let deduced_status = match relations
                        .iter()
                        .filter(|(_, r)| *r != GameRelationDb::Loser)
                        .count()
                    {
                        3 => GameStatusDb::CompletedFullDraw,
                        2 => GameStatusDb::CompletedSemiDraw,
                        1 => GameStatusDb::CompletedWin,
                        _ => {
                            return Err(anyhow::Error::msg(
                                "Invalid number of winners when finishing game",
                            ));
                        }
                    };

                    Games::update_many()
                        .filter(games::Column::Id.eq(game_id))
                        .col_expr(
                            games::Column::GameStatus,
                            Expr::Value(deduced_status.into()),
                        )
                        .exec(txn)
                        .await?;

                    Ok(())
                })
            })
            .await?;

        let game = self
            .find_game(path_params.game_id)
            .await?
            .expect("Game just verified to exist");

        self.game_events
            .send(GameEvent {
                game_id: path_params.game_id,
                event: WsEventEnded(Box::new(models::WsEventEnded::new(
                    "".into(),
                    models::WsGameEnded {
                        game: self.map_game_to_dto(game, true).await,
                    },
                ))),
            })
            .await;

        Ok(GamesFinishGameInResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }

    async fn games_get_game_state(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &GamesGetGameStatePathParams,
    ) -> Result<GamesGetGameStateResponse, anyhow::Error> {
        let game = match games::Entity::load()
            .filter_by_id(path_params.game_id)
            .with(players::Entity)
            .with((players::Entity, users::Entity))
            .one(&self.db)
            .await?
        {
            Some(game) => game,
            None => {
                return Ok(GamesGetGameStateResponse::Status404_NotFound(
                    error_404_miss(),
                ));
            }
        };

        let game_state = self.map_game_to_dto(game, true).await;

        Ok(GamesGetGameStateResponse::Status200_TheRequestHasSucceeded(
            game_state,
        ))
    }

    async fn games_join_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &GamesJoinGamePathParams,
        body: &JoinGamePayload,
    ) -> Result<GamesJoinGameResponse, anyhow::Error> {
        let Some(game) = games::Entity::load()
            .filter_by_id(path_params.game_id)
            .one(&self.db)
            .await?
        else {
            return Ok(GamesJoinGameResponse::Status404_NotFound(error_404_miss()));
        };

        let Some(user) = users::Entity::find_by_id(claims.sub).one(&self.db).await? else {
            return Ok(GamesJoinGameResponse::Status404_NotFound(error_404_miss()));
        };

        if game.game_status != GameStatusDb::Waiting {
            info!("Game {} is not in waiting status", path_params.game_id);
            return Ok(GamesJoinGameResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        };

        let new_player = players::ActiveModel {
            user_id: Set(user.id),
            game_id: Set(path_params.game_id),
            color: Set(match body.color {
                PlayerColor::White => ColorDb::White,
                PlayerColor::Grey => ColorDb::Gray,
                PlayerColor::Black => ColorDb::Black,
            }),
            relation: Set(players::GameRelationDb::None),
        };

        players::Entity::insert(new_player).exec(&self.db).await?;

        let players = self.find_players_for_game(path_params.game_id).await?;
        let mapped = self
            .players_to_dto(&players, game.names_masked.unwrap_or(false))
            .await;

        self.game_events
            .send(GameEvent {
                game_id: path_params.game_id,
                event: WsEventJoined(Box::new(models::WsEventJoined::new(
                    "".into(),
                    WsPlayerJoined::new(user.username, body.color.clone(), mapped.clone()),
                ))),
            })
            .await;

        Ok(GamesJoinGameResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }

    async fn games_leave_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &GamesLeaveGamePathParams,
    ) -> Result<GamesLeaveGameResponse, anyhow::Error> {
        let game = match games::Entity::load()
            .filter_by_id(path_params.game_id)
            .one(&self.db)
            .await?
        {
            Some(game) => game,
            None => {
                return Ok(GamesLeaveGameResponse::Status404_NotFound(error_404_miss()));
            }
        };

        if game.game_status != GameStatusDb::Waiting {
            return Ok(GamesLeaveGameResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        };

        let player = players::Entity::load()
            .with(users::Entity)
            .filter(players::Column::GameId.eq(path_params.game_id))
            .filter(players::Column::UserId.eq(claims.sub))
            .one(&self.db)
            .await?;

        let player = match player {
            Some(player) => player,
            None => {
                return Ok(GamesLeaveGameResponse::Status403_Forbidden(
                    error_403_forbidden(),
                ));
            }
        };

        players::Entity::delete_many()
            .filter(players::Column::GameId.eq(player.game_id))
            .filter(players::Column::UserId.eq(player.user_id))
            .exec(&self.db)
            .await?;

        let players = self.find_players_for_game(path_params.game_id).await?;
        let mapped = self
            .players_to_dto(&players, game.names_masked.unwrap_or(false))
            .await;

        self.game_events
            .send(GameEvent {
                game_id: path_params.game_id,
                event: WsEventLeft(Box::new(models::WsEventLeft::new(
                    "".into(),
                    WsPlayerLeft {
                        username: player.users.as_ref().unwrap().username.clone(),
                        color: match player.color {
                            ColorDb::White => PlayerColor::White,
                            ColorDb::Gray => PlayerColor::Grey,
                            ColorDb::Black => PlayerColor::Black,
                        },
                        players: mapped,
                    },
                ))),
            })
            .await;

        Ok(GamesLeaveGameResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }

    async fn games_list_games(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        query_params: &models::GamesListGamesQueryParams,
    ) -> Result<GamesListGamesResponse, anyhow::Error> {
        let mut builder = games::Entity::load()
            .with(players::Entity)
            .with((players::Entity, users::Entity));

        if let Some(filter) = query_params.status {
            match filter {
                FilterGames::Waiting => {
                    builder = builder.filter(games::Column::GameStatus.eq(GameStatusDb::Waiting));
                }
                FilterGames::InProgress => {
                    builder =
                        builder.filter(games::Column::GameStatus.eq(GameStatusDb::InProgress));
                }
                FilterGames::Finished => {
                    builder = builder.filter(
                        games::Column::GameStatus
                            .eq(GameStatusDb::CompletedWin)
                            .or(games::Column::GameStatus.eq(GameStatusDb::CompletedSemiDraw))
                            .or(games::Column::GameStatus.eq(GameStatusDb::CompletedFullDraw)),
                    );
                }
            }
        }

        if let Some(tournament_id) = query_params.tournament_id {
            builder = builder.filter(games::Column::TournamentId.eq(tournament_id));
        }

        let results = builder.order_by_id_desc().all(&self.db).await?;

        let games: Vec<GameState> = join_all(
            results
                .into_iter()
                .map(|record| self.map_game_to_dto(record, true)),
        )
        .await;

        Ok(GamesListGamesResponse::Status200_TheRequestHasSucceeded(
            games,
        ))
    }

    async fn games_make_move(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &GamesMakeMovePathParams,
        body: &MakeMovePayload,
    ) -> Result<GamesMakeMoveResponse, anyhow::Error> {
        let Some(game) = self.find_game(path_params.game_id).await? else {
            return Ok(GamesMakeMoveResponse::Status404_NotFound(error_404_miss()));
        };

        if game.game_status != GameStatusDb::InProgress {
            info!("Game {} is not in progress", path_params.game_id);
            return Ok(GamesMakeMoveResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        };

        let color = game.players.iter().find_map(|x| {
            if x.user_id == claims.sub {
                Some(match x.color {
                    ColorDb::White => White,
                    ColorDb::Gray => Gray,
                    ColorDb::Black => Black,
                })
            } else {
                None
            }
        });

        if color.is_none() {
            return Ok(GamesMakeMoveResponse::Status403_Forbidden(
                error_403_forbidden(),
            ));
        }

        let mut board =
            TriHexChess::new_with_fen(&game.fen.as_ref(), false).map_err(err_encapsulate)?;

        if board.is_terminal() {
            info!("Game {} is already over", path_params.game_id);
            return Ok(GamesMakeMoveResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        }

        let prev_fen = game.fen.clone();
        let prev_turn_counter = board.state.turn_counter;
        let prev_turn = board.get_turn().expect("Not terminal, so turn must exist");

        if let Some(current_turn) = board.get_turn() {
            if current_turn != color.unwrap() {
                info!(
                    "It's not user {}'s turn in game {}",
                    claims.sub, path_params.game_id
                );
                return Ok(GamesMakeMoveResponse::Status403_Forbidden(
                    error_403_forbidden(),
                ));
            }
        }

        let mut game_store = ChessMoveStore::default();

        board.update_pseudo_moves(&mut game_store, true);

        let maybe_chess_move = game_store.iter().find_map(|m| {
            if m.from != MemoryPos(body.from_index) || m.to != MemoryPos(body.to_index) {
                return None;
            }

            let internal_promotion_piece = match m.move_type {
                MoveType::Promotion(p) => Some(p),
                MoveType::EnPassantPromotion(prom) => Some(prom.get().1),
                _ => None,
            };

            match (body.promotion, internal_promotion_piece) {
                // Both API and internal say it's a promotion
                (Some(req_prom), Some(int_prom)) => {
                    let req_piece = match req_prom {
                        models::PromotionPiece::Queen => Piece::Queen,
                        models::PromotionPiece::Rook => Piece::Rook,
                        models::PromotionPiece::Bishop => Piece::Bishop,
                        models::PromotionPiece::Knight => Piece::Knight,
                    };
                    if int_prom == req_piece {
                        Some(m.clone())
                    } else {
                        None
                    }
                }

                // API has no promotion but internal does → reject
                (None, Some(_)) => None,

                // Neither is a promotion
                (None, None) => Some(m.clone()),

                // API provides a promotion but internal does not → reject
                (Some(_), None) => None,
            }
        });

        let chess_move = match maybe_chess_move {
            Some(mv) => mv,
            None => {
                info!(
                    "Invalid move in game with id {} by user {}. Api Body: {:?}",
                    path_params.game_id, claims.sub, body
                );
                return Ok(GamesMakeMoveResponse::Status409_Conflict(
                    error_409_conflict(),
                ));
            }
        };

        board.play_move_mut_with_store(&chess_move, &mut game_store, None);

        let new_fen = board.to_fen();

        // start a transaction to update the game's FEN and add the move to history
        self.db
            .transaction::<_, _, anyhow::Error>(|txn| {
                let game_id = path_params.game_id;
                let move_uci = chess_move.notation_uci();
                let new_fen = new_fen.clone();

                Box::pin(async move {
                    // Update the game's FEN
                    Games::update_many()
                        .filter(games::Column::Id.eq(game_id))
                        .col_expr(games::Column::Fen, Expr::value(new_fen.clone()))
                        .exec(txn)
                        .await
                        .map_err(err_encapsulate)?;

                    // Insert the move into history
                    crate::entity::history::Entity::insert(crate::entity::history::ActiveModel {
                        fen: Set(prev_fen),
                        move_uci: Set(move_uci),
                        turn: Set(prev_turn_counter as i64),
                        color: Set(prev_turn.into()),
                        game_id: Set(game_id),
                        ..Default::default()
                    })
                    .exec(txn)
                    .await
                    .map_err(err_encapsulate)?;

                    Ok(())
                })
            })
            .await?;

        let game_db = self
            .find_game(path_params.game_id)
            .await?
            .expect("Game just verified to exist");

        let game_state = self.map_game_to_dto(game_db.clone(), true).await;

        // send the move event to WS subscribers
        self.game_events
            .send(GameEvent {
                game_id: path_params.game_id,
                event: WsEventMoveMade(Box::new(models::WsEventMoveMade::new(
                    "".into(),
                    WsMoveMade {
                        r#move: models::Move {
                            from: memory_pos_to_qrs(chess_move.from),
                            to: memory_pos_to_qrs(chess_move.to),
                            move_uci: chess_move.notation_uci(),
                        },
                        new_fen,
                        new_turn: board.get_turn().map_or(PlayerColor::White, |c| match c {
                            White => PlayerColor::White,
                            Gray => PlayerColor::Grey,
                            Black => PlayerColor::Black,
                        }),
                    },
                ))),
            })
            .await;

        if board.is_terminal() {
            self.finish_terminal_game(path_params.game_id, &board)
                .await
                .map_err(err_encapsulate)?;
        } else if (board.state.turn_counter > 500) {
            info!(
                "Game {} exceeded maximum turn count, finishing as full draw",
                path_params.game_id
            );
            self.set_finished_game(
                vec![White, Gray, Black],
                GameRelationDb::Draw,
                path_params.game_id,
            )
            .await
            .map_err(err_encapsulate)?;
        } else if (board.can_game_end_early()) {
            info!(
                "Game {} reached early draw condition, finishing as full draw",
                path_params.game_id
            );

            self.set_finished_game(
                vec![White, Gray, Black],
                GameRelationDb::Draw,
                path_params.game_id,
            )
            .await
            .map_err(err_encapsulate)?;
        } else {
            let _ = self
                .bot_processing_tx
                .send(BotGameContext {
                    game: game_db,
                    board: board.clone(),
                })
                .await;
        }

        Ok(GamesMakeMoveResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }

    async fn games_start_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &GamesStartGamePathParams,
    ) -> Result<GamesStartGameResponse, anyhow::Error> {
        let Some(mut game) = self.find_game(path_params.game_id).await? else {
            return Ok(GamesStartGameResponse::Status404_NotFound(error_404_miss()));
        };

        if game.game_status != GameStatusDb::Waiting {
            info!("Game {} is not in waiting status", path_params.game_id);
            return Ok(GamesStartGameResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        };

        if game.owner_id != Some(claims.sub) {
            info!(
                "User {} is not the owner of game {}",
                claims.sub, path_params.game_id
            );
            return Ok(GamesStartGameResponse::Status403_Forbidden(
                error_403_forbidden(),
            ));
        }

        let player_count = game.players.len();

        if player_count != 3 {
            info!(
                "Not all player slots are filled for game {}. 3 != {}",
                path_params.game_id, player_count
            );
            return Ok(GamesStartGameResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        }

        Games::update_many()
            .filter(games::Column::Id.eq(path_params.game_id))
            .col_expr(
                games::Column::GameStatus,
                Expr::Value(GameStatusDb::InProgress.into()),
            )
            .exec(&self.db)
            .await?;

        game.game_status = GameStatusDb::InProgress;

        let game_state = self.map_game_to_dto(game.clone(), true).await;

        self.game_events
            .send(GameEvent {
                game_id: path_params.game_id,
                event: WsEventStarted(Box::new(models::WsEventStarted::new(
                    "".into(),
                    WsGameStarted::new(game_state.clone()),
                ))),
            })
            .await;

        // Check if the white player is a bot and trigger bot move if so
        for player in game.players.iter() {
            if player.color == ColorDb::White {
                if let HasOne::Loaded(user) = &player.users {
                    if user.user_type == UserTypeDb::Bot {
                        info!(
                            "Starting bot processing for game {} after start (white bot)",
                            path_params.game_id
                        );

                        let board = TriHexChess::new_with_fen(&game_state.fen.as_ref(), false)
                            .expect("First FEN should be valid");

                        let game = self
                            .find_game(path_params.game_id)
                            .await?
                            .expect("Game just verified to exist");

                        let _ = self
                            .bot_processing_tx
                            .send(BotGameContext {
                                game: game.clone(),
                                board: board.clone(),
                            })
                            .await;
                    }
                }
            }
        }

        Ok(GamesStartGameResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }

    async fn games_subscribe_to_game_events(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &GamesSubscribeToGameEventsPathParams,
    ) -> Result<GamesSubscribeToGameEventsResponse, anyhow::Error> {
        error!("This function is not supported in HTTP mode.");
        Err(anyhow::Error::msg(
            "This function is not supported in HTTP mode.",
        ))
    }
}
