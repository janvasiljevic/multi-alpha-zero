use sea_orm::ColumnTrait;
use crate::entity::history;
use crate::server::{Claims, ServerImpl};
use anyhow::Error;
use api_autogen::apis::game_history::HistoryGetGameHistoryResponse;
use api_autogen::models::{GameHistory, HistoryGetGameHistoryPathParams};
use async_trait::async_trait;
use axum::http::Method;
use axum_extra::extract::{CookieJar, Host};
use sea_orm::EntityTrait;
use sea_orm::QueryFilter;

#[allow(unused_variables)]
#[async_trait]
impl api_autogen::apis::game_history::GameHistory<anyhow::Error> for ServerImpl {
    type Claims = Claims;

    async fn history_get_game_history(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &HistoryGetGameHistoryPathParams,
    ) -> Result<HistoryGetGameHistoryResponse, Error> {
        let Some(game) = self.find_game(path_params.game_id).await? else {
            return Ok(HistoryGetGameHistoryResponse::Status404_NotFound(
                api_autogen::models::NotFoundError {
                    message: "Game not found".to_string(),
                },
            ));
        };

        let history = history::Entity::find()
            .filter(history::Column::GameId.eq(game.id))
            .all(&self.db)
            .await?;

        let mut history_entries : Vec<_> = history
            .into_iter()
            .map(|h| api_autogen::models::HistoryItem {
                turn_counter: h.turn as i32,
                fen: h.fen,
                move_uci: h.move_uci,
                color: h.color.into(),
            })
            .collect();

        // sort by turn_counter ascending and by color (White, Gray, Black)
        history_entries.sort_by(|a, b| {
            if a.turn_counter != b.turn_counter {
                a.turn_counter.cmp(&b.turn_counter)
            } else {
                a.color.cmp(&b.color)
            }
        });
        
        let max_turn = history_entries.iter().map(|h| h.turn_counter).max().unwrap_or(0);

        Ok(
            HistoryGetGameHistoryResponse::Status200_TheRequestHasSucceeded(GameHistory {
                history: history_entries,
                game: self.map_game_to_dto(game, true).await,
                max_turns: max_turn,
            }),
        )
    }
}
