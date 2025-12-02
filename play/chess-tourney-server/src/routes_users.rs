use crate::entity::db_color::ColorDb;
use crate::entity::games::GameStatusDb;
use crate::entity::users::UserTypeDb;
use crate::entity::{players, users};
use crate::errors::{error_403_forbidden, error_404_miss};
use crate::game_events::GameEvent;
use crate::server::{Claims, ServerImpl};
use anyhow::Error;
use api_autogen::apis::user_management::{
    UsersListBotUsersResponse, UsersListUsersResponse, UsersRemoveUserFromGameResponse,
};
use api_autogen::models;
use api_autogen::models::WsEvent::WsEventLeft;
use api_autogen::models::{PlayerColor, UsersRemoveUserFromGamePathParams, WsPlayerLeft};
use async_trait::async_trait;
use axum::http::Method;
use axum_extra::extract::{CookieJar, Host};
use sea_orm::ColumnTrait;
use sea_orm::EntityTrait;
use sea_orm::QueryFilter;
use tracing::info;

#[allow(unused_variables)]
#[async_trait]
impl api_autogen::apis::user_management::UserManagement<anyhow::Error> for ServerImpl {
    type Claims = Claims;

    async fn users_list_bot_users(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<UsersListBotUsersResponse, Error> {
        let users = users::Entity::find()
            .filter(users::Column::UserType.eq(crate::entity::users::UserTypeDb::Bot))
            .all(&self.db)
            .await?;

        Ok(UsersListBotUsersResponse::Status200_TheRequestHasSucceeded(
            users.into_iter().map(|u| u.into()).collect(),
        ))
    }

    async fn users_list_users(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<UsersListUsersResponse, Error> {
        let users = users::Entity::find().all(&self.db).await?;

        Ok(UsersListUsersResponse::Status200_TheRequestHasSucceeded(
            users.into_iter().map(|u| u.into()).collect(),
        ))
    }

    async fn users_remove_user_from_game(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &UsersRemoveUserFromGamePathParams,
    ) -> Result<UsersRemoveUserFromGameResponse, Error> {
        let Some(game) = self.find_game(path_params.game_id).await? else {
            return Ok(UsersRemoveUserFromGameResponse::Status404_NotFound(
                crate::errors::error_404_miss(),
            ));
        };

        if game.game_status != GameStatusDb::Waiting {
            info!(
                "Cannot remove user {} from game {} as it is not in waiting state",
                path_params.user_id, path_params.game_id
            );
            return Ok(UsersRemoveUserFromGameResponse::Status403_Forbidden(
                error_403_forbidden(),
            ));
        }

        if claims.userType != UserTypeDb::Admin {
            if let Some(game_owner_id) = game.owner_id {
                if game_owner_id != claims.sub {
                    info!(
                        "User {} unauthorized to remove user {} from game {}",
                        claims.sub, path_params.user_id, path_params.game_id
                    );
                    return Ok(UsersRemoveUserFromGameResponse::Status403_Forbidden(
                        error_403_forbidden(),
                    ));
                }
            }
        } else {
            info!(
                "Admin user {} removing user {} from game {}",
                claims.sub, path_params.user_id, path_params.game_id
            );
        }

        let Some(player) = players::Entity::load()
            .with(users::Entity)
            .filter(players::Column::GameId.eq(path_params.game_id))
            .filter(players::Column::UserId.eq(path_params.user_id))
            .one(&self.db)
            .await?
        else {
            return Ok(UsersRemoveUserFromGameResponse::Status404_NotFound(
                error_404_miss(),
            ));
        };

        players::Entity::delete_by_id((path_params.game_id, path_params.user_id))
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

        Ok(UsersRemoveUserFromGameResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }
}
