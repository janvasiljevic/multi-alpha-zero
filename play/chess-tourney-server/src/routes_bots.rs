use crate::entity::db_color::ColorDb;
use crate::entity::games::GameStatusDb;
use crate::entity::users::UserTypeDb;
use crate::entity::{bot, players, users};
use crate::errors::{
    err_encapsulate, error_400_bad_request, error_400_bad_request_with_msg, error_403_forbidden,
    error_404_miss, error_409_conflict,
};
use crate::game_events::GameEvent;
use crate::server::{Claims, ServerImpl};
use anyhow::Error;
use api_autogen::apis::bot_management::{
    BotsAssignBotResponse, BotsCreateResponse, BotsDeleteResponse, BotsListBotsResponse,
    BotsModelKeysResponse, BotsUpdateResponse,
};
use api_autogen::models;
use api_autogen::models::WsEvent::WsEventJoined;
use api_autogen::models::{
    AssignBotPayload, BotsDeletePathParams, BotsUpdatePathParams, CreateBot, PlayerColor, ReadBot,
    UpdateBot, WsPlayerJoined,
};
use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHasher};
use async_trait::async_trait;
use axum::http::Method;
use axum_extra::extract::{CookieJar, Host};
use log::warn;
use sea_orm::{EntityLoaderTrait, NotSet, Unchanged};
use sea_orm::{EntityTrait, Set, TransactionTrait};
use tracing::info;

#[allow(unused_variables)]
#[async_trait]
impl api_autogen::apis::bot_management::BotManagement<anyhow::Error> for ServerImpl {
    type Claims = Claims;

    async fn bots_assign_bot(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &AssignBotPayload,
    ) -> Result<BotsAssignBotResponse, Error> {
        let Some(game) = self.find_game(body.game_id).await? else {
            info!("Game {} not found", body.game_id);
            return Ok(BotsAssignBotResponse::Status404_NotFound(error_404_miss()));
        };

        if game.game_status != GameStatusDb::Waiting {
            info!("Game {} is not in waiting status", body.game_id);
            return Ok(BotsAssignBotResponse::Status409_Conflict(
                error_409_conflict(),
            ));
        }

        if claims.userType != UserTypeDb::Admin {
            if let Some(game_owner_id) = game.owner_id {
                if game_owner_id != claims.sub {
                    info!(
                        "User {} is not the owner of game {}",
                        claims.sub, body.game_id
                    );
                    return Ok(BotsAssignBotResponse::Status403_Forbidden(
                        error_403_forbidden(),
                    ));
                }
            }
        } else {
            info!(
                "Admin user {} assigning bot to game {}",
                claims.sub, body.game_id
            );
        }

        let Some(bot_user) = users::Entity::load()
            .with(players::Entity)
            .filter_by_id(body.bot_id)
            .one(&self.db)
            .await?
        else {
            info!("Bot user {} not found", body.bot_id);
            return Ok(BotsAssignBotResponse::Status404_NotFound(error_404_miss()));
        };

        // Check the bot is indeed a bot
        if bot_user.user_type != UserTypeDb::Bot {
            info!("User {} is not a bot", body.bot_id);
            return Ok(BotsAssignBotResponse::Status404_NotFound(error_404_miss()));
        }

        let new_bot_player = players::ActiveModel {
            user_id: Set(bot_user.id),
            game_id: Set(game.id),
            color: Set(match body.color {
                PlayerColor::White => ColorDb::White,
                PlayerColor::Grey => ColorDb::Gray,
                PlayerColor::Black => ColorDb::Black,
            }),
            relation: Set(players::GameRelationDb::None),
        };

        players::Entity::insert(new_bot_player)
            .exec(&self.db)
            .await?;

        let players = self.find_players_for_game(game.id).await?;
        let players = self.players_to_dto(&players, game.names_masked.unwrap_or(false)).await;

        self.game_events
            .send(GameEvent {
                game_id: game.id,
                event: WsEventJoined(Box::new(models::WsEventJoined::new(
                    "".into(),
                    WsPlayerJoined::new(bot_user.username, body.color.clone(), players),
                ))),
            })
            .await;

        Ok(BotsAssignBotResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }

    async fn bots_create(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &CreateBot,
    ) -> Result<BotsCreateResponse, Error> {
        if claims.userType != UserTypeDb::Admin {
            return Ok(BotsCreateResponse::Status403_Forbidden(
                error_403_forbidden(),
            ));
        }

        if let Some(key) = body.model_key.clone() {
            if !self.model_service.get_all_model_keys().contains(&key) {
                warn!("Model key {} does not exist", key);
                return Ok(BotsCreateResponse::Status409_Conflict(error_409_conflict()));
            }
        }

        let (user, bot) = self
            .db
            .transaction::<_, _, anyhow::Error>(|txn| {
                let body = body.clone();

                Box::pin(async move {
                    let salt = SaltString::generate(&mut OsRng);

                    let password_hash = Argon2::default()
                        .hash_password(body.password.as_bytes(), &salt)
                        .map_err(err_encapsulate)?
                        .to_string();

                    let new_user = users::ActiveModel {
                        id: Default::default(),
                        username: Set(body.username.clone()),
                        password_hash: Set(password_hash),
                        lichess_rating: Set(None),
                        chess_com_rating: Set(None),
                        fide_rating: Set(None),
                        experience_with_chess: Set(None),
                        user_type: Set(UserTypeDb::Bot),
                    };

                    let inserted_user = users::Entity::insert(new_user)
                        .exec_with_returning(txn)
                        .await?;

                    let new_bot = bot::ActiveModel {
                        id: Default::default(),
                        user_id: Set(inserted_user.id),
                        model_key: Set(body.model_key.clone()),
                        playouts: Set(body.playouts_per_move),
                        exploration_factor: Set(body.exploration_factor),
                        virtual_loss_weight: Set(body.virtual_loss),
                        contempt: Set(body.contempt),
                    };

                    let inserted_bot = bot::Entity::insert(new_bot)
                        .exec_with_returning(txn)
                        .await?;

                    Ok((inserted_user, inserted_bot))
                })
            })
            .await?;

        Ok(BotsCreateResponse::Status200_TheRequestHasSucceeded(
            ReadBot {
                user_id: user.id,
                bot_id: bot.id,
                username: user.username,
                model_key: bot.model_key,
                playouts_per_move: bot.playouts,
                exploration_factor: bot.exploration_factor,
                virtual_loss: bot.virtual_loss_weight,
                contempt: bot.contempt,
            },
        ))
    }

    async fn bots_delete(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &BotsDeletePathParams,
    ) -> Result<BotsDeleteResponse, Error> {
        if claims.userType != UserTypeDb::Admin {
            return Ok(BotsDeleteResponse::Status403_Forbidden(
                error_403_forbidden(),
            ));
        }

        let Some(bot_record) = bot::Entity::load()
            .with(users::Entity)
            .filter_by_id(path_params.bot_id)
            .one(&self.db)
            .await?
        else {
            return Ok(BotsDeleteResponse::Status404_NotFound(error_404_miss()));
        };

        let updated_user = users::ActiveModel {
            id: Set(bot_record.user_id),
            username: Set(format!("{} (deleted bot)", bot_record.user.as_ref().unwrap().username)),
            password_hash: NotSet,
            lichess_rating: NotSet,
            chess_com_rating: NotSet,
            fide_rating: NotSet,
            experience_with_chess: NotSet,
            user_type: Set(UserTypeDb::Normal)
        };

        self.db
            .transaction::<_, _, anyhow::Error>(|txn| {
                let updated_user = updated_user.clone();
                let bot_id = bot_record.id;
                Box::pin(async move {
                    users::Entity::update(updated_user).exec(txn).await?;
                    bot::Entity::delete_by_id(bot_id).exec(txn).await?;
                    Ok(())
                })
            })
            .await?;

        info!("Deleted bot {} and updated user {}", bot_record.id, bot_record.user.as_ref().unwrap().username);

        Ok(BotsDeleteResponse::Status204_ThereIsNoContentToSendForThisRequest)
    }

    async fn bots_list_bots(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<BotsListBotsResponse, Error> {
        let bots_list = bot::Entity::load()
            .with(users::Entity)
            .all(&self.db)
            .await?
            .into_iter()
            .map(Into::into)
            .collect::<Vec<ReadBot>>();

        Ok(BotsListBotsResponse::Status200_TheRequestHasSucceeded(
            bots_list,
        ))
    }

    async fn bots_model_keys(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<BotsModelKeysResponse, Error> {
        let model_keys = self.model_service.get_all_model_keys();

        Ok(BotsModelKeysResponse::Status200_TheRequestHasSucceeded(
            model_keys,
        ))
    }

    async fn bots_update(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        path_params: &BotsUpdatePathParams,
        body: &UpdateBot,
    ) -> Result<BotsUpdateResponse, Error> {
        // check admin
        if claims.userType != UserTypeDb::Admin {
            return Ok(BotsUpdateResponse::Status403_Forbidden(
                error_403_forbidden(),
            ));
        }

        if let Some(key) = body.model_key.clone() {
            if !self.model_service.get_all_model_keys().contains(&key) {
                warn!("Model key {} does not exist", key);
                return Ok(BotsUpdateResponse::Status400_BadRequest(
                    error_400_bad_request_with_msg("Model key does not exist".into()),
                ));
            }
        }

        let Some(bot_record) = bot::Entity::find_by_id(path_params.bot_id)
            .one(&self.db)
            .await?
        else {
            return Ok(BotsUpdateResponse::Status404_NotFound(error_404_miss()));
        };

        let updated_bot = bot::ActiveModel {
            id: Set(bot_record.id),
            user_id: Set(bot_record.user_id),
            model_key: Set(body.model_key.clone()),
            playouts: Set(body.playouts_per_move),
            exploration_factor: Set(body.exploration_factor),
            virtual_loss_weight: Set(body.virtual_loss),
            contempt: Set(body.contempt),
        };

        let updated_user = users::ActiveModel {
            id: Set(bot_record.user_id),
            username: Set(body.username.clone()),
            password_hash: NotSet,
            lichess_rating: NotSet,
            chess_com_rating: NotSet,
            fide_rating: NotSet,
            experience_with_chess: NotSet,
            user_type: NotSet,
        };

        self.db
            .transaction::<_, _, anyhow::Error>(|txn| {
                let updated_bot = updated_bot.clone();
                let updated_user = updated_user.clone();
                Box::pin(async move {
                    users::Entity::update(updated_user).exec(txn).await?;
                    bot::Entity::update(updated_bot).exec(txn).await?;
                    Ok(())
                })
            })
            .await?;

        let updated_bot: ReadBot = bot::Entity::load()
            .with(users::Entity)
            .filter_by_id(path_params.bot_id)
            .one(&self.db)
            .await?
            .expect("Bot just updated must exist")
            .into();

        info!("Updated bot to {:?}", updated_bot);

        Ok(BotsUpdateResponse::Status200_TheRequestHasSucceeded(
            updated_bot,
        ))
    }
}
