use crate::entity::tournament;
use crate::server::{Claims, ServerImpl};
use anyhow::Error;
use api_autogen::apis::tournament_management::{
    TournamentsCreateTournamentResponse, TournamentsListTournamentsResponse,
};
use api_autogen::models::{CreateTournament, ReadTournament};
use async_trait::async_trait;
use axum::http::Method;
use axum_extra::extract::{CookieJar, Host};
use sea_orm::EntityTrait;

#[allow(unused_variables)]
#[async_trait]
impl api_autogen::apis::tournament_management::TournamentManagement<anyhow::Error> for ServerImpl {
    type Claims = Claims;

    async fn tournaments_create_tournament(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &CreateTournament,
    ) -> Result<TournamentsCreateTournamentResponse, Error> {
        let new_tournament = tournament::ActiveModel {
            name: sea_orm::Set(body.name.clone()),
            id: Default::default(),
        };

        let res = tournament::Entity::insert(new_tournament)
            .exec(&self.db)
            .await?;

        let created_tournament = ReadTournament {
            id: res.last_insert_id,
            name: body.name.clone(),
        };

        Ok(
            TournamentsCreateTournamentResponse::Status200_TheRequestHasSucceeded(
                created_tournament,
            ),
        )
    }

    async fn tournaments_list_tournaments(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<TournamentsListTournamentsResponse, Error> {
        let tournaments = tournament::Entity::find()
            .all(&self.db)
            .await?
            .iter()
            .map(|t| ReadTournament {
                id: t.id,
                name: t.name.clone(),
            })
            .collect();

        Ok(TournamentsListTournamentsResponse::Status200_TheRequestHasSucceeded(tournaments))
    }
}
