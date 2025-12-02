use crate::entity::{games, players, users};
use crate::server::{Claims, ServerImpl};
use anyhow::Error;
use api_autogen::apis::leaderboard::LeaderboardGetLeaderboardResponse;
use api_autogen::models::{LeaderboardGetLeaderboardQueryParams, LeaderboardSortBy};
use async_trait::async_trait;
use axum::http::Method;
use axum_extra::extract::{CookieJar, Host};

#[allow(unused_variables)]
#[async_trait]
impl api_autogen::apis::leaderboard::Leaderboard<anyhow::Error> for ServerImpl {
    type Claims = Claims;

    async fn leaderboard_get_leaderboard(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        query_params: &LeaderboardGetLeaderboardQueryParams,
    ) -> Result<LeaderboardGetLeaderboardResponse, Error> {
        let users = users::Entity::load()
            .with(players::Entity)
            .with((players::Entity, games::Entity))
            .all(&self.db)
            .await?;

        let mut leaderboard_entries = Vec::new();

        for user in users {
            if !query_params.include_bots && user.user_type == crate::entity::users::UserTypeDb::Bot
            {
                continue;
            }

            let player_stats = user.players.iter().fold(
                (0, 0, 0, 0),
                |(wins, losses, draws, games_played), player| {
                    if let Some(tournament_id) = query_params.tournament_id {
                        if player
                            .games
                            .as_ref()
                            .map_or(true, |g| g.tournament_id != Some(tournament_id))
                        {
                            return (wins, losses, draws, games_played);
                        }
                    }

                    match player.relation {
                        players::GameRelationDb::Winner => {
                            (wins + 1, losses, draws, games_played + 1)
                        }
                        players::GameRelationDb::Loser => {
                            (wins, losses + 1, draws, games_played + 1)
                        }
                        players::GameRelationDb::Draw => {
                            (wins, losses, draws + 1, games_played + 1)
                        }
                        players::GameRelationDb::None => (wins, losses, draws, games_played),
                    }
                },
            );

            if player_stats.3 == 0 {
                continue; // Skip users with no games played
            }


            leaderboard_entries.push(api_autogen::models::LeaderboardEntry {
                user_id: user.id,
                is_bot: user.user_type == crate::entity::users::UserTypeDb::Bot,
                username: user.username.clone(),
                games_played: player_stats.3,
                wins: player_stats.0,
                win_rate: if player_stats.3 > 0 {
                    player_stats.0 as f32 / player_stats.3 as f32
                } else {
                    0.0
                },
                losses: player_stats.1,
                loss_rate: if player_stats.3 > 0 {
                    player_stats.1 as f32 / player_stats.3 as f32
                } else {
                    0.0
                },
                draws: player_stats.2,
            });
        }

        match query_params.sort_by {
            LeaderboardSortBy::Wins => {
                leaderboard_entries.sort_by(|a, b| b.wins.cmp(&a.wins));
            }
            LeaderboardSortBy::GamesPlayed => {
                leaderboard_entries.sort_by(|a, b| b.games_played.cmp(&a.games_played));
            }
            LeaderboardSortBy::WinRate => {
                leaderboard_entries.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap());
            }
            LeaderboardSortBy::LossRate => {
                leaderboard_entries.sort_by(|a, b| b.loss_rate.partial_cmp(&a.loss_rate).unwrap());
            }
            LeaderboardSortBy::Losses => {
                leaderboard_entries.sort_by(|a, b| b.losses.cmp(&a.losses));
            }
            LeaderboardSortBy::Draws => {
                leaderboard_entries.sort_by(|a, b| b.draws.cmp(&a.draws));
            }
        }

        Ok(
            LeaderboardGetLeaderboardResponse::Status200_TheRequestHasSucceeded(
                leaderboard_entries,
            ),
        )
    }
}
