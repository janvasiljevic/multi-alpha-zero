use crate::entity::prelude::Users;
use crate::entity::users;
use crate::entity::users::UserTypeDb;
use crate::errors::{err_encapsulate, error_401_unauthorized, error_404_miss, error_409_conflict};
use crate::server::{Claims, ServerImpl};
use anyhow::Error;
use api_autogen::apis::authentication::{
    AuthLoginResponse, AuthMeResponse, AuthRegisterResponse, AuthUpdateProfileResponse,
};
use api_autogen::apis::BasicAuthKind;
use api_autogen::models::{LoginPayload, LoginResponse, MeUser, RegisterPayload, UpdateProfilePayload, UserCreated};
use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use async_trait::async_trait;
use axum::http::{HeaderMap, Method};
use axum_extra::extract::{CookieJar, Host};
use chrono::Utc;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use sea_orm::{ActiveModelTrait, EntityTrait, IntoActiveModel, Set};

impl ServerImpl {
    pub fn create_jwt_token(&self, user: &users::Model) -> Result<String, anyhow::Error> {
        let claims = Claims {
            sub: user.id,
            exp: (Utc::now() + chrono::Duration::days(7)).timestamp() as usize,
            userType: user.user_type.clone(),
            username: user.username.clone(),
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_ref()),
        )?;

        Ok(token)
    }
}

#[allow(unused_variables)]
#[async_trait]
impl api_autogen::apis::authentication::Authentication<anyhow::Error> for ServerImpl {
    type Claims = Claims;

    async fn auth_login(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        body: &LoginPayload,
    ) -> Result<AuthLoginResponse, anyhow::Error> {
        let user = match Users::find_by_username(body.username.clone())
            .one(&self.db)
            .await?
        {
            Some(user) => user,
            None => {
                return Ok(AuthLoginResponse::Status401_UnauthorizedAccess(
                    error_401_unauthorized(),
                ));
            }
        };

        let is_valid = match PasswordHash::new(&user.password_hash) {
            Ok(parsed_hash) => Argon2::default()
                .verify_password(body.password.as_bytes(), &parsed_hash)
                .is_ok(),
            Err(_) => false,
        };

        if !is_valid {
            return Ok(AuthLoginResponse::Status401_UnauthorizedAccess(
                error_401_unauthorized(),
            ));
        }

        let token = self.create_jwt_token(&user)?;

        Ok(AuthLoginResponse::Status200_TheRequestHasSucceeded(
            LoginResponse {
                token,
                user: user.into(),
            },
        ))
    }

    async fn auth_me(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
    ) -> Result<AuthMeResponse, anyhow::Error> {
        let user = match Users::find_by_id(claims.sub).one(&self.db).await? {
            Some(user) => user,
            None => {
                return Ok(AuthMeResponse::Status404_NotFound(error_404_miss()));
            }
        };

        Ok(AuthMeResponse::Status200_TheRequestHasSucceeded(
            user.into()
        ))
    }

    async fn auth_register(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        body: &RegisterPayload,
    ) -> Result<AuthRegisterResponse, anyhow::Error> {
        match Users::find_by_username(body.username.clone())
            .one(&self.db)
            .await?
        {
            Some(_) => {
                return Ok(AuthRegisterResponse::Status409_Conflict(
                    error_409_conflict(),
                ));
            }
            None => {}
        };

        let salt = SaltString::generate(&mut OsRng);

        let password_hash = Argon2::default()
            .hash_password(body.password.as_bytes(), &salt)
            .map_err(err_encapsulate)?
            .to_string();

        let inserted = Users::insert(users::ActiveModel {
            username: Set(body.username.to_owned()),
            password_hash: Set(password_hash.to_owned()),
            user_type: Set(UserTypeDb::Normal),
            experience_with_chess: Set(body.experience_with_chess.clone()),
            chess_com_rating: Set(body.chess_com_rating.clone()),
            lichess_rating: Set(body.lichess_rating.clone()),
            fide_rating: Set(body.fide_rating.clone()),
            id: Default::default(),
        })
        .exec_with_returning(&self.db)
        .await?;

        let token = self.create_jwt_token(&inserted)?;

        Ok(AuthRegisterResponse::Status201_TheRequestHasSucceededAndANewResourceHasBeenCreatedAsAResult(
            UserCreated {
                user: inserted.into(),
                token,
            },
        ))
    }

    async fn auth_update_profile(
        &self,
        method: &Method,
        host: &Host,
        cookies: &CookieJar,
        claims: &Self::Claims,
        body: &UpdateProfilePayload,
    ) -> Result<AuthUpdateProfileResponse, Error> {
        let Some(user) = Users::find_by_id(claims.sub).one(&self.db).await? else {
            return Ok(AuthUpdateProfileResponse::Status404_NotFound(
                error_404_miss(),
            ));
        };

        let mut user = user.into_active_model();

        if let Some(exp) = &body.experience_with_chess {
            user.experience_with_chess = Set(Some(*exp));
        } else {
            user.experience_with_chess = Set(None);
        }

        if let Some(rating) = &body.chess_com_rating {
            user.chess_com_rating = Set(Some(*rating));
        } else {
            user.chess_com_rating = Set(None);
        }

        if let Some(rating) = &body.lichess_rating {
            user.lichess_rating = Set(Some(*rating));
        } else {
            user.lichess_rating = Set(None);
        }

        if let Some(rating) = &body.fide_rating {
            user.fide_rating = Set(Some(*rating));
        } else {
            user.fide_rating = Set(None);
        }

        let updated_user = user.update(&self.db).await?;

        Ok(AuthUpdateProfileResponse::Status200_TheRequestHasSucceeded(
            updated_user.into(),
        ))
    }
}

#[async_trait::async_trait]
impl api_autogen::apis::ApiAuthBasic for ServerImpl {
    type Claims = Claims;

    async fn extract_claims_from_auth_header(
        &self,
        kind: BasicAuthKind,
        headers: &HeaderMap,
        _key: &str,
    ) -> Option<Self::Claims> {
        if !matches!(kind, BasicAuthKind::Bearer) {
            return None;
        }

        if let Some(ws_protocol_header) = headers.get(http::header::SEC_WEBSOCKET_PROTOCOL) {
            // Needs to start with "Authorization, "
            if let Ok(header_str) = ws_protocol_header.to_str() {
                if let Some(token) = header_str.strip_prefix("Authorization, ") {
                    let token_data = decode::<Claims>(
                        token,
                        &DecodingKey::from_secret(self.jwt_secret.as_ref()),
                        &Validation::default(),
                    )
                    .ok()?;
                    return Some(token_data.claims);
                }
            }
        }

        let auth_header = headers.get(http::header::AUTHORIZATION)?.to_str().ok()?;

        let token = auth_header.strip_prefix("Bearer ")?;

        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_ref()),
            &Validation::default(),
        )
        .ok()?;

        Some(token_data.claims)
    }
}
