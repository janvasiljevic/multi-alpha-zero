#![allow(unused_qualifications)]

use http::HeaderValue;
use validator::Validate;

#[cfg(feature = "server")]
use crate::header;
use crate::{models, types::*};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct BotsDeletePathParams {
    pub bot_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct BotsUpdatePathParams {
    pub bot_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct HistoryGetGameHistoryPathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesFinishGameInPathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesGetGameStatePathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesJoinGamePathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesLeaveGamePathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesListGamesQueryParams {
    #[serde(rename = "status")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<models::FilterGames>,
    #[serde(rename = "tournamentId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tournament_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesMakeMovePathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesStartGamePathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GamesSubscribeToGameEventsPathParams {
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct LeaderboardGetLeaderboardQueryParams {
    #[serde(rename = "includeBots")]
    pub include_bots: bool,
    #[serde(rename = "sortBy")]
    pub sort_by: models::LeaderboardSortBy,
    #[serde(rename = "tournamentId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tournament_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct UsersRemoveUserFromGamePathParams {
    pub user_id: i64,
    pub game_id: i64,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct AssignBotPayload {
    #[serde(rename = "gameId")]
    pub game_id: i64,

    #[serde(rename = "botId")]
    pub bot_id: i64,

    #[serde(rename = "color")]
    pub color: models::PlayerColor,
}

impl AssignBotPayload {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(game_id: i64, bot_id: i64, color: models::PlayerColor) -> AssignBotPayload {
        AssignBotPayload {
            game_id,
            bot_id,
            color,
        }
    }
}

/// Converts the AssignBotPayload value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for AssignBotPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("gameId".to_string()),
            Some(self.game_id.to_string()),
            Some("botId".to_string()),
            Some(self.bot_id.to_string()),
            // Skipping color in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a AssignBotPayload value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for AssignBotPayload {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub game_id: Vec<i64>,
            pub bot_id: Vec<i64>,
            pub color: Vec<models::PlayerColor>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing AssignBotPayload".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "gameId" => intermediate_rep.game_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "botId" => intermediate_rep.bot_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "color" => intermediate_rep.color.push(
                        <models::PlayerColor as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing AssignBotPayload".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(AssignBotPayload {
            game_id: intermediate_rep
                .game_id
                .into_iter()
                .next()
                .ok_or_else(|| "gameId missing in AssignBotPayload".to_string())?,
            bot_id: intermediate_rep
                .bot_id
                .into_iter()
                .next()
                .ok_or_else(|| "botId missing in AssignBotPayload".to_string())?,
            color: intermediate_rep
                .color
                .into_iter()
                .next()
                .ok_or_else(|| "color missing in AssignBotPayload".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<AssignBotPayload> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<AssignBotPayload>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<AssignBotPayload>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for AssignBotPayload - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<AssignBotPayload> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <AssignBotPayload as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into AssignBotPayload - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Bad Request
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct BadRequestError {
    #[serde(rename = "message")]
    pub message: String,
}

impl BadRequestError {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(message: String) -> BadRequestError {
        BadRequestError { message }
    }
}

/// Converts the BadRequestError value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for BadRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("message".to_string()), Some(self.message.to_string())];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a BadRequestError value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for BadRequestError {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub message: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing BadRequestError".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "message" => intermediate_rep.message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing BadRequestError".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(BadRequestError {
            message: intermediate_rep
                .message
                .into_iter()
                .next()
                .ok_or_else(|| "message missing in BadRequestError".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<BadRequestError> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<BadRequestError>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<BadRequestError>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for BadRequestError - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<BadRequestError> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <BadRequestError as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into BadRequestError - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Bot {
    #[serde(rename = "user_id")]
    pub user_id: i64,

    #[serde(rename = "bot_id")]
    pub bot_id: i64,

    #[serde(rename = "username")]
    #[validate(length(min = 3))]
    pub username: String,

    #[serde(rename = "model_key")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,

    #[serde(rename = "playouts_per_move")]
    #[validate(range(min = 1u32))]
    pub playouts_per_move: u32,

    #[serde(rename = "exploration_factor")]
    #[validate(range(min = 0f32, max = 20f32))]
    pub exploration_factor: f32,

    #[serde(rename = "virtual_loss")]
    #[validate(range(min = 0f32, max = 5f32))]
    pub virtual_loss: f32,

    #[serde(rename = "contempt")]
    #[validate(
            range(min = -3f32, max = 3f32),
        )]
    pub contempt: f32,

    #[serde(rename = "password")]
    #[validate(length(min = 1))]
    pub password: String,
}

impl Bot {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        user_id: i64,
        bot_id: i64,
        username: String,
        playouts_per_move: u32,
        exploration_factor: f32,
        virtual_loss: f32,
        contempt: f32,
        password: String,
    ) -> Bot {
        Bot {
            user_id,
            bot_id,
            username,
            model_key: None,
            playouts_per_move,
            exploration_factor,
            virtual_loss,
            contempt,
            password,
        }
    }
}

/// Converts the Bot value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Bot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("user_id".to_string()),
            Some(self.user_id.to_string()),
            Some("bot_id".to_string()),
            Some(self.bot_id.to_string()),
            Some("username".to_string()),
            Some(self.username.to_string()),
            self.model_key
                .as_ref()
                .map(|model_key| ["model_key".to_string(), model_key.to_string()].join(",")),
            Some("playouts_per_move".to_string()),
            Some(self.playouts_per_move.to_string()),
            Some("exploration_factor".to_string()),
            Some(self.exploration_factor.to_string()),
            Some("virtual_loss".to_string()),
            Some(self.virtual_loss.to_string()),
            Some("contempt".to_string()),
            Some(self.contempt.to_string()),
            Some("password".to_string()),
            Some(self.password.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Bot value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Bot {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub user_id: Vec<i64>,
            pub bot_id: Vec<i64>,
            pub username: Vec<String>,
            pub model_key: Vec<String>,
            pub playouts_per_move: Vec<u32>,
            pub exploration_factor: Vec<f32>,
            pub virtual_loss: Vec<f32>,
            pub contempt: Vec<f32>,
            pub password: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err("Missing value while parsing Bot".to_string())
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "user_id" => intermediate_rep.user_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "bot_id" => intermediate_rep.bot_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "model_key" => intermediate_rep.model_key.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "playouts_per_move" => intermediate_rep.playouts_per_move.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "exploration_factor" => intermediate_rep.exploration_factor.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "virtual_loss" => intermediate_rep.virtual_loss.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "contempt" => intermediate_rep.contempt.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "password" => intermediate_rep.password.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Bot".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(Bot {
            user_id: intermediate_rep
                .user_id
                .into_iter()
                .next()
                .ok_or_else(|| "user_id missing in Bot".to_string())?,
            bot_id: intermediate_rep
                .bot_id
                .into_iter()
                .next()
                .ok_or_else(|| "bot_id missing in Bot".to_string())?,
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in Bot".to_string())?,
            model_key: intermediate_rep.model_key.into_iter().next(),
            playouts_per_move: intermediate_rep
                .playouts_per_move
                .into_iter()
                .next()
                .ok_or_else(|| "playouts_per_move missing in Bot".to_string())?,
            exploration_factor: intermediate_rep
                .exploration_factor
                .into_iter()
                .next()
                .ok_or_else(|| "exploration_factor missing in Bot".to_string())?,
            virtual_loss: intermediate_rep
                .virtual_loss
                .into_iter()
                .next()
                .ok_or_else(|| "virtual_loss missing in Bot".to_string())?,
            contempt: intermediate_rep
                .contempt
                .into_iter()
                .next()
                .ok_or_else(|| "contempt missing in Bot".to_string())?,
            password: intermediate_rep
                .password
                .into_iter()
                .next()
                .ok_or_else(|| "password missing in Bot".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<Bot> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<Bot>> for HeaderValue {
    type Error = String;

    fn try_from(hdr_value: header::IntoHeaderValue<Bot>) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for Bot - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<Bot> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => match <Bot as std::str::FromStr>::from_str(value) {
                std::result::Result::Ok(value) => {
                    std::result::Result::Ok(header::IntoHeaderValue(value))
                }
                std::result::Result::Err(err) => std::result::Result::Err(format!(
                    "Unable to convert header value '{}' into Bot - {}",
                    value, err
                )),
            },
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Conflict
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct ConflictError {
    #[serde(rename = "message")]
    pub message: String,
}

impl ConflictError {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(message: String) -> ConflictError {
        ConflictError { message }
    }
}

/// Converts the ConflictError value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for ConflictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("message".to_string()), Some(self.message.to_string())];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a ConflictError value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for ConflictError {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub message: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing ConflictError".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "message" => intermediate_rep.message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing ConflictError".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(ConflictError {
            message: intermediate_rep
                .message
                .into_iter()
                .next()
                .ok_or_else(|| "message missing in ConflictError".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<ConflictError> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<ConflictError>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<ConflictError>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for ConflictError - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<ConflictError> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <ConflictError as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into ConflictError - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct CreateBot {
    #[serde(rename = "username")]
    #[validate(length(min = 3))]
    pub username: String,

    #[serde(rename = "model_key")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,

    #[serde(rename = "playouts_per_move")]
    #[validate(range(min = 1u32))]
    pub playouts_per_move: u32,

    #[serde(rename = "exploration_factor")]
    #[validate(range(min = 0f32, max = 20f32))]
    pub exploration_factor: f32,

    #[serde(rename = "virtual_loss")]
    #[validate(range(min = 0f32, max = 5f32))]
    pub virtual_loss: f32,

    #[serde(rename = "contempt")]
    #[validate(
            range(min = -3f32, max = 3f32),
        )]
    pub contempt: f32,

    #[serde(rename = "password")]
    #[validate(length(min = 1))]
    pub password: String,
}

impl CreateBot {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        username: String,
        playouts_per_move: u32,
        exploration_factor: f32,
        virtual_loss: f32,
        contempt: f32,
        password: String,
    ) -> CreateBot {
        CreateBot {
            username,
            model_key: None,
            playouts_per_move,
            exploration_factor,
            virtual_loss,
            contempt,
            password,
        }
    }
}

/// Converts the CreateBot value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for CreateBot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("username".to_string()),
            Some(self.username.to_string()),
            self.model_key
                .as_ref()
                .map(|model_key| ["model_key".to_string(), model_key.to_string()].join(",")),
            Some("playouts_per_move".to_string()),
            Some(self.playouts_per_move.to_string()),
            Some("exploration_factor".to_string()),
            Some(self.exploration_factor.to_string()),
            Some("virtual_loss".to_string()),
            Some(self.virtual_loss.to_string()),
            Some("contempt".to_string()),
            Some(self.contempt.to_string()),
            Some("password".to_string()),
            Some(self.password.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a CreateBot value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for CreateBot {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub username: Vec<String>,
            pub model_key: Vec<String>,
            pub playouts_per_move: Vec<u32>,
            pub exploration_factor: Vec<f32>,
            pub virtual_loss: Vec<f32>,
            pub contempt: Vec<f32>,
            pub password: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing CreateBot".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "model_key" => intermediate_rep.model_key.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "playouts_per_move" => intermediate_rep.playouts_per_move.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "exploration_factor" => intermediate_rep.exploration_factor.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "virtual_loss" => intermediate_rep.virtual_loss.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "contempt" => intermediate_rep.contempt.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "password" => intermediate_rep.password.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing CreateBot".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(CreateBot {
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in CreateBot".to_string())?,
            model_key: intermediate_rep.model_key.into_iter().next(),
            playouts_per_move: intermediate_rep
                .playouts_per_move
                .into_iter()
                .next()
                .ok_or_else(|| "playouts_per_move missing in CreateBot".to_string())?,
            exploration_factor: intermediate_rep
                .exploration_factor
                .into_iter()
                .next()
                .ok_or_else(|| "exploration_factor missing in CreateBot".to_string())?,
            virtual_loss: intermediate_rep
                .virtual_loss
                .into_iter()
                .next()
                .ok_or_else(|| "virtual_loss missing in CreateBot".to_string())?,
            contempt: intermediate_rep
                .contempt
                .into_iter()
                .next()
                .ok_or_else(|| "contempt missing in CreateBot".to_string())?,
            password: intermediate_rep
                .password
                .into_iter()
                .next()
                .ok_or_else(|| "password missing in CreateBot".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<CreateBot> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<CreateBot>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<CreateBot>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for CreateBot - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<CreateBot> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <CreateBot as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into CreateBot - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct CreateGame {
    #[serde(rename = "name")]
    pub name: String,

    #[serde(rename = "suggested_move_time_seconds")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_move_time_seconds: Option<i32>,

    #[serde(rename = "names_masked")]
    pub names_masked: bool,

    #[serde(rename = "material_masked")]
    pub material_masked: bool,

    #[serde(rename = "tournamentId")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tournament_id: Option<i64>,

    #[serde(rename = "training_mode")]
    pub training_mode: bool,
}

impl CreateGame {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        name: String,
        names_masked: bool,
        material_masked: bool,
        training_mode: bool,
    ) -> CreateGame {
        CreateGame {
            name,
            suggested_move_time_seconds: None,
            names_masked,
            material_masked,
            tournament_id: None,
            training_mode,
        }
    }
}

/// Converts the CreateGame value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for CreateGame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("name".to_string()),
            Some(self.name.to_string()),
            self.suggested_move_time_seconds
                .as_ref()
                .map(|suggested_move_time_seconds| {
                    [
                        "suggested_move_time_seconds".to_string(),
                        suggested_move_time_seconds.to_string(),
                    ]
                    .join(",")
                }),
            Some("names_masked".to_string()),
            Some(self.names_masked.to_string()),
            Some("material_masked".to_string()),
            Some(self.material_masked.to_string()),
            self.tournament_id.as_ref().map(|tournament_id| {
                ["tournamentId".to_string(), tournament_id.to_string()].join(",")
            }),
            Some("training_mode".to_string()),
            Some(self.training_mode.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a CreateGame value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for CreateGame {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub name: Vec<String>,
            pub suggested_move_time_seconds: Vec<i32>,
            pub names_masked: Vec<bool>,
            pub material_masked: Vec<bool>,
            pub tournament_id: Vec<i64>,
            pub training_mode: Vec<bool>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing CreateGame".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "name" => intermediate_rep.name.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "suggested_move_time_seconds" => {
                        intermediate_rep.suggested_move_time_seconds.push(
                            <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                        )
                    }
                    #[allow(clippy::redundant_clone)]
                    "names_masked" => intermediate_rep.names_masked.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "material_masked" => intermediate_rep.material_masked.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "tournamentId" => intermediate_rep.tournament_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "training_mode" => intermediate_rep.training_mode.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing CreateGame".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(CreateGame {
            name: intermediate_rep
                .name
                .into_iter()
                .next()
                .ok_or_else(|| "name missing in CreateGame".to_string())?,
            suggested_move_time_seconds: intermediate_rep
                .suggested_move_time_seconds
                .into_iter()
                .next(),
            names_masked: intermediate_rep
                .names_masked
                .into_iter()
                .next()
                .ok_or_else(|| "names_masked missing in CreateGame".to_string())?,
            material_masked: intermediate_rep
                .material_masked
                .into_iter()
                .next()
                .ok_or_else(|| "material_masked missing in CreateGame".to_string())?,
            tournament_id: intermediate_rep.tournament_id.into_iter().next(),
            training_mode: intermediate_rep
                .training_mode
                .into_iter()
                .next()
                .ok_or_else(|| "training_mode missing in CreateGame".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<CreateGame> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<CreateGame>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<CreateGame>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for CreateGame - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<CreateGame> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <CreateGame as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into CreateGame - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct CreateTournament {
    #[serde(rename = "name")]
    pub name: String,
}

impl CreateTournament {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(name: String) -> CreateTournament {
        CreateTournament { name }
    }
}

/// Converts the CreateTournament value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for CreateTournament {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("name".to_string()), Some(self.name.to_string())];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a CreateTournament value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for CreateTournament {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub name: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing CreateTournament".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "name" => intermediate_rep.name.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing CreateTournament".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(CreateTournament {
            name: intermediate_rep
                .name
                .into_iter()
                .next()
                .ok_or_else(|| "name missing in CreateTournament".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<CreateTournament> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<CreateTournament>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<CreateTournament>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for CreateTournament - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<CreateTournament> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <CreateTournament as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into CreateTournament - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum FilterGames {
    #[serde(rename = "Waiting")]
    Waiting,
    #[serde(rename = "InProgress")]
    InProgress,
    #[serde(rename = "Finished")]
    Finished,
}

impl std::fmt::Display for FilterGames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            FilterGames::Waiting => write!(f, "Waiting"),
            FilterGames::InProgress => write!(f, "InProgress"),
            FilterGames::Finished => write!(f, "Finished"),
        }
    }
}

impl std::str::FromStr for FilterGames {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Waiting" => std::result::Result::Ok(FilterGames::Waiting),
            "InProgress" => std::result::Result::Ok(FilterGames::InProgress),
            "Finished" => std::result::Result::Ok(FilterGames::Finished),
            _ => std::result::Result::Err(format!("Value not valid: {}", s)),
        }
    }
}

/// Only available for admins
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct FinishGameIn {
    #[serde(rename = "white")]
    pub white: models::GameRelation,

    #[serde(rename = "grey")]
    pub grey: models::GameRelation,

    #[serde(rename = "black")]
    pub black: models::GameRelation,
}

impl FinishGameIn {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        white: models::GameRelation,
        grey: models::GameRelation,
        black: models::GameRelation,
    ) -> FinishGameIn {
        FinishGameIn { white, grey, black }
    }
}

/// Converts the FinishGameIn value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for FinishGameIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping white in query parameter serialization

            // Skipping grey in query parameter serialization

            // Skipping black in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a FinishGameIn value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for FinishGameIn {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub white: Vec<models::GameRelation>,
            pub grey: Vec<models::GameRelation>,
            pub black: Vec<models::GameRelation>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing FinishGameIn".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "white" => intermediate_rep.white.push(
                        <models::GameRelation as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "grey" => intermediate_rep.grey.push(
                        <models::GameRelation as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "black" => intermediate_rep.black.push(
                        <models::GameRelation as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing FinishGameIn".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(FinishGameIn {
            white: intermediate_rep
                .white
                .into_iter()
                .next()
                .ok_or_else(|| "white missing in FinishGameIn".to_string())?,
            grey: intermediate_rep
                .grey
                .into_iter()
                .next()
                .ok_or_else(|| "grey missing in FinishGameIn".to_string())?,
            black: intermediate_rep
                .black
                .into_iter()
                .next()
                .ok_or_else(|| "black missing in FinishGameIn".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<FinishGameIn> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<FinishGameIn>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<FinishGameIn>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for FinishGameIn - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<FinishGameIn> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <FinishGameIn as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into FinishGameIn - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Forbidden
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct ForbiddenError {
    #[serde(rename = "message")]
    pub message: String,
}

impl ForbiddenError {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(message: String) -> ForbiddenError {
        ForbiddenError { message }
    }
}

/// Converts the ForbiddenError value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for ForbiddenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("message".to_string()), Some(self.message.to_string())];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a ForbiddenError value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for ForbiddenError {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub message: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing ForbiddenError".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "message" => intermediate_rep.message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing ForbiddenError".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(ForbiddenError {
            message: intermediate_rep
                .message
                .into_iter()
                .next()
                .ok_or_else(|| "message missing in ForbiddenError".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<ForbiddenError> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<ForbiddenError>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<ForbiddenError>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for ForbiddenError - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<ForbiddenError> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <ForbiddenError as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into ForbiddenError - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GameHistory {
    #[serde(rename = "history")]
    pub history: Vec<models::HistoryItem>,

    #[serde(rename = "game")]
    pub game: models::GameState,

    #[serde(rename = "maxTurns")]
    pub max_turns: i32,
}

impl GameHistory {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        history: Vec<models::HistoryItem>,
        game: models::GameState,
        max_turns: i32,
    ) -> GameHistory {
        GameHistory {
            history,
            game,
            max_turns,
        }
    }
}

/// Converts the GameHistory value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for GameHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping history in query parameter serialization

            // Skipping game in query parameter serialization
            Some("maxTurns".to_string()),
            Some(self.max_turns.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a GameHistory value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for GameHistory {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub history: Vec<Vec<models::HistoryItem>>,
            pub game: Vec<models::GameState>,
            pub max_turns: Vec<i32>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing GameHistory".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    "history" => {
                        return std::result::Result::Err(
                            "Parsing a container in this style is not supported in GameHistory"
                                .to_string(),
                        )
                    }
                    #[allow(clippy::redundant_clone)]
                    "game" => intermediate_rep.game.push(
                        <models::GameState as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "maxTurns" => intermediate_rep.max_turns.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing GameHistory".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(GameHistory {
            history: intermediate_rep
                .history
                .into_iter()
                .next()
                .ok_or_else(|| "history missing in GameHistory".to_string())?,
            game: intermediate_rep
                .game
                .into_iter()
                .next()
                .ok_or_else(|| "game missing in GameHistory".to_string())?,
            max_turns: intermediate_rep
                .max_turns
                .into_iter()
                .next()
                .ok_or_else(|| "maxTurns missing in GameHistory".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<GameHistory> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<GameHistory>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<GameHistory>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for GameHistory - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<GameHistory> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <GameHistory as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into GameHistory - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum GameRelation {
    #[serde(rename = "winner")]
    Winner,
    #[serde(rename = "loser")]
    Loser,
    #[serde(rename = "draw")]
    Draw,
}

impl std::fmt::Display for GameRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            GameRelation::Winner => write!(f, "winner"),
            GameRelation::Loser => write!(f, "loser"),
            GameRelation::Draw => write!(f, "draw"),
        }
    }
}

impl std::str::FromStr for GameRelation {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "winner" => std::result::Result::Ok(GameRelation::Winner),
            "loser" => std::result::Result::Ok(GameRelation::Loser),
            "draw" => std::result::Result::Ok(GameRelation::Draw),
            _ => std::result::Result::Err(format!("Value not valid: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct GameState {
    #[serde(rename = "gameId")]
    pub game_id: i64,

    #[serde(rename = "name")]
    pub name: String,

    #[serde(rename = "fen")]
    pub fen: String,

    #[serde(rename = "ownerId")]
    pub owner_id: i64,

    #[serde(rename = "ownerUsername")]
    pub owner_username: String,

    #[serde(rename = "players")]
    pub players: Nullable<models::PlayerUpdate>,

    #[serde(rename = "status")]
    pub status: models::GameStatus,

    #[serde(rename = "players_masked")]
    pub players_masked: bool,

    #[serde(rename = "material_masked")]
    pub material_masked: bool,

    #[serde(rename = "training_mode")]
    pub training_mode: bool,

    #[serde(rename = "suggested_move_time_seconds")]
    pub suggested_move_time_seconds: Nullable<i32>,

    #[serde(rename = "tournamentId")]
    pub tournament_id: Nullable<i64>,
}

impl GameState {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        game_id: i64,
        name: String,
        fen: String,
        owner_id: i64,
        owner_username: String,
        players: Nullable<models::PlayerUpdate>,
        status: models::GameStatus,
        players_masked: bool,
        material_masked: bool,
        training_mode: bool,
        suggested_move_time_seconds: Nullable<i32>,
        tournament_id: Nullable<i64>,
    ) -> GameState {
        GameState {
            game_id,
            name,
            fen,
            owner_id,
            owner_username,
            players,
            status,
            players_masked,
            material_masked,
            training_mode,
            suggested_move_time_seconds,
            tournament_id,
        }
    }
}

/// Converts the GameState value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for GameState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("gameId".to_string()),
            Some(self.game_id.to_string()),
            Some("name".to_string()),
            Some(self.name.to_string()),
            Some("fen".to_string()),
            Some(self.fen.to_string()),
            Some("ownerId".to_string()),
            Some(self.owner_id.to_string()),
            Some("ownerUsername".to_string()),
            Some(self.owner_username.to_string()),
            // Skipping players in query parameter serialization

            // Skipping status in query parameter serialization
            Some("players_masked".to_string()),
            Some(self.players_masked.to_string()),
            Some("material_masked".to_string()),
            Some(self.material_masked.to_string()),
            Some("training_mode".to_string()),
            Some(self.training_mode.to_string()),
            Some("suggested_move_time_seconds".to_string()),
            Some(
                self.suggested_move_time_seconds
                    .as_ref()
                    .map_or("null".to_string(), |x| x.to_string()),
            ),
            Some("tournamentId".to_string()),
            Some(
                self.tournament_id
                    .as_ref()
                    .map_or("null".to_string(), |x| x.to_string()),
            ),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a GameState value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for GameState {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub game_id: Vec<i64>,
            pub name: Vec<String>,
            pub fen: Vec<String>,
            pub owner_id: Vec<i64>,
            pub owner_username: Vec<String>,
            pub players: Vec<models::PlayerUpdate>,
            pub status: Vec<models::GameStatus>,
            pub players_masked: Vec<bool>,
            pub material_masked: Vec<bool>,
            pub training_mode: Vec<bool>,
            pub suggested_move_time_seconds: Vec<i32>,
            pub tournament_id: Vec<i64>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing GameState".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "gameId" => intermediate_rep.game_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "name" => intermediate_rep.name.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "fen" => intermediate_rep.fen.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "ownerId" => intermediate_rep.owner_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "ownerUsername" => intermediate_rep.owner_username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "players" => {
                        return std::result::Result::Err(
                            "Parsing a nullable type in this style is not supported in GameState"
                                .to_string(),
                        )
                    }
                    #[allow(clippy::redundant_clone)]
                    "status" => intermediate_rep.status.push(
                        <models::GameStatus as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "players_masked" => intermediate_rep.players_masked.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "material_masked" => intermediate_rep.material_masked.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "training_mode" => intermediate_rep.training_mode.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "suggested_move_time_seconds" => {
                        return std::result::Result::Err(
                            "Parsing a nullable type in this style is not supported in GameState"
                                .to_string(),
                        )
                    }
                    "tournamentId" => {
                        return std::result::Result::Err(
                            "Parsing a nullable type in this style is not supported in GameState"
                                .to_string(),
                        )
                    }
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing GameState".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(GameState {
            game_id: intermediate_rep
                .game_id
                .into_iter()
                .next()
                .ok_or_else(|| "gameId missing in GameState".to_string())?,
            name: intermediate_rep
                .name
                .into_iter()
                .next()
                .ok_or_else(|| "name missing in GameState".to_string())?,
            fen: intermediate_rep
                .fen
                .into_iter()
                .next()
                .ok_or_else(|| "fen missing in GameState".to_string())?,
            owner_id: intermediate_rep
                .owner_id
                .into_iter()
                .next()
                .ok_or_else(|| "ownerId missing in GameState".to_string())?,
            owner_username: intermediate_rep
                .owner_username
                .into_iter()
                .next()
                .ok_or_else(|| "ownerUsername missing in GameState".to_string())?,
            players: std::result::Result::Err(
                "Nullable types not supported in GameState".to_string(),
            )?,
            status: intermediate_rep
                .status
                .into_iter()
                .next()
                .ok_or_else(|| "status missing in GameState".to_string())?,
            players_masked: intermediate_rep
                .players_masked
                .into_iter()
                .next()
                .ok_or_else(|| "players_masked missing in GameState".to_string())?,
            material_masked: intermediate_rep
                .material_masked
                .into_iter()
                .next()
                .ok_or_else(|| "material_masked missing in GameState".to_string())?,
            training_mode: intermediate_rep
                .training_mode
                .into_iter()
                .next()
                .ok_or_else(|| "training_mode missing in GameState".to_string())?,
            suggested_move_time_seconds: std::result::Result::Err(
                "Nullable types not supported in GameState".to_string(),
            )?,
            tournament_id: std::result::Result::Err(
                "Nullable types not supported in GameState".to_string(),
            )?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<GameState> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<GameState>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<GameState>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for GameState - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<GameState> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <GameState as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into GameState - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum GameStatus {
    #[serde(rename = "Waiting")]
    Waiting,
    #[serde(rename = "InProgress")]
    InProgress,
    #[serde(rename = "FinishedWin")]
    FinishedWin,
    #[serde(rename = "FinishedDraw")]
    FinishedDraw,
    #[serde(rename = "FinishedSemiDraw")]
    FinishedSemiDraw,
}

impl std::fmt::Display for GameStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            GameStatus::Waiting => write!(f, "Waiting"),
            GameStatus::InProgress => write!(f, "InProgress"),
            GameStatus::FinishedWin => write!(f, "FinishedWin"),
            GameStatus::FinishedDraw => write!(f, "FinishedDraw"),
            GameStatus::FinishedSemiDraw => write!(f, "FinishedSemiDraw"),
        }
    }
}

impl std::str::FromStr for GameStatus {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Waiting" => std::result::Result::Ok(GameStatus::Waiting),
            "InProgress" => std::result::Result::Ok(GameStatus::InProgress),
            "FinishedWin" => std::result::Result::Ok(GameStatus::FinishedWin),
            "FinishedDraw" => std::result::Result::Ok(GameStatus::FinishedDraw),
            "FinishedSemiDraw" => std::result::Result::Ok(GameStatus::FinishedSemiDraw),
            _ => std::result::Result::Err(format!("Value not valid: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct HistoryItem {
    #[serde(rename = "turn_counter")]
    pub turn_counter: i32,

    #[serde(rename = "fen")]
    pub fen: String,

    #[serde(rename = "move_uci")]
    pub move_uci: String,

    #[serde(rename = "color")]
    pub color: models::PlayerColor,
}

impl HistoryItem {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        turn_counter: i32,
        fen: String,
        move_uci: String,
        color: models::PlayerColor,
    ) -> HistoryItem {
        HistoryItem {
            turn_counter,
            fen,
            move_uci,
            color,
        }
    }
}

/// Converts the HistoryItem value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for HistoryItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("turn_counter".to_string()),
            Some(self.turn_counter.to_string()),
            Some("fen".to_string()),
            Some(self.fen.to_string()),
            Some("move_uci".to_string()),
            Some(self.move_uci.to_string()),
            // Skipping color in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a HistoryItem value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for HistoryItem {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub turn_counter: Vec<i32>,
            pub fen: Vec<String>,
            pub move_uci: Vec<String>,
            pub color: Vec<models::PlayerColor>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing HistoryItem".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "turn_counter" => intermediate_rep.turn_counter.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "fen" => intermediate_rep.fen.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "move_uci" => intermediate_rep.move_uci.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "color" => intermediate_rep.color.push(
                        <models::PlayerColor as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing HistoryItem".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(HistoryItem {
            turn_counter: intermediate_rep
                .turn_counter
                .into_iter()
                .next()
                .ok_or_else(|| "turn_counter missing in HistoryItem".to_string())?,
            fen: intermediate_rep
                .fen
                .into_iter()
                .next()
                .ok_or_else(|| "fen missing in HistoryItem".to_string())?,
            move_uci: intermediate_rep
                .move_uci
                .into_iter()
                .next()
                .ok_or_else(|| "move_uci missing in HistoryItem".to_string())?,
            color: intermediate_rep
                .color
                .into_iter()
                .next()
                .ok_or_else(|| "color missing in HistoryItem".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<HistoryItem> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<HistoryItem>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<HistoryItem>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for HistoryItem - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<HistoryItem> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <HistoryItem as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into HistoryItem - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// ISE
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct InternalError {
    #[serde(rename = "message")]
    pub message: String,
}

impl InternalError {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(message: String) -> InternalError {
        InternalError { message }
    }
}

/// Converts the InternalError value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for InternalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("message".to_string()), Some(self.message.to_string())];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a InternalError value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for InternalError {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub message: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing InternalError".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "message" => intermediate_rep.message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing InternalError".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(InternalError {
            message: intermediate_rep
                .message
                .into_iter()
                .next()
                .ok_or_else(|| "message missing in InternalError".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<InternalError> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<InternalError>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<InternalError>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for InternalError - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<InternalError> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <InternalError as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into InternalError - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct JoinGamePayload {
    #[serde(rename = "color")]
    pub color: models::PlayerColor,
}

impl JoinGamePayload {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(color: models::PlayerColor) -> JoinGamePayload {
        JoinGamePayload { color }
    }
}

/// Converts the JoinGamePayload value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for JoinGamePayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping color in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a JoinGamePayload value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for JoinGamePayload {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub color: Vec<models::PlayerColor>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing JoinGamePayload".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "color" => intermediate_rep.color.push(
                        <models::PlayerColor as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing JoinGamePayload".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(JoinGamePayload {
            color: intermediate_rep
                .color
                .into_iter()
                .next()
                .ok_or_else(|| "color missing in JoinGamePayload".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<JoinGamePayload> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<JoinGamePayload>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<JoinGamePayload>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for JoinGamePayload - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<JoinGamePayload> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <JoinGamePayload as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into JoinGamePayload - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct LeaderboardEntry {
    #[serde(rename = "userId")]
    pub user_id: i64,

    #[serde(rename = "isBot")]
    pub is_bot: bool,

    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "gamesPlayed")]
    pub games_played: i32,

    #[serde(rename = "wins")]
    pub wins: i32,

    #[serde(rename = "winRate")]
    pub win_rate: f32,

    #[serde(rename = "losses")]
    pub losses: i32,

    #[serde(rename = "lossRate")]
    pub loss_rate: f32,

    #[serde(rename = "draws")]
    pub draws: i32,
}

impl LeaderboardEntry {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        user_id: i64,
        is_bot: bool,
        username: String,
        games_played: i32,
        wins: i32,
        win_rate: f32,
        losses: i32,
        loss_rate: f32,
        draws: i32,
    ) -> LeaderboardEntry {
        LeaderboardEntry {
            user_id,
            is_bot,
            username,
            games_played,
            wins,
            win_rate,
            losses,
            loss_rate,
            draws,
        }
    }
}

/// Converts the LeaderboardEntry value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for LeaderboardEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("userId".to_string()),
            Some(self.user_id.to_string()),
            Some("isBot".to_string()),
            Some(self.is_bot.to_string()),
            Some("username".to_string()),
            Some(self.username.to_string()),
            Some("gamesPlayed".to_string()),
            Some(self.games_played.to_string()),
            Some("wins".to_string()),
            Some(self.wins.to_string()),
            Some("winRate".to_string()),
            Some(self.win_rate.to_string()),
            Some("losses".to_string()),
            Some(self.losses.to_string()),
            Some("lossRate".to_string()),
            Some(self.loss_rate.to_string()),
            Some("draws".to_string()),
            Some(self.draws.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a LeaderboardEntry value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for LeaderboardEntry {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub user_id: Vec<i64>,
            pub is_bot: Vec<bool>,
            pub username: Vec<String>,
            pub games_played: Vec<i32>,
            pub wins: Vec<i32>,
            pub win_rate: Vec<f32>,
            pub losses: Vec<i32>,
            pub loss_rate: Vec<f32>,
            pub draws: Vec<i32>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing LeaderboardEntry".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "userId" => intermediate_rep.user_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "isBot" => intermediate_rep.is_bot.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "gamesPlayed" => intermediate_rep.games_played.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "wins" => intermediate_rep.wins.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "winRate" => intermediate_rep.win_rate.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "losses" => intermediate_rep.losses.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "lossRate" => intermediate_rep.loss_rate.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "draws" => intermediate_rep.draws.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing LeaderboardEntry".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(LeaderboardEntry {
            user_id: intermediate_rep
                .user_id
                .into_iter()
                .next()
                .ok_or_else(|| "userId missing in LeaderboardEntry".to_string())?,
            is_bot: intermediate_rep
                .is_bot
                .into_iter()
                .next()
                .ok_or_else(|| "isBot missing in LeaderboardEntry".to_string())?,
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in LeaderboardEntry".to_string())?,
            games_played: intermediate_rep
                .games_played
                .into_iter()
                .next()
                .ok_or_else(|| "gamesPlayed missing in LeaderboardEntry".to_string())?,
            wins: intermediate_rep
                .wins
                .into_iter()
                .next()
                .ok_or_else(|| "wins missing in LeaderboardEntry".to_string())?,
            win_rate: intermediate_rep
                .win_rate
                .into_iter()
                .next()
                .ok_or_else(|| "winRate missing in LeaderboardEntry".to_string())?,
            losses: intermediate_rep
                .losses
                .into_iter()
                .next()
                .ok_or_else(|| "losses missing in LeaderboardEntry".to_string())?,
            loss_rate: intermediate_rep
                .loss_rate
                .into_iter()
                .next()
                .ok_or_else(|| "lossRate missing in LeaderboardEntry".to_string())?,
            draws: intermediate_rep
                .draws
                .into_iter()
                .next()
                .ok_or_else(|| "draws missing in LeaderboardEntry".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<LeaderboardEntry> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<LeaderboardEntry>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<LeaderboardEntry>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for LeaderboardEntry - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<LeaderboardEntry> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <LeaderboardEntry as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into LeaderboardEntry - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum LeaderboardSortBy {
    #[serde(rename = "Wins")]
    Wins,
    #[serde(rename = "GamesPlayed")]
    GamesPlayed,
    #[serde(rename = "WinRate")]
    WinRate,
    #[serde(rename = "LossRate")]
    LossRate,
    #[serde(rename = "Losses")]
    Losses,
    #[serde(rename = "Draws")]
    Draws,
}

impl std::fmt::Display for LeaderboardSortBy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            LeaderboardSortBy::Wins => write!(f, "Wins"),
            LeaderboardSortBy::GamesPlayed => write!(f, "GamesPlayed"),
            LeaderboardSortBy::WinRate => write!(f, "WinRate"),
            LeaderboardSortBy::LossRate => write!(f, "LossRate"),
            LeaderboardSortBy::Losses => write!(f, "Losses"),
            LeaderboardSortBy::Draws => write!(f, "Draws"),
        }
    }
}

impl std::str::FromStr for LeaderboardSortBy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Wins" => std::result::Result::Ok(LeaderboardSortBy::Wins),
            "GamesPlayed" => std::result::Result::Ok(LeaderboardSortBy::GamesPlayed),
            "WinRate" => std::result::Result::Ok(LeaderboardSortBy::WinRate),
            "LossRate" => std::result::Result::Ok(LeaderboardSortBy::LossRate),
            "Losses" => std::result::Result::Ok(LeaderboardSortBy::Losses),
            "Draws" => std::result::Result::Ok(LeaderboardSortBy::Draws),
            _ => std::result::Result::Err(format!("Value not valid: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct LoginPayload {
    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "password")]
    pub password: String,
}

impl LoginPayload {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(username: String, password: String) -> LoginPayload {
        LoginPayload { username, password }
    }
}

/// Converts the LoginPayload value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for LoginPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("username".to_string()),
            Some(self.username.to_string()),
            Some("password".to_string()),
            Some(self.password.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a LoginPayload value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for LoginPayload {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub username: Vec<String>,
            pub password: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing LoginPayload".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "password" => intermediate_rep.password.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing LoginPayload".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(LoginPayload {
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in LoginPayload".to_string())?,
            password: intermediate_rep
                .password
                .into_iter()
                .next()
                .ok_or_else(|| "password missing in LoginPayload".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<LoginPayload> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<LoginPayload>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<LoginPayload>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for LoginPayload - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<LoginPayload> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <LoginPayload as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into LoginPayload - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct LoginResponse {
    /// The JWT token to be used for authenticating subsequent requests.
    #[serde(rename = "token")]
    pub token: String,

    #[serde(rename = "user")]
    pub user: models::User,
}

impl LoginResponse {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(token: String, user: models::User) -> LoginResponse {
        LoginResponse { token, user }
    }
}

/// Converts the LoginResponse value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for LoginResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("token".to_string()),
            Some(self.token.to_string()),
            // Skipping user in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a LoginResponse value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for LoginResponse {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub token: Vec<String>,
            pub user: Vec<models::User>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing LoginResponse".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "token" => intermediate_rep.token.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "user" => intermediate_rep.user.push(
                        <models::User as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing LoginResponse".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(LoginResponse {
            token: intermediate_rep
                .token
                .into_iter()
                .next()
                .ok_or_else(|| "token missing in LoginResponse".to_string())?,
            user: intermediate_rep
                .user
                .into_iter()
                .next()
                .ok_or_else(|| "user missing in LoginResponse".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<LoginResponse> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<LoginResponse>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<LoginResponse>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for LoginResponse - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<LoginResponse> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <LoginResponse as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into LoginResponse - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct MakeMovePayload {
    #[serde(rename = "fromIndex")]
    #[validate(range(min = 0u8, max = 96u8))]
    pub from_index: u8,

    #[serde(rename = "toIndex")]
    #[validate(range(min = 0u8, max = 96u8))]
    pub to_index: u8,

    #[serde(rename = "promotion")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub promotion: Option<models::PromotionPiece>,
}

impl MakeMovePayload {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(from_index: u8, to_index: u8) -> MakeMovePayload {
        MakeMovePayload {
            from_index,
            to_index,
            promotion: None,
        }
    }
}

/// Converts the MakeMovePayload value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for MakeMovePayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("fromIndex".to_string()),
            Some(self.from_index.to_string()),
            Some("toIndex".to_string()),
            Some(self.to_index.to_string()),
            // Skipping promotion in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a MakeMovePayload value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for MakeMovePayload {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub from_index: Vec<u8>,
            pub to_index: Vec<u8>,
            pub promotion: Vec<models::PromotionPiece>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing MakeMovePayload".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "fromIndex" => intermediate_rep
                        .from_index
                        .push(<u8 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "toIndex" => intermediate_rep
                        .to_index
                        .push(<u8 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    #[allow(clippy::redundant_clone)]
                    "promotion" => intermediate_rep.promotion.push(
                        <models::PromotionPiece as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing MakeMovePayload".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(MakeMovePayload {
            from_index: intermediate_rep
                .from_index
                .into_iter()
                .next()
                .ok_or_else(|| "fromIndex missing in MakeMovePayload".to_string())?,
            to_index: intermediate_rep
                .to_index
                .into_iter()
                .next()
                .ok_or_else(|| "toIndex missing in MakeMovePayload".to_string())?,
            promotion: intermediate_rep.promotion.into_iter().next(),
        })
    }
}

// Methods for converting between header::IntoHeaderValue<MakeMovePayload> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<MakeMovePayload>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<MakeMovePayload>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for MakeMovePayload - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<MakeMovePayload> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <MakeMovePayload as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into MakeMovePayload - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct MeUser {
    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "type")]
    pub r#type: models::UserType,

    #[serde(rename = "lichess_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lichess_rating: Option<u16>,

    #[serde(rename = "chess_com_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chess_com_rating: Option<u16>,

    #[serde(rename = "fide_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fide_rating: Option<u16>,

    #[serde(rename = "experience_with_chess")]
    #[validate(range(min = 0u8, max = 10u8))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experience_with_chess: Option<u8>,
}

impl MeUser {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(id: i64, username: String, r#type: models::UserType) -> MeUser {
        MeUser {
            id,
            username,
            r#type,
            lichess_rating: None,
            chess_com_rating: None,
            fide_rating: None,
            experience_with_chess: None,
        }
    }
}

/// Converts the MeUser value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for MeUser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("id".to_string()),
            Some(self.id.to_string()),
            Some("username".to_string()),
            Some(self.username.to_string()),
            // Skipping type in query parameter serialization
            self.lichess_rating.as_ref().map(|lichess_rating| {
                ["lichess_rating".to_string(), lichess_rating.to_string()].join(",")
            }),
            self.chess_com_rating.as_ref().map(|chess_com_rating| {
                ["chess_com_rating".to_string(), chess_com_rating.to_string()].join(",")
            }),
            self.fide_rating
                .as_ref()
                .map(|fide_rating| ["fide_rating".to_string(), fide_rating.to_string()].join(",")),
            self.experience_with_chess
                .as_ref()
                .map(|experience_with_chess| {
                    [
                        "experience_with_chess".to_string(),
                        experience_with_chess.to_string(),
                    ]
                    .join(",")
                }),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a MeUser value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for MeUser {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub id: Vec<i64>,
            pub username: Vec<String>,
            pub r#type: Vec<models::UserType>,
            pub lichess_rating: Vec<u16>,
            pub chess_com_rating: Vec<u16>,
            pub fide_rating: Vec<u16>,
            pub experience_with_chess: Vec<u8>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing MeUser".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "id" => intermediate_rep.id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "type" => intermediate_rep.r#type.push(
                        <models::UserType as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "lichess_rating" => intermediate_rep.lichess_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "chess_com_rating" => intermediate_rep.chess_com_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "fide_rating" => intermediate_rep.fide_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "experience_with_chess" => intermediate_rep
                        .experience_with_chess
                        .push(<u8 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing MeUser".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(MeUser {
            id: intermediate_rep
                .id
                .into_iter()
                .next()
                .ok_or_else(|| "id missing in MeUser".to_string())?,
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in MeUser".to_string())?,
            r#type: intermediate_rep
                .r#type
                .into_iter()
                .next()
                .ok_or_else(|| "type missing in MeUser".to_string())?,
            lichess_rating: intermediate_rep.lichess_rating.into_iter().next(),
            chess_com_rating: intermediate_rep.chess_com_rating.into_iter().next(),
            fide_rating: intermediate_rep.fide_rating.into_iter().next(),
            experience_with_chess: intermediate_rep.experience_with_chess.into_iter().next(),
        })
    }
}

// Methods for converting between header::IntoHeaderValue<MeUser> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<MeUser>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<MeUser>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for MeUser - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<MeUser> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <MeUser as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into MeUser - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Move {
    #[serde(rename = "from")]
    pub from: models::Position,

    #[serde(rename = "to")]
    pub to: models::Position,

    #[serde(rename = "move_uci")]
    pub move_uci: String,
}

impl Move {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(from: models::Position, to: models::Position, move_uci: String) -> Move {
        Move { from, to, move_uci }
    }
}

/// Converts the Move value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping from in query parameter serialization

            // Skipping to in query parameter serialization
            Some("move_uci".to_string()),
            Some(self.move_uci.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Move value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Move {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub from: Vec<models::Position>,
            pub to: Vec<models::Position>,
            pub move_uci: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err("Missing value while parsing Move".to_string())
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "from" => intermediate_rep.from.push(
                        <models::Position as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "to" => intermediate_rep.to.push(
                        <models::Position as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "move_uci" => intermediate_rep.move_uci.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Move".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(Move {
            from: intermediate_rep
                .from
                .into_iter()
                .next()
                .ok_or_else(|| "from missing in Move".to_string())?,
            to: intermediate_rep
                .to
                .into_iter()
                .next()
                .ok_or_else(|| "to missing in Move".to_string())?,
            move_uci: intermediate_rep
                .move_uci
                .into_iter()
                .next()
                .ok_or_else(|| "move_uci missing in Move".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<Move> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<Move>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<Move>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for Move - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<Move> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => match <Move as std::str::FromStr>::from_str(value) {
                std::result::Result::Ok(value) => {
                    std::result::Result::Ok(header::IntoHeaderValue(value))
                }
                std::result::Result::Err(err) => std::result::Result::Err(format!(
                    "Unable to convert header value '{}' into Move - {}",
                    value, err
                )),
            },
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Not Found
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct NotFoundError {
    #[serde(rename = "message")]
    pub message: String,
}

impl NotFoundError {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(message: String) -> NotFoundError {
        NotFoundError { message }
    }
}

/// Converts the NotFoundError value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for NotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("message".to_string()), Some(self.message.to_string())];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a NotFoundError value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for NotFoundError {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub message: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing NotFoundError".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "message" => intermediate_rep.message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing NotFoundError".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(NotFoundError {
            message: intermediate_rep
                .message
                .into_iter()
                .next()
                .ok_or_else(|| "message missing in NotFoundError".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<NotFoundError> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<NotFoundError>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<NotFoundError>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for NotFoundError - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<NotFoundError> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <NotFoundError as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into NotFoundError - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Player {
    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "relation")]
    pub relation: Nullable<models::GameRelation>,

    #[serde(rename = "isOwner")]
    pub is_owner: bool,

    #[serde(rename = "isConnectedToGame")]
    pub is_connected_to_game: bool,
}

impl Player {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        id: i64,
        username: String,
        relation: Nullable<models::GameRelation>,
        is_owner: bool,
        is_connected_to_game: bool,
    ) -> Player {
        Player {
            id,
            username,
            relation,
            is_owner,
            is_connected_to_game,
        }
    }
}

/// Converts the Player value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Player {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("id".to_string()),
            Some(self.id.to_string()),
            Some("username".to_string()),
            Some(self.username.to_string()),
            // Skipping relation in query parameter serialization
            Some("isOwner".to_string()),
            Some(self.is_owner.to_string()),
            Some("isConnectedToGame".to_string()),
            Some(self.is_connected_to_game.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Player value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Player {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub id: Vec<i64>,
            pub username: Vec<String>,
            pub relation: Vec<models::GameRelation>,
            pub is_owner: Vec<bool>,
            pub is_connected_to_game: Vec<bool>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing Player".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "id" => intermediate_rep.id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    "relation" => {
                        return std::result::Result::Err(
                            "Parsing a nullable type in this style is not supported in Player"
                                .to_string(),
                        )
                    }
                    #[allow(clippy::redundant_clone)]
                    "isOwner" => intermediate_rep.is_owner.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "isConnectedToGame" => intermediate_rep.is_connected_to_game.push(
                        <bool as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Player".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(Player {
            id: intermediate_rep
                .id
                .into_iter()
                .next()
                .ok_or_else(|| "id missing in Player".to_string())?,
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in Player".to_string())?,
            relation: std::result::Result::Err(
                "Nullable types not supported in Player".to_string(),
            )?,
            is_owner: intermediate_rep
                .is_owner
                .into_iter()
                .next()
                .ok_or_else(|| "isOwner missing in Player".to_string())?,
            is_connected_to_game: intermediate_rep
                .is_connected_to_game
                .into_iter()
                .next()
                .ok_or_else(|| "isConnectedToGame missing in Player".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<Player> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<Player>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<Player>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for Player - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<Player> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <Player as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into Player - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum PlayerColor {
    #[serde(rename = "White")]
    White,
    #[serde(rename = "Grey")]
    Grey,
    #[serde(rename = "Black")]
    Black,
}

impl std::fmt::Display for PlayerColor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            PlayerColor::White => write!(f, "White"),
            PlayerColor::Grey => write!(f, "Grey"),
            PlayerColor::Black => write!(f, "Black"),
        }
    }
}

impl std::str::FromStr for PlayerColor {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "White" => std::result::Result::Ok(PlayerColor::White),
            "Grey" => std::result::Result::Ok(PlayerColor::Grey),
            "Black" => std::result::Result::Ok(PlayerColor::Black),
            _ => std::result::Result::Err(format!("Value not valid: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct PlayerUpdate {
    #[serde(rename = "white")]
    pub white: Nullable<models::Player>,

    #[serde(rename = "grey")]
    pub grey: Nullable<models::Player>,

    #[serde(rename = "black")]
    pub black: Nullable<models::Player>,
}

impl PlayerUpdate {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        white: Nullable<models::Player>,
        grey: Nullable<models::Player>,
        black: Nullable<models::Player>,
    ) -> PlayerUpdate {
        PlayerUpdate { white, grey, black }
    }
}

/// Converts the PlayerUpdate value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for PlayerUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping white in query parameter serialization

            // Skipping grey in query parameter serialization

            // Skipping black in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a PlayerUpdate value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for PlayerUpdate {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub white: Vec<models::Player>,
            pub grey: Vec<models::Player>,
            pub black: Vec<models::Player>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing PlayerUpdate".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    "white" => return std::result::Result::Err(
                        "Parsing a nullable type in this style is not supported in PlayerUpdate"
                            .to_string(),
                    ),
                    "grey" => return std::result::Result::Err(
                        "Parsing a nullable type in this style is not supported in PlayerUpdate"
                            .to_string(),
                    ),
                    "black" => return std::result::Result::Err(
                        "Parsing a nullable type in this style is not supported in PlayerUpdate"
                            .to_string(),
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing PlayerUpdate".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(PlayerUpdate {
            white: std::result::Result::Err(
                "Nullable types not supported in PlayerUpdate".to_string(),
            )?,
            grey: std::result::Result::Err(
                "Nullable types not supported in PlayerUpdate".to_string(),
            )?,
            black: std::result::Result::Err(
                "Nullable types not supported in PlayerUpdate".to_string(),
            )?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<PlayerUpdate> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<PlayerUpdate>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<PlayerUpdate>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for PlayerUpdate - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<PlayerUpdate> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <PlayerUpdate as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into PlayerUpdate - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Position {
    #[serde(rename = "q")]
    pub q: i32,

    #[serde(rename = "r")]
    pub r: i32,

    #[serde(rename = "s")]
    pub s: i32,

    #[serde(rename = "i")]
    pub i: i32,
}

impl Position {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(q: i32, r: i32, s: i32, i: i32) -> Position {
        Position { q, r, s, i }
    }
}

/// Converts the Position value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("q".to_string()),
            Some(self.q.to_string()),
            Some("r".to_string()),
            Some(self.r.to_string()),
            Some("s".to_string()),
            Some(self.s.to_string()),
            Some("i".to_string()),
            Some(self.i.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Position value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Position {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub q: Vec<i32>,
            pub r: Vec<i32>,
            pub s: Vec<i32>,
            pub i: Vec<i32>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing Position".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "q" => intermediate_rep.q.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "r" => intermediate_rep.r.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "s" => intermediate_rep.s.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "i" => intermediate_rep.i.push(
                        <i32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Position".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(Position {
            q: intermediate_rep
                .q
                .into_iter()
                .next()
                .ok_or_else(|| "q missing in Position".to_string())?,
            r: intermediate_rep
                .r
                .into_iter()
                .next()
                .ok_or_else(|| "r missing in Position".to_string())?,
            s: intermediate_rep
                .s
                .into_iter()
                .next()
                .ok_or_else(|| "s missing in Position".to_string())?,
            i: intermediate_rep
                .i
                .into_iter()
                .next()
                .ok_or_else(|| "i missing in Position".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<Position> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<Position>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<Position>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for Position - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<Position> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <Position as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into Position - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum PromotionPiece {
    #[serde(rename = "Queen")]
    Queen,
    #[serde(rename = "Rook")]
    Rook,
    #[serde(rename = "Bishop")]
    Bishop,
    #[serde(rename = "Knight")]
    Knight,
}

impl std::fmt::Display for PromotionPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            PromotionPiece::Queen => write!(f, "Queen"),
            PromotionPiece::Rook => write!(f, "Rook"),
            PromotionPiece::Bishop => write!(f, "Bishop"),
            PromotionPiece::Knight => write!(f, "Knight"),
        }
    }
}

impl std::str::FromStr for PromotionPiece {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Queen" => std::result::Result::Ok(PromotionPiece::Queen),
            "Rook" => std::result::Result::Ok(PromotionPiece::Rook),
            "Bishop" => std::result::Result::Ok(PromotionPiece::Bishop),
            "Knight" => std::result::Result::Ok(PromotionPiece::Knight),
            _ => std::result::Result::Err(format!("Value not valid: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct ReadBot {
    #[serde(rename = "user_id")]
    pub user_id: i64,

    #[serde(rename = "bot_id")]
    pub bot_id: i64,

    #[serde(rename = "username")]
    #[validate(length(min = 3))]
    pub username: String,

    #[serde(rename = "model_key")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,

    #[serde(rename = "playouts_per_move")]
    #[validate(range(min = 1u32))]
    pub playouts_per_move: u32,

    #[serde(rename = "exploration_factor")]
    #[validate(range(min = 0f32, max = 20f32))]
    pub exploration_factor: f32,

    #[serde(rename = "virtual_loss")]
    #[validate(range(min = 0f32, max = 5f32))]
    pub virtual_loss: f32,

    #[serde(rename = "contempt")]
    #[validate(
            range(min = -3f32, max = 3f32),
        )]
    pub contempt: f32,
}

impl ReadBot {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        user_id: i64,
        bot_id: i64,
        username: String,
        playouts_per_move: u32,
        exploration_factor: f32,
        virtual_loss: f32,
        contempt: f32,
    ) -> ReadBot {
        ReadBot {
            user_id,
            bot_id,
            username,
            model_key: None,
            playouts_per_move,
            exploration_factor,
            virtual_loss,
            contempt,
        }
    }
}

/// Converts the ReadBot value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for ReadBot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("user_id".to_string()),
            Some(self.user_id.to_string()),
            Some("bot_id".to_string()),
            Some(self.bot_id.to_string()),
            Some("username".to_string()),
            Some(self.username.to_string()),
            self.model_key
                .as_ref()
                .map(|model_key| ["model_key".to_string(), model_key.to_string()].join(",")),
            Some("playouts_per_move".to_string()),
            Some(self.playouts_per_move.to_string()),
            Some("exploration_factor".to_string()),
            Some(self.exploration_factor.to_string()),
            Some("virtual_loss".to_string()),
            Some(self.virtual_loss.to_string()),
            Some("contempt".to_string()),
            Some(self.contempt.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a ReadBot value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for ReadBot {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub user_id: Vec<i64>,
            pub bot_id: Vec<i64>,
            pub username: Vec<String>,
            pub model_key: Vec<String>,
            pub playouts_per_move: Vec<u32>,
            pub exploration_factor: Vec<f32>,
            pub virtual_loss: Vec<f32>,
            pub contempt: Vec<f32>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing ReadBot".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "user_id" => intermediate_rep.user_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "bot_id" => intermediate_rep.bot_id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "model_key" => intermediate_rep.model_key.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "playouts_per_move" => intermediate_rep.playouts_per_move.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "exploration_factor" => intermediate_rep.exploration_factor.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "virtual_loss" => intermediate_rep.virtual_loss.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "contempt" => intermediate_rep.contempt.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing ReadBot".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(ReadBot {
            user_id: intermediate_rep
                .user_id
                .into_iter()
                .next()
                .ok_or_else(|| "user_id missing in ReadBot".to_string())?,
            bot_id: intermediate_rep
                .bot_id
                .into_iter()
                .next()
                .ok_or_else(|| "bot_id missing in ReadBot".to_string())?,
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in ReadBot".to_string())?,
            model_key: intermediate_rep.model_key.into_iter().next(),
            playouts_per_move: intermediate_rep
                .playouts_per_move
                .into_iter()
                .next()
                .ok_or_else(|| "playouts_per_move missing in ReadBot".to_string())?,
            exploration_factor: intermediate_rep
                .exploration_factor
                .into_iter()
                .next()
                .ok_or_else(|| "exploration_factor missing in ReadBot".to_string())?,
            virtual_loss: intermediate_rep
                .virtual_loss
                .into_iter()
                .next()
                .ok_or_else(|| "virtual_loss missing in ReadBot".to_string())?,
            contempt: intermediate_rep
                .contempt
                .into_iter()
                .next()
                .ok_or_else(|| "contempt missing in ReadBot".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<ReadBot> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<ReadBot>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<ReadBot>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for ReadBot - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<ReadBot> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <ReadBot as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into ReadBot - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct ReadTournament {
    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "name")]
    pub name: String,
}

impl ReadTournament {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(id: i64, name: String) -> ReadTournament {
        ReadTournament { id, name }
    }
}

/// Converts the ReadTournament value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for ReadTournament {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("id".to_string()),
            Some(self.id.to_string()),
            Some("name".to_string()),
            Some(self.name.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a ReadTournament value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for ReadTournament {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub id: Vec<i64>,
            pub name: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing ReadTournament".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "id" => intermediate_rep.id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "name" => intermediate_rep.name.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing ReadTournament".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(ReadTournament {
            id: intermediate_rep
                .id
                .into_iter()
                .next()
                .ok_or_else(|| "id missing in ReadTournament".to_string())?,
            name: intermediate_rep
                .name
                .into_iter()
                .next()
                .ok_or_else(|| "name missing in ReadTournament".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<ReadTournament> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<ReadTournament>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<ReadTournament>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for ReadTournament - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<ReadTournament> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <ReadTournament as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into ReadTournament - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct RegisterPayload {
    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "password")]
    pub password: String,

    #[serde(rename = "lichess_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lichess_rating: Option<u16>,

    #[serde(rename = "chess_com_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chess_com_rating: Option<u16>,

    #[serde(rename = "fide_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fide_rating: Option<u16>,

    #[serde(rename = "experience_with_chess")]
    #[validate(range(min = 0u8, max = 10u8))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experience_with_chess: Option<u8>,
}

impl RegisterPayload {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(username: String, password: String) -> RegisterPayload {
        RegisterPayload {
            username,
            password,
            lichess_rating: None,
            chess_com_rating: None,
            fide_rating: None,
            experience_with_chess: None,
        }
    }
}

/// Converts the RegisterPayload value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for RegisterPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("username".to_string()),
            Some(self.username.to_string()),
            Some("password".to_string()),
            Some(self.password.to_string()),
            self.lichess_rating.as_ref().map(|lichess_rating| {
                ["lichess_rating".to_string(), lichess_rating.to_string()].join(",")
            }),
            self.chess_com_rating.as_ref().map(|chess_com_rating| {
                ["chess_com_rating".to_string(), chess_com_rating.to_string()].join(",")
            }),
            self.fide_rating
                .as_ref()
                .map(|fide_rating| ["fide_rating".to_string(), fide_rating.to_string()].join(",")),
            self.experience_with_chess
                .as_ref()
                .map(|experience_with_chess| {
                    [
                        "experience_with_chess".to_string(),
                        experience_with_chess.to_string(),
                    ]
                    .join(",")
                }),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a RegisterPayload value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for RegisterPayload {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub username: Vec<String>,
            pub password: Vec<String>,
            pub lichess_rating: Vec<u16>,
            pub chess_com_rating: Vec<u16>,
            pub fide_rating: Vec<u16>,
            pub experience_with_chess: Vec<u8>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing RegisterPayload".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "password" => intermediate_rep.password.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "lichess_rating" => intermediate_rep.lichess_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "chess_com_rating" => intermediate_rep.chess_com_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "fide_rating" => intermediate_rep.fide_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "experience_with_chess" => intermediate_rep
                        .experience_with_chess
                        .push(<u8 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing RegisterPayload".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(RegisterPayload {
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in RegisterPayload".to_string())?,
            password: intermediate_rep
                .password
                .into_iter()
                .next()
                .ok_or_else(|| "password missing in RegisterPayload".to_string())?,
            lichess_rating: intermediate_rep.lichess_rating.into_iter().next(),
            chess_com_rating: intermediate_rep.chess_com_rating.into_iter().next(),
            fide_rating: intermediate_rep.fide_rating.into_iter().next(),
            experience_with_chess: intermediate_rep.experience_with_chess.into_iter().next(),
        })
    }
}

// Methods for converting between header::IntoHeaderValue<RegisterPayload> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<RegisterPayload>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<RegisterPayload>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for RegisterPayload - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<RegisterPayload> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <RegisterPayload as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into RegisterPayload - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct Tournament {
    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "name")]
    pub name: String,
}

impl Tournament {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(id: i64, name: String) -> Tournament {
        Tournament { id, name }
    }
}

/// Converts the Tournament value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for Tournament {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("id".to_string()),
            Some(self.id.to_string()),
            Some("name".to_string()),
            Some(self.name.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a Tournament value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for Tournament {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub id: Vec<i64>,
            pub name: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing Tournament".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "id" => intermediate_rep.id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "name" => intermediate_rep.name.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing Tournament".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(Tournament {
            id: intermediate_rep
                .id
                .into_iter()
                .next()
                .ok_or_else(|| "id missing in Tournament".to_string())?,
            name: intermediate_rep
                .name
                .into_iter()
                .next()
                .ok_or_else(|| "name missing in Tournament".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<Tournament> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<Tournament>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<Tournament>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for Tournament - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<Tournament> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <Tournament as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into Tournament - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Unauthorized access
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct UnauthorizedError {
    #[serde(rename = "message")]
    pub message: String,
}

impl UnauthorizedError {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(message: String) -> UnauthorizedError {
        UnauthorizedError { message }
    }
}

/// Converts the UnauthorizedError value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for UnauthorizedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> =
            vec![Some("message".to_string()), Some(self.message.to_string())];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a UnauthorizedError value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for UnauthorizedError {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub message: Vec<String>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing UnauthorizedError".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "message" => intermediate_rep.message.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing UnauthorizedError".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(UnauthorizedError {
            message: intermediate_rep
                .message
                .into_iter()
                .next()
                .ok_or_else(|| "message missing in UnauthorizedError".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<UnauthorizedError> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<UnauthorizedError>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<UnauthorizedError>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for UnauthorizedError - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<UnauthorizedError> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <UnauthorizedError as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into UnauthorizedError - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct UpdateBot {
    #[serde(rename = "username")]
    #[validate(length(min = 3))]
    pub username: String,

    #[serde(rename = "model_key")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_key: Option<String>,

    #[serde(rename = "playouts_per_move")]
    #[validate(range(min = 1u32))]
    pub playouts_per_move: u32,

    #[serde(rename = "exploration_factor")]
    #[validate(range(min = 0f32, max = 20f32))]
    pub exploration_factor: f32,

    #[serde(rename = "virtual_loss")]
    #[validate(range(min = 0f32, max = 5f32))]
    pub virtual_loss: f32,

    #[serde(rename = "contempt")]
    #[validate(
            range(min = -3f32, max = 3f32),
        )]
    pub contempt: f32,
}

impl UpdateBot {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        username: String,
        playouts_per_move: u32,
        exploration_factor: f32,
        virtual_loss: f32,
        contempt: f32,
    ) -> UpdateBot {
        UpdateBot {
            username,
            model_key: None,
            playouts_per_move,
            exploration_factor,
            virtual_loss,
            contempt,
        }
    }
}

/// Converts the UpdateBot value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for UpdateBot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("username".to_string()),
            Some(self.username.to_string()),
            self.model_key
                .as_ref()
                .map(|model_key| ["model_key".to_string(), model_key.to_string()].join(",")),
            Some("playouts_per_move".to_string()),
            Some(self.playouts_per_move.to_string()),
            Some("exploration_factor".to_string()),
            Some(self.exploration_factor.to_string()),
            Some("virtual_loss".to_string()),
            Some(self.virtual_loss.to_string()),
            Some("contempt".to_string()),
            Some(self.contempt.to_string()),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a UpdateBot value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for UpdateBot {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub username: Vec<String>,
            pub model_key: Vec<String>,
            pub playouts_per_move: Vec<u32>,
            pub exploration_factor: Vec<f32>,
            pub virtual_loss: Vec<f32>,
            pub contempt: Vec<f32>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing UpdateBot".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "model_key" => intermediate_rep.model_key.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "playouts_per_move" => intermediate_rep.playouts_per_move.push(
                        <u32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "exploration_factor" => intermediate_rep.exploration_factor.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "virtual_loss" => intermediate_rep.virtual_loss.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "contempt" => intermediate_rep.contempt.push(
                        <f32 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing UpdateBot".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(UpdateBot {
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in UpdateBot".to_string())?,
            model_key: intermediate_rep.model_key.into_iter().next(),
            playouts_per_move: intermediate_rep
                .playouts_per_move
                .into_iter()
                .next()
                .ok_or_else(|| "playouts_per_move missing in UpdateBot".to_string())?,
            exploration_factor: intermediate_rep
                .exploration_factor
                .into_iter()
                .next()
                .ok_or_else(|| "exploration_factor missing in UpdateBot".to_string())?,
            virtual_loss: intermediate_rep
                .virtual_loss
                .into_iter()
                .next()
                .ok_or_else(|| "virtual_loss missing in UpdateBot".to_string())?,
            contempt: intermediate_rep
                .contempt
                .into_iter()
                .next()
                .ok_or_else(|| "contempt missing in UpdateBot".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<UpdateBot> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<UpdateBot>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<UpdateBot>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for UpdateBot - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<UpdateBot> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <UpdateBot as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into UpdateBot - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct UpdateProfilePayload {
    #[serde(rename = "lichess_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lichess_rating: Option<u16>,

    #[serde(rename = "chess_com_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chess_com_rating: Option<u16>,

    #[serde(rename = "fide_rating")]
    #[validate(range(min = 100u16, max = 3000u16))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fide_rating: Option<u16>,

    #[serde(rename = "experience_with_chess")]
    #[validate(range(min = 0u8, max = 10u8))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experience_with_chess: Option<u8>,
}

impl UpdateProfilePayload {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new() -> UpdateProfilePayload {
        UpdateProfilePayload {
            lichess_rating: None,
            chess_com_rating: None,
            fide_rating: None,
            experience_with_chess: None,
        }
    }
}

/// Converts the UpdateProfilePayload value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for UpdateProfilePayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            self.lichess_rating.as_ref().map(|lichess_rating| {
                ["lichess_rating".to_string(), lichess_rating.to_string()].join(",")
            }),
            self.chess_com_rating.as_ref().map(|chess_com_rating| {
                ["chess_com_rating".to_string(), chess_com_rating.to_string()].join(",")
            }),
            self.fide_rating
                .as_ref()
                .map(|fide_rating| ["fide_rating".to_string(), fide_rating.to_string()].join(",")),
            self.experience_with_chess
                .as_ref()
                .map(|experience_with_chess| {
                    [
                        "experience_with_chess".to_string(),
                        experience_with_chess.to_string(),
                    ]
                    .join(",")
                }),
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a UpdateProfilePayload value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for UpdateProfilePayload {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub lichess_rating: Vec<u16>,
            pub chess_com_rating: Vec<u16>,
            pub fide_rating: Vec<u16>,
            pub experience_with_chess: Vec<u8>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing UpdateProfilePayload".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "lichess_rating" => intermediate_rep.lichess_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "chess_com_rating" => intermediate_rep.chess_com_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "fide_rating" => intermediate_rep.fide_rating.push(
                        <u16 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "experience_with_chess" => intermediate_rep
                        .experience_with_chess
                        .push(<u8 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing UpdateProfilePayload".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(UpdateProfilePayload {
            lichess_rating: intermediate_rep.lichess_rating.into_iter().next(),
            chess_com_rating: intermediate_rep.chess_com_rating.into_iter().next(),
            fide_rating: intermediate_rep.fide_rating.into_iter().next(),
            experience_with_chess: intermediate_rep.experience_with_chess.into_iter().next(),
        })
    }
}

// Methods for converting between header::IntoHeaderValue<UpdateProfilePayload> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<UpdateProfilePayload>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<UpdateProfilePayload>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for UpdateProfilePayload - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<UpdateProfilePayload> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <UpdateProfilePayload as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into UpdateProfilePayload - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct User {
    #[serde(rename = "id")]
    pub id: i64,

    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "type")]
    pub r#type: models::UserType,
}

impl User {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(id: i64, username: String, r#type: models::UserType) -> User {
        User {
            id,
            username,
            r#type,
        }
    }
}

/// Converts the User value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for User {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("id".to_string()),
            Some(self.id.to_string()),
            Some("username".to_string()),
            Some(self.username.to_string()),
            // Skipping type in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a User value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for User {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub id: Vec<i64>,
            pub username: Vec<String>,
            pub r#type: Vec<models::UserType>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err("Missing value while parsing User".to_string())
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "id" => intermediate_rep.id.push(
                        <i64 as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "type" => intermediate_rep.r#type.push(
                        <models::UserType as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing User".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(User {
            id: intermediate_rep
                .id
                .into_iter()
                .next()
                .ok_or_else(|| "id missing in User".to_string())?,
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in User".to_string())?,
            r#type: intermediate_rep
                .r#type
                .into_iter()
                .next()
                .ok_or_else(|| "type missing in User".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<User> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<User>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<User>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for User - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<User> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => match <User as std::str::FromStr>::from_str(value) {
                std::result::Result::Ok(value) => {
                    std::result::Result::Ok(header::IntoHeaderValue(value))
                }
                std::result::Result::Err(err) => std::result::Result::Err(format!(
                    "Unable to convert header value '{}' into User - {}",
                    value, err
                )),
            },
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct UserCreated {
    /// The JWT token to be used for authenticating subsequent requests.
    #[serde(rename = "token")]
    pub token: String,

    #[serde(rename = "user")]
    pub user: models::User,
}

impl UserCreated {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(token: String, user: models::User) -> UserCreated {
        UserCreated { token, user }
    }
}

/// Converts the UserCreated value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for UserCreated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("token".to_string()),
            Some(self.token.to_string()),
            // Skipping user in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a UserCreated value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for UserCreated {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub token: Vec<String>,
            pub user: Vec<models::User>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing UserCreated".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "token" => intermediate_rep.token.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "user" => intermediate_rep.user.push(
                        <models::User as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing UserCreated".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(UserCreated {
            token: intermediate_rep
                .token
                .into_iter()
                .next()
                .ok_or_else(|| "token missing in UserCreated".to_string())?,
            user: intermediate_rep
                .user
                .into_iter()
                .next()
                .ok_or_else(|| "user missing in UserCreated".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<UserCreated> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<UserCreated>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<UserCreated>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for UserCreated - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<UserCreated> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <UserCreated as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into UserCreated - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

/// Enumeration of values.
/// Since this enum's variants do not hold data, we can easily define them as `#[repr(C)]`
/// which helps with FFI.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[cfg_attr(feature = "conversion", derive(frunk_enum_derive::LabelledGenericEnum))]
pub enum UserType {
    #[serde(rename = "Regular")]
    Regular,
    #[serde(rename = "Bot")]
    Bot,
    #[serde(rename = "Admin")]
    Admin,
}

impl std::fmt::Display for UserType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            UserType::Regular => write!(f, "Regular"),
            UserType::Bot => write!(f, "Bot"),
            UserType::Admin => write!(f, "Admin"),
        }
    }
}

impl std::str::FromStr for UserType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Regular" => std::result::Result::Ok(UserType::Regular),
            "Bot" => std::result::Result::Ok(UserType::Bot),
            "Admin" => std::result::Result::Ok(UserType::Admin),
            _ => std::result::Result::Err(format!("Value not valid: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(tag = "kind")]
#[allow(non_camel_case_types)]
pub enum WsEvent {
    WsEventJoined(Box<models::WsEventJoined>),
    WsEventLeft(Box<models::WsEventLeft>),
    WsEventStarted(Box<models::WsEventStarted>),
    WsEventEnded(Box<models::WsEventEnded>),
    WsEventMoveMade(Box<models::WsEventMoveMade>),
    WsEventOnJoin(Box<models::WsEventOnJoin>),
    WsEventPlayerUpdate(Box<models::WsEventPlayerUpdate>),
}

impl validator::Validate for WsEvent {
    fn validate(&self) -> std::result::Result<(), validator::ValidationErrors> {
        match self {
            Self::WsEventJoined(x) => x.validate(),
            Self::WsEventLeft(x) => x.validate(),
            Self::WsEventStarted(x) => x.validate(),
            Self::WsEventEnded(x) => x.validate(),
            Self::WsEventMoveMade(x) => x.validate(),
            Self::WsEventOnJoin(x) => x.validate(),
            Self::WsEventPlayerUpdate(x) => x.validate(),
        }
    }
}

impl serde::Serialize for WsEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::WsEventJoined(x) => x.serialize(serializer),
            Self::WsEventLeft(x) => x.serialize(serializer),
            Self::WsEventStarted(x) => x.serialize(serializer),
            Self::WsEventEnded(x) => x.serialize(serializer),
            Self::WsEventMoveMade(x) => x.serialize(serializer),
            Self::WsEventOnJoin(x) => x.serialize(serializer),
            Self::WsEventPlayerUpdate(x) => x.serialize(serializer),
        }
    }
}

impl From<models::WsEventJoined> for WsEvent {
    fn from(value: models::WsEventJoined) -> Self {
        Self::WsEventJoined(Box::new(value))
    }
}
impl From<models::WsEventLeft> for WsEvent {
    fn from(value: models::WsEventLeft) -> Self {
        Self::WsEventLeft(Box::new(value))
    }
}
impl From<models::WsEventStarted> for WsEvent {
    fn from(value: models::WsEventStarted) -> Self {
        Self::WsEventStarted(Box::new(value))
    }
}
impl From<models::WsEventEnded> for WsEvent {
    fn from(value: models::WsEventEnded) -> Self {
        Self::WsEventEnded(Box::new(value))
    }
}
impl From<models::WsEventMoveMade> for WsEvent {
    fn from(value: models::WsEventMoveMade) -> Self {
        Self::WsEventMoveMade(Box::new(value))
    }
}
impl From<models::WsEventOnJoin> for WsEvent {
    fn from(value: models::WsEventOnJoin) -> Self {
        Self::WsEventOnJoin(Box::new(value))
    }
}
impl From<models::WsEventPlayerUpdate> for WsEvent {
    fn from(value: models::WsEventPlayerUpdate) -> Self {
        Self::WsEventPlayerUpdate(Box::new(value))
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEvent value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEvent {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsEventEnded {
    /// Note: inline enums are not fully supported by openapi-generator
    #[serde(default = "WsEventEnded::_name_for_kind")]
    #[serde(serialize_with = "WsEventEnded::_serialize_kind")]
    #[serde(rename = "kind")]
    pub kind: String,

    #[serde(rename = "value")]
    pub value: models::WsGameEnded,
}

impl WsEventEnded {
    fn _name_for_kind() -> String {
        String::from("WsEventEnded")
    }

    fn _serialize_kind<S>(_: &String, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(&Self::_name_for_kind())
    }
}

impl WsEventEnded {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(kind: String, value: models::WsGameEnded) -> WsEventEnded {
        WsEventEnded { kind, value }
    }
}

/// Converts the WsEventEnded value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsEventEnded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("kind".to_string()),
            Some(self.kind.to_string()),
            // Skipping value in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEventEnded value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEventEnded {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub kind: Vec<String>,
            pub value: Vec<models::WsGameEnded>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsEventEnded".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "kind" => intermediate_rep.kind.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "value" => intermediate_rep.value.push(
                        <models::WsGameEnded as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsEventEnded".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsEventEnded {
            kind: intermediate_rep
                .kind
                .into_iter()
                .next()
                .ok_or_else(|| "kind missing in WsEventEnded".to_string())?,
            value: intermediate_rep
                .value
                .into_iter()
                .next()
                .ok_or_else(|| "value missing in WsEventEnded".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsEventEnded> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsEventEnded>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsEventEnded>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsEventEnded - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsEventEnded> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsEventEnded as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsEventEnded - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsEventJoined {
    /// Note: inline enums are not fully supported by openapi-generator
    #[serde(default = "WsEventJoined::_name_for_kind")]
    #[serde(serialize_with = "WsEventJoined::_serialize_kind")]
    #[serde(rename = "kind")]
    pub kind: String,

    #[serde(rename = "value")]
    pub value: models::WsPlayerJoined,
}

impl WsEventJoined {
    fn _name_for_kind() -> String {
        String::from("WsEventJoined")
    }

    fn _serialize_kind<S>(_: &String, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(&Self::_name_for_kind())
    }
}

impl WsEventJoined {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(kind: String, value: models::WsPlayerJoined) -> WsEventJoined {
        WsEventJoined { kind, value }
    }
}

/// Converts the WsEventJoined value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsEventJoined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("kind".to_string()),
            Some(self.kind.to_string()),
            // Skipping value in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEventJoined value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEventJoined {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub kind: Vec<String>,
            pub value: Vec<models::WsPlayerJoined>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsEventJoined".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "kind" => intermediate_rep.kind.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "value" => intermediate_rep.value.push(
                        <models::WsPlayerJoined as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsEventJoined".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsEventJoined {
            kind: intermediate_rep
                .kind
                .into_iter()
                .next()
                .ok_or_else(|| "kind missing in WsEventJoined".to_string())?,
            value: intermediate_rep
                .value
                .into_iter()
                .next()
                .ok_or_else(|| "value missing in WsEventJoined".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsEventJoined> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsEventJoined>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsEventJoined>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsEventJoined - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsEventJoined> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsEventJoined as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsEventJoined - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsEventLeft {
    /// Note: inline enums are not fully supported by openapi-generator
    #[serde(default = "WsEventLeft::_name_for_kind")]
    #[serde(serialize_with = "WsEventLeft::_serialize_kind")]
    #[serde(rename = "kind")]
    pub kind: String,

    #[serde(rename = "value")]
    pub value: models::WsPlayerLeft,
}

impl WsEventLeft {
    fn _name_for_kind() -> String {
        String::from("WsEventLeft")
    }

    fn _serialize_kind<S>(_: &String, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(&Self::_name_for_kind())
    }
}

impl WsEventLeft {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(kind: String, value: models::WsPlayerLeft) -> WsEventLeft {
        WsEventLeft { kind, value }
    }
}

/// Converts the WsEventLeft value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsEventLeft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("kind".to_string()),
            Some(self.kind.to_string()),
            // Skipping value in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEventLeft value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEventLeft {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub kind: Vec<String>,
            pub value: Vec<models::WsPlayerLeft>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsEventLeft".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "kind" => intermediate_rep.kind.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "value" => intermediate_rep.value.push(
                        <models::WsPlayerLeft as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsEventLeft".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsEventLeft {
            kind: intermediate_rep
                .kind
                .into_iter()
                .next()
                .ok_or_else(|| "kind missing in WsEventLeft".to_string())?,
            value: intermediate_rep
                .value
                .into_iter()
                .next()
                .ok_or_else(|| "value missing in WsEventLeft".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsEventLeft> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsEventLeft>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsEventLeft>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsEventLeft - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsEventLeft> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsEventLeft as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsEventLeft - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsEventMoveMade {
    /// Note: inline enums are not fully supported by openapi-generator
    #[serde(default = "WsEventMoveMade::_name_for_kind")]
    #[serde(serialize_with = "WsEventMoveMade::_serialize_kind")]
    #[serde(rename = "kind")]
    pub kind: String,

    #[serde(rename = "value")]
    pub value: models::WsMoveMade,
}

impl WsEventMoveMade {
    fn _name_for_kind() -> String {
        String::from("WsEventMoveMade")
    }

    fn _serialize_kind<S>(_: &String, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(&Self::_name_for_kind())
    }
}

impl WsEventMoveMade {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(kind: String, value: models::WsMoveMade) -> WsEventMoveMade {
        WsEventMoveMade { kind, value }
    }
}

/// Converts the WsEventMoveMade value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsEventMoveMade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("kind".to_string()),
            Some(self.kind.to_string()),
            // Skipping value in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEventMoveMade value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEventMoveMade {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub kind: Vec<String>,
            pub value: Vec<models::WsMoveMade>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsEventMoveMade".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "kind" => intermediate_rep.kind.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "value" => intermediate_rep.value.push(
                        <models::WsMoveMade as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsEventMoveMade".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsEventMoveMade {
            kind: intermediate_rep
                .kind
                .into_iter()
                .next()
                .ok_or_else(|| "kind missing in WsEventMoveMade".to_string())?,
            value: intermediate_rep
                .value
                .into_iter()
                .next()
                .ok_or_else(|| "value missing in WsEventMoveMade".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsEventMoveMade> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsEventMoveMade>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsEventMoveMade>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsEventMoveMade - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsEventMoveMade> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsEventMoveMade as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsEventMoveMade - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsEventOnJoin {
    /// Note: inline enums are not fully supported by openapi-generator
    #[serde(default = "WsEventOnJoin::_name_for_kind")]
    #[serde(serialize_with = "WsEventOnJoin::_serialize_kind")]
    #[serde(rename = "kind")]
    pub kind: String,

    #[serde(rename = "value")]
    pub value: models::WsOnJoinMessage,
}

impl WsEventOnJoin {
    fn _name_for_kind() -> String {
        String::from("WsEventOnJoin")
    }

    fn _serialize_kind<S>(_: &String, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(&Self::_name_for_kind())
    }
}

impl WsEventOnJoin {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(kind: String, value: models::WsOnJoinMessage) -> WsEventOnJoin {
        WsEventOnJoin { kind, value }
    }
}

/// Converts the WsEventOnJoin value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsEventOnJoin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("kind".to_string()),
            Some(self.kind.to_string()),
            // Skipping value in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEventOnJoin value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEventOnJoin {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub kind: Vec<String>,
            pub value: Vec<models::WsOnJoinMessage>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsEventOnJoin".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "kind" => intermediate_rep.kind.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "value" => intermediate_rep.value.push(
                        <models::WsOnJoinMessage as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsEventOnJoin".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsEventOnJoin {
            kind: intermediate_rep
                .kind
                .into_iter()
                .next()
                .ok_or_else(|| "kind missing in WsEventOnJoin".to_string())?,
            value: intermediate_rep
                .value
                .into_iter()
                .next()
                .ok_or_else(|| "value missing in WsEventOnJoin".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsEventOnJoin> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsEventOnJoin>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsEventOnJoin>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsEventOnJoin - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsEventOnJoin> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsEventOnJoin as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsEventOnJoin - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsEventPlayerUpdate {
    /// Note: inline enums are not fully supported by openapi-generator
    #[serde(default = "WsEventPlayerUpdate::_name_for_kind")]
    #[serde(serialize_with = "WsEventPlayerUpdate::_serialize_kind")]
    #[serde(rename = "kind")]
    pub kind: String,

    #[serde(rename = "value")]
    pub value: models::WsPlayerUpdate,
}

impl WsEventPlayerUpdate {
    fn _name_for_kind() -> String {
        String::from("WsEventPlayerUpdate")
    }

    fn _serialize_kind<S>(_: &String, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(&Self::_name_for_kind())
    }
}

impl WsEventPlayerUpdate {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(kind: String, value: models::WsPlayerUpdate) -> WsEventPlayerUpdate {
        WsEventPlayerUpdate { kind, value }
    }
}

/// Converts the WsEventPlayerUpdate value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsEventPlayerUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("kind".to_string()),
            Some(self.kind.to_string()),
            // Skipping value in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEventPlayerUpdate value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEventPlayerUpdate {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub kind: Vec<String>,
            pub value: Vec<models::WsPlayerUpdate>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsEventPlayerUpdate".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "kind" => intermediate_rep.kind.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "value" => intermediate_rep.value.push(
                        <models::WsPlayerUpdate as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsEventPlayerUpdate".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsEventPlayerUpdate {
            kind: intermediate_rep
                .kind
                .into_iter()
                .next()
                .ok_or_else(|| "kind missing in WsEventPlayerUpdate".to_string())?,
            value: intermediate_rep
                .value
                .into_iter()
                .next()
                .ok_or_else(|| "value missing in WsEventPlayerUpdate".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsEventPlayerUpdate> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsEventPlayerUpdate>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsEventPlayerUpdate>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsEventPlayerUpdate - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsEventPlayerUpdate> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsEventPlayerUpdate as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsEventPlayerUpdate - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsEventStarted {
    /// Note: inline enums are not fully supported by openapi-generator
    #[serde(default = "WsEventStarted::_name_for_kind")]
    #[serde(serialize_with = "WsEventStarted::_serialize_kind")]
    #[serde(rename = "kind")]
    pub kind: String,

    #[serde(rename = "value")]
    pub value: models::WsGameStarted,
}

impl WsEventStarted {
    fn _name_for_kind() -> String {
        String::from("WsEventStarted")
    }

    fn _serialize_kind<S>(_: &String, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        s.serialize_str(&Self::_name_for_kind())
    }
}

impl WsEventStarted {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(kind: String, value: models::WsGameStarted) -> WsEventStarted {
        WsEventStarted { kind, value }
    }
}

/// Converts the WsEventStarted value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsEventStarted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("kind".to_string()),
            Some(self.kind.to_string()),
            // Skipping value in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsEventStarted value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsEventStarted {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub kind: Vec<String>,
            pub value: Vec<models::WsGameStarted>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsEventStarted".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "kind" => intermediate_rep.kind.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "value" => intermediate_rep.value.push(
                        <models::WsGameStarted as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsEventStarted".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsEventStarted {
            kind: intermediate_rep
                .kind
                .into_iter()
                .next()
                .ok_or_else(|| "kind missing in WsEventStarted".to_string())?,
            value: intermediate_rep
                .value
                .into_iter()
                .next()
                .ok_or_else(|| "value missing in WsEventStarted".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsEventStarted> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsEventStarted>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsEventStarted>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsEventStarted - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsEventStarted> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsEventStarted as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsEventStarted - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsGameEnded {
    #[serde(rename = "game")]
    pub game: models::GameState,
}

impl WsGameEnded {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(game: models::GameState) -> WsGameEnded {
        WsGameEnded { game }
    }
}

/// Converts the WsGameEnded value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsGameEnded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping game in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsGameEnded value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsGameEnded {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub game: Vec<models::GameState>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsGameEnded".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "game" => intermediate_rep.game.push(
                        <models::GameState as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsGameEnded".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsGameEnded {
            game: intermediate_rep
                .game
                .into_iter()
                .next()
                .ok_or_else(|| "game missing in WsGameEnded".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsGameEnded> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsGameEnded>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsGameEnded>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsGameEnded - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsGameEnded> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsGameEnded as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsGameEnded - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsGameStarted {
    #[serde(rename = "game")]
    pub game: models::GameState,
}

impl WsGameStarted {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(game: models::GameState) -> WsGameStarted {
        WsGameStarted { game }
    }
}

/// Converts the WsGameStarted value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsGameStarted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping game in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsGameStarted value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsGameStarted {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub game: Vec<models::GameState>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsGameStarted".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "game" => intermediate_rep.game.push(
                        <models::GameState as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsGameStarted".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsGameStarted {
            game: intermediate_rep
                .game
                .into_iter()
                .next()
                .ok_or_else(|| "game missing in WsGameStarted".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsGameStarted> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsGameStarted>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsGameStarted>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsGameStarted - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsGameStarted> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsGameStarted as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsGameStarted - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsMoveMade {
    #[serde(rename = "move")]
    pub r#move: models::Move,

    #[serde(rename = "newFen")]
    pub new_fen: String,

    #[serde(rename = "newTurn")]
    pub new_turn: models::PlayerColor,
}

impl WsMoveMade {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(r#move: models::Move, new_fen: String, new_turn: models::PlayerColor) -> WsMoveMade {
        WsMoveMade {
            r#move,
            new_fen,
            new_turn,
        }
    }
}

/// Converts the WsMoveMade value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsMoveMade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping move in query parameter serialization
            Some("newFen".to_string()),
            Some(self.new_fen.to_string()),
            // Skipping newTurn in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsMoveMade value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsMoveMade {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub r#move: Vec<models::Move>,
            pub new_fen: Vec<String>,
            pub new_turn: Vec<models::PlayerColor>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsMoveMade".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "move" => intermediate_rep.r#move.push(
                        <models::Move as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "newFen" => intermediate_rep.new_fen.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "newTurn" => intermediate_rep.new_turn.push(
                        <models::PlayerColor as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsMoveMade".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsMoveMade {
            r#move: intermediate_rep
                .r#move
                .into_iter()
                .next()
                .ok_or_else(|| "move missing in WsMoveMade".to_string())?,
            new_fen: intermediate_rep
                .new_fen
                .into_iter()
                .next()
                .ok_or_else(|| "newFen missing in WsMoveMade".to_string())?,
            new_turn: intermediate_rep
                .new_turn
                .into_iter()
                .next()
                .ok_or_else(|| "newTurn missing in WsMoveMade".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsMoveMade> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsMoveMade>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsMoveMade>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsMoveMade - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsMoveMade> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsMoveMade as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsMoveMade - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsOnJoinMessage {
    #[serde(rename = "game")]
    pub game: models::GameState,
}

impl WsOnJoinMessage {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(game: models::GameState) -> WsOnJoinMessage {
        WsOnJoinMessage { game }
    }
}

/// Converts the WsOnJoinMessage value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsOnJoinMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping game in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsOnJoinMessage value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsOnJoinMessage {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub game: Vec<models::GameState>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsOnJoinMessage".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "game" => intermediate_rep.game.push(
                        <models::GameState as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsOnJoinMessage".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsOnJoinMessage {
            game: intermediate_rep
                .game
                .into_iter()
                .next()
                .ok_or_else(|| "game missing in WsOnJoinMessage".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsOnJoinMessage> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsOnJoinMessage>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsOnJoinMessage>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsOnJoinMessage - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsOnJoinMessage> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsOnJoinMessage as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsOnJoinMessage - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsPlayerJoined {
    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "color")]
    pub color: models::PlayerColor,

    #[serde(rename = "players")]
    pub players: models::PlayerUpdate,
}

impl WsPlayerJoined {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        username: String,
        color: models::PlayerColor,
        players: models::PlayerUpdate,
    ) -> WsPlayerJoined {
        WsPlayerJoined {
            username,
            color,
            players,
        }
    }
}

/// Converts the WsPlayerJoined value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsPlayerJoined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("username".to_string()),
            Some(self.username.to_string()),
            // Skipping color in query parameter serialization

            // Skipping players in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsPlayerJoined value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsPlayerJoined {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub username: Vec<String>,
            pub color: Vec<models::PlayerColor>,
            pub players: Vec<models::PlayerUpdate>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsPlayerJoined".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "color" => intermediate_rep.color.push(
                        <models::PlayerColor as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "players" => intermediate_rep.players.push(
                        <models::PlayerUpdate as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsPlayerJoined".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsPlayerJoined {
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in WsPlayerJoined".to_string())?,
            color: intermediate_rep
                .color
                .into_iter()
                .next()
                .ok_or_else(|| "color missing in WsPlayerJoined".to_string())?,
            players: intermediate_rep
                .players
                .into_iter()
                .next()
                .ok_or_else(|| "players missing in WsPlayerJoined".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsPlayerJoined> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsPlayerJoined>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsPlayerJoined>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsPlayerJoined - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsPlayerJoined> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsPlayerJoined as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsPlayerJoined - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsPlayerLeft {
    #[serde(rename = "username")]
    pub username: String,

    #[serde(rename = "color")]
    pub color: models::PlayerColor,

    #[serde(rename = "players")]
    pub players: models::PlayerUpdate,
}

impl WsPlayerLeft {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(
        username: String,
        color: models::PlayerColor,
        players: models::PlayerUpdate,
    ) -> WsPlayerLeft {
        WsPlayerLeft {
            username,
            color,
            players,
        }
    }
}

/// Converts the WsPlayerLeft value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsPlayerLeft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            Some("username".to_string()),
            Some(self.username.to_string()),
            // Skipping color in query parameter serialization

            // Skipping players in query parameter serialization
        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsPlayerLeft value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsPlayerLeft {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub username: Vec<String>,
            pub color: Vec<models::PlayerColor>,
            pub players: Vec<models::PlayerUpdate>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsPlayerLeft".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "username" => intermediate_rep.username.push(
                        <String as std::str::FromStr>::from_str(val).map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "color" => intermediate_rep.color.push(
                        <models::PlayerColor as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    #[allow(clippy::redundant_clone)]
                    "players" => intermediate_rep.players.push(
                        <models::PlayerUpdate as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsPlayerLeft".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsPlayerLeft {
            username: intermediate_rep
                .username
                .into_iter()
                .next()
                .ok_or_else(|| "username missing in WsPlayerLeft".to_string())?,
            color: intermediate_rep
                .color
                .into_iter()
                .next()
                .ok_or_else(|| "color missing in WsPlayerLeft".to_string())?,
            players: intermediate_rep
                .players
                .into_iter()
                .next()
                .ok_or_else(|| "players missing in WsPlayerLeft".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsPlayerLeft> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsPlayerLeft>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsPlayerLeft>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsPlayerLeft - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsPlayerLeft> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsPlayerLeft as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsPlayerLeft - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, validator::Validate)]
#[cfg_attr(feature = "conversion", derive(frunk::LabelledGeneric))]
pub struct WsPlayerUpdate {
    #[serde(rename = "players")]
    pub players: models::PlayerUpdate,
}

impl WsPlayerUpdate {
    #[allow(clippy::new_without_default, clippy::too_many_arguments)]
    pub fn new(players: models::PlayerUpdate) -> WsPlayerUpdate {
        WsPlayerUpdate { players }
    }
}

/// Converts the WsPlayerUpdate value to the Query Parameters representation (style=form, explode=false)
/// specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde serializer
impl std::fmt::Display for WsPlayerUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: Vec<Option<String>> = vec![
            // Skipping players in query parameter serialization

        ];

        write!(
            f,
            "{}",
            params.into_iter().flatten().collect::<Vec<_>>().join(",")
        )
    }
}

/// Converts Query Parameters representation (style=form, explode=false) to a WsPlayerUpdate value
/// as specified in https://swagger.io/docs/specification/serialization/
/// Should be implemented in a serde deserializer
impl std::str::FromStr for WsPlayerUpdate {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        /// An intermediate representation of the struct to use for parsing.
        #[derive(Default)]
        #[allow(dead_code)]
        struct IntermediateRep {
            pub players: Vec<models::PlayerUpdate>,
        }

        let mut intermediate_rep = IntermediateRep::default();

        // Parse into intermediate representation
        let mut string_iter = s.split(',');
        let mut key_result = string_iter.next();

        while key_result.is_some() {
            let val = match string_iter.next() {
                Some(x) => x,
                None => {
                    return std::result::Result::Err(
                        "Missing value while parsing WsPlayerUpdate".to_string(),
                    )
                }
            };

            if let Some(key) = key_result {
                #[allow(clippy::match_single_binding)]
                match key {
                    #[allow(clippy::redundant_clone)]
                    "players" => intermediate_rep.players.push(
                        <models::PlayerUpdate as std::str::FromStr>::from_str(val)
                            .map_err(|x| x.to_string())?,
                    ),
                    _ => {
                        return std::result::Result::Err(
                            "Unexpected key while parsing WsPlayerUpdate".to_string(),
                        )
                    }
                }
            }

            // Get the next key
            key_result = string_iter.next();
        }

        // Use the intermediate representation to return the struct
        std::result::Result::Ok(WsPlayerUpdate {
            players: intermediate_rep
                .players
                .into_iter()
                .next()
                .ok_or_else(|| "players missing in WsPlayerUpdate".to_string())?,
        })
    }
}

// Methods for converting between header::IntoHeaderValue<WsPlayerUpdate> and HeaderValue

#[cfg(feature = "server")]
impl std::convert::TryFrom<header::IntoHeaderValue<WsPlayerUpdate>> for HeaderValue {
    type Error = String;

    fn try_from(
        hdr_value: header::IntoHeaderValue<WsPlayerUpdate>,
    ) -> std::result::Result<Self, Self::Error> {
        let hdr_value = hdr_value.to_string();
        match HeaderValue::from_str(&hdr_value) {
            std::result::Result::Ok(value) => std::result::Result::Ok(value),
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Invalid header value for WsPlayerUpdate - value: {} is invalid {}",
                hdr_value, e
            )),
        }
    }
}

#[cfg(feature = "server")]
impl std::convert::TryFrom<HeaderValue> for header::IntoHeaderValue<WsPlayerUpdate> {
    type Error = String;

    fn try_from(hdr_value: HeaderValue) -> std::result::Result<Self, Self::Error> {
        match hdr_value.to_str() {
            std::result::Result::Ok(value) => {
                match <WsPlayerUpdate as std::str::FromStr>::from_str(value) {
                    std::result::Result::Ok(value) => {
                        std::result::Result::Ok(header::IntoHeaderValue(value))
                    }
                    std::result::Result::Err(err) => std::result::Result::Err(format!(
                        "Unable to convert header value '{}' into WsPlayerUpdate - {}",
                        value, err
                    )),
                }
            }
            std::result::Result::Err(e) => std::result::Result::Err(format!(
                "Unable to convert header: {:?} to string: {}",
                hdr_value, e
            )),
        }
    }
}
