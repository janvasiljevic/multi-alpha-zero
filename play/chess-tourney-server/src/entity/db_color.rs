use api_autogen::models::PlayerColor;
use game_tri_chess::basics::Color;
use sea_orm::prelude::StringLen;
use sea_orm::{DeriveActiveEnum, EnumIter};

#[derive(EnumIter, DeriveActiveEnum, Clone, Debug, PartialEq, Eq)]
#[sea_orm(
    rs_type = "String",
    db_type = "String(StringLen::None)",
    rename_all = "camelCase"
)]
pub enum ColorDb {
    White,
    Gray,
    Black,
}

impl From<Color> for ColorDb {
    fn from(value: Color) -> Self {
        match value {
            Color::White => ColorDb::White,
            Color::Gray => ColorDb::Gray,
            Color::Black => ColorDb::Black,
        }
    }
}

impl From<ColorDb> for Color {
    fn from(value: ColorDb) -> Self {
        match value {
            ColorDb::White => Color::White,
            ColorDb::Gray => Color::Gray,
            ColorDb::Black => Color::Black,
        }
    }
}

impl From<PlayerColor> for ColorDb {
    fn from(value: PlayerColor) -> Self {
        match value {
            PlayerColor::White => ColorDb::White,
            PlayerColor::Grey => ColorDb::Gray,
            PlayerColor::Black => ColorDb::Black,
        }
    }
}

impl From<ColorDb> for PlayerColor {
    fn from(value: ColorDb) -> Self {
        match value {
            ColorDb::White => PlayerColor::White,
            ColorDb::Gray => PlayerColor::Grey,
            ColorDb::Black => PlayerColor::Black,
        }
    }
}