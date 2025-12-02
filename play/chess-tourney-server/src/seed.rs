use sea_orm::QueryFilter;
use crate::entity::users;
use crate::entity::users::UserTypeDb;
use crate::server::ServerImpl;
use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHasher};
use sea_orm::ColumnTrait;
use sea_orm::{EntityTrait, Set};
use tracing::info;

impl ServerImpl {
    pub fn seed_default_user(&self) {
        let argon = Argon2::default();
        let salt = SaltString::generate(&mut OsRng);

        let users = vec![
            users::ActiveModel {
                id: Default::default(),
                username: Set("user".to_string()),
                password_hash: Set(argon
                    .hash_password("password".as_bytes(), &salt)
                    .unwrap()
                    .to_string()),
                lichess_rating: Default::default(),
                chess_com_rating: Default::default(),
                fide_rating: Default::default(),
                experience_with_chess: Default::default(),
                user_type: Set(UserTypeDb::Normal),
            },
            users::ActiveModel {
                id: Default::default(),
                username: Set("admin".to_string()),
                password_hash: Set(argon
                    .hash_password("adminpass".as_bytes(), &salt)
                    .unwrap()
                    .to_string()),
                lichess_rating: Default::default(),
                chess_com_rating: Default::default(),
                fide_rating: Default::default(),
                experience_with_chess: Default::default(),
                user_type: Set(UserTypeDb::Admin),
            },
        ];

        for user in users {
            let db = self.db.clone();
            let username = user.username.clone();
            tokio::spawn(async move {
                let existing = users::Entity::find()
                    .filter(users::Column::Username.eq(username.as_ref().clone()))
                    .one(&db)
                    .await
                    .unwrap();

                if existing.is_none() {
                    users::Entity::insert(user).exec(&db).await.unwrap();
                    info!("Seeded default user: {}", username.as_ref());
                }
            });
        }
    }
}
