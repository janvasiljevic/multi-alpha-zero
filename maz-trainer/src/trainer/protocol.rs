use crate::trainer::protocol::training::training_coordinator_client::TrainingCoordinatorClient;
use crate::trainer::protocol::training::{
    InitialSettingsResponse, SetSettingsRequest, TrainModelRequest, TrainModelResponse,
};
use maz_config::config::{AppConfig, TrainingConfig};
use std::time::Duration;
use tonic::transport::Channel;
use tonic::Request;

pub mod training {
    tonic::include_proto!("training");
}

async fn create_channel(
    server_addr: &'static str,
    timeout: Option<u64>,
) -> Result<Channel, Box<dyn std::error::Error>> {
    let span = tracing::span!(tracing::Level::INFO, "create_channel");
    let _enter = span.enter();

    match Channel::from_static(server_addr)
        .connect_timeout(
            timeout
                .map(Duration::from_secs)
                .unwrap_or(Duration::from_secs(1)),
        )
        .connect()
        .await
    {
        Ok(channel) => Ok(channel),
        Err(e) => {
            tracing::error!(
                "Failed to connect to training coordinator at {}: {}",
                server_addr,
                e
            );
            Err(Box::new(e))
        }
    }
}

pub async fn send_initial_settings(
    server_addr: &'static str,
    config: &AppConfig,
    base_directory: String,
    current_training_step: u64,
) -> Result<InitialSettingsResponse, Box<dyn std::error::Error>> {
    let span = tracing::span!(tracing::Level::INFO, "send_initial_settings");
    let _enter = span.enter();

    let channel = create_channel(server_addr, None).await?;

    let mut client = TrainingCoordinatorClient::new(channel);

    let request = Request::new(SetSettingsRequest {
        batch_size: config.performance.gpu_batch_size as u64,
        base_directory,
        max_samples: config.training.samples,
        learning_rate: config.training.learning_rate,
        model_key: config.game.name.to_internal_name().to_string(),
        current_training_step,
        weight_decay: config.training.weight_decay,
        policy_loss_weight: config.training.policy_loss_weight,
        q_value_loss_weight: config.training.q_value_loss_weight,
        z_value_loss_weight: config.training.z_value_loss_weight,
        num_of_aux_features: config.training.number_of_aux_features as u64,
        warmup_steps: config.training.warmup_steps as u64
    });

    Ok(client.set_initial_settings(request).await?.into_inner())
}

pub async fn train_model(
    server_addr: &'static str,
    train_config: &TrainingConfig,
) -> Result<TrainModelResponse, Box<dyn std::error::Error>> {
    let span = tracing::span!(tracing::Level::INFO, "train_model");
    let _enter = span.enter();

    let channel = create_channel(server_addr, Some(60 * 60)).await?;

    let mut client = TrainingCoordinatorClient::new(channel);

    let request = Request::new(TrainModelRequest {
        batch_size: train_config.batch_size as u32,
        epochs: train_config.epochs as u32,
    });

    Ok(client.train_model(request).await?.into_inner())
}
