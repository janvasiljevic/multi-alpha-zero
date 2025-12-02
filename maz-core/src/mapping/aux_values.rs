use std::sync::OnceLock;
use tracing::info;

// This is a little hacky, would need refactorization later...
static NUM_OF_AUX_FEATURES: OnceLock<usize> = OnceLock::new();

pub fn set_num_of_aux_features(num: usize) {
    info!("Setting NUM_OF_AUX_FEATURES to {num}");
    if NUM_OF_AUX_FEATURES.set(num).is_err() {
        panic!("NUM_OF_AUX_FEATURES has already been set");
    }
}

pub fn get_num_of_aux_features() -> usize {
    *NUM_OF_AUX_FEATURES
        .get()
        .expect("NUM_OF_AUX_FEATURES has not been set")
}
