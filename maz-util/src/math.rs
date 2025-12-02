#[derive(Debug, Default, Clone, Copy)]
pub struct RunningAverage {
    count: u64,
    mean: f32,
}

impl RunningAverage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_sample(&mut self, value: f32) {
        self.count += 1;
        let n = self.count as f32;

        self.mean += (value - self.mean) / n;
    }

    pub fn get_average(&self) -> f32 {
        self.mean
    }

    pub fn get_count(&self) -> u64 {
        self.count
    }
}

pub fn safe_div(a: f64, b: f64) -> f64 {
    if b > 0.0 { a / b } else { 0.0 }
}


/// Slow implementation of softmax that allocates no extra memory.
pub fn softmax_in_place(slice: &mut [f32]) {
    let max = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let mut sum = 0.0;
    for v in slice.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    assert!(
        sum > 0.0,
        "Softmax input sum must be strictly positive, was {sum}. Input: {:?}.  Max: {}", slice, max
    );
    for v in slice.iter_mut() {
        *v /= sum;
    }
}

/// Takes a policy that has already been **softmaxed** and applies a temperature to it.
/// Softmax properties -> Policy must sum to 1.0 before applying this function and all values must be finite and positive.
pub fn apply_temperature_to_softmax_policy(slice: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }

    // Check that it roughly sums to 1.0, this is a precondition for the softmax to work correctly.
    debug_assert!(
        (0.99..=1.01).contains(&slice.iter().sum::<f32>()),
        "Expected sum 1.0, got {} from {:?}",
        slice.iter().sum::<f32>(),
        slice
    );

    debug_assert!(
        temperature > 0.0 && temperature.is_finite(),
        "Temperature must be finite and positive, got {temperature}"
    );

    let mut prev_sum = 0.0;
    let mut sum = 0.0;

    for v in slice.iter_mut() {
        prev_sum += *v;
        *v = v.powf(1.0 / temperature);
        sum += *v;
    }

    debug_assert!(
        0.99 < prev_sum && prev_sum < 1.01,
        "Expected sum 1.0, got {prev_sum} from {slice:?}"
    );

    for v in slice {
        *v /= sum;
    }
}
