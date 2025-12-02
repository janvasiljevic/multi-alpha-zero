use rand::distr::Distribution;
use rand::Rng;
use rand_distr::Gamma;


const ALPHA_MIN: f32 = 0.1;
const SUM_MIN: f32 = 0.00000001;

/// Variant of [rand::distributions::Dirichlet] that never generates NaNs, even when `alpha` is low
#[derive(Debug, Copy, Clone)]
pub struct StableDirichletFast {
    alpha: f32,
    len: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct DirichletErrorFast;

impl StableDirichletFast {
    pub fn new(alpha: f32, len: usize) -> Result<Self, DirichletErrorFast> {
        if alpha > 0.0 && len > 0 {
            Ok(Self { alpha, len })
        } else {
            Err(DirichletErrorFast)
        }
    }

    /// Samples from the distribution, writing the results directly into the target slice.
    /// This avoids heap allocation.
    ///
    /// # Panics
    /// Panics if `target.len()` does not match `self.len`.
    pub fn sample_into<R: Rng + ?Sized>(&self, rng: &mut R, target: &mut [f32]) {
        assert_eq!(
            self.len,
            target.len(),
            "Target slice length must match distribution length"
        );

        if self.len == 0 {
            return;
        }
        if self.len == 1 {
            target[0] = 1.0;
            return;
        }

        if self.alpha > ALPHA_MIN {
            let gamma = Gamma::new(self.alpha, 1.0).unwrap();
            let mut sum = 0.0;

            for val in target.iter_mut() {
                let v = gamma.sample(rng);
                *val = v;
                sum += v;
            }

            if sum > SUM_MIN {
                for val in target.iter_mut() {
                    *val /= sum;
                }
                return;
            }
        }

        // Fallback: generate a maximally concentrated sample
        // First, zero out the buffer
        target.fill(0.0);
        // Then set one random element to 1.0
        let index = rng.random_range(0..self.len);
        target[index] = 1.0;
    }
}

// Add this to the bottom of your file
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rng;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_constructor_valid() {
        assert!(StableDirichletFast::new(1.0, 3).is_ok());
        assert!(StableDirichletFast::new(0.1, 1).is_ok());
    }

    #[test]
    fn test_constructor_invalid() {
        // Invalid alpha
        assert!(StableDirichletFast::new(0.0, 3).is_err());
        assert!(StableDirichletFast::new(-1.0, 3).is_err());

        // Invalid len
        assert!(StableDirichletFast::new(1.0, 0).is_err());
    }

    #[test]
    #[should_panic(expected = "Target slice length must match distribution length")]
    fn test_sample_into_len_mismatch_panic() {
        let mut rng = rng();
        let dist = StableDirichletFast::new(1.0, 5).unwrap();
        let mut target = vec![0.0; 4]; // Mismatched length
        dist.sample_into(&mut rng, &mut target);
    }

    #[test]
    fn test_sample_into_len_zero() {
        let mut rng = rng();
        // Although the constructor prevents len=0, we can test the internal logic
        // by creating the struct directly if it were public.
        // Since it's not, we can assume the guard in `new` is sufficient.
        // However, the `sample_into` method does have a guard for `len == 0`.
        let dist = StableDirichletFast { alpha: 1.0, len: 0 };
        let mut target = [];
        dist.sample_into(&mut rng, &mut target); // Should not panic and do nothing
        assert_eq!(target.len(), 0);
    }

    #[test]
    fn test_sample_into_len_one() {
        let mut rng = rng();
        let dist = StableDirichletFast::new(1.0, 1).unwrap();
        let mut target = vec![0.0; 1];
        dist.sample_into(&mut rng, &mut target);
        assert_eq!(target, vec![1.0]);
    }

    #[test]
    fn test_standard_sample_properties() {
        let mut rng = rng();
        let len = 10;
        // Use an alpha well above the fallback threshold
        let dist = StableDirichletFast::new(0.25, len).unwrap();
        let mut target = vec![0.0; len];

        // Run a few times to be sure
        for _ in 0..10 {
            dist.sample_into(&mut rng, &mut target);

            let mut sum = 0.0;
            for &val in target.iter() {
                // All values should be valid probabilities
                assert!(val >= 0.0 && val <= 1.0);
                assert!(!val.is_nan());
                sum += val;
            }

            // The sum of all values should be 1.0
            assert!((sum - 1.0).abs() < EPSILON, "Sum was {}", sum);

            println!("Sampled values: {:?}", target);
        }
    }

    #[test]
    fn test_fallback_sample_properties() {
        let mut rng = rng();
        let len = 20;

        let dist = StableDirichletFast::new(ALPHA_MIN / 2.0, len).unwrap();
        let mut target = vec![0.0; len];

        for _ in 0..10 {
            dist.sample_into(&mut rng, &mut target);

            let sum: f32 = target.iter().sum();
            assert!((sum - 1.0).abs() < EPSILON, "Sum was {}", sum);


            let ones = target.iter().filter(|&&v| v == 1.0).count();
            let zeros = target.iter().filter(|&&v| v == 0.0).count();

            assert_eq!(ones, 1, "Expected exactly one 1.0 in fallback mode");
            assert_eq!(zeros, len - 1, "Expected the rest to be 0.0");
        }
    }
}