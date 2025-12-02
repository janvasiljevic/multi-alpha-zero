use rasciigraph::plot;
use std::fmt::{Display, Formatter};

pub fn display_option<T: Display>(value: Option<T>) -> impl Display {
    struct Wrapper<T>(Option<T>);
    impl<T: Display> Display for Wrapper<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match &self.0 {
                None => write!(f, "None"),
                Some(value) => write!(f, "Some({})", value),
            }
        }
    }
    Wrapper(value)
}

pub fn display_option_empty<T: Display>(value: Option<T>) -> impl Display {
    struct Wrapper<T>(Option<T>);
    impl<T: Display> Display for Wrapper<T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            match &self.0 {
                None => write!(f, ""),
                Some(value) => write!(f, "{}", value),
            }
        }
    }
    Wrapper(value)
}

pub fn ascii_art_policy(policy: &Vec<f32>, number_of_bins: usize) -> String {
    let mut bins = vec![1.0f64; number_of_bins];

    for &p in policy {
        let idx = (p * number_of_bins as f32).min((number_of_bins - 1) as f32) as usize;
        bins[idx] += 1.0;
    }

    bins.iter_mut().for_each(|b| {
        if *b > 0.0 {
            *b = (*b).log(10.0);
        }
    });

    plot(
        bins,
        rasciigraph::Config::default()
            .with_offset(5)
            .with_height(3)
            .with_caption(format!(
                "log_10 PI dist. | Max prob.: {:.3}",
                policy.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            )),
    )
}
