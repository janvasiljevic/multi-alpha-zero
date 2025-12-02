use skillratings::weng_lin::{
    expected_score_multi_team, WengLinConfig, WengLinRating,
};

fn main() {
    let uncertainty = 0.71;

    let abs110 = WengLinRating {
        rating: 29.49,
        uncertainty,
    };

    let abs80 = WengLinRating {
        rating: 27.25,
        uncertainty,
    };

    let mcts12800 = WengLinRating {
        rating: 25.18,
        uncertainty,
    };

    let exp = expected_score_multi_team(
        &[&vec![abs110], &vec![abs80], &vec![mcts12800]],
        &WengLinConfig {
            beta: 4.0,
            uncertainty_tolerance: 0.0,
        }
    );

    println!("{:?}", exp);

    let exp_2 = expected_score_multi_team(
        &[&vec![abs110], &vec![abs80], &vec![mcts12800]],
        &WengLinConfig {
            beta: 0.0,
            uncertainty_tolerance: 1.0,
        }
    );

    println!("{:?}", exp_2);
}
