use skillratings::weng_lin::WengLinRating;

pub fn generate_ascii_arena_table(final_results: Vec<(String, WengLinRating)>) {
    println!("\n--- Final Tournament Standings ---");
    println!(
        "{:<4} {:<30} {:<20} {:<20} {:<20}",
        "Rank", "Player", "Rating (μ)", "Uncertainty (σ)", "Conservative Rating"
    );
    println!("{}", "-".repeat(98));
    for (i, (name, rating)) in final_results.iter().enumerate() {
        println!(
            "{:<4} {:<30} {:<20.4} {:<20.4} {:<20.4}",
            i + 1,
            name,
            rating.rating,
            rating.uncertainty,
            rating.rating - 3.0 * rating.uncertainty
        );
    }
}
