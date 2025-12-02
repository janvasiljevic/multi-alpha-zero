pub use crate::vis_arrow::{ArrowProps, Point, VisArrow, create_arrow_component};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use svg2pdf::{ConversionOptions, PageOptions};
use usvg::{Tree, fontdb};

pub mod parquet_reader;
pub mod vis_arrow;
pub mod vis_turn;

pub fn convert_svg_to_pdf(input_path: &str, output_path: &str) -> Result<String, String> {
    println!(
        "Converting {} to {} using embedded font...",
        input_path, output_path
    );

    let mut font_db = fontdb::Database::new();

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut font_path = PathBuf::from(manifest_dir);
    font_path.push("font/RobotoMono-Bold.ttf");

    let mut font_path_2 = PathBuf::from(manifest_dir);
    font_path_2.push("font/Inter_18pt-Regular.ttf");

    font_db
        .load_font_file(&font_path)
        .map_err(|e| format!("Failed to load font file at {:?}: {}", font_path, e))?;

    font_db
        .load_font_file(&font_path_2)
        .map_err(|e| format!("Failed to load font file at {:?}: {}", font_path_2, e))?;

    let svg_data = fs::read(input_path).map_err(|e| e.to_string())?;

    let opts = usvg::Options {
        fontdb: Arc::new(font_db),
        ..usvg::Options::default()
    };

    let tree = Tree::from_data(&svg_data, &opts).map_err(|e| e.to_string())?;

    let pdf_data = svg2pdf::to_pdf(&tree, ConversionOptions::default(), PageOptions::default())
        .map_err(|e| format!("SVG to PDF conversion failed: {}", e))?;

    fs::write(output_path, pdf_data).map_err(|e| e.to_string())?;

    Ok(output_path.to_string())
}
