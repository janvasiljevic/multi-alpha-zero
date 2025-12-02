fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../shared/training.proto");

    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(
            &[
                "../shared/training.proto",
                "../shared/health.proto"
            ],
            &["../shared"],
        )?;

    Ok(())
}