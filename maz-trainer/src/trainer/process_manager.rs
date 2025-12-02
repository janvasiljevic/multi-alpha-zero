use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;
use tonic::transport::Channel;

pub mod health {
    tonic::include_proto!("grpc.health.v1"); // The package name in your .proto file
}

use health::health_check_response::ServingStatus;
use health::health_client::HealthClient;
use health::HealthCheckRequest;

pub struct ManagedChildProcess {
    name: String,
    child: Child,
}

impl ManagedChildProcess {
    pub fn new(name: String, child: Child) -> Self {
        tracing::info!("[{}] Process started with PID: {}", name, child.id());
        Self { name, child }
    }
}

impl Drop for ManagedChildProcess {
    fn drop(&mut self) {
        let span = tracing::span!(
            tracing::Level::INFO,
            "ManagedChildProcessDrop",
            name = self.name
        );
        let _enter = span.enter();

        tracing::info!("Attempting to shut down child process...");

        match self.child.kill() {
            Ok(_) => {
                tracing::info!("Successfully sent kill signal. Waiting for process to exit...");

                match self.child.wait() {
                    Ok(status) => tracing::info!("Child process exited with status: {}", status),
                    Err(e) => tracing::error!("Failed to wait for child process: {}", e),
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to kill child process (it may have already exited): {}",
                    e
                );
            }
        }
    }
}

pub async fn launch_python_server(
    server_addr: &'static str,
    python_interpreter_path: String,
) -> Result<ManagedChildProcess, Box<dyn std::error::Error>> {
    let span = tracing::span!(tracing::Level::INFO, "launch_python_server");
    let _enter = span.enter();

    // Get the directory of the Rust crate's Cargo.toml
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let project_root = manifest_dir
        .parent()
        .ok_or("Failed to get parent directory of CARGO_MANIFEST_DIR")?;

    tracing::info!(
        "CARGO_MANIFEST_DIR: {:?}, Project root: {:?}",
        manifest_dir,
        project_root
    );

    let python_project_dir = project_root.join("python");

    let py_server_path = python_project_dir.join("main.py");
    if !py_server_path.exists() {
        return Err(format!("Python server script not found at: {py_server_path:?}").into());
    }

    let python_executable = project_root.join(python_interpreter_path);

    if !python_executable.exists() {
        return Err(
            format!("Python executable not found in venv at: {python_executable:?}").into(),
        );
    }

    tracing::info!(
        "Using Python executable: {:?}. Starting Python server at: {:?}",
        python_executable,
        py_server_path
    );

    // The current_dir() call is important so python can find its own modules
    let child = Command::new(&python_executable)
        .arg(&py_server_path)
        .current_dir(&python_project_dir)
        .env("GRPC_VERBOSITY", "error")
        // .env("GRPC_VERBOSITY", "debug")
        // .env("GRPC_TRACE", "all")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    let mut managed_child = ManagedChildProcess::new("PythonServer".to_string(), child);

    tracing::info!("Waiting for Python server to become healthy...");

    let max_retries = 300;

    for i in 0..max_retries {
        if let Some(exit_status) = managed_child.child.try_wait()? {
            return Err(format!(
                "Python server process exited prematurely with status: {exit_status}"
            )
            .into());
        }

        if let Ok(channel) = Channel::from_static(server_addr)
            .connect_timeout(Duration::from_secs(1))
            .connect()
            .await
        {
            let mut health_client = HealthClient::new(channel);

            let request = HealthCheckRequest {
                service: "".to_string(),
            };

            if let Ok(response) = health_client.check(request).await {
                if response.into_inner().status == ServingStatus::Serving as i32 {
                    tracing::info!("Python server is healthy and ready!");
                    return Ok(managed_child);
                }
            }
        }

        if i % 10 == 0 {
            tracing::warn!(
                "Health probe {}/{} failed. Retrying every 500ms...",
                i + 1,
                max_retries
            );
        }

        sleep(Duration::from_millis(500)).await;
    }

    Err("Python server failed to start and become healthy in time.".into())
}
