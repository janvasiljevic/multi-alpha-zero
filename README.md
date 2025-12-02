# Multiplayer Alpha Zero

This repository contains an implementation of a multiplayer version of the Alpha Zero algorithm, designed to handle games with more than two players. 
It was developed as part of my Master Thesis available [NOT YET PUBLISHED]().

Self-Play server is implemented in Rust and model training is done using Python and PyTorch.
Client-Server communication is handled via gRPC, samples are stored in Parquet files and model inference is done using ONNX.

Example config is shown in `config.yaml` file.

## Repository structure

The project is organized as a Cargo workspace, with some non Rust projects included:

- `/maz-trainer` Self play server and search logic
- `/maz-core` Traits and implementations for board and mappers that need to be implemented for each game
- `/game` Implementations of games and associated other game specific logic such as oracles.
- `/maz-util` Utility crate with common code
- `/analysis` Rust code for analysing and visualizing results 
- `/maz-arena` Pit trained models against each other
- `/python` The python client for training models
- `/shared` Proto files shared between Rust server and Python client
- `/runs` Empty folder for saving training runs
- `/play` Three servers and frontends for playing against the trained models. All follow Rust + React structure.
    - `/chess-tourney-(frontend|server)` For playing Three-Way Chess against the trained models and other players
    - `/hex-(frontend|server)` For playing Three-player Hex against the trained models and visualizing MCTS
    - `/chess-(frontend|server)` For playing Three-Way Chess against the trained models, visualizing MCTS and other debug features
- `/cache` For storing `tensorrt` engine caches
- `/docs` Documentation files