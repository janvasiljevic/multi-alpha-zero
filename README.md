# Multiplayer Alpha Zero

This repository contains an implementation of a multiplayer version of the Alpha Zero algorithm, designed to handle games with more than two players. 
It was developed as part of my Master Thesis available [NOT YET PUBLISHED]().

Self-Play server is implemented in Rust and model training is done using Python and PyTorch.
Client-Server communication is handled via gRPC, samples are stored in Parquet files and model inference is done using ONNX.

Example config is shown in `config.yaml` file.