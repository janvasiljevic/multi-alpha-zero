#!/bin/bash
set -e

PROTO_DIR="./../shared"
GENERATED_DIR="./generated"

echo "Setting up output directory: ${GENERATED_DIR}"
mkdir -p ${GENERATED_DIR}
touch ${GENERATED_DIR}/__init__.py

echo "Generating Python gRPC code from training.proto..."

uv run python -m grpc_tools.protoc \
  -I generated=${PROTO_DIR} \
  --python_out=. \
  --pyi_out=. \
  --grpc_python_out=. \
  ${PROTO_DIR}/training.proto \
  ${PROTO_DIR}/health.proto

echo "Code generation complete."