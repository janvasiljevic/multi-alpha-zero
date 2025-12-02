import logging
import socket
from concurrent import futures

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from generated import training_pb2
from generated import training_pb2_grpc
from logging_setup import setup_logging
from server import TrainingCoordinatorService

import torch.multiprocessing as mp


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a TCP port is in use on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def serve():
    setup_logging(tag="server.py")

    try:
        mp.set_start_method('spawn', force=True)
        logging.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        # This can happen if the context is already set.
        # It's fine if 'spawn' is already the method.
        pass

    """Starts the gRPC server."""
    keepalive_options = [
        # Send a PING every 30 seconds if there are no calls
        ('grpc.keepalive_time_ms', 30000),
        # Wait 10 seconds for a PING ACK before closing the connection
        ('grpc.keepalive_timeout_ms', 10000),
        # Do NOT send pings if there is an active stream/RPC on the connection.
        ('grpc.keepalive_permit_without_calls', 0),
        # Number of bad PINGs to tolerate before closing connection
        ('grpc.http2.max_pings_without_data', 2),
        # Minimum time between PINGs to prevent spamming
        ('grpc.http2.min_time_between_pings_ms', 10000),
    ]

    # Test: Not using keepalive options for now?

    # Keep thread pool size to 1 to avoid issues with multiple threads
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1)) #, options=keepalive_options)

    training_pb2_grpc.add_TrainingCoordinatorServicer_to_server(
        TrainingCoordinatorService(), server
    )

    # Create a health servicer and add it to the server
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    service_name = training_pb2.DESCRIPTOR.services_by_name['TrainingCoordinator'].full_name
    health_servicer.set(service_name, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    # check if port 50051 is already in use
    port = 50051

    if is_port_in_use(port):
        logging.error(f"Port {port} is already in use. Exiting.")

        exit(1)

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logging.info(f"Python Training Server started on port {port}.")
    logging.info(f"Service '{service_name}' is marked as SERVING.")

    server.wait_for_termination()


if __name__ == '__main__':
    serve()
