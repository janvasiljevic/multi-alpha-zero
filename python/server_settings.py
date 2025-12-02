from pathlib import Path

import grpc

from generated import training_pb2


class ServerSettings:
    def __init__(self, grpc_msg: training_pb2.SetSettingsRequest):
        """
        Initialize the server settings with a gRPC message.

        :param grpc_msg: A SetSettingsRequest protobuf message.
        """
        self.base_directory: str = grpc_msg.base_directory
        self.max_samples: int = grpc_msg.max_samples
        self.batch_size: int = grpc_msg.batch_size

        self.samples_path = Path(self.base_directory) / "samples_new"
        self.pytorch_dir = Path(self.base_directory) / "torch"
        self.optimizer_dir = Path(self.base_directory) / "optimizer"
        self.onnx_dir = Path(self.base_directory) / "onnx"
        self.tensorboard_dir = Path(self.base_directory) / "tensorboard"

        for path in [self.samples_path, self.pytorch_dir, self.onnx_dir, self.optimizer_dir, self.tensorboard_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in self.__dict__.items() if not k.startswith('_')])})"


def settings_not_set_error():
    """
    Raises an error indicating that the initial settings have not been set.
    This is used to ensure that the server settings are initialized before any training data is sent.
    """
    raise grpc.RpcError(
        grpc.StatusCode.FAILED_PRECONDITION,
        "Initial settings have not been set. Please call SetInitialSettings first."
    )
