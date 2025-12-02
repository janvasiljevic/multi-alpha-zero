from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SetSettingsRequest(_message.Message):
    __slots__ = ("base_directory", "max_samples", "batch_size", "learning_rate", "model_key", "current_training_step", "weight_decay", "policy_loss_weight", "z_value_loss_weight", "q_value_loss_weight", "num_of_aux_features", "warmup_steps")
    BASE_DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    MAX_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_TRAINING_STEP_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    POLICY_LOSS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    Z_VALUE_LOSS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    Q_VALUE_LOSS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    NUM_OF_AUX_FEATURES_FIELD_NUMBER: _ClassVar[int]
    WARMUP_STEPS_FIELD_NUMBER: _ClassVar[int]
    base_directory: str
    max_samples: int
    batch_size: int
    learning_rate: float
    model_key: str
    current_training_step: int
    weight_decay: float
    policy_loss_weight: float
    z_value_loss_weight: float
    q_value_loss_weight: float
    num_of_aux_features: int
    warmup_steps: int
    def __init__(self, base_directory: _Optional[str] = ..., max_samples: _Optional[int] = ..., batch_size: _Optional[int] = ..., learning_rate: _Optional[float] = ..., model_key: _Optional[str] = ..., current_training_step: _Optional[int] = ..., weight_decay: _Optional[float] = ..., policy_loss_weight: _Optional[float] = ..., z_value_loss_weight: _Optional[float] = ..., q_value_loss_weight: _Optional[float] = ..., num_of_aux_features: _Optional[int] = ..., warmup_steps: _Optional[int] = ...) -> None: ...

class InitialSettingsResponse(_message.Message):
    __slots__ = ("model_path",)
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    model_path: str
    def __init__(self, model_path: _Optional[str] = ...) -> None: ...

class TrainingSamplesUpdate(_message.Message):
    __slots__ = ("parquet_file_path",)
    PARQUET_FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    parquet_file_path: str
    def __init__(self, parquet_file_path: _Optional[str] = ...) -> None: ...

class SendTrainingDataRequest(_message.Message):
    __slots__ = ("update",)
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    update: TrainingSamplesUpdate
    def __init__(self, update: _Optional[_Union[TrainingSamplesUpdate, _Mapping]] = ...) -> None: ...

class SendTrainingDataResponse(_message.Message):
    __slots__ = ("samples_loaded_in_stream", "total_samples_in_memory")
    SAMPLES_LOADED_IN_STREAM_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SAMPLES_IN_MEMORY_FIELD_NUMBER: _ClassVar[int]
    samples_loaded_in_stream: int
    total_samples_in_memory: int
    def __init__(self, samples_loaded_in_stream: _Optional[int] = ..., total_samples_in_memory: _Optional[int] = ...) -> None: ...

class TrainModelRequest(_message.Message):
    __slots__ = ("epochs", "batch_size")
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    epochs: int
    batch_size: int
    def __init__(self, epochs: _Optional[int] = ..., batch_size: _Optional[int] = ...) -> None: ...

class TrainModelResponse(_message.Message):
    __slots__ = ("success", "new_model_path", "duration_seconds", "total_training_steps")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    NEW_MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TRAINING_STEPS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    new_model_path: str
    duration_seconds: float
    total_training_steps: int
    def __init__(self, success: bool = ..., new_model_path: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., total_training_steps: _Optional[int] = ...) -> None: ...

class TensorboardScalarRequest(_message.Message):
    __slots__ = ("tag", "value", "step")
    TAG_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    tag: str
    value: float
    step: int
    def __init__(self, tag: _Optional[str] = ..., value: _Optional[float] = ..., step: _Optional[int] = ...) -> None: ...

class TensorboardHistogramRequest(_message.Message):
    __slots__ = ("tag", "values", "step")
    TAG_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    tag: str
    values: _containers.RepeatedScalarFieldContainer[float]
    step: int
    def __init__(self, tag: _Optional[str] = ..., values: _Optional[_Iterable[float]] = ..., step: _Optional[int] = ...) -> None: ...

class TensorboardMultipleScalarsRequest(_message.Message):
    __slots__ = ("tag", "values", "step")
    class ValuesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TAG_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    tag: str
    values: _containers.ScalarMap[str, float]
    step: int
    def __init__(self, tag: _Optional[str] = ..., values: _Optional[_Mapping[str, float]] = ..., step: _Optional[int] = ...) -> None: ...

class TensorboardResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...
