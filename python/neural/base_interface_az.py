from abc import ABC, abstractmethod

from torch import nn


class AlphaZeroNet(nn.Module, ABC):
    @abstractmethod
    def state_shape(self):
        """
        The shape of the input state tensor.
        This should be overridden by subclasses to return the correct shape.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def policy_shape(self):
        """
        The shape of the output policy tensor.
        This should be overridden by subclasses to return the correct shape.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def value_shape(self):
        """
        The shape of the output value tensor.
        Default implementation returns a 1D tensor with 3 elements (for 3 players).
        Override if a different shape is needed.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def device(self):
        """
        The device on which the model is located.
        This should be overridden by subclasses to return the correct device.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def log_gradients(self, epoch: int):
        """
        Log gradients of the model's parameters.
        This method should be implemented by subclasses to log gradients.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def debug(self):
        """
        This can literally do anything for debugging purposes.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def pre_onnx_export(self):
        """
        If the model needs to do anything special before being exported to ONNX, do it here.
        """
        raise NotImplementedError("Subclasses must implement this method.")
