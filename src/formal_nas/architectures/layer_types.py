"""
Neural network layer type definitions.

This module defines the basic building blocks for neural architectures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class LayerType(Enum):
    """Supported layer types in the architecture."""
    INPUT = "input"
    CONV2D = "conv2d"
    DENSE = "dense"
    POOLING = "pooling"
    BATCH_NORM = "batch_norm"
    ACTIVATION = "activation"
    DROPOUT = "dropout"
    FLATTEN = "flatten"
    OUTPUT = "output"
    ADD = "add"      # Element-wise addition (Residuals)
    CONCAT = "concat" # Channel concatenation (Inception)


class ActivationType(Enum):
    """Supported activation functions."""
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    NONE = "none"


class PoolingType(Enum):
    """Supported pooling operations."""
    MAX = "max"
    AVERAGE = "average"
    GLOBAL_AVERAGE = "global_average"


@dataclass
class LayerConfig:
    """
    Base configuration for all layer types.
    
    This contains the common properties that all layers share.
    """
    layer_id: int
    layer_type: LayerType
    name: str
    
    def __str__(self) -> str:
        return f"{self.name} ({self.layer_type.value})"


@dataclass
class InputLayerConfig(LayerConfig):
    """Configuration for input layer."""
    input_shape: Tuple[int, ...]  # (channels, height, width) or (features,)
    
    def __init__(self, layer_id: int, input_shape: Tuple[int, ...], name: str = "input"):
        super().__init__(layer_id, LayerType.INPUT, name)
        self.input_shape = input_shape


@dataclass
class Conv2DLayerConfig(LayerConfig):
    """Configuration for 2D convolutional layer."""
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    
    def __init__(
        self,
        layer_id: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        name: Optional[str] = None
    ):
        name = name or f"conv2d_{layer_id}"
        super().__init__(layer_id, LayerType.CONV2D, name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def compute_output_size(self, input_height: int, input_width: int) -> Tuple[int, int]:
        """Compute output spatial dimensions."""
        out_height = (input_height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (input_width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return out_height, out_width
    
    def num_parameters(self) -> int:
        """Calculate number of parameters in this layer."""
        # weights + biases
        return (self.kernel_size[0] * self.kernel_size[1] * self.in_channels * self.out_channels + 
                self.out_channels)
    
    def num_macs(self, input_height: int, input_width: int) -> int:
        """Calculate number of multiply-accumulate operations."""
        out_height, out_width = self.compute_output_size(input_height, input_width)
        return (self.kernel_size[0] * self.kernel_size[1] * self.in_channels * 
                self.out_channels * out_height * out_width)


@dataclass
class DenseLayerConfig(LayerConfig):
    """Configuration for fully-connected (dense) layer."""
    input_dim: int
    output_dim: int
    
    def __init__(
        self,
        layer_id: int,
        input_dim: int,
        output_dim: int,
        name: Optional[str] = None
    ):
        name = name or f"dense_{layer_id}"
        super().__init__(layer_id, LayerType.DENSE, name)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def num_parameters(self) -> int:
        """Calculate number of parameters in this layer."""
        return self.input_dim * self.output_dim + self.output_dim
    
    def num_macs(self) -> int:
        """Calculate number of multiply-accumulate operations."""
        return self.input_dim * self.output_dim


@dataclass
class PoolingLayerConfig(LayerConfig):
    """Configuration for pooling layer."""
    pooling_type: PoolingType
    kernel_size: Tuple[int, int]
    stride: Optional[Tuple[int, int]] = None
    
    def __init__(
        self,
        layer_id: int,
        pooling_type: PoolingType,
        kernel_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]] = None,
        name: Optional[str] = None
    ):
        name = name or f"pool_{layer_id}"
        super().__init__(layer_id, LayerType.POOLING, name)
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
    
    def compute_output_size(self, input_height: int, input_width: int) -> Tuple[int, int]:
        """Compute output spatial dimensions."""
        out_height = (input_height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (input_width - self.kernel_size[1]) // self.stride[1] + 1
        return out_height, out_width


@dataclass
class BatchNormLayerConfig(LayerConfig):
    """Configuration for batch normalization layer."""
    num_features: int
    
    def __init__(
        self,
        layer_id: int,
        num_features: int,
        name: Optional[str] = None
    ):
        name = name or f"bn_{layer_id}"
        super().__init__(layer_id, LayerType.BATCH_NORM, name)
        self.num_features = num_features
    
    def num_parameters(self) -> int:
        """Calculate number of parameters (gamma, beta)."""
        return 2 * self.num_features


@dataclass
class ActivationLayerConfig(LayerConfig):
    """Configuration for activation layer."""
    activation_type: ActivationType
    
    def __init__(
        self,
        layer_id: int,
        activation_type: ActivationType,
        name: Optional[str] = None
    ):
        name = name or f"{activation_type.value}_{layer_id}"
        super().__init__(layer_id, LayerType.ACTIVATION, name)
        self.activation_type = activation_type


@dataclass
class FlattenLayerConfig(LayerConfig):
    """Configuration for flatten layer."""
    
    def __init__(self, layer_id: int, name: Optional[str] = None):
        name = name or f"flatten_{layer_id}"
        super().__init__(layer_id, LayerType.FLATTEN, name)


@dataclass
class DropoutLayerConfig(LayerConfig):
    """Configuration for dropout layer."""
    dropout_rate: float
    
    def __init__(
        self,
        layer_id: int,
        dropout_rate: float,
        name: Optional[str] = None
    ):
        name = name or f"dropout_{layer_id}"
        super().__init__(layer_id, LayerType.DROPOUT, name)
        self.dropout_rate = dropout_rate


@dataclass
class OutputLayerConfig(LayerConfig):
    """Configuration for output layer."""
    num_classes: int
    
    def __init__(
        self,
        layer_id: int,
        num_classes: int,
        name: str = "output"
    ):
        super().__init__(layer_id, LayerType.OUTPUT, name)
        self.num_classes = num_classes


@dataclass
class AddLayerConfig(LayerConfig):
    """Configuration for element-wise addition (Residual connection)."""
    
    def __init__(self, layer_id: int, name: Optional[str] = None):
        name = name or f"add_{layer_id}"
        super().__init__(layer_id, LayerType.ADD, name)


@dataclass
class ConcatLayerConfig(LayerConfig):
    """Configuration for concatenation layer."""
    
    def __init__(self, layer_id: int, name: Optional[str] = None):
        name = name or f"concat_{layer_id}"
        super().__init__(layer_id, LayerType.CONCAT, name)

