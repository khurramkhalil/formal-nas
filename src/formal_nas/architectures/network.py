"""
Neural network architecture representation.

This module defines how complete architectures are represented and manipulated.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .layer_types import (
    LayerConfig, LayerType, InputLayerConfig, Conv2DLayerConfig,
    DenseLayerConfig, PoolingLayerConfig, BatchNormLayerConfig,
    ActivationLayerConfig, FlattenLayerConfig, OutputLayerConfig
)


@dataclass
class NeuralArchitecture:
    """
    Representation of a complete neural network architecture.
    
    This class provides a unified interface for working with neural architectures
    and computing their properties.
    """
    
    layers: List[LayerConfig] = field(default_factory=list)
    name: str = "generated_architecture"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_layer(self, layer: LayerConfig):
        """Add a layer to the architecture."""
        self.layers.append(layer)
    
    def get_layer(self, layer_id: int) -> Optional[LayerConfig]:
        """Get a layer by its ID."""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        return None
    
    def get_layers_by_type(self, layer_type: LayerType) -> List[LayerConfig]:
        """Get all layers of a specific type."""
        return [layer for layer in self.layers if layer.layer_type == layer_type]
    
    @property
    def depth(self) -> int:
        """Total number of layers."""
        return len(self.layers)
    
    @property
    def num_conv_layers(self) -> int:
        """Number of convolutional layers."""
        return len(self.get_layers_by_type(LayerType.CONV2D))
    
    @property
    def num_dense_layers(self) -> int:
        """Number of dense layers."""
        return len(self.get_layers_by_type(LayerType.DENSE))
    
    @property
    def has_batch_norm(self) -> bool:
        """Check if architecture has batch normalization."""
        return len(self.get_layers_by_type(LayerType.BATCH_NORM)) > 0
    
    def total_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'num_parameters'):
                total += layer.num_parameters()
        return total
    
    def compute_macs(self, input_shape: Tuple[int, ...]) -> int:
        """
        Compute total multiply-accumulate operations.
        
        Args:
            input_shape: Input tensor shape (channels, height, width)
        """
        total_macs = 0
        current_shape = input_shape
        
        for layer in self.layers:
            if isinstance(layer, Conv2DLayerConfig):
                if len(current_shape) == 3:
                    _, h, w = current_shape
                    total_macs += layer.num_macs(h, w)
                    h_out, w_out = layer.compute_output_size(h, w)
                    current_shape = (layer.out_channels, h_out, w_out)
            
            elif isinstance(layer, DenseLayerConfig):
                total_macs += layer.num_macs()
                current_shape = (layer.output_dim,)
            
            elif isinstance(layer, PoolingLayerConfig):
                if len(current_shape) == 3:
                    c, h, w = current_shape
                    h_out, w_out = layer.compute_output_size(h, w)
                    current_shape = (c, h_out, w_out)
            
            elif isinstance(layer, FlattenLayerConfig):
                if len(current_shape) == 3:
                    c, h, w = current_shape
                    current_shape = (c * h * w,)
        
        return total_macs
    
    def get_topology_signature(self) -> str:
        """
        Generate a signature representing the architecture topology.
        
        This can be used for architecture comparison or caching.
        """
        signature_parts = []
        for layer in self.layers:
            if isinstance(layer, Conv2DLayerConfig):
                signature_parts.append(
                    f"C{layer.out_channels}K{layer.kernel_size[0]}"
                )
            elif isinstance(layer, DenseLayerConfig):
                signature_parts.append(f"D{layer.output_dim}")
            elif isinstance(layer, PoolingLayerConfig):
                signature_parts.append(f"P{layer.kernel_size[0]}")
            elif isinstance(layer, BatchNormLayerConfig):
                signature_parts.append("BN")
            elif isinstance(layer, ActivationLayerConfig):
                signature_parts.append(layer.activation_type.value[:1].upper())
            elif isinstance(layer, FlattenLayerConfig):
                signature_parts.append("F")
        
        return "-".join(signature_parts)
    
    def validate_connectivity(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that layer connections are valid.
        
        Returns:
            (is_valid, error_message)
        """
        if not self.layers:
            return False, "Architecture has no layers"
        
        if self.layers[0].layer_type != LayerType.INPUT:
            return False, "First layer must be INPUT"
        
        # Check for proper input/output dimension matching
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            # Additional validation logic can be added here
            # For now, basic type checking
            if isinstance(current_layer, Conv2DLayerConfig) and isinstance(next_layer, Conv2DLayerConfig):
                if current_layer.out_channels != next_layer.in_channels:
                    return False, f"Channel mismatch between {current_layer.name} and {next_layer.name}"
        
        return True, None
    
    def summary(self) -> str:
        """Generate a human-readable summary of the architecture."""
        lines = [f"Architecture: {self.name}"]
        lines.append("=" * 80)
        lines.append(f"{'Layer':<30} {'Type':<15} {'Details':<35}")
        lines.append("-" * 80)
        
        for layer in self.layers:
            details = ""
            if isinstance(layer, Conv2DLayerConfig):
                details = f"in={layer.in_channels}, out={layer.out_channels}, k={layer.kernel_size}"
            elif isinstance(layer, DenseLayerConfig):
                details = f"in={layer.input_dim}, out={layer.output_dim}"
            elif isinstance(layer, BatchNormLayerConfig):
                details = f"features={layer.num_features}"
            elif isinstance(layer, PoolingLayerConfig):
                details = f"type={layer.pooling_type.value}, k={layer.kernel_size}"
            
            lines.append(f"{layer.name:<30} {layer.layer_type.value:<15} {details:<35}")
        
        lines.append("-" * 80)
        lines.append(f"Total layers: {self.depth}")
        lines.append(f"Conv layers: {self.num_conv_layers}")
        lines.append(f"Dense layers: {self.num_dense_layers}")
        lines.append(f"Total parameters: {self.total_parameters():,}")
        lines.append(f"Topology: {self.get_topology_signature()}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary representation."""
        return {
            'name': self.name,
            'depth': self.depth,
            'topology': self.get_topology_signature(),
            'total_parameters': self.total_parameters(),
            'num_conv_layers': self.num_conv_layers,
            'num_dense_layers': self.num_dense_layers,
            'has_batch_norm': self.has_batch_norm,
            'layers': [
                {
                    'id': layer.layer_id,
                    'type': layer.layer_type.value,
                    'name': layer.name,
                }
                for layer in self.layers
            ],
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        return self.summary()
    
    def __repr__(self) -> str:
        return f"NeuralArchitecture(name='{self.name}', depth={self.depth}, topology='{self.get_topology_signature()}')"
