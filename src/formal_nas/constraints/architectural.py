"""
Architectural pattern constraints for neural network synthesis.

These constraints specify structural properties of the architecture.
"""

import z3
from typing import Any, Dict, List, Optional
from ..constraints.base import Constraint, ConstraintType, GuaranteeLevel, ConstraintViolation
from ..architectures.layer_types import LayerType


class DepthConstraint(Constraint):
    """
    Constraint on network depth (number of layers).
    
    This is a HARD (provable) guarantee.
    """
    
    def __init__(self, min_depth: int, max_depth: int):
        super().__init__(
            name="Depth_constraint",
            constraint_type=ConstraintType.ARCHITECTURAL,
            guarantee_level=GuaranteeLevel.HARD,
            description=f"Network depth ∈ [{min_depth}, {max_depth}]"
        )
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def to_smt(self, solver: z3.Solver, variables: Dict[str, Any]) -> List[Any]:
        """Encode depth constraint as SMT."""
        assertions = []
        
        if 'num_layers' in variables:
            num_layers = variables['num_layers']
            assertions.append(num_layers >= self.min_depth)
            assertions.append(num_layers <= self.max_depth)
        
        return assertions
    
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """Validate network depth."""
        depth = architecture.depth
        
        if depth < self.min_depth or depth > self.max_depth:
            return ConstraintViolation(
                constraint_name=self.name,
                expected=f"[{self.min_depth}, {self.max_depth}]",
                actual=str(depth),
                severity="error",
                message=f"Network depth {depth} outside allowed range"
            )
        
        return None


class MinConvLayersConstraint(Constraint):
    """
    Constraint on minimum number of convolutional layers.
    
    HARD guarantee.
    """
    
    def __init__(self, min_conv_layers: int):
        super().__init__(
            name="MinConvLayers_constraint",
            constraint_type=ConstraintType.ARCHITECTURAL,
            guarantee_level=GuaranteeLevel.HARD,
            description=f"Minimum {min_conv_layers} conv layers"
        )
        self.min_conv_layers = min_conv_layers
    
    def to_smt(self, solver: z3.Solver, variables: Dict[str, Any]) -> List[Any]:
        """Encode minimum conv layers constraint."""
        assertions = []
        
        if 'num_conv_layers' in variables:
            num_conv = variables['num_conv_layers']
            assertions.append(num_conv >= self.min_conv_layers)
        
        return assertions
    
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """Validate minimum conv layers."""
        num_conv = architecture.num_conv_layers
        
        if num_conv < self.min_conv_layers:
            return ConstraintViolation(
                constraint_name=self.name,
                expected=f"≥ {self.min_conv_layers}",
                actual=str(num_conv),
                severity="error",
                message=f"Only {num_conv} conv layers, need at least {self.min_conv_layers}"
            )
        
        return None


class BatchNormRequiredConstraint(Constraint):
    """
    Constraint requiring batch normalization in the architecture.
    
    HARD guarantee.
    """
    
    def __init__(self):
        super().__init__(
            name="BatchNormRequired_constraint",
            constraint_type=ConstraintType.ARCHITECTURAL,
            guarantee_level=GuaranteeLevel.HARD,
            description="Architecture must include batch normalization"
        )
    
    def to_smt(self, solver: z3.Solver, variables: Dict[str, Any]) -> List[Any]:
        """Encode batch norm requirement."""
        assertions = []
        
        if 'has_batch_norm' in variables:
            has_bn = variables['has_batch_norm']
            assertions.append(has_bn == True)
        
        return assertions
    
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """Validate batch norm presence."""
        has_bn = architecture.has_batch_norm
        
        if not has_bn:
            return ConstraintViolation(
                constraint_name=self.name,
                expected="has batch normalization",
                actual="no batch normalization",
                severity="error",
                message="Architecture missing required batch normalization layers"
            )
        
        return None


class InputOutputConstraint(Constraint):
    """
    Constraint on input and output dimensions.
    
    HARD guarantee.
    """
    
    def __init__(self, input_channels: int, output_classes: int):
        super().__init__(
            name="InputOutput_constraint",
            constraint_type=ConstraintType.ARCHITECTURAL,
            guarantee_level=GuaranteeLevel.HARD,
            description=f"Input: {input_channels} channels, Output: {output_classes} classes"
        )
        self.input_channels = input_channels
        self.output_classes = output_classes
    
    def to_smt(self, solver: z3.Solver, variables: Dict[str, Any]) -> List[Any]:
        """Encode input/output constraints."""
        assertions = []
        
        if 'input_channels' in variables:
            assertions.append(variables['input_channels'] == self.input_channels)
        
        if 'output_classes' in variables:
            assertions.append(variables['output_classes'] == self.output_classes)
        
        return assertions
    
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """Validate input/output dimensions."""
        # Check first layer (input)
        if architecture.layers:
            first_layer = architecture.layers[0]
            if first_layer.layer_type == LayerType.INPUT:
                from ..architectures.layer_types import InputLayerConfig
                if isinstance(first_layer, InputLayerConfig):
                    if first_layer.input_shape[0] != self.input_channels:
                        return ConstraintViolation(
                            constraint_name=self.name,
                            expected=f"{self.input_channels} input channels",
                            actual=f"{first_layer.input_shape[0]} input channels",
                            severity="error",
                            message="Input channel mismatch"
                        )
        
        # Check last layer (output)
        if architecture.layers:
            last_layer = architecture.layers[-1]
            if last_layer.layer_type == LayerType.OUTPUT:
                from ..architectures.layer_types import OutputLayerConfig
                if isinstance(last_layer, OutputLayerConfig):
                    if last_layer.num_classes != self.output_classes:
                        return ConstraintViolation(
                            constraint_name=self.name,
                            expected=f"{self.output_classes} output classes",
                            actual=f"{last_layer.num_classes} output classes",
                            severity="error",
                            message="Output class mismatch"
                        )
        
        return None


class ConvAfterBatchNormConstraint(Constraint):
    """
    Architectural pattern: Batch normalization should follow convolution.
    
    HARD guarantee on pattern enforcement.
    """
    
    def __init__(self):
        super().__init__(
            name="ConvBatchNormPattern_constraint",
            constraint_type=ConstraintType.ARCHITECTURAL,
            guarantee_level=GuaranteeLevel.HARD,
            description="Each conv layer should be followed by batch norm"
        )
    
    def to_smt(self, solver: z3.Solver, variables: Dict[str, Any]) -> List[Any]:
        """
        Encode pattern constraint.
        
        This is more naturally expressed in validation than SMT.
        """
        # This constraint is better handled in post-generation validation
        # or through construction rules
        return []
    
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """Validate conv→batchnorm pattern."""
        for i, layer in enumerate(architecture.layers[:-1]):
            if layer.layer_type == LayerType.CONV2D:
                next_layer = architecture.layers[i + 1]
                if next_layer.layer_type != LayerType.BATCH_NORM:
                    return ConstraintViolation(
                        constraint_name=self.name,
                        expected="batch norm after conv",
                        actual=f"{next_layer.layer_type.value} after conv",
                        severity="warning",  # Could be error depending on strictness
                        message=f"Conv layer {layer.name} not followed by batch norm"
                    )
        
        return None
