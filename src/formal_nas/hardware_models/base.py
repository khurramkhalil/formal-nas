"""
Base hardware model interface.

This module defines the abstract interface for hardware resource models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class HardwareType(Enum):
    """Types of hardware platforms."""
    FPGA = "fpga"
    GPU = "gpu"
    CPU = "cpu"
    ASIC = "asic"
    TPU = "tpu"


@dataclass
class ResourceEstimate:
    """
    Resource usage estimate with confidence bounds.
    
    This represents model-dependent guarantees (Level 2 guarantees).
    """
    luts: int
    dsp_blocks: int
    bram_bits: int
    power_watts: float
    latency_ms: float
    
    # Error bounds (for bounded guarantees)
    lut_error_bound: float = 0.10  # ±10%
    dsp_error_bound: float = 0.10
    bram_error_bound: float = 0.05
    power_error_bound: float = 0.15  # ±15%
    latency_error_bound: float = 0.10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'luts': self.luts,
            'dsp_blocks': self.dsp_blocks,
            'bram_bits': self.bram_bits,
            'power_watts': self.power_watts,
            'latency_ms': self.latency_ms,
            'error_bounds': {
                'lut': self.lut_error_bound,
                'dsp': self.dsp_error_bound,
                'bram': self.bram_error_bound,
                'power': self.power_error_bound,
                'latency': self.latency_error_bound,
            }
        }
    
    def conservative_upper_bounds(self) -> Dict[str, float]:
        """
        Get conservative upper bounds with safety margins.
        
        These are the values used in formal guarantees.
        """
        return {
            'luts': self.luts * (1 + self.lut_error_bound),
            'dsp_blocks': self.dsp_blocks * (1 + self.dsp_error_bound),
            'bram_bits': self.bram_bits * (1 + self.bram_error_bound),
            'power_watts': self.power_watts * (1 + self.power_error_bound),
            'latency_ms': self.latency_ms * (1 + self.latency_error_bound),
        }


class HardwareModel(ABC):
    """
    Abstract base class for hardware resource models.
    
    Hardware models provide estimates of resource usage for neural network
    operations on specific hardware platforms.
    """
    
    def __init__(self, platform_name: str, hardware_type: HardwareType):
        self.platform_name = platform_name
        self.hardware_type = hardware_type
        self.model_version = "1.0.0"
        self.calibration_data = {}
    
    @abstractmethod
    def estimate_conv2d_resources(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        input_height: int,
        input_width: int,
        **kwargs
    ) -> Dict[str, float]:
        """
        Estimate resources for a 2D convolution operation.
        
        Returns:
            Dictionary with resource estimates (luts, dsp, bram, etc.)
        """
        pass
    
    @abstractmethod
    def estimate_dense_resources(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ) -> Dict[str, float]:
        """
        Estimate resources for a fully-connected layer.
        
        Returns:
            Dictionary with resource estimates
        """
        pass
    
    @abstractmethod
    def estimate_architecture_resources(
        self,
        architecture: 'NeuralArchitecture'
    ) -> ResourceEstimate:
        """
        Estimate total resources for a complete architecture.
        
        Returns:
            Complete resource estimate with error bounds
        """
        pass
    
    @abstractmethod
    def get_platform_limits(self) -> Dict[str, float]:
        """
        Get the hardware resource limits for this platform.
        
        Returns:
            Dictionary with maximum available resources
        """
        pass
    
    def validate_resources(
        self,
        estimate: ResourceEstimate,
        limits: Dict[str, float]
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Validate that resource estimates are within platform limits.
        
        Args:
            estimate: Resource estimate to validate
            limits: Platform resource limits
            
        Returns:
            (is_valid, violations_dict)
        """
        violations = {}
        
        # Use conservative upper bounds for validation
        conservative = estimate.conservative_upper_bounds()
        
        if 'max_luts' in limits and conservative['luts'] > limits['max_luts']:
            violations['luts'] = f"Exceeds limit: {conservative['luts']:.0f} > {limits['max_luts']:.0f}"
        
        if 'max_dsp' in limits and conservative['dsp_blocks'] > limits['max_dsp']:
            violations['dsp'] = f"Exceeds limit: {conservative['dsp_blocks']:.0f} > {limits['max_dsp']:.0f}"
        
        if 'max_bram' in limits and conservative['bram_bits'] > limits['max_bram']:
            violations['bram'] = f"Exceeds limit: {conservative['bram_bits']:.0f} > {limits['max_bram']:.0f}"
        
        if 'max_power' in limits and conservative['power_watts'] > limits['max_power']:
            violations['power'] = f"Exceeds limit: {conservative['power_watts']:.2f}W > {limits['max_power']:.2f}W"
        
        return len(violations) == 0, violations
    
    def __str__(self) -> str:
        return f"HardwareModel({self.platform_name}, {self.hardware_type.value})"
    
    def __repr__(self) -> str:
        return f"HardwareModel(platform='{self.platform_name}', type={self.hardware_type}, version='{self.model_version}')"
