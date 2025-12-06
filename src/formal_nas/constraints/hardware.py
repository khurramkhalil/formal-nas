"""
Hardware-specific constraints for architecture synthesis.

These constraints ensure generated architectures meet hardware resource limits.
"""

import z3
from typing import Any, Dict, List, Optional
from ..constraints.base import Constraint, ConstraintType, GuaranteeLevel, ConstraintViolation
from ..hardware_models.base import HardwareModel, ResourceEstimate


class HardwareResourceConstraint(Constraint):
    """
    Constraint on hardware resource usage.
    
    This provides BOUNDED guarantees (Level 2) based on hardware models.
    """
    
    def __init__(
        self,
        name: str,
        resource_type: str,  # 'luts', 'dsp', 'bram', 'power'
        max_value: float,
        hardware_model: HardwareModel,
        description: Optional[str] = None
    ):
        super().__init__(
            name=name,
            constraint_type=ConstraintType.HARDWARE,
            guarantee_level=GuaranteeLevel.BOUNDED,  # Model-dependent
            description=description or f"{resource_type} ≤ {max_value}"
        )
        self.resource_type = resource_type
        self.max_value = max_value
        self.hardware_model = hardware_model
    
    def to_smt(self, solver: z3.Solver, variables: Dict[str, Any]) -> List[Any]:
        """
        Encode hardware resource constraint as SMT.
        
        Note: This is a simplified encoding. Full implementation would
        track resource usage through architecture construction.
        """
        assertions = []
        
        # Get the resource variable for this type
        resource_var_name = f"total_{self.resource_type}"
        if resource_var_name in variables:
            resource_var = variables[resource_var_name]
            
            # Conservative bound: use upper bound with safety margin
            conservative_limit = self.max_value * 0.9  # 10% safety margin
            assertions.append(resource_var <= conservative_limit)
        
        return assertions
    
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """
        Validate architecture against hardware resource constraint.
        
        Uses hardware model to estimate resources and checks against limits.
        """
        # Get resource estimate from hardware model
        estimate = self.hardware_model.estimate_architecture_resources(architecture)
        
        # Get conservative upper bound
        conservative_bounds = estimate.conservative_upper_bounds()
        
        # Check the specific resource type
        actual_value = conservative_bounds.get(self.resource_type, 0)
        
        if actual_value > self.max_value:
            return ConstraintViolation(
                constraint_name=self.name,
                expected=f"≤ {self.max_value}",
                actual=f"{actual_value:.2f}",
                severity="error",
                message=f"{self.resource_type} usage {actual_value:.2f} exceeds limit {self.max_value}"
            )
        
        return None


class LUTConstraint(HardwareResourceConstraint):
    """Constraint on LUT (Logic Unit) utilization."""
    
    def __init__(self, max_luts: int, hardware_model: HardwareModel):
        super().__init__(
            name="LUT_constraint",
            resource_type="luts",
            max_value=max_luts,
            hardware_model=hardware_model,
            description=f"LUT usage ≤ {max_luts}"
        )


class DSPConstraint(HardwareResourceConstraint):
    """Constraint on DSP block utilization."""
    
    def __init__(self, max_dsp: int, hardware_model: HardwareModel):
        super().__init__(
            name="DSP_constraint",
            resource_type="dsp_blocks",
            max_value=max_dsp,
            hardware_model=hardware_model,
            description=f"DSP blocks ≤ {max_dsp}"
        )


class BRAMConstraint(HardwareResourceConstraint):
    """Constraint on BRAM (Block RAM) utilization."""
    
    def __init__(self, max_bram_bits: int, hardware_model: HardwareModel):
        super().__init__(
            name="BRAM_constraint",
            resource_type="bram_bits",
            max_value=max_bram_bits,
            hardware_model=hardware_model,
            description=f"BRAM ≤ {max_bram_bits} bits"
        )


class PowerConstraint(HardwareResourceConstraint):
    """Constraint on power consumption."""
    
    def __init__(self, max_power_watts: float, hardware_model: HardwareModel):
        super().__init__(
            name="Power_constraint",
            resource_type="power_watts",
            max_value=max_power_watts,
            hardware_model=hardware_model,
            description=f"Power ≤ {max_power_watts}W"
        )


class LatencyConstraint(HardwareResourceConstraint):
    """Constraint on inference latency."""
    
    def __init__(self, max_latency_ms: float, hardware_model: HardwareModel):
        super().__init__(
            name="Latency_constraint",
            resource_type="latency_ms",
            max_value=max_latency_ms,
            hardware_model=hardware_model,
            description=f"Latency ≤ {max_latency_ms}ms"
        )
