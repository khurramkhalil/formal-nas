"""
Base classes for constraint specification.

This module defines the abstract interfaces for constraints that will be
used in the formal synthesis process.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints in the synthesis process."""
    HARDWARE = "hardware"
    ARCHITECTURAL = "architectural"
    TEMPORAL = "temporal"
    PERFORMANCE = "performance"


class GuaranteeLevel(Enum):
    """
    Levels of formal guarantees provided by constraints.
    
    HARD: Provably satisfied (structural properties)
    BOUNDED: Satisfied within model accuracy bounds (resource properties)
    SOFT: Best-effort satisfaction (optimization objectives)
    """
    HARD = "hard"
    BOUNDED = "bounded"
    SOFT = "soft"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation with details."""
    constraint_name: str
    expected: Any
    actual: Any
    severity: str  # "error", "warning"
    message: str


class Constraint(ABC):
    """
    Abstract base class for all constraints.
    
    Each constraint must be able to:
    1. Encode itself as SMT assertions
    2. Validate against an architecture
    3. Provide a description of what it constrains
    """
    
    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType,
        guarantee_level: GuaranteeLevel,
        description: Optional[str] = None
    ):
        self.name = name
        self.constraint_type = constraint_type
        self.guarantee_level = guarantee_level
        self.description = description or f"{name} constraint"
        
    @abstractmethod
    def to_smt(self, solver: Any, variables: Dict[str, Any]) -> List[Any]:
        """
        Convert this constraint to SMT assertions.
        
        Args:
            solver: The Z3 solver instance
            variables: Dictionary of SMT variables available for this constraint
            
        Returns:
            List of SMT assertions
        """
        pass
    
    @abstractmethod
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """
        Validate if an architecture satisfies this constraint.
        
        Args:
            architecture: The architecture to validate
            
        Returns:
            None if constraint is satisfied, ConstraintViolation otherwise
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} ({self.constraint_type.value}, {self.guarantee_level.value})"
    
    def __repr__(self) -> str:
        return f"Constraint(name='{self.name}', type={self.constraint_type}, level={self.guarantee_level})"


class CompositeConstraint(Constraint):
    """
    A constraint composed of multiple sub-constraints.
    
    This allows for hierarchical constraint specification and modular composition.
    """
    
    def __init__(
        self,
        name: str,
        constraints: List[Constraint],
        constraint_type: ConstraintType,
        guarantee_level: GuaranteeLevel,
        description: Optional[str] = None
    ):
        super().__init__(name, constraint_type, guarantee_level, description)
        self.constraints = constraints
        
    def to_smt(self, solver: Any, variables: Dict[str, Any]) -> List[Any]:
        """Combine SMT assertions from all sub-constraints."""
        assertions = []
        for constraint in self.constraints:
            assertions.extend(constraint.to_smt(solver, variables))
        return assertions
    
    def validate(self, architecture: 'NeuralArchitecture') -> Optional[ConstraintViolation]:
        """Validate all sub-constraints."""
        violations = []
        for constraint in self.constraints:
            violation = constraint.validate(architecture)
            if violation:
                violations.append(violation)
        
        if violations:
            # Return the first violation (could be modified to return all)
            return violations[0]
        return None
    
    def add_constraint(self, constraint: Constraint):
        """Add a new constraint to this composite."""
        self.constraints.append(constraint)
    
    def get_all_constraints(self) -> List[Constraint]:
        """Get all constraints including nested ones."""
        all_constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, CompositeConstraint):
                all_constraints.extend(constraint.get_all_constraints())
            else:
                all_constraints.append(constraint)
        return all_constraints


@dataclass
class ConstraintSpecification:
    """
    Complete specification of all constraints for architecture synthesis.
    
    This is the main interface for users to specify their requirements.
    """
    
    hardware_constraints: List[Constraint] = field(default_factory=list)
    architectural_constraints: List[Constraint] = field(default_factory=list)
    temporal_constraints: List[Constraint] = field(default_factory=list)
    performance_constraints: List[Constraint] = field(default_factory=list)
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the appropriate category."""
        if constraint.constraint_type == ConstraintType.HARDWARE:
            self.hardware_constraints.append(constraint)
        elif constraint.constraint_type == ConstraintType.ARCHITECTURAL:
            self.architectural_constraints.append(constraint)
        elif constraint.constraint_type == ConstraintType.TEMPORAL:
            self.temporal_constraints.append(constraint)
        elif constraint.constraint_type == ConstraintType.PERFORMANCE:
            self.performance_constraints.append(constraint)
    
    def get_all_constraints(self) -> List[Constraint]:
        """Get all constraints from all categories."""
        return (
            self.hardware_constraints +
            self.architectural_constraints +
            self.temporal_constraints +
            self.performance_constraints
        )
    
    def get_hard_constraints(self) -> List[Constraint]:
        """Get only hard (provable) constraints."""
        return [c for c in self.get_all_constraints() 
                if c.guarantee_level == GuaranteeLevel.HARD]
    
    def get_bounded_constraints(self) -> List[Constraint]:
        """Get bounded (model-dependent) constraints."""
        return [c for c in self.get_all_constraints() 
                if c.guarantee_level == GuaranteeLevel.BOUNDED]
    
    def validate_all(self, architecture: 'NeuralArchitecture') -> List[ConstraintViolation]:
        """Validate all constraints against an architecture."""
        violations = []
        for constraint in self.get_all_constraints():
            violation = constraint.validate(architecture)
            if violation:
                violations.append(violation)
        return violations
    
    def summary(self) -> str:
        """Generate a human-readable summary of all constraints."""
        lines = ["Constraint Specification:"]
        lines.append(f"  Hardware: {len(self.hardware_constraints)}")
        lines.append(f"  Architectural: {len(self.architectural_constraints)}")
        lines.append(f"  Temporal: {len(self.temporal_constraints)}")
        lines.append(f"  Performance: {len(self.performance_constraints)}")
        lines.append(f"  Total: {len(self.get_all_constraints())}")
        
        hard = len(self.get_hard_constraints())
        bounded = len(self.get_bounded_constraints())
        lines.append(f"\nGuarantee Levels:")
        lines.append(f"  Hard (provable): {hard}")
        lines.append(f"  Bounded (model-dependent): {bounded}")
        
        return "\n".join(lines)
