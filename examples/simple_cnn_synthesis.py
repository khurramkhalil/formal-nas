"""
Simple CNN Synthesis Example

This example demonstrates basic constraint-compliant architecture synthesis
for a small CNN without complex hardware constraints.
"""

import sys
sys.path.insert(0, '/home/claude/formal_nas_poc/src')

from formal_nas.synthesis.smt_solver import FormalArchitectureSynthesizer
from formal_nas.constraints.base import ConstraintSpecification
from formal_nas.constraints.architectural import (
    DepthConstraint, MinConvLayersConstraint,
    BatchNormRequiredConstraint, InputOutputConstraint
)
from formal_nas.hardware_models.fpga_models import SimpleFPGAModel
from formal_nas.constraints.hardware import LUTConstraint, PowerConstraint


def main():
    print("=" * 80)
    print("FORMAL NEURAL ARCHITECTURE SYNTHESIS - Proof of Concept")
    print("Simple CNN Example")
    print("=" * 80)
    print()
    
    # Define hardware platform (simplified FPGA)
    print("Setting up hardware model...")
    hardware_model = SimpleFPGAModel(
        max_luts=100000,
        max_dsp=500,
        max_bram=10000000,
        max_power=10.0
    )
    print(f"  Platform: {hardware_model.platform_name}")
    print(f"  Max LUTs: {hardware_model.max_luts:,}")
    print(f"  Max Power: {hardware_model.max_power}W")
    print()
    
    # Define constraints
    print("Defining constraints...")
    constraint_spec = ConstraintSpecification()
    
    # Architectural constraints (HARD guarantees)
    constraint_spec.add_constraint(
        DepthConstraint(min_depth=5, max_depth=15)
    )
    constraint_spec.add_constraint(
        MinConvLayersConstraint(min_conv_layers=2)
    )
    constraint_spec.add_constraint(
        BatchNormRequiredConstraint()
    )
    constraint_spec.add_constraint(
        InputOutputConstraint(input_channels=3, output_classes=10)
    )
    
    # Hardware constraints (BOUNDED guarantees)
    constraint_spec.add_constraint(
        LUTConstraint(max_luts=80000, hardware_model=hardware_model)
    )
    constraint_spec.add_constraint(
        PowerConstraint(max_power_watts=8.0, hardware_model=hardware_model)
    )
    
    print(constraint_spec.summary())
    print()
    
    # Create synthesizer
    synthesizer = FormalArchitectureSynthesizer(
        hardware_model=hardware_model,
        constraint_spec=constraint_spec,
        input_channels=3,
        output_classes=10
    )
    
    # Synthesize architectures
    print("=" * 80)
    print("SYNTHESIS PHASE")
    print("=" * 80)
    print()
    
    architectures = synthesizer.synthesize(
        num_solutions=3,
        timeout_seconds=30
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    for i, (arch, cert) in enumerate(architectures, 1):
        print(f"\n{'─' * 80}")
        print(f"Architecture {i}")
        print(f"{'─' * 80}")
        print()
        
        print(arch.summary())
        print()
        
        print(cert)
        print()
        
        # Validate constraints
        violations = constraint_spec.validate_all(arch)
        if violations:
            print("⚠️  Constraint Violations:")
            for violation in violations:
                print(f"  - {violation.message}")
        else:
            print("✓ All constraints satisfied!")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully synthesized {len(architectures)} constraint-compliant architectures")
    print("All architectures have formal guarantees:")
    print("  • Structural properties (HARD): provably satisfied")
    print("  • Resource bounds (BOUNDED): satisfied within model error margins")
    print()


if __name__ == "__main__":
    main()
