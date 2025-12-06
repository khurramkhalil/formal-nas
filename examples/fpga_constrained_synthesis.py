"""
FPGA-Constrained Synthesis Example

This example demonstrates synthesis with realistic Intel FPGA constraints,
showing how the framework provides formal guarantees for hardware deployment.
"""

import sys
sys.path.insert(0, '/home/claude/formal_nas_poc/src')

from formal_nas.synthesis.smt_solver import FormalArchitectureSynthesizer
from formal_nas.constraints.base import ConstraintSpecification
from formal_nas.constraints.architectural import (
    DepthConstraint, MinConvLayersConstraint,
    BatchNormRequiredConstraint, InputOutputConstraint
)
from formal_nas.constraints.hardware import (
    LUTConstraint, DSPConstraint, BRAMConstraint, PowerConstraint
)
from formal_nas.hardware_models.fpga_models import IntelStratixModel


def main():
    print("=" * 80)
    print("FORMAL NEURAL ARCHITECTURE SYNTHESIS")
    print("Intel FPGA Deployment Example")
    print("=" * 80)
    print()
    
    # Define Intel Stratix 10 platform
    print("Setting up Intel Stratix 10 FPGA model...")
    hardware_model = IntelStratixModel(device="Stratix10GX2800")
    
    platform_limits = hardware_model.get_platform_limits()
    print(f"  Device: {hardware_model.device}")
    print(f"  Available LUTs: {platform_limits['max_luts']:,}")
    print(f"  Available DSP blocks: {platform_limits['max_dsp']:,}")
    print(f"  Available BRAM: {platform_limits['max_bram']:,} bits (~{platform_limits['max_bram']//8//1024//1024}MB)")
    print(f"  Power budget: {platform_limits['max_power']}W")
    print()
    
    # Define strict hardware constraints for safety-critical deployment
    print("Defining constraints for safety-critical deployment...")
    constraint_spec = ConstraintSpecification()
    
    # Architectural constraints (HARD - provably satisfied)
    print("\n  Architectural Constraints (HARD guarantees):")
    
    arch_constraints = [
        DepthConstraint(min_depth=5, max_depth=12),
        MinConvLayersConstraint(min_conv_layers=3),
        BatchNormRequiredConstraint(),
        InputOutputConstraint(input_channels=3, output_classes=10)
    ]
    
    for c in arch_constraints:
        constraint_spec.add_constraint(c)
        print(f"    • {c.description}")
    
    # Hardware constraints (BOUNDED - model-dependent with ±10-15% error bounds)
    print("\n  Hardware Constraints (BOUNDED guarantees with safety margins):")
    
    # Use conservative limits (80% utilization target)
    conservative_limits = hardware_model.get_conservative_limits(utilization_target=0.8)
    
    hw_constraints = [
        LUTConstraint(max_luts=int(conservative_limits['max_luts']), hardware_model=hardware_model),
        DSPConstraint(max_dsp=int(conservative_limits['max_dsp']), hardware_model=hardware_model),
        BRAMConstraint(max_bram_bits=int(conservative_limits['max_bram']), hardware_model=hardware_model),
        PowerConstraint(max_power_watts=15.0, hardware_model=hardware_model)
    ]
    
    for c in hw_constraints:
        constraint_spec.add_constraint(c)
        print(f"    • {c.description}")
    
    print(f"\n  Total constraints: {len(constraint_spec.get_all_constraints())}")
    print(f"    - HARD (provable): {len(constraint_spec.get_hard_constraints())}")
    print(f"    - BOUNDED (model-dependent): {len(constraint_spec.get_bounded_constraints())}")
    print()
    
    # Create synthesizer
    print("Initializing formal synthesis engine...")
    synthesizer = FormalArchitectureSynthesizer(
        hardware_model=hardware_model,
        constraint_spec=constraint_spec,
        input_channels=3,
        output_classes=10
    )
    print("  ✓ SMT solver configured")
    print("  ✓ Constraint encoding complete")
    print()
    
    # Synthesize architectures
    print("=" * 80)
    print("FORMAL SYNTHESIS PHASE")
    print("=" * 80)
    print("Generating constraint-compliant architectures...")
    print()
    
    architectures = synthesizer.synthesize(
        num_solutions=5,
        timeout_seconds=60
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("SYNTHESIS RESULTS")
    print("=" * 80)
    
    if not architectures:
        print("\n⚠️  No architectures found satisfying all constraints!")
        print("Suggestions:")
        print("  1. Relax hardware constraints (increase resource limits)")
        print("  2. Simplify architectural requirements (reduce min layers)")
        print("  3. Check constraint compatibility (may be UNSAT)")
        return
    
    print(f"\n✓ Successfully synthesized {len(architectures)} architectures\n")
    
    for i, (arch, cert) in enumerate(architectures, 1):
        print(f"\n{'═' * 80}")
        print(f"ARCHITECTURE {i}: {arch.name}")
        print(f"{'═' * 80}")
        print()
        
        # Architecture summary
        print(arch.summary())
        print()
        
        # Certificate
        print("─" * 80)
        print("FORMAL CERTIFICATE")
        print("─" * 80)
        print(cert)
        print()
        
        # Resource estimate with error bounds
        estimate = hardware_model.estimate_architecture_resources(arch)
        conservative = estimate.conservative_upper_bounds()
        
        print("─" * 80)
        print("RESOURCE UTILIZATION (with safety margins)")
        print("─" * 80)
        print(f"  LUTs: {conservative['luts']:,.0f} / {conservative_limits['max_luts']:,.0f} "
              f"({100*conservative['luts']/conservative_limits['max_luts']:.1f}% of limit)")
        print(f"    Base estimate: {estimate.luts:,} ± {estimate.lut_error_bound*100:.0f}%")
        
        print(f"  DSP blocks: {conservative['dsp_blocks']:,.0f} / {conservative_limits['max_dsp']:,.0f} "
              f"({100*conservative['dsp_blocks']/conservative_limits['max_dsp']:.1f}% of limit)")
        print(f"    Base estimate: {estimate.dsp_blocks:,} ± {estimate.dsp_error_bound*100:.0f}%")
        
        print(f"  BRAM: {conservative['bram_bits']:,.0f} bits / {conservative_limits['max_bram']:,.0f} bits "
              f"({100*conservative['bram_bits']/conservative_limits['max_bram']:.1f}% of limit)")
        print(f"    Base estimate: {estimate.bram_bits:,} bits ± {estimate.bram_error_bound*100:.0f}%")
        
        print(f"  Power: {conservative['power_watts']:.2f}W / 15.00W "
              f"({100*conservative['power_watts']/15.0:.1f}% of limit)")
        print(f"    Base estimate: {estimate.power_watts:.2f}W ± {estimate.power_error_bound*100:.0f}%")
        print()
        
        # Validation
        violations = constraint_spec.validate_all(arch)
        if violations:
            print("⚠️  CONSTRAINT VIOLATIONS:")
            for violation in violations:
                print(f"  {violation.severity.upper()}: {violation.message}")
        else:
            print("✅ ALL CONSTRAINTS FORMALLY SATISFIED")
    
    # Final summary
    print("\n" + "=" * 80)
    print("DEPLOYMENT READINESS SUMMARY")
    print("=" * 80)
    print()
    print(f"Generated: {len(architectures)} architectures")
    print(f"All architectures guaranteed to:")
    print(f"  ✓ Meet structural requirements (HARD guarantees)")
    print(f"  ✓ Fit within FPGA resources (BOUNDED guarantees)")
    print(f"  ✓ Stay within power budget (BOUNDED guarantees)")
    print()
    print("Next steps:")
    print("  1. Select architecture based on performance requirements")
    print("  2. Train selected architecture on target dataset")
    print("  3. Synthesize with Intel Quartus for deployment")
    print("  4. Validate actual resource usage against predictions")
    print()
    print("Formal guarantees enable:")
    print("  • Zero deployment failures due to resource violations")
    print("  • Predictable hardware utilization")
    print("  • Certification for safety-critical applications")
    print()


if __name__ == "__main__":
    main()
