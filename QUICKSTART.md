# Quick Start Guide

## 5-Minute Tour

### 1. Run the Examples

```bash
cd /home/claude/formal_nas_poc

# Simple example
python examples/simple_cnn_synthesis.py

# FPGA-constrained example  
python examples/fpga_constrained_synthesis.py
```

### 2. Understand the Output

The synthesis will:
1. Define constraints (structural + hardware)
2. Generate architectures that satisfy constraints
3. Provide formal certificates proving satisfaction
4. Estimate resource usage with error bounds

Example output:
```
Architecture: generated_arch
Total layers: 17
Conv layers: 3
Dense layers: 2
Total parameters: 357,258

Synthesis Certificate:
✓ All constraints satisfied
Resource Bounds (Conservative):
  luts: 75,432
  power: 12.3W
```

### 3. Explore the Code

**Main Components:**

```python
# 1. Define constraints
from formal_nas.constraints.architectural import DepthConstraint
constraint = DepthConstraint(min_depth=5, max_depth=15)

# 2. Define hardware
from formal_nas.hardware_models.fpga_models import IntelStratixModel
hw = IntelStratixModel(device="Stratix10GX2800")

# 3. Synthesize
from formal_nas.synthesis.smt_solver import FormalArchitectureSynthesizer
synth = FormalArchitectureSynthesizer(hw, constraints)
architectures = synth.synthesize(num_solutions=5)

# 4. Use results
for arch, cert in architectures:
    print(arch.summary())  # Architecture details
    print(cert)            # Formal certificate
```

## Project Structure

```
formal_nas_poc/
├── examples/              ← Start here!
├── src/formal_nas/       ← Core framework
│   ├── constraints/      ← Constraint types
│   ├── hardware_models/  ← FPGA models
│   ├── synthesis/        ← SMT solver
│   └── architectures/    ← Network representation
├── USAGE.md              ← Detailed guide
├── PROJECT_SUMMARY.md    ← What we built
└── README.md             ← Overview
```

## Key Files to Read

1. **examples/simple_cnn_synthesis.py** - Complete working example
2. **src/formal_nas/synthesis/smt_solver.py** - Main synthesis algorithm
3. **src/formal_nas/constraints/base.py** - Constraint framework
4. **src/formal_nas/hardware_models/fpga_models.py** - FPGA resource models
5. **USAGE.md** - Comprehensive usage guide

## Adding Your Own Constraint

```python
# In src/formal_nas/constraints/custom.py
from formal_nas.constraints.base import Constraint, ConstraintType, GuaranteeLevel

class MaxParametersConstraint(Constraint):
    def __init__(self, max_params):
        super().__init__(
            name="MaxParameters",
            constraint_type=ConstraintType.ARCHITECTURAL,
            guarantee_level=GuaranteeLevel.HARD,
            description=f"Params ≤ {max_params}"
        )
        self.max_params = max_params
    
    def to_smt(self, solver, variables):
        # Encode in SMT
        return [variables['total_params'] <= self.max_params]
    
    def validate(self, architecture):
        # Validate post-generation
        if architecture.total_parameters() > self.max_params:
            return ConstraintViolation(...)
        return None
```

## Common Tasks

### Generate Architectures for Specific Hardware

```python
# Intel Stratix 10 with strict constraints
hardware = IntelStratixModel("Stratix10GX2800")
limits = hardware.get_conservative_limits(utilization_target=0.7)

spec = ConstraintSpecification()
spec.add_constraint(LUTConstraint(limits['max_luts'], hardware))
spec.add_constraint(PowerConstraint(15.0, hardware))

synth = FormalArchitectureSynthesizer(hardware, spec)
archs = synth.synthesize()
```

### Validate Existing Architecture

```python
# Check if an architecture satisfies constraints
from formal_nas.architectures.network import NeuralArchitecture

arch = NeuralArchitecture()
# ... build architecture ...

violations = constraint_spec.validate_all(arch)
if not violations:
    print("✓ All constraints satisfied")
else:
    for v in violations:
        print(f"✗ {v.message}")
```

### Estimate Resources

```python
# Get resource estimates for an architecture
estimate = hardware_model.estimate_architecture_resources(arch)

print(f"LUTs: {estimate.luts} ± {estimate.lut_error_bound*100}%")
print(f"Power: {estimate.power_watts}W")

# Get conservative upper bounds
conservative = estimate.conservative_upper_bounds()
print(f"Conservative LUTs: {conservative['luts']}")
```

## Next Steps

1. **Run the examples** to see the framework in action
2. **Read USAGE.md** for detailed documentation
3. **Read PROJECT_SUMMARY.md** to understand what's implemented
4. **Modify examples** to test different constraints
5. **Add your own constraints** for your research

## Getting Help

- Check **USAGE.md** for detailed usage
- Read **PROJECT_SUMMARY.md** for architecture overview
- Look at **examples/** for working code
- Review **src/formal_nas/** for implementation details

## Limitations (Current PoC)

⚠️ **Using Mock Z3**: Install real Z3 with:
```bash
pip install z3-solver
rm src/z3.py  # Remove mock
```

⚠️ **Simple Architectures**: Currently generates basic CNNs
- No residual connections yet
- No complex skip paths yet
- Easy to extend!

## Success Metrics

You'll know it's working when:
- ✅ Examples run without errors
- ✅ Architectures are generated
- ✅ Certificates show constraint satisfaction
- ✅ Resource estimates are reasonable
- ✅ Validation passes

## Tips

1. **Start small**: Use SimpleFPGAModel for initial experiments
2. **Relax constraints**: If no solutions, loosen resource limits
3. **Check certificates**: Always verify constraint satisfaction
4. **Validate models**: Compare estimates with actual synthesis
5. **Read the code**: Framework is well-documented

---

**You're ready to start! Run the examples and explore.** 🚀
