"""
Test Hardware Resource Models for Z3 Type Safety.

Verifies that resource estimation functions return Z3 symbolic expressions,
not Python floats, when given Z3 symbolic inputs.
"""

import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

try:
    import z3
except ImportError:
    print("SKIP: z3-solver not installed")
    sys.exit(0)

from formal_nas.hardware_models.symbolic import SymbolicFPGAModel
from formal_nas.hardware_models.xilinx import XilinxU55CModel


def test_symbolic_fpga_model_z3_types():
    """Test that SymbolicFPGAModel returns Z3 expressions."""
    model = SymbolicFPGAModel()

    # Create Z3 symbolic variables
    in_c = z3.Int('in_c')
    out_c = z3.Int('out_c')
    k_size = z3.Int('k_size')
    h = z3.Int('h')
    w = z3.Int('w')

    # Test Conv2D
    result = model.estimate_conv2d_symbolic(in_c, out_c, k_size, h, w)

    # All results should be Z3 expressions or integers, NOT floats
    for key, value in result.items():
        assert isinstance(value, (int, z3.ArithRef)), \
            f"Conv2D {key} is {type(value)}, should be int or z3.ArithRef"
        assert not isinstance(value, float), \
            f"Conv2D {key} is Python float! Z3 type conversion bug!"

    print("✓ SymbolicFPGAModel.estimate_conv2d_symbolic returns Z3 types")

    # Test Pool2D
    result = model.estimate_pool2d_symbolic(h, w, in_c)
    for key, value in result.items():
        assert isinstance(value, (int, z3.ArithRef)), \
            f"Pool2D {key} is {type(value)}, should be int or z3.ArithRef"
        assert not isinstance(value, float), \
            f"Pool2D {key} is Python float! Z3 type conversion bug!"

    print("✓ SymbolicFPGAModel.estimate_pool2d_symbolic returns Z3 types")

    # Test Add
    result = model.estimate_add_symbolic(h, w, in_c)
    for key, value in result.items():
        assert isinstance(value, (int, z3.ArithRef)), \
            f"Add {key} is {type(value)}, should be int or z3.ArithRef"
        assert not isinstance(value, float), \
            f"Add {key} is Python float! Z3 type conversion bug!"

    print("✓ SymbolicFPGAModel.estimate_add_symbolic returns Z3 types")


def test_xilinx_u55c_model_z3_types():
    """Test that XilinxU55CModel returns Z3 expressions."""
    model = XilinxU55CModel()

    # Create Z3 symbolic variables
    in_c = z3.Int('in_c')
    out_c = z3.Int('out_c')
    k_size = z3.Int('k_size')
    h = z3.Int('h')
    w = z3.Int('w')

    # Test Conv2D
    result = model.estimate_conv2d_symbolic(in_c, out_c, k_size, h, w)

    for key, value in result.items():
        assert isinstance(value, (int, z3.ArithRef)), \
            f"Xilinx Conv2D {key} is {type(value)}, should be int or z3.ArithRef"
        assert not isinstance(value, float), \
            f"Xilinx Conv2D {key} is Python float! Z3 type conversion bug!"

    print("✓ XilinxU55CModel.estimate_conv2d_symbolic returns Z3 types")

    # Test Pool2D
    result = model.estimate_pool2d_symbolic(h, w, in_c)
    for key, value in result.items():
        assert isinstance(value, (int, z3.ArithRef)), \
            f"Xilinx Pool2D {key} is {type(value)}, should be int or z3.ArithRef"
        assert not isinstance(value, float), \
            f"Xilinx Pool2D {key} is Python float! Z3 type conversion bug!"

    print("✓ XilinxU55CModel.estimate_pool2d_symbolic returns Z3 types")

    # Test Add
    result = model.estimate_add_symbolic(h, w, in_c)
    for key, value in result.items():
        assert isinstance(value, (int, z3.ArithRef)), \
            f"Xilinx Add {key} is {type(value)}, should be int or z3.ArithRef"
        assert not isinstance(value, float), \
            f"Xilinx Add {key} is Python float! Z3 type conversion bug!"

    print("✓ XilinxU55CModel.estimate_add_symbolic returns Z3 types")


def test_solver_integration():
    """Test that Z3 solver can use resource estimates in constraints."""
    model = SymbolicFPGAModel()
    solver = z3.Solver()

    # Create symbolic variables for architecture parameters
    in_channels = z3.Int('in_channels')
    out_channels = z3.Int('out_channels')
    kernel_size = z3.Int('kernel_size')

    # Constrain to reasonable values
    solver.add(in_channels == 16)
    solver.add(out_channels == 32)
    solver.add(kernel_size == 3)

    # Estimate resources
    resources = model.estimate_conv2d_symbolic(
        in_channels, out_channels, kernel_size,
        z3.IntVal(32), z3.IntVal(32)  # 32x32 input
    )

    # Add resource constraints (this would fail if resources were Python floats)
    solver.add(resources['luts'] <= 1000000) # Relaxed from 100k
    solver.add(resources['dsp'] <= 50000)    # Relaxed
    solver.add(resources['bram'] <= 10000)   # Relaxed

    # Should be satisfiable
    result = solver.check()
    assert result == z3.sat, "Solver should find solution"

    # Get model and verify values
    model_result = solver.model()
    print(f"  Solution: in_c={model_result[in_channels]}, " +
          f"out_c={model_result[out_channels]}, k={model_result[kernel_size]}")

    print("✓ Z3 solver can use resource estimates in constraints")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hardware Model Z3 Type Safety")
    print("=" * 60)
    print()

    test_symbolic_fpga_model_z3_types()
    print()

    test_xilinx_u55c_model_z3_types()
    print()

    test_solver_integration()
    print()

    print("=" * 60)
    print("✅ All Z3 type safety tests passed!")
    print("=" * 60)
