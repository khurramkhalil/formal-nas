"""
Symbolic shape inference logic for Formal NAS.

This module provides Z3-based functions to calculate tensor output shapes
symbolically. This enables 'Correct-by-Construction' synthesis where the
solver guarantees dimensional consistency.
"""

import z3
from typing import Tuple, Union, Any

# Type alias for Z3 arithmetic expressions or finding integers
SymbolicInt = Union[int, z3.ArithRef]

def get_conv2d_output_shape(
    h_in: SymbolicInt,
    w_in: SymbolicInt,
    kernel_size: SymbolicInt,
    stride: SymbolicInt = 1,
    padding: SymbolicInt = 0,
    dilation: SymbolicInt = 1
) -> Tuple[SymbolicInt, SymbolicInt]:
    """
    Calculate output spatial dimensions for 2D convolution symbolically.
    
    Formula: H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    
    For SMT synthesis, we typically enforce:
    1. Output dimension > 0
    2. Kernel size <= Input size (after padding)
    """
    # Effective kernel size
    k_eff = dilation * (kernel_size - 1) + 1
    
    # Numerator for the division
    h_num = h_in + 2 * padding - k_eff
    w_num = w_in + 2 * padding - k_eff

    # In Z3, integer division is truncated (like C++ /).
    # Formula mapping: floor((X)/S + 1) -> (X)/S + 1
    # Note: This assumes exact integer arithmetic.
    # For formal correctness, h_num and w_num must be >= 0 to avoid negative dimensions.
    # When using with Z3, callers should add constraints: h_num >= 0, w_num >= 0
    # or use is_valid_conv_config() to ensure valid configurations.

    h_out = (h_num / stride) + 1
    w_out = (w_num / stride) + 1
    
    return h_out, w_out

def get_pool2d_output_shape(
    h_in: SymbolicInt,
    w_in: SymbolicInt,
    kernel_size: SymbolicInt,
    stride: SymbolicInt = None,
    padding: SymbolicInt = 0
) -> Tuple[SymbolicInt, SymbolicInt]:
    """
    Calculate output spatial dimensions for 2D pooling.
    If stride is None, it defaults to kernel_size.
    """
    if stride is None:
        stride = kernel_size
        
    # Same formula as Conv2D usually (floor)
    # PyTorch default: floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
    # Pooling usually dilation=1

    h_num = h_in + 2*padding - kernel_size
    w_num = w_in + 2*padding - kernel_size

    # For formal correctness, h_num and w_num must be >= 0 to avoid negative dimensions.
    # When using with Z3, callers should add constraints: h_num >= 0, w_num >= 0
    # or use is_valid_pool_config() to ensure valid configurations.

    h_out = (h_num / stride) + 1
    w_out = (w_num / stride) + 1
    
    return h_out, w_out

def is_valid_conv_config(
    h_in: SymbolicInt,
    w_in: SymbolicInt,
    kernel_size: SymbolicInt,
    padding: SymbolicInt = 0,
    dilation: SymbolicInt = 1
) -> z3.BoolRef:
    """
    Generate Z3 constraint ensuring convolution configuration is valid.

    Checks:
    1. Input spatial dims are positive
    2. Effective kernel fits within padded input (accounting for dilation)
    3. Output dims will be positive (h_num >= 0, w_num >= 0)
    """
    # Effective kernel size with dilation
    k_eff = dilation * (kernel_size - 1) + 1

    # Kernel fits check: K_eff <= H + 2P, which ensures h_num >= 0
    param_check = (k_eff <= h_in + 2 * padding)
    param_check_w = (k_eff <= w_in + 2 * padding)

    # Positive dims
    pos_check = z3.And(h_in > 0, w_in > 0, kernel_size > 0, dilation > 0)

    return z3.And(param_check, param_check_w, pos_check)

def is_valid_pool_config(
    h_in: SymbolicInt,
    w_in: SymbolicInt,
    kernel_size: SymbolicInt,
    padding: SymbolicInt = 0
) -> z3.BoolRef:
    """
    Generate Z3 constraint ensuring pooling configuration is valid.

    Checks:
    1. Input spatial dims are positive
    2. Kernel fits within padded input
    3. Output dims will be positive (h_num >= 0, w_num >= 0)
    """
    # Kernel fits check: K <= H + 2P, which ensures h_num >= 0
    param_check = (kernel_size <= h_in + 2 * padding)
    param_check_w = (kernel_size <= w_in + 2 * padding)

    # Positive dims
    pos_check = z3.And(h_in > 0, w_in > 0, kernel_size > 0)

    return z3.And(param_check, param_check_w, pos_check)

def get_flattened_size(
    channels: SymbolicInt,
    height: SymbolicInt,
    width: SymbolicInt
) -> SymbolicInt:
    """Calculate flattened size for Dense layer input."""
    return channels * height * width
