"""
Symbolic FPGA Resource Model for SMT Synthesis.

This module adapts the IntelStratixModel to work with Z3 symbolic variables.
Instead of returning concrete floats, it returns Z3 arithmetic expressions.
"""

import z3
try:
    from z3 import ArithRef
except ImportError:
    # Build-time mock if z3 not installed, though we expect it is
    ArithRef = object

from typing import Dict, Any, Union
from .base import HardwareType

SymbolicNum = Union[int, float, ArithRef]

class SymbolicFPGAModel:
    """
    Symbolic version of IntelStratixModel.
    Generates Z3 expressions for resource usage.
    """
    
    # Constants from Stratix 10 Model
    LUTS_PER_MAC = 15
    DSP_PER_MAC_GROUP = 0.5
    BRAM_BITS_PER_PARAM = 16
    POWER_PER_MAC_MW = 0.002
    
    def estimate_conv2d_symbolic(
        self,
        in_channels: SymbolicNum,
        out_channels: SymbolicNum,
        kernel_size: SymbolicNum,
        input_height: SymbolicNum,
        input_width: SymbolicNum,
        padding: int = 1
    ) -> Dict[str, SymbolicNum]:
        """
        Symbolic Resource Estimation for Conv2D.
        """
        # Symbolic Output Shape
        # H_out approx H_in for stride 1, padding 1 (simplified for resource est)
        out_height = input_height 
        out_width = input_width
        
        # MACs = K*K * Cin * Cout * H * W
        # Using Z3 multiplication
        k_sq = kernel_size * kernel_size
        cin_cout = in_channels * out_channels
        hw = out_height * out_width
        
        total_macs = k_sq * cin_cout * hw
        
        # Resources
        # Note: Must use integer arithmetic to preserve Z3 symbolic types
        # Division by Python int/float converts Z3.ArithRef to Python float!
        # CRITICAL: Use integer division (//) or multiply constraints instead

        # LUTs: 0.15 * MACs = (MACs * 15) / 100
        # Use integer division to keep Z3 type
        luts = (total_macs * 15) / 100 # 15% overhead
        # dsp = total_macs (1 DSP per MAC usually, but we can model sharing)
        dsp = total_macs / 100 # 1 DSP per 100 ops? No, usually 1:1 if fully parallel. 
                               # But here we assume ReuseFactor. Let's say 1 DSP handles many ops.
        
        # BRAM: Weights
        # weights = in_c * out_c * k * k
        in_c = in_channels # Alias for clarity with the snippet
        out_c = out_channels # Alias for clarity with the snippet
        k_size = kernel_size # Alias for clarity with the snippet
        weights = in_c * out_c * k_size * k_size
        branch_mem = weights
        bram = branch_mem / 1000 # Heuristic
        # Power: 0.002 mW per MAC => 2e-9 W per MAC
        # Keeping everything in simplified units (e.g. uW or abstract cost)
        # Let's say Power Units = MACs * 2
        power = total_macs * 2 
        
        return {
            'luts': luts,
            'dsp': dsp,
            'bram': bram,
            'power': power,
            'ops': total_macs * 2, # 2 ops per MAC
            'bytes': weights * 2 + total_macs * 2 * 2 # Crude approx: Read weights + (Read In + Write Out) per MAC
        }

    def estimate_pool2d_symbolic(
        self,
        input_height: SymbolicNum,
        input_width: SymbolicNum,
        channels: SymbolicNum
    ) -> Dict[str, SymbolicNum]:
        """Pooling is cheap on FPGA (usually valid bit logic or line buffers)."""
        # Minimal logic - use integer division to preserve Z3 types
        h = input_height # Alias for clarity with the snippet
        w = input_width # Alias for clarity with the snippet
        c = channels # Alias for clarity with the snippet
        return {
            'luts': h * w * c / 10, # Heuristic
            'dsp': 0,
            'bram': (w * c) / 1000, # Line buffer
            'power': 0,
            'ops': h*w*c,
            'bytes': h*w*c
        }
    
    def estimate_add_symbolic(
        self,
        height: SymbolicNum,
        width: SymbolicNum,
        channels: SymbolicNum
    ) -> Dict[str, SymbolicNum]:
        """Element-wise add."""
        h = height # Alias for clarity with the snippet
        w = width # Alias for clarity with the snippet
        c = channels # Alias for clarity with the snippet
        return {
            'luts': h * w * c / 20,
            'dsp': 0, 
            'bram': 0,
            'power': 0,
            'ops': h*w*c,
            'bytes': h*w*c * 2
        }

    def estimate_concat_symbolic(self) -> Dict[str, SymbolicNum]:
        """Concat is free in hardware (wiring)."""
        return {'luts': 0, 'dsp': 0, 'bram': 0, 'power': 0, 'ops': 0, 'bytes': 0}
