# Hardware Resource Modeling Analysis

## Critical Issues Found

### 1. **Z3 Type Incompatibility: Floating-Point Division** ⚠️ CRITICAL

**Location**: `src/formal_nas/hardware_models/symbolic.py` lines 63, 66, 70
**Location**: `src/formal_nas/hardware_models/xilinx.py` lines 105, 119, 121, 129

**Problem**:
```python
# symbolic.py
luts = total_macs * 15 / 100  # Z3 ArithRef / int → Python float! ❌
dsp = total_macs * 5 / 1000   # Z3 ArithRef / int → Python float! ❌
bram = weights * 16 * 12 / 10 # Z3 ArithRef / int → Python float! ❌

# xilinx.py
bram_blocks = (num_weights * 8) / 36000  # Z3 ArithRef / int → Python float! ❌
luts = h * w * c * 0.1                   # Z3 ArithRef * float → Python float! ❌
```

**Why This is Critical**:
- When you divide a Z3 symbolic integer by a Python int/float, Python's `__truediv__` returns a **Python float**, not a Z3 expression
- This breaks the SMT solver because constraints become: `solver.add(42.5 <= 100000)` instead of `solver.add(total_macs * 15 <= 100000 * 100)`
- Result: **Resource constraints are ignored**, architectures may exceed FPGA capacity

**Correct Fix**:
```python
# Use integer arithmetic to keep Z3 expressions
luts = (total_macs * 15) // 100  # Integer division keeps Z3 type
# Or multiply the constraint instead:
# solver.add(luts * 100 <= limit * 15)
```

---

### 2. **Resource Estimation Accuracy Issues**

#### 2.1 Xilinx Model: Unrealistic Parallelism Assumption

**Location**: `xilinx.py` lines 88-109

**Problem**:
```python
parallel_mults = kernel_size * kernel_size * in_channels * out_channels
# ...
return {
    'dsp': parallel_mults * self.DSP_EFFICIENCY,  # Assumes FULL parallelism!
}
```

**Example**:
- Conv 3×3, 16→32 channels: `parallel_mults = 3 * 3 * 16 * 32 = 4,608 DSPs`
- **U55C only has 9,024 DSPs total!**
- A single layer would consume **51% of all DSPs** on the chip

**Reality**:
- HLS uses **reuse factor** (temporal multiplexing)
- With ReuseFactor=128 (from generate_hls.py), actual DSP usage = `4608 / 128 = 36 DSPs`
- Current model **overestimates by 128×**

**Impact**:
- Solver rejects valid architectures (too conservative)
- Or if constraints are floats (bug #1), accepts invalid ones

---

#### 2.2 Symbolic Model: Hardcoded Intel Stratix Constants

**Location**: `symbolic.py` lines 27-30

**Problem**:
```python
LUTS_PER_MAC = 15        # Intel Stratix constant
DSP_PER_MAC_GROUP = 0.5  # Intel Stratix constant
```

Used with Xilinx U55C target:
- Intel and Xilinx have different architectures
- Intel uses ALMs (Adaptive Logic Modules)
- Xilinx uses SLICEs with 6-input LUTs
- DSP blocks are completely different (Intel Variable Precision vs Xilinx DSP58)

**Impact**:
- Resource estimates are **architecturally incorrect** for Xilinx
- LUT usage likely underestimated for Xilinx
- BRAM usage formula doesn't account for Xilinx RAMB36 organization

---

#### 2.3 No Accounting for IO Stream Mode

**Problem**:
After fixing HLS to use `io_stream`, the resource model still assumes `io_parallel`:
- **io_parallel**: Uses registers (counted in LUTs/FFs)
- **io_stream**: Uses FIFOs (counted in BRAM/URAM)

**Missing**:
- FIFO depth estimation for streaming layers
- BRAM usage for line buffers in streaming convolution
- URAM usage for large feature map buffers

**Impact**:
- BRAM estimates too low for streaming designs
- May synthesize architecture that runs out of BRAM at implementation

---

### 3. **Missing Resource Types**

#### 3.1 Flip-Flops (FFs)
- Xilinx designs are often **FF-limited**, not LUT-limited
- Streaming designs use many pipeline registers
- **No FF modeling at all**

#### 3.2 UltraRAM (URAM)
- U55C has 960 URAM blocks (288Kb each)
- Often used for weight storage in large models
- **Not modeled**

#### 3.3 Routing Congestion
- High-utilization designs (>80%) often fail due to routing
- **Not modeled**

---

### 4. **Incomplete Pooling Model**

**Location**: `xilinx.py` lines 116-125

**Problem**:
```python
def estimate_pool2d_symbolic(self, h, w, c):
    return {
        'luts': h * w * c * 0.1,  # Magic number, what does 0.1 mean?
        'dsp': 0,
        'bram': (w * c * 8) / 36000,  # Line buffer - but what about pooling window?
    }
```

**Issues**:
1. **Magic constant 0.1** - No justification, likely wrong
2. **Line buffer only** - Doesn't account for pooling window buffer
3. **No kernel size parameter** - 2×2 pool vs 3×3 pool have same cost?

**Reality**:
- MaxPool needs comparators (LUT cost)
- AvgPool needs adders/dividers (higher LUT cost, possibly DSP)
- Different kernel sizes need different buffer sizes

---

## Correctness Assessment

### What's Correct ✅

1. **Conceptual approach**: Using symbolic Z3 expressions for SMT synthesis
2. **Weight storage**: `num_weights * bits_per_weight` is correct
3. **Total MACs calculation**: Formula is correct
4. **Zero cost for concat**: Correct (just wire connections)

### What's Wrong ❌

1. **Z3 type safety**: Floating-point division breaks symbolic expressions
2. **Parallelism assumption**: Assumes full parallelization (no reuse factor)
3. **Platform specificity**: Using Intel constants for Xilinx chips
4. **Missing resources**: No FFs, URAM, routing
5. **IO mode**: Doesn't account for streaming vs parallel
6. **Magic numbers**: Unexplained constants (0.1, 0.05, etc.)

---

## Verification Against Real HLS Synthesis

To check accuracy, let's compare against actual Vitis HLS results:

### Test Case: Simple Conv2D
- **Config**: 3×3 conv, 16 input channels, 16 output channels, 32×32 input
- **Current Model Prediction**:
  - `parallel_mults = 9 * 16 * 16 = 2,304`
  - **DSPs: 2,304** (from xilinx.py)
  - **LUTs: 11,520** (2304 * 5)

- **Actual HLS with ReuseFactor=128, io_stream**:
  - **DSPs: ~18-36** (2304 / 128 = 18, with some overhead)
  - **LUTs: ~5,000-10,000** (control logic + streaming)
  - **BRAM: ~10-20 blocks** (FIFOs + weights)

**Error**: Model overestimates DSPs by **64-128×**

---

## Recommended Fixes

### Priority 1: Fix Z3 Type Safety (CRITICAL)
```python
# Before (WRONG):
luts = total_macs * 15 / 100  # Returns Python float!

# After (CORRECT):
luts = (total_macs * 15) // 100  # Integer division, keeps Z3 type
# Or even better, use constraint transformation:
# solver.add(luts * 100 <= limit * 15)
```

### Priority 2: Add Reuse Factor
```python
def estimate_conv2d_symbolic(self, ..., reuse_factor: int = 1):
    parallel_mults = (kernel_size * kernel_size * in_channels * out_channels)
    actual_dsp = parallel_mults // reuse_factor  # Temporal reuse
```

### Priority 3: Xilinx-Specific Calibration
```python
# Calibrate constants from actual Vitis HLS synthesis runs
XILINX_LUTS_PER_DSP_MAC = 5  # Control overhead
XILINX_LUTS_PER_ADD = 1       # Single LUT per adder
XILINX_BRAM_36K_BITS = 36864  # Actual BRAM size
```

### Priority 4: Add Missing Resources
```python
return {
    'luts': ...,
    'ffs': ...,      # NEW
    'dsp': ...,
    'bram': ...,
    'uram': ...,     # NEW
}
```

---

## Impact on Current System

### If Bug #1 is Active (Z3 type conversion):
- ❌ **Resource constraints completely broken**
- Solver ignores resource limits
- May generate unsynthesizable architectures

### If Only Accuracy Issues (2-4):
- ⚠️ **Over-conservative** (rejects valid designs)
- But safe (won't generate infeasible designs)

### Current State:
Based on the code, **Bug #1 IS ACTIVE** - the Z3 type conversion issue is present in both `symbolic.py` and `xilinx.py`. This is **CRITICAL** and explains potential synthesis failures.

---

## Testing Recommendations

1. **Unit test**: Verify Z3 types
   ```python
   result = model.estimate_conv2d_symbolic(...)
   assert isinstance(result['luts'], z3.ArithRef), "Must be Z3 expr!"
   ```

2. **Integration test**: Synthesize known architecture, compare HLS results

3. **Regression test**: Ensure resource constraints actually prevent oversized designs

---

## Next Steps

1. Fix Z3 type safety (critical bug)
2. Calibrate constants from real HLS runs
3. Add reuse factor parameter
4. Add FF and URAM modeling
5. Create validation suite comparing model vs actual HLS

