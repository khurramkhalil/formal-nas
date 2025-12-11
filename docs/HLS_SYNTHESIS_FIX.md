# HLS Synthesis Issue Fix for VITAS on FPGA

## Problem Description

When synthesizing neural network architectures with `hls4ml` for FPGA deployment (particularly on Xilinx U55C via NRPU), the synthesis process was failing with:

- **3,655,743 instructions** reported during C/RTL synthesis
- Extremely long synthesis times (hours) or complete failure
- Massive resource usage exceeding FPGA capacity
- Out of memory errors during synthesis

## Root Cause

The issue was caused by **complete array partitioning** of all intermediate layer buffers:

```cpp
layer2_t layer2_out[32*32*16];  // 16,384 elements
#pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0  // Creates 16,384 registers!
```

This happened because:

1. **No `io_type` specified** in hls4ml conversion → defaulted to `io_parallel`
2. **`io_parallel` mode** generates complete array partitioning for all buffers
3. **DATAFLOW disabled** by compatibility patches, preventing streaming optimization
4. **Result**: Millions of registers/flip-flops, extremely complex design

### Why Complete Partitioning Fails

For a 32×32×16 feature map:
- 32 × 32 × 16 = **16,384 individual registers** per layer
- Multiple layers → hundreds of thousands of registers
- All data sits in registers → no streaming, no pipelining
- Synthesis complexity explodes exponentially

## The Fix

### 1. Enable Streaming IO (`io_stream`)

Modified `scripts/generate_hls.py` to use streaming interfaces:

```python
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    input_shape=(3, 32, 32),
    hls_config=config,
    output_dir=args.output_dir,
    part='xcu55c-fsvh2892-2L-e',
    project_name=args.project_name,
    io_type='io_stream'  # CRITICAL FIX
)
```

### 2. Keep DATAFLOW Enabled

Commented out the DATAFLOW-disabling patch for streaming mode:

```python
# For io_stream mode, DATAFLOW is REQUIRED for proper streaming operation
# content = content.replace("#pragma HLS DATAFLOW", "//#pragma HLS DATAFLOW")  # DISABLED
```

### 3. Remove Complete Partitioning for Intermediate Buffers

Added automatic patch to comment out complete partitioning pragmas:

```python
if "layer" in line and "_out" in line and "ARRAY_PARTITION" in line and "complete" in line:
    new_lines.append("//" + line + "  // Disabled: Use streaming FIFOs instead")
```

## Benefits

With streaming IO:

✅ **FIFOs instead of registers** - Intermediate data flows through small FIFOs
✅ **DATAFLOW pipelining** - Layers execute in parallel pipeline
✅ **Reduced synthesis time** - From hours/failure to minutes
✅ **Lower resource usage** - Uses Block RAM for FIFOs instead of registers
✅ **Better throughput** - Pipeline allows continuous data flow

## Usage

### Regenerate HLS Projects

```bash
python scripts/generate_hls.py \
    --arch "|skip_connect~0|+|nor_conv_3x3~0|skip_connect~1|+|nor_conv_3x3~0|skip_connect~1|skip_connect~2|" \
    --project-name fpga_best_arch \
    --output-dir hls_project
```

The script now automatically:
- Uses `io_stream` mode
- Keeps DATAFLOW enabled
- Removes complete partitioning from intermediate buffers
- Applies Vitis 2023.1 compatibility patches

### Running Synthesis

```bash
cd hls_project
source /tools/Xilinx/Vitis_HLS/2023.1/settings64.sh
vitis_hls -f build_prj.tcl
```

Expected output:
- **Much lower instruction count** (typically < 100k instructions)
- **Faster synthesis** (10-30 minutes instead of hours)
- **Successful synthesis** without resource exhaustion

## Technical Details

### io_parallel vs io_stream

| Feature | io_parallel | io_stream |
|---------|------------|-----------|
| Interface | Arrays with complete partitioning | `hls::stream<>` FIFOs |
| Memory | Registers/FFs | Block RAM |
| Dataflow | Not recommended | Required |
| Resource usage | Very high | Moderate |
| Synthesis time | Very long | Reasonable |
| Best for | Tiny networks (<1K params) | Any real network |

### Why This Wasn't Caught Earlier

The original patches (for Vitis 2023.1 compatibility) disabled DATAFLOW to avoid "strict dataflow checking" crashes. However, this made `io_parallel` even worse by preventing any optimization. The proper fix is to use `io_stream` where DATAFLOW is required and works correctly.

## Verification

To verify the fix worked:

1. Check synthesis log for instruction count:
   - ❌ Before: "3,655,743 instructions"
   - ✅ After: "< 100,000 instructions"

2. Check generated firmware:
   - ❌ Before: `#pragma HLS ARRAY_PARTITION variable=layer2_out complete`
   - ✅ After: `// #pragma HLS ARRAY_PARTITION...` (commented out)

3. Check synthesis time:
   - ❌ Before: Hours or failure
   - ✅ After: 10-30 minutes

## Related Files

- `scripts/generate_hls.py` - Main fix applied here
- `examples/fpga_export_demo.py` - Updated example
- `hls_project/` - Example project directory (regenerate with fix)

## References

- hls4ml documentation: https://fastmachinelearning.org/hls4ml/
- Xilinx HLS Optimization: UG1399
- Streaming interfaces: UG902 (High-Level Synthesis)
