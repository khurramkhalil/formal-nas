# Formal Neural Architecture Synthesis (Research Framework)

> **Novelty Engine**: Symbolic DAG Synthesis with Formal Guarantees.

This research project implements a **Correct-by-Construction** framework for Neural Architecture Synthesis (NAS). Unlike traditional NAS which relies on Reinforcement Learning or Evolutionary Algorithms to *search* a space, we use **SMT Solvers (Z3)** to *synthesize* architectures that are mathematically proven to satisfy structural, dimensional, and hardware constraints.

## 🚀 Key Research Capabilities

The "Novelty Engine" allows you to:

1.  **Synthesize Arbitrary DAGs**: Instead of tuning parameters of a fixed template, the solver "invents" topologies (nodes + edges) directly in logic.
2.  **Enforce Dimensional Validity (ShapeLogic)**: We encoded a custom **Theory of Shapes** into SMT. The solver strictly enforces tensor compatibility (e.g., ensuring `Conv2D` output shapes match `Add` layer inputs), guaranteeing every generated graph is valid.
3.  **Specify Design Rules via Temporal Logic (TWTL)**: 
    > *"Always ensure a Convolution is eventually followed by a Pooling layer"*
    
    This allows high-level design intent without manually specifying layers.
4.  **Optimize for Hardware (Symbolic Models)**: The solver "sees" the FPGA LUT/DSP cost of every decision (as symbolic variables) and optimizes topology to fit the device budget.

---

## ⚡ Quick Start

### 1. Installation

- Python 3.8+
- [Z3 Solver](https://github.com/Z3Prover/z3) (`pip install z3-solver`)
- [PyTorch](https://pytorch.org/) (`pip install torch`)
- *Optional (for FPGA Export)*: [hls4ml](https://fastmachinelearning.org/hls4ml/) (`pip install hls4ml onnx`)

### 2. Run the Demo

We have created a demo script (`examples/dag_synthesis_demo.py`) that showcases all key features: Arbitrary Graph Synthesis, Dimensional Checking, Hardware Constraints, and Temporal Logic.

```bash
python examples/dag_synthesis_demo.py
```

**Example Output:**
The demo finds an architecture that fits a tight 4000 LUT budget and satisfies *"Always Conv -> Eventually Pool"*:

```text
✅ Solution Found!
Estimated Resources: LUTs=1924, DSPs=61, BRAMs=7065

Synthesized Graph:
------------------
Node 0: INPUT    | In: 0    | Shape: (3, 32, 32)
Node 2: POOL     | In: 0    | Shape: (3, 16, 16)  <-- Solver chose to Pool early!
Node 3: CONV     | In: 2    | Shape: (16, 18, 18)
Node 4: POOL     | In: 3    | Shape: (16, 9, 9)
Node 5: OUTPUT   | In: 4    | Shape: (16, 9, 9)
------------------
```

---

## 📚 Usage Guide

The core class is `DAGEncoding`. Here is how to use it to discover architectures.

### 1. Basic DAG Synthesis

```python
import z3
from formal_nas.synthesis.dag_encoding import DAGEncoding

solver = z3.Solver()
# Create an encoding for a graph with max 10 nodes
encoding = DAGEncoding(solver, max_nodes=10, input_channels=3)

# Add a constraint: Must have at least one Residual Connection (Add layer)
# OP_ADD is 3 (check source for constants)
has_add = z3.Or([op == 3 for op in encoding.node_ops])
solver.add(has_add)

if solver.check() == z3.sat:
    model = solver.model()
    arch = encoding.decode_architecture(model)
    print(arch)
```

### 2. Adding Hardware Constraints

Pass a dictionary of resource limits. The solver will find a topology that strictly respects these bounds.

```python
# Limit FPGA resources
limits = {
    "luts": 5000,   # Very tight limit!
    "dsp": 100,
    "bram": 50000
}
encoding = DAGEncoding(solver, max_nodes=10, input_channels=3, resource_limits=limits)
```

### 3. Using Temporal Logic (TWTL)

Use the `logic.temporal` module to enforce design patterns.

```python
from formal_nas.logic.temporal import Always, Eventually, Implies, IsOp, Next

# Rule: "If Conv (1), then eventually Pool (2)"
rule = Always(Implies(
    IsOp(1), 
    Next(Eventually(IsOp(2)))
))

# Encode the rule starting at time step 0 (Node 0)
solver.add(rule.encode(solver, encoding, 0))
```

### 4. Advanced: Roofline & Scalability

- **Roofline Model**: Enforce memory bandwidth limits by setting `min_intensity` (Ops/Byte).
  ```python
  limits = {"luts": 20000, "min_intensity": 50} 
  # Solver will reject architectures that are too memory-hungry
  ```

- **Scalability**: Use `formal_nas.architectures.hierarchical` to synthesize a **Cell** and stack it into a **SuperNet** (ResNet-like).

- **Baselines**: Use `formal_nas.synthesis.baselines.RandomSearchBaseline` to compare SMT efficiency against Random Search.

---

## 🏗️ Project Architecture

```
formal_nas/
├── src/formal_nas/
│   ├── synthesis/            # The Core Engine
│   │   ├── dag_encoding.py   # Symbolic DAG & Shape Logic (The "Brain")
│   │   └── smt_solver.py     # High-level orchestration
│   ├── logic/                # Formal Logic Modules
│   │   ├── shape_inference.py # Tensor dimension math
│   │   └── temporal.py       # TWTL/LTL parser
│   ├── hardware_models/      # Symbolic & Concrete Models
│   │   ├── symbolic.py       # Z3-based resource estimation
│   │   └── fpga_models.py    # Concrete Intel Stratix 10 models
├── examples/
│   ├── dag_synthesis_demo.py # <-- Start here
└── tests/                    # Verification tests
```

## Citation

If you use this work, please cite:

```bibtex
@article{formal_nas_2025,
  title={Correct-by-Constuction Neural Architecture Synthesis with Temporal Logic},
  author={[Your Name]},
  year={2025}
}
```
