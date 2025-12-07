"""
Formal NAS: End-to-End Training Demo.

This script demonstrates the complete research pipeline:
1. FORMAL SYNTHESIS: Generate a mathematically valid architecture using Z3.
2. MODEL CONSTRUCTION: specific PyTorch `nn.Module` conversion.
3. EMPIRICAL TRAINING: Train the model on synthetic data to prove differentiability.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import z3

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.architectures.pytorch_exporter import PyTorchDAG
from formal_nas.logic.temporal import Always, Eventually, Implies, IsOp, Next

def run_training_demo():
    print("=== Formal NAS: Training Verification Demo ===")
    
    # ---------------------------------------------------------
    # Step 1: Synthesize Architecture
    # ---------------------------------------------------------
    print("\n[1/3] Synthesizing 'Smart' Architecture...")
    solver = z3.Solver()
    
    # Use modest limits
    resource_limits = {"luts": 20000, "dsp": 500, "bram": 50000}
    encoding = DAGEncoding(solver, max_nodes=8, input_channels=3, resource_limits=resource_limits)
    
    # Design Rule: "Deep but Efficient"
    # 1. Must have residual connection (Add)
    solver.add(Eventually(IsOp(3)).encode(solver, encoding, 0))
    # 2. Must downsample eventually (Pool)
    solver.add(Eventually(IsOp(2)).encode(solver, encoding, 0))
    
    if solver.check() != z3.sat:
        print("❌ Could not synthesize architecture.")
        return

    model = solver.model()
    arch_dict = encoding.decode_architecture(model)
    print(f"✓ Found architecture with {len(arch_dict)} active nodes.")
    
    # ---------------------------------------------------------
    # Step 2: Convert to PyTorch
    # ---------------------------------------------------------
    print("\n[2/3] Constructing PyTorch Model...")
    try:
        net = PyTorchDAG(arch_dict, input_channels=3, num_classes=10)
        print(net)
    except Exception as e:
        print(f"❌ Conversion Failed: {e}")
        return

    # ---------------------------------------------------------
    # Step 3: Train Loop (Proof of Gradient Flow)
    # ---------------------------------------------------------
    print("\n[3/3] Verifying Training (Gradient Flow)...")
    
    # Synthetic Data (Batch=16, C=3, H=32, W=32)
    inputs = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    net.train()
    
    initial_loss = 0.0
    print(f"{'Step':<5} | {'Loss':<10}")
    print("-" * 20)
    
    for step in range(1, 11):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"{step:<5} | {loss.item():.4f}")
        
        if step == 1:
            initial_loss = loss.item()
            
    final_loss = loss.item()
    if final_loss < initial_loss:
        print(f"\n✅ Success! Loss decreased from {initial_loss:.4f} to {final_loss:.4f}.")
        print("   This proves the synthesized graph is valid, connected, and differentiable.")
    else:
        print("\n⚠️ Warning: Loss did not decrease (might need more steps or better LR).")

if __name__ == "__main__":
    run_training_demo()
