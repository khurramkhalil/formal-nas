"""
CIFAR-10 Benchmark Suite for Formal NAS.

This script runs a full experimental campaign with WandB Logging:
1. Sweeps through Resource Budgets (LUTs).
2. Synthesizes a Hardware-Constrained Architecture for each budget.
3. Trains the architecture on CIFAR-10 to convergence.
4. Logs EVERYTHING to Weights & Biases for publication plots.

Usage: python experiments/cifar10_benchmark.py
"""

import os
import sys
import csv
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import z3
import wandb
from dotenv import load_dotenv

# Ensure source code is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from formal_nas.synthesis.dag_encoding import DAGEncoding
from formal_nas.architectures.pytorch_exporter import PyTorchDAG
from formal_nas.logic.temporal import Eventually, IsOp

# Load Secrets
load_dotenv()

# --- Configuration ---
LUT_BUDGETS = [2000, 5000, 10000, 20000, 50000]
EPOCHS = 100 # Increased for convergence
BATCH_SIZE = 128
LEARNING_RATE = 0.1
PROJECT_NAME = os.getenv("WANDB_PROJECT", "formal-nas-cifar10")

def get_cifar10_loaders():
    print("Loading CIFAR-10...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_and_evaluate(net, trainloader, testloader, device, epochs, run_name):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Log Data Granularly (per 50 steps)
            if i % 50 == 0:
                wandb.log({
                    "batch_loss": loss.item(), 
                    "epoch": epoch + i/len(trainloader)
                })
        
        scheduler.step()
        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(trainloader)
        
        # Test Loop
        net.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_acc = 100 * correct_test / total_test
        if test_acc > best_acc:
            best_acc = test_acc
            # Log best model? Optional
        
        # Log Epoch Stats
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "lr": scheduler.get_last_lr()[0]
        })
        
        print(f"[{run_name}] Epoch {epoch+1}: Test Acc={test_acc:.2f}%")
        
    duration = time.time() - start_time
    return best_acc, duration

def run_benchmark():
    print(f"=== Formal NAS: CIFAR-10 Benchmark ({PROJECT_NAME}) ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainloader, testloader = get_cifar10_loaders()
    
    for limit in LUT_BUDGETS:
        run_name = f"arch_lut_{limit}"
        print(f"\n--- Starting Run: {run_name} ---")
        
        # 1. Synthesize
        solver = z3.Solver()
        res_limits = {"luts": limit, "dsp": 1000, "bram": 5000}
        
        encoding = DAGEncoding(solver, max_nodes=10, input_channels=3, 
                               resource_limits=res_limits, hw_model_type="xilinx_u55c")
        
        # Enforce Conv
        solver.add(Eventually(IsOp(1)).encode(solver, encoding, 0))
        
        if solver.check() == z3.sat:
            model = solver.model()
            arch_dict = encoding.decode_architecture(model)
            
            # Extract Graph Stats for Plotting
            num_nodes = len(arch_dict)
            conv_count = sum(1 for n in arch_dict if n['op'] == 1)
            
            # 2. Init WandB Run
            wandb.init(
                project=PROJECT_NAME,
                name=run_name,
                config={
                    "lut_limit": limit,
                    "max_nodes": 10,
                    "input_res": 32,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LEARNING_RATE,
                    # Graph Metrics
                    "graph_nodes": num_nodes,
                    "graph_convs": conv_count,
                    "graph_json": json.dumps(str(arch_dict)) 
                }
            )
            
            # 3. Train
            try:
                net = PyTorchDAG(arch_dict, input_channels=3, num_classes=10)
                
                # Watch model gradients
                wandb.watch(net, log="all", log_freq=100)
                
                acc, duration = train_and_evaluate(net, trainloader, testloader, device, EPOCHS, run_name)
                
                # Log Final Summary Metrics
                wandb.log({
                    "final_test_acc": acc,
                    "training_duration": duration,
                    "lut_limit_x": limit # Helpful for X-axis plotting
                })
                
                print(f"Run Complete. Final Acc: {acc:.2f}%")
                
            except Exception as e:
                print(f"Error: {e}")
                wandb.log({"error": str(e)})
            
            wandb.finish()
            
        else:
            print(f"UNSAT for limit {limit}")

if __name__ == "__main__":
    run_benchmark()
