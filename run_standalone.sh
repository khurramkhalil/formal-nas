#!/bin/bash
echo "=== Launching Formal NAS Standalone Search ==="

# 1. Cleaning Up
echo "[1/5] Cleanup: Deleting old jobs..."
kubectl delete job formal-nas-standalone --ignore-not-found

# 2. Applying Job
echo "[2/5] Deploying: Deploying 'standalone_job.yaml'..."
kubectl apply -f k8s/standalone_job.yaml

# 3. Waiting for Pod
echo "[3/5] Waiting: Waiting for pod creation..."
sleep 5
POD_NAME=$(kubectl get pods -l job-name=formal-nas-standalone -o jsonpath="{.items[0].metadata.name}")
echo "Found Pod: $POD_NAME"

# 4. Waiting for Initialization (Data Download)
echo "[4/5] Monitoring: Waiting for InitContainer (Data Download ~2min)..."
# We wait for the MAIN container to be 'Running' (which means Init is done)
kubectl wait --for=condition=Ready pod/$POD_NAME --timeout=300s

# 5. Injecting Code
echo "[5/5] Injecting: Injecting Codebase (during 30s sleep window)..."
# Ensure we copy into the WORKSPACE dir mounted at /workspace
# Local src -> Remote /workspace/src
kubectl cp src $POD_NAME:/workspace/src
kubectl cp experiments $POD_NAME:/workspace/experiments
kubectl cp .env $POD_NAME:/workspace/.env

echo "=== Injection Complete! Tailing Logs... ==="
kubectl logs -f $POD_NAME
