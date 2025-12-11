FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# (Torch is already in the base image)
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1. Clone TransNAS-Bench API
RUN git clone https://github.com/yawen-d/TransNASBench.git /app/TransNASBench

# 2. Add Data Files (EARLY LAYERS for Caching)
# Ensure these files exist in the build context!
COPY transnas-bench_v10141024.pth /app/transnas-bench_v10141024.pth
COPY NAS-Bench-201-v1_1-096897.pth /app/NAS-Bench-201-v1_1-096897.pth

# Install zip for export script (Done here to preserve cache of above layers)
RUN apt-get update && apt-get install -y zip && rm -rf /var/lib/apt/lists/*

# 3. Copy Local Codebase (Changes Frequently)
COPY src /app/src
COPY experiments /app/experiments
COPY scripts /app/scripts
COPY .env /app/.env

# Setup Environment Variables
ENV PYTHONPATH="${PYTHONPATH}:/app/src:/app/TransNASBench"

# Make entrypoint executable
RUN chmod +x /app/scripts/entrypoint_full_benchmark.sh

# Default Command: Run Full Benchmark Suite
CMD ["/app/scripts/entrypoint_full_benchmark.sh"]
