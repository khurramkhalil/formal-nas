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
RUN pip install --no-cache-dir \
    wandb \
    z3-solver \
    scipy \
    gdown \
    python-dotenv \
    matplotlib \
    networkx

# 1. Clone TransNAS-Bench API
RUN git clone https://github.com/yawen-d/TransNASBench.git /app/TransNASBench

# 2. Add TransNAS-Bench-101 Data (Local Copy)
# This assumes the data is in 'transnas_data/TransNASBench-file' relative to build context
COPY transnas_data/transnas-bench_v10141024.pth /app/transnas-bench_v10141024.pth

# 3. Copy Local Codebase
COPY src /app/src
COPY experiments /app/experiments
COPY .env /app/.env

# Setup Environment Variables
ENV PYTHONPATH="${PYTHONPATH}:/app/src:/app/TransNASBench"
ENV WANDB_API_KEY="" 
# (User needs to pass WANDB_API_KEY at runtime)

# Create symlink for code compatibility
RUN ln -s /app/data/TransNASBench-file/transnas-bench_v10141024.pth /app/transnas-bench_v10141024.pth

# Default Command
CMD ["python", "experiments/run_unified_search.py"]
