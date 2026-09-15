import time
import numpy as np
import torch

# 1. Setup size (5000x5000 is large enough to see a clear GPU advantage)
size = 5000
print(f"Generating {size}x{size} matrices...")

# ==========================================
# BENCHMARK 1: NumPy (Runs on CPU)
# ==========================================
# We use float32 to ensure an exact data type match with PyTorch
np_a = np.random.randn(size, size).astype(np.float32)
np_b = np.random.randn(size, size).astype(np.float32)

start_time = time.time()
np_result = np.dot(np_a, np_b)
numpy_duration = time.time() - start_time
print(f"NumPy (CPU) Time: {numpy_duration:.4f} seconds")

# ==========================================
# BENCHMARK 2: PyTorch (Runs on GPU via MPS)
# ==========================================
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Found {device} device.")

# Allocate directly on the MPS device
pt_a = torch.randn(size, size, device=device)
pt_b = torch.randn(size, size, device=device)

# Warm-up run (Crucial for MPS to compile kernels out of the timing loop)
start_time = time.time()
_ = torch.mm(pt_a, pt_b)
torch.mps.synchronize()
torch_duration = time.time() - start_time
print(f"PyTorch (MPS GPU) Warm-up Time: {torch_duration:.4f} seconds")

# Actual timed run
start_time = time.time()
pt_result = torch.mm(pt_a, pt_b)
torch.mps.synchronize()  # Force CPU to wait for GPU to finish computing
torch_duration = time.time() - start_time
print(f"PyTorch (MPS GPU) Time: {torch_duration:.4f} seconds")

# ==========================================
# RESULTS COMPARISON
# ==========================================
speedup = numpy_duration / torch_duration
print(f"--> PyTorch on MPS is {speedup:.1f}x faster than NumPy for this operation!")
