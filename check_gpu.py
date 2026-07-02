import torch
import sys

def main():
    print("=== PyTorch GPU Diagnostics ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs found: {device_count}")
        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    else:
        print("\nPyTorch cannot detect a GPU.")
        print("This usually means one of the following:")
        print("  1. The NVIDIA driver is not installed or requires an update.")
        print("  2. The NVIDIA Display service is not running.")
        print("  3. The GPU is disabled in Device Manager.")
        print("\nPlease try running 'nvidia-smi' from an Administrator Command Prompt.")

if __name__ == '__main__':
    main()
