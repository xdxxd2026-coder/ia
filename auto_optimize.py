# auto_optimize.py
import subprocess
import sys

def optimize_system():
    """Otimiza o sistema para performance máxima"""
    
    optimizations = {
        "memory": "sudo sysctl -w vm.swappiness=10",
        "gpu": "nvidia-smi --auto-boost-default=0",
        "cuda": "export CUDA_LAUNCH_BLOCKING=1"
    }
    
    for name, cmd in optimizations.items():
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✓ Otimização {name} aplicada")
        except:
            print(f"✗ Falha na otimização {name}")
    
    # Configurações específicas do PyTorch
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    print("Sistema otimizado para NeuroGenesis AI")