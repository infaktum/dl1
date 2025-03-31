import torch
import torchvision
import os
import platform

def device():
    return "cuda" if torch.cuda.is_available else "cpu"
    
def check_gpu():
    if torch.cuda.device_count() == 0:
        print("Keine GPU gefunden, verwende CPU")
        return 'cpu'
    else:
        print(f"GPU gefunden: {torch.cuda.get_device_name(0)}")
        return 'cuda'

        
def gpu_info():
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA devices count: {torch.cuda.device_count()}')
        print(f'CUDA current device: {torch.cuda.current_device()}')
        print(f'CUDA device 0: {torch.cuda.device(0)}')
        print(f'CUDA device name: {torch.cuda.get_device_name(0)}')


def sys_info():
    print(f'OS: {platform.system()}')
    print(f'OS version: {platform.version()}')
    print(f'CPU: {platform.processor()}')
    print(f'CPU count: {os.cpu_count()}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Torchvision version: {torchvision.__version__}')
    print(f'Python version: {platform.python_version()}')



if __name__ == '__main__':
    sys_info()
    gpu_info()


