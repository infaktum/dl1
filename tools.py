import torch

def device():
    return "cuda" if torch.cuda.is_available else "cpu"
    
def check_gpu(device = None):
    if torch.cuda.device_count() == 0:
        print("Keine GPU gefunden, verwende CPU")
        return 'cpu'
    else:
        print(f"GPU gefunden: {torch.cuda.get_device_name(0)}")
        return 'cuda'

        
def print_gpu_info():
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'CUDA devices count: {torch.cuda.device_count()}')
    print(f'CUDA current device: {torch.cuda.current_device()}')
    print(f'CUDA device 0: {torch.cuda.device(0)}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')

if __name__ == '__main__':
    check_gpu()


