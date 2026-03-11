import sys

def check_environment():
    print("="*40)
    print("   MEDICALNET REQUIREMENTS DIAGNOSTIC   ")
    print("="*40)

    # 1. Check Python Version
    py_version = sys.version.split()[0]
    print(f"[Python] Current: {py_version} | Required: 3.7.0")

    # 2. Check PyTorch, CUDA, and cuDNN
    try:
        import torch
        print(f"[PyTorch] Current: {torch.__version__} | Required: 0.4.1")
        
        if torch.cuda.is_available():
            print(f"[CUDA] Current: {torch.version.cuda} | Required: 9.0")
            print(f"[cuDNN] Current: {torch.backends.cudnn.version()} | Required: 7.0.5")
            print("-" * 40)
            print(f"GPUs Detected: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("[!] CUDA is NOT available. PyTorch cannot see your GPUs.")
            
    except ImportError:
        print("[!] PyTorch is not installed in this environment.")

    print("="*40)

if __name__ == "__main__":
    check_environment()