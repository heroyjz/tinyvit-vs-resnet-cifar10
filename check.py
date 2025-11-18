import torch
from torchvision.models import resnet18


print("torch version:", getattr(torch, "__version__", None))
print("torch.version.cuda:", getattr(torch.version, "cuda", None))
print("cuda available:", torch.cuda.is_available() if "cuda" in dir(torch) else None)
print("cudnn available:", torch.backends.cudnn.is_available() if hasattr(torch.backends, "cudnn") else None)




m = resnet18(weights=None)   # 或 weights='IMAGENET1K_V1'
print(m)
