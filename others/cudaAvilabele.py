import torch
print("PyTorch 版本:", torch.__version__)
print("编译 CUDA:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())
