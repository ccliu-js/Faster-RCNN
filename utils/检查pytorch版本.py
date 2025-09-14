import torch
print("PyTorch version:", torch.__version__)
print("bianyi CUDA:", torch.version.cuda)  # 如果是 None 就说明没带 CUDA
