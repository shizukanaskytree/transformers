import torch

size1 = torch.Size([3, 4])
size2 = torch.Size([3, 4])
size3 = torch.Size([2, 5])

if size1 == size2:
    print("size1 and size2 are the same.")

if size1 == size3:
    print("size1 and size3 are the same.")
else:
    print("size1 and size3 are different.")
