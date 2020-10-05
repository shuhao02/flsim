import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def trans_torch(x):
    x = torch.FloatTensor(x).t()
    return x
x = (torch.FloatTensor([[1,2,3],[4,5,6]]))
x_ = trans_torch(x)
print(x_)
# A = [1,2,3,4]
# B = [(a, "2") for i, a in enumerate(A)]
# C = [client for client, _ in [B[0]]]
# print(C)