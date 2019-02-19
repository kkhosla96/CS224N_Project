import sys
import torch

sys.path.append("..")

from CNN import CNN

batch_size = 32
in_channels = 1
height = 5
embed_size = 50

x = torch.zeros([batch_size, in_channels, height, embed_size])
cnn = CNN(height, embed_size)
f = cnn.forward(x)
print(f)
print(f.size())
