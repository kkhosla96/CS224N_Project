import sys
import torch

sys.path.append("..")

from CNN import CNN

batch_size = 32
in_channels = 1
height = 4
embed_size = 50

'''
can't test this until write the vocab class, since
the CNN will take in a list of list of strings. see
the CNN file for more details.

x = torch.zeros([batch_size, in_channels, height, embed_size])
cnn = CNN(height, embed_size)
f = cnn.forward(x)
print(f)
print(f.size())
'''