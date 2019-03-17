import sys
import torch

sys.path.append("..")

from CNN import DeepCNN
from Vocab import Vocab

height = 4
embed_size = 50
number_words = 1000

embeddings = torch.nn.Embedding(number_words, embed_size)
print(embeddings.weight.type())
vocab = Vocab("sample_vocabulary.txt", 4)
cnn = DeepCNN(vocab, embeddings)
terms = [["this", "vocabulary"], ["it", "bitch"], ["hello"]]
f = cnn.forward(terms)
print(f)
print(f.size())
