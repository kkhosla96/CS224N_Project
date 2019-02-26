import sys
import torch.nn as nn

sys.path.append("..")

from LSTM import LSTM
from Vocab import Vocab

height = 4
embed_size = 50
number_words = 1000


vocab = Vocab("sample_vocabulary.txt", 3)
embeddings = nn.Embedding(number_words, embed_size, padding_idx=vocab["<pad>"])
LSTM = LSTM(vocab=vocab, embeddings=embeddings, bidirectional=True)
terms = [["this", "vocabulary"], ["it", "hello", "this"], ["i"], ["i", "hope", "you", "appreciate"]]
f = LSTM(terms)
print(f)
print(f.size())
