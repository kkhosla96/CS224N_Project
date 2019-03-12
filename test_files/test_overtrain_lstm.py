import sys
import torch
import matplotlib.pyplot as plt

sys.path.append("..")

from LSTM import LSTM
from WordVectorParser import WordVectorParser


test_file = "fasttext_pretrained_candidates.vec"

wvp = WordVectorParser(test_file)
vocab = wvp.get_vocab()
embedding_layer = wvp.get_embedding_layer()

train_X = [["carbohydrate"], ["calories"], ["cell", "cycle"], ["primary", "structure"], ["structural", "isomers"], ["textbooks"], ["many", "tires"], ["many"], ["groups"], ["surround"]]
train_y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# train_X = [["carbohydrate"], ["cell", "cycle"], ["primary", "structure"], ["textbooks"], ["groups"]]
# train_y = [1, 1, 1, 0, 0]
# train_X = [["carbohydrate"], ["calories"], ["cell", "cycle"], ["many"], ["textbooks"]]
# train_y = [1, 1, 1, 0, 0]

lstm = LSTM(vocab=vocab, embeddings=embedding_layer)
losses = lstm.train_on_data(train_X, train_y, num_epochs=200, lr=.001, batch_size=2)
print(lstm.forward(train_X))
plt.plot(losses)
plt.show()


