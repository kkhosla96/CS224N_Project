import sys
import torch
import matplotlib.pyplot as plt

sys.path.append("..")

from CNN import CNN
from WordVectorParser import WordVectorParser


test_file = "fasttext_pretrained_candidates.vec"

wvp = WordVectorParser(test_file)
vocab = wvp.get_vocab()
embedding_layer = wvp.get_embedding_layer()

train_X = [["carbohydrate"], ["calories"], ["cell", "cycle"], ["primary", "structure"], ["structural", "isomers"], ["textbooks"], ["many", "tires"], ["many"], ["groups"], ["surround"]]
train_y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# train_X = [["carbohydrate"], ["cell", "cycle"], ["primary", "structure"], ["textbooks"], ["groups"]]
# train_y = [1, 1, 1, 0, 0]

cnn = CNN(vocab, 3, embedding_layer)
losses = cnn.train_on_data(train_X, train_y, num_epochs=200, lr=.01, momentum=0.99)
print(cnn.forward(train_X))
plt.plot(losses)
plt.show()


