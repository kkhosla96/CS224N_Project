import sys
import torch

sys.path.append("..")

from CNN import CNN
from WordVectorParser import WordVectorParser


test_file = "fasttext_pretrained_candidates.vec"

wvp = WordVectorParser(test_file)
vocab = wvp.get_vocab()
embedding_layer = wvp.get_embedding_layer()
print(embedding_layer.weight.type())

train_X = [["carbohydrate"], ["calories"], ["cell", "cycle"], ["textbooks"], ["many", "tires"]]
train_y = [1, 1, 1, 0, 0]

cnn = CNN(vocab, 3, embedding_layer)
cnn.train_on_data(train_X, train_y, num_epochs=3000)
print(cnn.forward(train_X))



