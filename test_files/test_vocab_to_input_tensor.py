import sys
sys.path.append("..")
from WordVectorParser import WordVectorParser

test_file = "fasttext_pretrained_candidates.vec"

terms = [["carbohydrate"], ["cell", "cycle"], ["primary", "structure"], ["textbooks"], ["groups"]]

wvp = WordVectorParser(test_file)
vocab = wvp.get_vocab()
embeddings = wvp.get_embeddings()
embedding_layer = wvp.get_embedding_layer()

input_tensor = vocab.to_input_tensor(terms)
print(input_tensor)
print(embedding_layer(input_tensor))


