import sys
sys.path.append("..")
from WordVectorParser import WordVectorParser

test_file = "fasttext_pretrained_candidates.vec"

wvp = WordVectorParser(test_file)
vocab = wvp.get_vocab()
embeddings = wvp.get_embeddings()

my_word = "carbohydrate"
word_id = vocab[my_word]
embedding_for_word = embeddings[word_id]

print(my_word)
print(word_id)
print(embedding_for_word)
print(embeddings.size())