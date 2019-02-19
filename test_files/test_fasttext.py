from gensim.models.wrappers import FastText

model = FastText.load_fasttext_format('../fastText-0.2.0/model.bin')

print(model.most_similar('hemoglobin'))
print(model.most_similar('carbohydrate'))
print(model.most_similar('keratin'))