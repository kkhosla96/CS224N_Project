import pickle

import nltk, re, pprint
from nltk import word_tokenize
import re

chapter1_path = "../data/textbook_sentences/openstax_biology_chapter_1_sentences.txt"
chapter2_path = "../data/textbook_sentences/openstax_biology_chapter_2_sentences.txt"
chapter3_path = "../data/textbook_sentences/openstax_biology_chapter_3_sentences.txt"
outputFile = "../data/textbook_sentences/openstax_biology_chapters_123_sentences.txt"
    

 #classifier code for sentence delimiter prediction found here
 #https://www.nltk.org/book/ch06.html#sec-further-examples-of-supervised-classification 

def punct_features(tokens, i):
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
            'prev-word': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}

    
def segment_sentences(words, classifier):
    start = 0
    sents = []
    for i, word in enumerate(words):
        if word in '.?!' and i != len(words) - 1 and classifier.classify(punct_features(words, i)) == True:
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return [' '.join(x) for x in sents]


sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in sents:
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)

    
featuresets = [(punct_features(tokens, i), (i in boundaries))
                for i in range(1, len(tokens)-1)
                if tokens[i] in '.?!']

size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier, test_set))

with open(chapter1_path, "r") as ch1:
    with open(chapter2_path, "r") as ch2:
        with open(chapter3_path, "r") as ch3:
            data = (' '.join(ch1.readlines())).replace('\n', '') + (' '.join(ch2.readlines())).replace('\n', '') + (' '.join(ch3.readlines())).replace('\n', '')
            #print(data[0:500])
            #split_data = re.split('\.|!|\?',data)
            #print(split_data[0:10])
            split_data = re.split('(\W)', data)
            #print(split_data[:10])
            split_data_no_spaces = [x for x in split_data if x and x != " " and x != "  " and x != "   "]
            #print(split_data_no_spaces[:10])
            sentences = segment_sentences(split_data_no_spaces, classifier)
            #print(sentences[:10])

with open(outputFile, "w") as output:
    for s in sentences:
        output.write(s + "\n")








