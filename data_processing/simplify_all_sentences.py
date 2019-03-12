import re
import sys
if ".." not in sys.path:
	sys.path.append("..")
from utils import normalize

dataFile = "../data/textbook_sentences/openstax_biology_chapters_123_sentences.txt"
outputFile = "../data/textbook_sentences/openstax_biology_chapters_123_sentences_simple.txt"


out = open(outputFile, 'w')

with open(dataFile) as f:
	line = f.readline()
	while line:
		#sentence = line.split("\t")[1]
		sentence = line.strip()
		cleaned = normalize(sentence)
		out.write(cleaned + "\n")
		line = f.readline()

out.close()
